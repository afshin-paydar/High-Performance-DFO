/*
 * High-Performance DFO/uDFO Kernels
 *
 * Based on original DFO by Mohammad Majid al-Rifaie
 * al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on
 * Computer Science and Information Systems (pp. 529-538). IEEE.
 *
 * Design notes:
 * - Positions, fitness, bounds: double everywhere
 * - curand_uniform_double for all position/fitness-affecting RNG draws
 * - uDFO exploitation-probability computation stays float32 (O(1) values,
 *   float32 precision is adequate for the threshold comparison)
 * - Fitness reductions use clean __syncthreads() all the way down (no warp unroll)
 *   to avoid out-of-bounds shared memory reads when blockDim.x == 32.
 * - Global-best search uses a two-phase reduction so N > 1024 is handled correctly:
 *     Phase 1 (kernelReduceToBlockBest): each block of DFO_BLOCK_SIZE particles
 *             reduces to one (fitness, index) pair.
 *     Phase 2 (kernelReduceFinalBest):  single block reduces the phase-1 results.
 * - Population update uses Gauss-Seidel (sequential fly loop, single block):
 *   D threads handle all dimensions of each fly in parallel; __syncthreads()
 *   after each fly ensures fly i+1 sees fly i's written positions, exactly
 *   matching the Python reference sweep order.
 * - atomicAdd(double*) requires SM 6.0+; GTX 1080 Ti is SM 6.1 ✓
 */

#ifndef DFO_KERNELS_CUH
#define DFO_KERNELS_CUH

#include "dfo_common.cuh"

// ---------------------------------------------------------------------------
// Device helper: double atomic min (CAS loop, works for any double value)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        if (__longlong_as_double((long long)assumed) <= val) break;
        old = atomicCAS(addr_ull, assumed,
                        (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// Device helper: double atomic max
// ---------------------------------------------------------------------------
__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        if (__longlong_as_double((long long)assumed) >= val) break;
        old = atomicCAS(addr_ull, assumed,
                        (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
}

//=============================================================================
// Kernel: Initialize RNG states
//=============================================================================
__global__ void kernelInitRNG(
    curandState* __restrict__ states,
    unsigned long long seed,
    int totalStates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalStates) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

//=============================================================================
// Kernel: Initialize population randomly within bounds (double positions)
//=============================================================================
__global__ void kernelInitPopulation(
    double* __restrict__ positions,
    curandState* __restrict__ rngStates,
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
        int dimIdx = idx % D;
        double lower = d_lowerBounds[dimIdx];
        double upper = d_upperBounds[dimIdx];

        curandState localState = rngStates[idx];
        double r = curand_uniform_double(&localState);
        rngStates[idx] = localState;

        positions[idx] = lower + r * (upper - lower);
    }
}

//=============================================================================
// Fitness evaluation kernels
//
// Layout: one block per particle (blockIdx.x == particle index).
// Threads reduce over dimensions using shared memory.
// Reduction uses a clean power-of-2 loop with __syncthreads() at every step
// to avoid out-of-bounds shared memory reads in the warp tail.
//=============================================================================

// ---------------------------------------------------------------------------
// Sphere: f(x) = sum(x_i^2)
// ---------------------------------------------------------------------------
__global__ void kernelEvaluateFitnessSphere(
    const double* __restrict__ positions,
    double*       __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ double sdata[];

    int particleIdx = blockIdx.x;
    int tid         = threadIdx.x;
    if (particleIdx >= N) return;

    double sum = 0.0;
    for (int d = tid; d < D; d += blockDim.x) {
        double x = positions[particleIdx * D + d];
        sum += x * x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) fitness[particleIdx] = sdata[0];
}

// ---------------------------------------------------------------------------
// Rastrigin: f(x) = 10D + sum(x_i^2 - 10*cos(2*pi*x_i))
// ---------------------------------------------------------------------------
__global__ void kernelEvaluateFitnessRastrigin(
    const double* __restrict__ positions,
    double*       __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ double sdata[];

    int particleIdx = blockIdx.x;
    int tid         = threadIdx.x;
    if (particleIdx >= N) return;

    const double TWO_PI = 6.28318530717958647692;

    double sum = 0.0;
    for (int d = tid; d < D; d += blockDim.x) {
        double x = positions[particleIdx * D + d];
        sum += x * x - 10.0 * cos(TWO_PI * x) + 10.0;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) fitness[particleIdx] = sdata[0];
}

// ---------------------------------------------------------------------------
// Rosenbrock: f(x) = sum_{i=0}^{D-2}( 100*(x[i+1]-x[i]^2)^2 + (x[i]-1)^2 )
// ---------------------------------------------------------------------------
__global__ void kernelEvaluateFitnessRosenbrock(
    const double* __restrict__ positions,
    double*       __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ double sdata[];

    int particleIdx = blockIdx.x;
    int tid         = threadIdx.x;
    if (particleIdx >= N) return;

    const double* x = positions + particleIdx * D;
    double sum = 0.0;
    for (int d = tid; d < D - 1; d += blockDim.x) {
        double xi  = x[d];
        double xi1 = x[d + 1];
        double t1  = xi1 - xi * xi;
        double t2  = xi - 1.0;
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) fitness[particleIdx] = sdata[0];
}

// ---------------------------------------------------------------------------
// Ackley: f(x) = -20*exp(-0.2*sqrt(sum(x^2)/D)) - exp(sum(cos(2pi*x))/D) + 20+e
// Two shared-memory arrays: one for sum-of-squares, one for sum-of-cosines.
// ---------------------------------------------------------------------------
__global__ void kernelEvaluateFitnessAckley(
    const double* __restrict__ positions,
    double*       __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ double sdata[];
    double* sdata_sq  = sdata;
    double* sdata_cos = sdata + blockDim.x;

    int particleIdx = blockIdx.x;
    int tid         = threadIdx.x;
    if (particleIdx >= N) return;

    const double TWO_PI = 6.28318530717958647692;

    double sum_sq = 0.0, sum_cos = 0.0;
    for (int d = tid; d < D; d += blockDim.x) {
        double x = positions[particleIdx * D + d];
        sum_sq  += x * x;
        sum_cos += cos(TWO_PI * x);
    }
    sdata_sq[tid]  = sum_sq;
    sdata_cos[tid] = sum_cos;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_sq[tid]  += sdata_sq[tid  + s];
            sdata_cos[tid] += sdata_cos[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        double val = -20.0 * exp(-0.2 * sqrt(sdata_sq[0] / D))
                     - exp(sdata_cos[0] / D)
                     + 20.0 + 2.718281828459045235360287;
        fitness[particleIdx] = val;
    }
}

//=============================================================================
// Two-phase global-best search — handles any N, no single-block limitation.
//
// Phase 1 (kernelReduceToBlockBest):
//   Launch ceil(N / DFO_BLOCK_SIZE) blocks of DFO_BLOCK_SIZE threads each.
//   Each block reduces its DFO_BLOCK_SIZE particles to one (fitness, index) pair
//   stored in partialFitness[blockIdx.x] / partialIdx[blockIdx.x].
//
// Phase 2 (kernelReduceFinalBest):
//   Single block reduces the ceil(N/DFO_BLOCK_SIZE) partial results to the
//   global best and writes directly to device scalars globalBestIdx /
//   globalBestFitness (no atomicExch needed — single block, thread 0 writes).
//=============================================================================

__global__ void kernelReduceToBlockBest(
    const double* __restrict__ fitness,
    double*       __restrict__ partialFitness,
    int*          __restrict__ partialIdx,
    int  N,
    bool minimize
) {
    extern __shared__ double sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double myVal = minimize ? DBL_MAX : -DBL_MAX;
    int    myIdx = -1;

    if (gid < N) {
        myVal = fitness[gid];
        myIdx = gid;
    }

    sdata[tid] = myVal;
    sidx[tid]  = myIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool cond = minimize ? (sdata[tid + s] < sdata[tid])
                                 : (sdata[tid + s] > sdata[tid]);
            if (cond) {
                sdata[tid] = sdata[tid + s];
                sidx[tid]  = sidx[tid  + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialFitness[blockIdx.x] = sdata[0];
        partialIdx[blockIdx.x]     = sidx[0];
    }
}

__global__ void kernelReduceFinalBest(
    const double* __restrict__ partialFitness,
    const int*    __restrict__ partialIdx,
    int*          __restrict__ globalBestIdx,
    double*       __restrict__ globalBestFitness,
    int   numPartials,
    bool  minimize
) {
    extern __shared__ double sdata[];
    int* sidx = (int*)&sdata[blockDim.x];

    int tid = threadIdx.x;

    double myVal = (tid < numPartials) ? partialFitness[tid]
                                       : (minimize ? DBL_MAX : -DBL_MAX);
    int    myIdx = (tid < numPartials) ? partialIdx[tid] : -1;

    sdata[tid] = myVal;
    sidx[tid]  = myIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            bool cond = minimize ? (sdata[tid + s] < sdata[tid])
                                 : (sdata[tid + s] > sdata[tid]);
            if (cond) {
                sdata[tid] = sdata[tid + s];
                sidx[tid]  = sidx[tid  + s];
            }
        }
        __syncthreads();
    }

    // Single block → thread 0 writes directly; no atomicExch needed.
    if (tid == 0) {
        *globalBestFitness = sdata[0];
        *globalBestIdx     = sidx[0];
    }
}

//=============================================================================
// Kernel: Find best neighbor for each particle (ring topology)
//=============================================================================
__global__ void kernelFindBestNeighbors(
    const double* __restrict__ fitness,
    int*          __restrict__ bestNeighborIdx,
    int  N,
    bool minimize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int left  = (idx - 1 + N) % N;
        int right = (idx + 1) % N;

        double leftFit  = fitness[left];
        double rightFit = fitness[right];

        // Strict '<' (left wins ties) matches Python:
        //   bNeighbour = right if fitness[right]<fitness[left] else left
        bool selectRight = minimize ? (rightFit < leftFit) : (rightFit > leftFit);
        bestNeighborIdx[idx] = selectRight ? right : left;
    }
}

//=============================================================================
// Kernel: Standard DFO position update — Gauss-Seidel (exact Python match)
//
// Processes flies sequentially i=0..N-1 inside a single block.
// D threads handle all dimensions of fly i in parallel.
//
// __syncthreads() is called UNCONDITIONALLY at the end of every loop
// iteration: it is a full execution and memory barrier for all threads in the
// block, ensuring every thread has committed its write to global memory before
// any thread proceeds to the next fly.  This guarantees fly i+1 sees fly i's
// written positions — exactly reproducing Python's in-order sweep.
//
// Gauss-Seidel asymmetry (matches Python):
//   - Left neighbour (index i-1): already updated in the current sweep
//     → fly i reads the freshly written position of i-1.
//   - Right neighbour (index i+1): not yet updated in the current sweep
//     → fly i reads the old position of i+1.
//   - Global best (globalBestIdx): never written (elitist skip)
//     → fly i always reads the pre-iteration best position.
//
// Requires D <= DFO_MAX_DIMS (currently 1024). The single-block design caps
// blockSize at 1024; if DFO_MAX_DIMS were raised above 1024 without switching
// to a multi-block approach, dimensions >= 1024 would silently not be updated.
//
// positions is NOT __restrict__ (same array is read and written in-place).
//
// Launch config: <<<1, nextPow2(D)>>>
//=============================================================================
__global__ void kernelUpdateDFO_GaussSeidel(
    double*      positions,
    const int*   __restrict__ bestNeighborIdx,
    curandState* __restrict__ rngStates,
    int    globalBestIdx,
    double delta,
    int    N,
    int    D
) {
    int dimIdx = threadIdx.x;

    for (int i = 0; i < N; i++) {
        if (i != globalBestIdx && dimIdx < D) {
            int    idx   = i * D + dimIdx;
            double lower = d_lowerBounds[dimIdx];
            double upper = d_upperBounds[dimIdx];

            curandState localState = rngStates[idx];

            int    neighborIdx = bestNeighborIdx[i];
            double x_neighbor  = positions[neighborIdx * D + dimIdx];
            double x_best      = positions[globalBestIdx * D + dimIdx];
            double x_current   = positions[idx];

            double r = curand_uniform_double(&localState);
            double x_new;
            if (r < delta) {
                double r2 = curand_uniform_double(&localState);
                x_new = lower + r2 * (upper - lower);
            } else {
                double u = curand_uniform_double(&localState);
                x_new = x_neighbor + u * (x_best - x_current);
                if (x_new < lower || x_new > upper) {
                    double r3 = curand_uniform_double(&localState);
                    x_new = lower + r3 * (upper - lower);
                }
            }

            rngStates[idx] = localState;
            positions[idx] = x_new;
        }
        __syncthreads();   // unconditional — all threads barrier here every iteration
    }
}

//=============================================================================
// Kernel: uDFO position update — Gauss-Seidel (all variants)
//
// Same Gauss-Seidel structure as kernelUpdateDFO_GaussSeidel: sequential
// loop over flies, D threads per fly, unconditional __syncthreads() at the
// bottom of every iteration.
//
// Gauss-Seidel asymmetry (matches Python):
//   - Left neighbour (index i-1): already updated in the current sweep.
//   - Right neighbour (index i+1): not yet updated.
//   - Global best (globalBestIdx): never written (elitist skip).
//
// The exploitation-probability computation uses float32 (O(1) scaled values,
// float32 precision is adequate for the threshold comparison).
//
// Requires D <= DFO_MAX_DIMS (1024). positions is NOT __restrict__.
// variant must be UDFO_1000P, UDFO_1500P, or UDFO_Z5 (not STANDARD).
//
// Launch config: <<<1, nextPow2(D)>>>
//=============================================================================
__global__ void kernelUpdateUDFO_GaussSeidel(
    double*      positions,
    const int*   __restrict__ bestNeighborIdx,
    curandState* __restrict__ rngStates,
    int        globalBestIdx,
    int        N,
    int        D,
    DFOVariant variant
) {
    int dimIdx = threadIdx.x;

    for (int i = 0; i < N; i++) {
        if (i != globalBestIdx && dimIdx < D) {
            int    idx   = i * D + dimIdx;
            double lower = d_lowerBounds[dimIdx];
            double upper = d_upperBounds[dimIdx];

            curandState localState = rngStates[idx];

            int    neighborIdx = bestNeighborIdx[i];
            double x_neighbor  = positions[neighborIdx * D + dimIdx];
            double x_best      = positions[globalBestIdx * D + dimIdx];
            double x_current   = positions[idx];

            // Compute exploitation probability p in float32
            double scale = fabs(x_neighbor - x_best);
            float  p;
            if (scale < 1e-10) {
                p = 1.0f;
            } else {
                float x_scaled     = (float)((x_current - x_best) / scale);
                float lower_scaled = (float)((lower     - x_best) / scale);
                float upper_scaled = (float)((upper     - x_best) / scale);
                float L = fabsf(x_scaled - lower_scaled);
                float R = fabsf(upper_scaled - x_scaled);
                p = calcExploitationProb(x_scaled, L, R);
            }
            float  dynamicDelta = calcDynamicDelta(p, variant);
            double r = curand_uniform_double(&localState);

            double x_new;
            if (r < (double)dynamicDelta) {
                if (variant == DFOVariant::UDFO_Z5) {
                    double z5_lower, z5_upper;
                    // x_neighbor reflects Gauss-Seidel ordering: may be already-updated in
                    // current sweep (left neighbour) or pre-sweep (right neighbour).
                    if (x_neighbor >= x_best) { z5_lower = x_neighbor; z5_upper = upper;      }
                    else                       { z5_lower = lower;      z5_upper = x_neighbor; }
                    double r2 = curand_uniform_double(&localState);
                    x_new = (z5_lower < z5_upper)
                            ? z5_lower + r2 * (z5_upper - z5_lower)
                            : lower    + r2 * (upper    - lower);
                } else {
                    double r2 = curand_uniform_double(&localState);
                    x_new = lower + r2 * (upper - lower);
                }
            } else {
                double u = curand_uniform_double(&localState);
                x_new = x_neighbor + u * (x_best - x_current);
                if (x_new < lower || x_new > upper) {
                    double r3 = curand_uniform_double(&localState);
                    x_new = lower + r3 * (upper - lower);
                }
            }

            rngStates[idx] = localState;
            positions[idx] = x_new;
        }
        __syncthreads();   // unconditional barrier — all threads every iteration
    }
}

//=============================================================================
// Kernel: Copy best solution to output buffer
//=============================================================================
__global__ void kernelCopyBestSolution(
    const double* __restrict__ positions,
    double*       __restrict__ bestPosition,
    int globalBestIdx,
    int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < D) {
        bestPosition[tid] = positions[globalBestIdx * D + tid];
    }
}

//=============================================================================
// DEBUG Kernel: Per-iteration population health statistics
//
// Launched with N threads (one per particle).
//
// d_out layout (double[8]):
//   [0]  min fitness    (verified independently of kernelReduceFinalBest)
//   [1]  max fitness
//   [2]  sum |pos[i,0] - pos[best,0]|  (dim-0 spread proxy)
//   [3]  count of "fully collapsed" particles (position indistinguishable
//        from best in all D dims within 10*DBL_EPSILON)
//   [4]  count of particles with bit-identical fitness to best
//   [5]  sum |x_best[d] - x_i[d]| over all (i,d)  — total movement potential
//        → approaches 0 when the update formula is dead
//   [6]  count of (i,d) pairs where |x_best[d]-x_i[d]| < 1e-14
//        (float64 dead zone; should be near 0 after convergence)
//
// Min/max use a double CAS loop.
// Sums/counts use atomicAdd(double*) — supported on SM 6.0+ (GTX 1080 Ti ✓).
//=============================================================================
__global__ void kernelComputeDebugStats(
    const double* __restrict__ positions,
    const double* __restrict__ fitness,
    double*       __restrict__ d_out,   // double[8], caller pre-initialises [0]=DBL_MAX, [1]=-DBL_MAX
    int globalBestIdx,
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double fit_i = fitness[i];

    // [0] min fitness
    atomicMinDouble(&d_out[0], fit_i);

    // [1] max fitness
    atomicMaxDouble(&d_out[1], fit_i);

    // [2] sum |pos[i,0] - pos[best,0]|
    double diff0 = fabs(positions[i * D] - positions[globalBestIdx * D]);
    atomicAdd(&d_out[2], diff0);

    // [3] collapsed count, [5] movement potential, [6] dead-zone count
    bool   collapsed       = true;
    double total_move_pot  = 0.0;
    double dead_zone_count = 0.0;
    for (int d = 0; d < D; d++) {
        double xb    = positions[globalBestIdx * D + d];
        double xi    = positions[i * D + d];
        double adiff = fabs(xb - xi);
        total_move_pot += adiff;
        if (adiff < 1e-14) dead_zone_count += 1.0;
        double thresh = fmax(1e-14, 10.0 * DBL_EPSILON * fabs(xb));
        if (adiff >= thresh) collapsed = false;
    }
    if (collapsed) atomicAdd(&d_out[3], 1.0);

    // [4] exact fitness match (bit-identical double)
    if (fit_i == fitness[globalBestIdx]) atomicAdd(&d_out[4], 1.0);

    atomicAdd(&d_out[5], total_move_pot);
    atomicAdd(&d_out[6], dead_zone_count);
}

#endif // DFO_KERNELS_CUH
