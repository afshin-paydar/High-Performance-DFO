/*
 * High-Performance DFO/uDFO Kernels
 *
 * Based on original DFO by Mohammad Majid al-Rifaie
 * al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.
 * Design notes:
 * - Positions, fitness, bounds: double everywhere
 * - curand_uniform_double for all position/fitness-affecting RNG draws
 * - uDFO exploitation-probability computation stays float32 (O(1) values,
 *   float32 precision is adequate for the threshold comparison)
 * - Warp reductions use volatile double* + __syncwarp() (required SM 6.1+)
 * - kernelFindGlobalBest uses a single block → direct store from thread 0,
 *   no atomicExch needed (no double atomicExch exists in CUDA)
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
// Warp unroll uses volatile double* + __syncwarp() for correctness on SM 6.1+.
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

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile double* vs = sdata;
        if (blockDim.x >= 64) vs[tid] += vs[tid + 32]; __syncwarp();
        if (blockDim.x >= 32) vs[tid] += vs[tid + 16]; __syncwarp();
        if (blockDim.x >= 16) vs[tid] += vs[tid +  8]; __syncwarp();
        if (blockDim.x >=  8) vs[tid] += vs[tid +  4]; __syncwarp();
        if (blockDim.x >=  4) vs[tid] += vs[tid +  2]; __syncwarp();
        if (blockDim.x >=  2) vs[tid] += vs[tid +  1]; __syncwarp();
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

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile double* vs = sdata;
        if (blockDim.x >= 64) vs[tid] += vs[tid + 32]; __syncwarp();
        if (blockDim.x >= 32) vs[tid] += vs[tid + 16]; __syncwarp();
        if (blockDim.x >= 16) vs[tid] += vs[tid +  8]; __syncwarp();
        if (blockDim.x >=  8) vs[tid] += vs[tid +  4]; __syncwarp();
        if (blockDim.x >=  4) vs[tid] += vs[tid +  2]; __syncwarp();
        if (blockDim.x >=  2) vs[tid] += vs[tid +  1]; __syncwarp();
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

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile double* vs = sdata;
        if (blockDim.x >= 64) vs[tid] += vs[tid + 32]; __syncwarp();
        if (blockDim.x >= 32) vs[tid] += vs[tid + 16]; __syncwarp();
        if (blockDim.x >= 16) vs[tid] += vs[tid +  8]; __syncwarp();
        if (blockDim.x >=  8) vs[tid] += vs[tid +  4]; __syncwarp();
        if (blockDim.x >=  4) vs[tid] += vs[tid +  2]; __syncwarp();
        if (blockDim.x >=  2) vs[tid] += vs[tid +  1]; __syncwarp();
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

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata_sq[tid]  += sdata_sq[tid  + s];
            sdata_cos[tid] += sdata_cos[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile double* vsq  = sdata_sq;
        volatile double* vcos = sdata_cos;
        if (blockDim.x >= 64) { vsq[tid] += vsq[tid+32]; vcos[tid] += vcos[tid+32]; } __syncwarp();
        if (blockDim.x >= 32) { vsq[tid] += vsq[tid+16]; vcos[tid] += vcos[tid+16]; } __syncwarp();
        if (blockDim.x >= 16) { vsq[tid] += vsq[tid+ 8]; vcos[tid] += vcos[tid+ 8]; } __syncwarp();
        if (blockDim.x >=  8) { vsq[tid] += vsq[tid+ 4]; vcos[tid] += vcos[tid+ 4]; } __syncwarp();
        if (blockDim.x >=  4) { vsq[tid] += vsq[tid+ 2]; vcos[tid] += vcos[tid+ 2]; } __syncwarp();
        if (blockDim.x >=  2) { vsq[tid] += vsq[tid+ 1]; vcos[tid] += vcos[tid+ 1]; } __syncwarp();
    }
    if (tid == 0) {
        double val = -20.0 * exp(-0.2 * sqrt(sdata_sq[0] / D))
                     - exp(sdata_cos[0] / D)
                     + 20.0 + 2.718281828459045235360287;
        fitness[particleIdx] = val;
    }
}

//=============================================================================
// Kernel: Find global best particle (single block, minimize or maximize)
//
// Uses shared memory reduction.  Always launched with numBlocks=1, so no
// inter-block communication is needed — thread 0 writes directly to device
// scalars (no atomicExch, which has no double overload in CUDA).
//
// __syncwarp() between volatile warp-unroll steps required on SM 6.1+.
//=============================================================================
__global__ void kernelFindGlobalBest(
    const double* __restrict__ fitness,
    int*          __restrict__ globalBestIdx,
    double*       __restrict__ globalBestFitness,
    int  N,
    bool minimize
) {
    extern __shared__ double sdata[];
    int* sidx = (int*)&sdata[blockDim.x];  // packed after the double array

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double myVal = minimize ? DBL_MAX : -DBL_MAX;
    int    myIdx = -1;

    if (idx < N) {
        myVal = fitness[idx];
        myIdx = idx;
    }

    sdata[tid] = myVal;
    sidx[tid]  = myIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
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

    if (tid < 32) {
        volatile double* vs  = sdata;
        volatile int*    vsi = sidx;

        #define WARP_REDUCE_STEP(offset) \
            if (tid < (offset)) { \
                bool _c = minimize ? (vs[tid + (offset)] < vs[tid]) \
                                   : (vs[tid + (offset)] > vs[tid]); \
                if (_c) { vs[tid] = vs[tid + (offset)]; vsi[tid] = vsi[tid + (offset)]; } \
            } \
            __syncwarp();

        WARP_REDUCE_STEP(32)
        WARP_REDUCE_STEP(16)
        WARP_REDUCE_STEP(8)
        WARP_REDUCE_STEP(4)
        WARP_REDUCE_STEP(2)
        WARP_REDUCE_STEP(1)

        #undef WARP_REDUCE_STEP
    }

    // Single block → only thread 0 writes; direct store is safe.
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
// Kernel: Standard DFO position update (double precision)
//
// Gauss-Seidel ordering: 1 block, D threads, flies 0..N-1 in sequence.
// __syncthreads() after each fly guarantees all D writes are globally visible
// before the next fly reads its (potentially just-updated) neighbour — this
// exactly replicates the Python sequential for-loop behaviour.
//=============================================================================
__global__ void kernelUpdateDFO(
    double*       __restrict__ positions,
    const double* __restrict__ fitness,        // unused but kept for signature symmetry
    const int*    __restrict__ bestNeighborIdx,
    curandState*  __restrict__ rngStates,
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

            double r = curand_uniform_double(&localState);
            if (r < delta) {
                // Disturbance: reinitialise uniformly in bounds
                double r2 = curand_uniform_double(&localState);
                positions[idx] = lower + r2 * (upper - lower);
            } else {
                int    neighborIdx = bestNeighborIdx[i];
                double x_neighbor  = positions[neighborIdx * D + dimIdx];
                double x_best      = positions[globalBestIdx * D + dimIdx];
                double x_current   = positions[idx];  // read before write

                double u     = curand_uniform_double(&localState);
                double x_new = x_neighbor + u * (x_best - x_current);

                if (x_new < lower || x_new > upper) {
                    double r3 = curand_uniform_double(&localState);
                    x_new = lower + r3 * (upper - lower);
                }
                positions[idx] = x_new;
            }

            rngStates[idx] = localState;
        }
        __syncthreads();  // Gauss-Seidel barrier: fly i fully written before fly i+1 reads
    }
}

//=============================================================================
// Kernel: uDFO position update with dynamic delta (double precision)
//
// The exploitation-probability computation (calcExploitationProb / calcDynamicDelta)
// uses float32 internally — those values are O(1) probabilities and float32
// precision is fully adequate for the threshold comparison.
//=============================================================================
__global__ void kernelUpdateUDFO(
    double*       __restrict__ positions,
    const double* __restrict__ fitness,
    const int*    __restrict__ bestNeighborIdx,
    curandState*  __restrict__ rngStates,
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

            if (r < (double)dynamicDelta) {
                if (variant == DFOVariant::UDFO_Z5) {
                    double z5_lower, z5_upper;
                    if (x_neighbor >= x_best) { z5_lower = x_neighbor; z5_upper = upper;      }
                    else                       { z5_lower = lower;      z5_upper = x_neighbor; }

                    double r2 = curand_uniform_double(&localState);
                    positions[idx] = (z5_lower < z5_upper)
                                     ? z5_lower + r2 * (z5_upper - z5_lower)
                                     : lower    + r2 * (upper    - lower);
                } else {
                    double r2 = curand_uniform_double(&localState);
                    positions[idx] = lower + r2 * (upper - lower);
                }
            } else {
                double u     = curand_uniform_double(&localState);
                double x_new = x_neighbor + u * (x_best - x_current);

                if (x_new < lower || x_new > upper) {
                    double r3 = curand_uniform_double(&localState);
                    x_new = lower + r3 * (upper - lower);
                }
                positions[idx] = x_new;
            }

            rngStates[idx] = localState;
        }
        __syncthreads();
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
//   [0]  min fitness    (verified independently of kernelFindGlobalBest)
//   [1]  max fitness
//   [2]  sum |pos[i,0] - pos[best,0]|  (dim-0 spread proxy)
//   [3]  count of "fully collapsed" particles (position indistinguishable
//        from best in all D dims within 10*DBL_EPSILON)
//   [4]  count of particles with bit-identical fitness to best
//   [5]  sum |x_best[d] - x_i[d]| over all (i,d)  — total movement potential
//        → approaches 0 when the update formula is dead
//   [6]  count of (i,d) pairs where |x_best[d]-x_i[d]| < 1e-14
//        (float64 dead zone; should be near 0 after conversion)
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
