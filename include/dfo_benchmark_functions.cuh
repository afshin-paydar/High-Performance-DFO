/*
 * GPU Benchmark Functions for DFO/uDFO
 *
 * Contains CUDA device implementations of standard optimization benchmarks
 */

#ifndef DFO_BENCHMARK_FUNCTIONS_CUH
#define DFO_BENCHMARK_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cmath>

// Constants
#define DFO_PI 3.14159265358979323846f
#define DFO_E  2.71828182845904523536f

//=============================================================================
// Unimodal Functions
//=============================================================================

// F1: Sphere Function
// f(x) = sum(x_i^2)
// Global minimum: f(0,...,0) = 0
// Bounds: [-5.12, 5.12]^D
__device__ __forceinline__ float benchmarkSphere(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

// F2: Elliptic Function
// f(x) = sum(10^(6*(i-1)/(D-1)) * x_i^2)
// Global minimum: f(0,...,0) = 0
// Bounds: [-100, 100]^D
__device__ __forceinline__ float benchmarkElliptic(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float coef = powf(10.0f, 6.0f * i / (D - 1));
        sum += coef * x[i] * x[i];
    }
    return sum;
}

// F3: Schwefel 1.2
// f(x) = sum(sum(x_j, j<=i)^2, i=1..D)
// Global minimum: f(0,...,0) = 0
// Bounds: [-100, 100]^D
__device__ __forceinline__ float benchmarkSchwefel12(const float* x, int D) {
    float sum = 0.0f;
    float inner = 0.0f;
    for (int i = 0; i < D; i++) {
        inner += x[i];
        sum += inner * inner;
    }
    return sum;
}

// F4: Schwefel 2.21
// f(x) = max(|x_i|)
// Global minimum: f(0,...,0) = 0
// Bounds: [-100, 100]^D
__device__ __forceinline__ float benchmarkSchwefel221(const float* x, int D) {
    float maxVal = fabsf(x[0]);
    for (int i = 1; i < D; i++) {
        maxVal = fmaxf(maxVal, fabsf(x[i]));
    }
    return maxVal;
}

// F5: Schwefel 2.22
// f(x) = sum(|x_i|) + prod(|x_i|)
// Global minimum: f(0,...,0) = 0
// Bounds: [-10, 10]^D
__device__ __forceinline__ float benchmarkSchwefel222(const float* x, int D) {
    float sum = 0.0f;
    float prod = 1.0f;
    for (int i = 0; i < D; i++) {
        float absX = fabsf(x[i]);
        sum += absX;
        prod *= absX;
    }
    return sum + prod;
}

// F6: Quadric
// f(x) = sum((sum(x_j, j<=i))^2)
// Global minimum: f(0,...,0) = 0
// Bounds: [-100, 100]^D
__device__ __forceinline__ float benchmarkQuadric(const float* x, int D) {
    float sum = 0.0f;
    float inner = 0.0f;
    for (int i = 0; i < D; i++) {
        inner += x[i];
        sum += inner * inner;
    }
    return sum;
}

//=============================================================================
// Multimodal Functions
//=============================================================================

// F7: Rastrigin Function
// f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))
// Global minimum: f(0,...,0) = 0
// Bounds: [-5.12, 5.12]^D
__device__ __forceinline__ float benchmarkRastrigin(const float* x, int D) {
    float sum = 10.0f * D;
    for (int i = 0; i < D; i++) {
        sum += x[i] * x[i] - 10.0f * cosf(2.0f * DFO_PI * x[i]);
    }
    return sum;
}

// F8: Ackley Function
// f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/D)) - exp(sum(cos(2*pi*x_i))/D) + 20 + e
// Global minimum: f(0,...,0) = 0
// Bounds: [-32, 32]^D
__device__ __forceinline__ float benchmarkAckley(const float* x, int D) {
    float sumSq = 0.0f;
    float sumCos = 0.0f;
    for (int i = 0; i < D; i++) {
        sumSq += x[i] * x[i];
        sumCos += cosf(2.0f * DFO_PI * x[i]);
    }
    return -20.0f * expf(-0.2f * sqrtf(sumSq / D))
           - expf(sumCos / D) + 20.0f + DFO_E;
}

// F9: Griewank Function
// f(x) = 1 + sum(x_i^2/4000) - prod(cos(x_i/sqrt(i+1)))
// Global minimum: f(0,...,0) = 0
// Bounds: [-600, 600]^D
__device__ __forceinline__ float benchmarkGriewank(const float* x, int D) {
    float sum = 0.0f;
    float prod = 1.0f;
    for (int i = 0; i < D; i++) {
        sum += x[i] * x[i] / 4000.0f;
        prod *= cosf(x[i] / sqrtf(i + 1.0f));
    }
    return 1.0f + sum - prod;
}

// F10: Rosenbrock Function
// f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
// Global minimum: f(1,...,1) = 0
// Bounds: [-5, 10]^D
__device__ __forceinline__ float benchmarkRosenbrock(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D - 1; i++) {
        float t1 = x[i + 1] - x[i] * x[i];
        float t2 = 1.0f - x[i];
        sum += 100.0f * t1 * t1 + t2 * t2;
    }
    return sum;
}

// F11: Schwefel Function
// f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|)))
// Global minimum: f(420.9687,...,420.9687) = 0
// Bounds: [-500, 500]^D
__device__ __forceinline__ float benchmarkSchwefel(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += x[i] * sinf(sqrtf(fabsf(x[i])));
    }
    return 418.9829f * D - sum;
}

// F12: Weierstrass Function
// f(x) = sum(sum(a^k * cos(2*pi*b^k*(x_i+0.5)))) - D*sum(a^k*cos(pi*b^k))
// a = 0.5, b = 3, k = 0..20
// Global minimum: f(0,...,0) = 0
// Bounds: [-0.5, 0.5]^D
__device__ __forceinline__ float benchmarkWeierstrass(const float* x, int D) {
    const float a = 0.5f;
    const float b = 3.0f;
    const int kmax = 20;

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < D; i++) {
        for (int k = 0; k <= kmax; k++) {
            float ak = powf(a, k);
            float bk = powf(b, k);
            sum1 += ak * cosf(2.0f * DFO_PI * bk * (x[i] + 0.5f));
        }
    }

    for (int k = 0; k <= kmax; k++) {
        float ak = powf(a, k);
        float bk = powf(b, k);
        sum2 += ak * cosf(DFO_PI * bk);
    }

    return sum1 - D * sum2;
}

// F13: Alpine Function
// f(x) = sum(|x_i * sin(x_i) + 0.1 * x_i|)
// Global minimum: f(0,...,0) = 0
// Bounds: [-10, 10]^D
__device__ __forceinline__ float benchmarkAlpine(const float* x, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += fabsf(x[i] * sinf(x[i]) + 0.1f * x[i]);
    }
    return sum;
}

// F14: Levy Function
// Complex formula - see implementation
// Global minimum: f(1,...,1) = 0
// Bounds: [-10, 10]^D
__device__ __forceinline__ float benchmarkLevy(const float* x, int D) {
    // w_i = 1 + (x_i - 1)/4
    auto w = [](float xi) { return 1.0f + (xi - 1.0f) / 4.0f; };

    float w1 = w(x[0]);
    float wD = w(x[D - 1]);

    float term1 = sinf(DFO_PI * w1) * sinf(DFO_PI * w1);
    float term3 = (wD - 1.0f) * (wD - 1.0f) * (1.0f + sinf(2.0f * DFO_PI * wD) * sinf(2.0f * DFO_PI * wD));

    float sum = 0.0f;
    for (int i = 0; i < D - 1; i++) {
        float wi = w(x[i]);
        sum += (wi - 1.0f) * (wi - 1.0f) * (1.0f + 10.0f * sinf(DFO_PI * wi + 1.0f) * sinf(DFO_PI * wi + 1.0f));
    }

    return term1 + sum + term3;
}

// F15: Zakharov Function
// f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
// Global minimum: f(0,...,0) = 0
// Bounds: [-5, 10]^D
__device__ __forceinline__ float benchmarkZakharov(const float* x, int D) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int i = 0; i < D; i++) {
        sum1 += x[i] * x[i];
        sum2 += 0.5f * (i + 1) * x[i];
    }

    return sum1 + sum2 * sum2 + sum2 * sum2 * sum2 * sum2;
}

//=============================================================================
// Parallel Evaluation Kernels
//=============================================================================

// Generic parallel reduction kernel for benchmark functions
template<typename FitnessFunc>
__global__ void kernelEvaluateBenchmark(
    const float* __restrict__ positions,
    float* __restrict__ fitness,
    FitnessFunc func,
    int N,
    int D
) {
    int particleIdx = blockIdx.x;
    if (particleIdx >= N) return;

    // Each block handles one particle
    const float* x = positions + particleIdx * D;
    fitness[particleIdx] = func(x, D);
}

// Optimized parallel Sphere evaluation
__global__ void kernelEvalSphereOptimized(
    const float* __restrict__ positions,
    float* __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ float sdata[];

    int particleIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (particleIdx >= N) return;

    // Each thread handles multiple dimensions
    float localSum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float x = positions[particleIdx * D + d];
        localSum += x * x;
    }

    sdata[tid] = localSum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        fitness[particleIdx] = sdata[0];
    }
}

// Optimized parallel Rastrigin evaluation
__global__ void kernelEvalRastriginOptimized(
    const float* __restrict__ positions,
    float* __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ float sdata[];

    int particleIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (particleIdx >= N) return;

    float localSum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float x = positions[particleIdx * D + d];
        localSum += x * x - 10.0f * __cosf(2.0f * DFO_PI * x) + 10.0f;
    }

    sdata[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        fitness[particleIdx] = sdata[0];
    }
}

// Optimized parallel Ackley evaluation
__global__ void kernelEvalAckleyOptimized(
    const float* __restrict__ positions,
    float* __restrict__ fitness,
    int N,
    int D
) {
    extern __shared__ float sdata[];
    float* sdata_cos = &sdata[blockDim.x];

    int particleIdx = blockIdx.x;
    int tid = threadIdx.x;

    if (particleIdx >= N) return;

    float localSumSq = 0.0f;
    float localSumCos = 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        float x = positions[particleIdx * D + d];
        localSumSq += x * x;
        localSumCos += __cosf(2.0f * DFO_PI * x);
    }

    sdata[tid] = localSumSq;
    sdata_cos[tid] = localSumCos;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata_cos[tid] += sdata_cos[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sumSq = sdata[0];
        float sumCos = sdata_cos[0];
        float invD = 1.0f / D;
        fitness[particleIdx] = -20.0f * __expf(-0.2f * sqrtf(sumSq * invD))
                              - __expf(sumCos * invD) + 20.0f + DFO_E;
    }
}

#endif // DFO_BENCHMARK_FUNCTIONS_CUH
