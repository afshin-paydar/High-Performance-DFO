/*
 * High-Performance DFO/uDFO
 * Common headers and structures
 *
 *
 * Based on original DFO by Mohammad Majid al-Rifaie
 * al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.
 */

#ifndef DFO_COMMON_CUH
#define DFO_COMMON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <ctime>

// Compile-time configuration
#define DFO_WARP_SIZE   32
#define DFO_MAX_DIMS  1024
#define DFO_BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Algorithm variants
enum class DFOVariant {
    STANDARD,    // Standard DFO with fixed delta
    UDFO_1000P,  // uDFO with delta_dynamic = 1/(1000p)
    UDFO_1500P,  // uDFO with delta_dynamic = 1/(1500p)
    UDFO_Z5      // uDFO with zone-5 relocation
};

// Optimization direction
enum class OptimizationType {
    MINIMIZE,
    MAXIMIZE
};

// Configuration structure
struct DFOConfig {
    int  populationSize;   // N: number of flies
    int  dimensions;       // D: problem dimensions
    int  maxIterations;    // Maximum iterations
    double delta;          // Disturbance threshold (for standard DFO)
    DFOVariant      variant;
    OptimizationType optType;
    unsigned long long seed;
    bool debug;            // Enable per-iteration diagnostic output (--debug)
    int  debugInterval;    // How often to print debug stats (default: 100)

    // Optional per-dimension bounds (host side; actual bounds go via setBounds())
    double* lowerBounds;
    double* upperBounds;

    __host__ DFOConfig()
        : populationSize(100),
          dimensions(30),
          maxIterations(1000),
          delta(0.001),
          variant(DFOVariant::STANDARD),
          optType(OptimizationType::MINIMIZE),
          seed(static_cast<unsigned long long>(time(nullptr))),
          debug(false),
          debugInterval(100),
          lowerBounds(nullptr),
          upperBounds(nullptr) {}
};

// Result structure
// Use DFOOptimizer::getBestSolution() to retrieve the best position as a host vector.
struct DFOResult {
    double  bestFitness;
    int     iterations;
    float   elapsedTimeMs;
};

// Device-side particle layout (row-major: particle i, dim d = [i*D + d])
struct ParticleData {
    double*      positions;
    double*      fitness;
    int*         bestNeighborIdx;
    curandState* rngStates;
};

// --------------------------------------------------------------------------
// Device constants
// bounds and delta are double to match Python float64 arithmetic exactly.
// --------------------------------------------------------------------------
__constant__ double d_lowerBounds[DFO_MAX_DIMS];
__constant__ double d_upperBounds[DFO_MAX_DIMS];
__constant__ int    d_N;
__constant__ int    d_D;
__constant__ double d_delta;

// --------------------------------------------------------------------------
// uDFO helpers — probability/delta computations stay float32 since the
// values involved (probabilities, scaled positions) are O(1) and float32
// precision is adequate for the dynamic delta formula.
// --------------------------------------------------------------------------
__device__ __forceinline__ float fast_log(float x) {
    return __logf(x);
}

__device__ __forceinline__ float calcExploitationProb(float x, float L, float R) {
    if (L < 1.0f || R < 1.0f) {
        if (L < 0.5f && R < 0.5f) return 0.0f;
        if (R >= 0.5f && R < 1.0f && L >= 0.0f && L <= 1.0f)
            return (2.0f * R - 1.0f - fast_log(2.0f * R)) / (L + R);
        return 0.5f;
    }
    float sum = L + R;
    float p = (sum - 1.0f - fast_log(2.0f * L)) / sum;
    return fminf(fmaxf(p, 0.0f), 1.0f);
}

__device__ __forceinline__ float calcDynamicDelta(float p, DFOVariant variant) {
    if (p <= 0.0f) return 1.0f;
    switch (variant) {
        case DFOVariant::UDFO_1000P: return 1.0f / (1000.0f * p);
        case DFOVariant::UDFO_1500P: return 1.0f / (1500.0f * p);
        // UDFO_Z5 uses the same 1500p delta formula as UDFO_1500P.
        // The Z5 variant is distinguished solely by its zone-5 relocation strategy
        // (see kernelUpdateUDFO_Jacobi), not by a different delta scaling.
        // Reference: al-Rifaie (2021), Entropy 23(8), 977.
        case DFOVariant::UDFO_Z5:    return 1.0f / (1500.0f * p);
        default:                     return (float)d_delta;
    }
}

#endif // DFO_COMMON_CUH
