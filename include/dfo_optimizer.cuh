/*
 * High-Performance DFO/uDFO Class
 *
 * Based on original DFO by Mohammad Majid al-Rifaie
 * al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.
 */

#ifndef DFO_OPTIMIZER_CUH
#define DFO_OPTIMIZER_CUH

#include "dfo_common.cuh"
#include "dfo_kernels.cuh"
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>

// Fitness function types
enum class FitnessFunction {
    SPHERE,
    RASTRIGIN,
    ROSENBROCK,
    ACKLEY,
    CUSTOM
};

class DFOOptimizer {
public:
    explicit DFOOptimizer(const DFOConfig& config);
    ~DFOOptimizer();

    // Main optimization function
    DFOResult optimize(FitnessFunction fitnessType);

    // Get best solution (copies from device to host)
    std::vector<double> getBestSolution();

    // Get all positions (for debugging/visualization)
    std::vector<double> getAllPositions();

    // Get all fitness values
    std::vector<double> getAllFitness();

    // Reset optimizer state for a fresh run (keeps allocations)
    void reset();

    // Set per-dimension bounds (host arrays, copied to device constants)
    void setBounds(const double* lower, const double* upper);

    // Set uniform bounds for all dimensions
    void setUniformBounds(double lower, double upper);

private:
    DFOConfig config_;

    // Device memory — all positions and fitness use double
    double*      d_positions_;
    double*      d_fitness_;
    int*         d_bestNeighborIdx_;
    curandState* d_rngStates_;
    double*      d_bestPosition_;

    // Device scalars
    int*    d_globalBestIdx_;
    double* d_globalBestFitness_;

    // Debug stats buffer on device (8 doubles, see kernelComputeDebugStats)
    double* d_debugStats_;

    // Host-side cached results
    int    h_globalBestIdx_;
    double h_globalBestFitness_;

    // CUDA streams
    cudaStream_t computeStream_;
    cudaStream_t copyStream_;

    // Internal helpers
    void allocateMemory();
    void freeMemory();
    void initializePopulation();
    void evaluateFitness(FitnessFunction fitnessType);
    void findGlobalBest();
    void findBestNeighbors();
    void updatePopulation();
    void debugPrintStats(int iter, FitnessFunction fitnessType);

    bool isInitialized_;
};

//=============================================================================
// Implementation
//=============================================================================

inline DFOOptimizer::DFOOptimizer(const DFOConfig& config)
    : config_(config),
      d_positions_(nullptr),
      d_fitness_(nullptr),
      d_bestNeighborIdx_(nullptr),
      d_rngStates_(nullptr),
      d_bestPosition_(nullptr),
      d_globalBestIdx_(nullptr),
      d_globalBestFitness_(nullptr),
      d_debugStats_(nullptr),
      h_globalBestIdx_(0),
      h_globalBestFitness_(DBL_MAX),
      isInitialized_(false)
{
    if (config_.dimensions > DFO_MAX_DIMS) {
        fprintf(stderr, "Error: dimensions (%d) exceeds maximum (%d)\n",
                config_.dimensions, DFO_MAX_DIMS);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaStreamCreate(&computeStream_));
    CUDA_CHECK(cudaStreamCreate(&copyStream_));
    allocateMemory();
}

inline DFOOptimizer::~DFOOptimizer() {
    freeMemory();
    cudaStreamDestroy(computeStream_);
    cudaStreamDestroy(copyStream_);
}

inline void DFOOptimizer::allocateMemory() {
    int N = config_.populationSize;
    int D = config_.dimensions;

    CUDA_CHECK(cudaMalloc(&d_positions_,         (size_t)N * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitness_,           (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bestNeighborIdx_,   (size_t)N     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rngStates_,         (size_t)N * D * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_bestPosition_,      (size_t)D     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_globalBestIdx_,                     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_globalBestFitness_,                 sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_debugStats_,        8             * sizeof(double)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_N,     &N,             sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_D,     &D,             sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_delta, &config_.delta, sizeof(double)));
}

inline void DFOOptimizer::freeMemory() {
    if (d_positions_)         cudaFree(d_positions_);
    if (d_fitness_)           cudaFree(d_fitness_);
    if (d_bestNeighborIdx_)   cudaFree(d_bestNeighborIdx_);
    if (d_rngStates_)         cudaFree(d_rngStates_);
    if (d_bestPosition_)      cudaFree(d_bestPosition_);
    if (d_globalBestIdx_)     cudaFree(d_globalBestIdx_);
    if (d_globalBestFitness_) cudaFree(d_globalBestFitness_);
    if (d_debugStats_)        cudaFree(d_debugStats_);
}

inline void DFOOptimizer::setBounds(const double* lower, const double* upper) {
    int D = config_.dimensions;
    CUDA_CHECK(cudaMemcpyToSymbol(d_lowerBounds, lower, D * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_upperBounds, upper, D * sizeof(double)));
}

inline void DFOOptimizer::setUniformBounds(double lower, double upper) {
    int D = config_.dimensions;
    std::vector<double> lo(D, lower), hi(D, upper);
    setBounds(lo.data(), hi.data());
}

inline void DFOOptimizer::initializePopulation() {
    int N = config_.populationSize;
    int D = config_.dimensions;
    int total = N * D;

    int blockSize = DFO_BLOCK_SIZE;
    int numBlocks = (total + blockSize - 1) / blockSize;

    kernelInitRNG<<<numBlocks, blockSize, 0, computeStream_>>>(
        d_rngStates_, config_.seed, total);

    kernelInitPopulation<<<numBlocks, blockSize, 0, computeStream_>>>(
        d_positions_, d_rngStates_, N, D);

    CUDA_CHECK(cudaStreamSynchronize(computeStream_));
    isInitialized_ = true;
}

inline void DFOOptimizer::evaluateFitness(FitnessFunction fitnessType) {
    int N = config_.populationSize;
    int D = config_.dimensions;

    // Block size: smallest power-of-2 >= D, capped at 256
    int blockSize = 1;
    while (blockSize < D && blockSize < 256) blockSize <<= 1;

    // Ackley needs two shared arrays (sum-of-sq and sum-of-cos)
    int sharedMult    = (fitnessType == FitnessFunction::ACKLEY) ? 2 : 1;
    size_t sharedMem  = (size_t)sharedMult * blockSize * sizeof(double);

    switch (fitnessType) {
        case FitnessFunction::SPHERE:
            kernelEvaluateFitnessSphere<<<N, blockSize, sharedMem, computeStream_>>>(
                d_positions_, d_fitness_, N, D);
            break;
        case FitnessFunction::RASTRIGIN:
            kernelEvaluateFitnessRastrigin<<<N, blockSize, sharedMem, computeStream_>>>(
                d_positions_, d_fitness_, N, D);
            break;
        case FitnessFunction::ROSENBROCK:
            kernelEvaluateFitnessRosenbrock<<<N, blockSize, sharedMem, computeStream_>>>(
                d_positions_, d_fitness_, N, D);
            break;
        case FitnessFunction::ACKLEY:
            kernelEvaluateFitnessAckley<<<N, blockSize, sharedMem, computeStream_>>>(
                d_positions_, d_fitness_, N, D);
            break;
        default:
            kernelEvaluateFitnessSphere<<<N, blockSize, sharedMem, computeStream_>>>(
                d_positions_, d_fitness_, N, D);
    }
}

inline void DFOOptimizer::findGlobalBest() {
    int N = config_.populationSize;

    // Single block; size = smallest power-of-2 >= N, capped at 1024
    int blockSize = 64;
    while (blockSize < N && blockSize < 1024) blockSize <<= 1;

    // Shared memory: blockSize doubles (values) + blockSize ints (indices)
    // Pad int array to start at a double-aligned offset
    size_t sharedMem = (size_t)blockSize * sizeof(double)
                     + (size_t)blockSize * sizeof(int);

    bool minimize = (config_.optType == OptimizationType::MINIMIZE);

    // Initialise device scalar before kernel so the warp reduction has a
    // defined sentinel to overwrite.
    double initFitness = minimize ? DBL_MAX : -DBL_MAX;
    CUDA_CHECK(cudaMemcpyAsync(d_globalBestFitness_, &initFitness, sizeof(double),
                               cudaMemcpyHostToDevice, computeStream_));

    kernelFindGlobalBest<<<1, blockSize, sharedMem, computeStream_>>>(
        d_fitness_, d_globalBestIdx_, d_globalBestFitness_, N, minimize);

    // Cross-stream ordering: copyStream_ must wait for the kernel
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreate(&ev));
    CUDA_CHECK(cudaEventRecord(ev, computeStream_));
    CUDA_CHECK(cudaStreamWaitEvent(copyStream_, ev, 0));
    CUDA_CHECK(cudaEventDestroy(ev));

    CUDA_CHECK(cudaMemcpyAsync(&h_globalBestIdx_, d_globalBestIdx_,
                               sizeof(int),    cudaMemcpyDeviceToHost, copyStream_));
    CUDA_CHECK(cudaMemcpyAsync(&h_globalBestFitness_, d_globalBestFitness_,
                               sizeof(double), cudaMemcpyDeviceToHost, copyStream_));
}

inline void DFOOptimizer::findBestNeighbors() {
    int N = config_.populationSize;
    int blockSize = DFO_BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    bool minimize = (config_.optType == OptimizationType::MINIMIZE);

    kernelFindBestNeighbors<<<numBlocks, blockSize, 0, computeStream_>>>(
        d_fitness_, d_bestNeighborIdx_, N, minimize);
}

inline void DFOOptimizer::updatePopulation() {
    int N = config_.populationSize;
    int D = config_.dimensions;

    // copyStream_ must have finished delivering h_globalBestIdx_
    CUDA_CHECK(cudaStreamSynchronize(copyStream_));

    // Block size: smallest power-of-2 >= D, capped at 1024
    int blockSize = 1;
    while (blockSize < D && blockSize < 1024) blockSize <<= 1;

    if (config_.variant == DFOVariant::STANDARD) {
        kernelUpdateDFO<<<1, blockSize, 0, computeStream_>>>(
            d_positions_, d_fitness_, d_bestNeighborIdx_, d_rngStates_,
            h_globalBestIdx_, config_.delta, N, D);
    } else {
        kernelUpdateUDFO<<<1, blockSize, 0, computeStream_>>>(
            d_positions_, d_fitness_, d_bestNeighborIdx_, d_rngStates_,
            h_globalBestIdx_, N, D, config_.variant);
    }
}

// ---------------------------------------------------------------------------
// debugPrintStats — GPU diagnostic kernel + CPU post-processing
// ---------------------------------------------------------------------------
inline void DFOOptimizer::debugPrintStats(int iter, FitnessFunction /*fitnessType*/) {
    int N = config_.populationSize;
    int D = config_.dimensions;

    // ---- GPU: kernelComputeDebugStats -------------------------------------
    // Initialise: [0]=DBL_MAX, [1]=-DBL_MAX, [2..7]=0
    double initBuf[8];
    initBuf[0] = DBL_MAX;
    initBuf[1] = -DBL_MAX;
    for (int k = 2; k < 8; k++) initBuf[k] = 0.0;
    CUDA_CHECK(cudaMemcpy(d_debugStats_, initBuf, 8 * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Ensure evaluateFitness + findGlobalBest are complete before reading
    CUDA_CHECK(cudaStreamSynchronize(computeStream_));

    int blockSize = DFO_BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;
    kernelComputeDebugStats<<<numBlocks, blockSize, 0, computeStream_>>>(
        d_positions_, d_fitness_, d_debugStats_, h_globalBestIdx_, N, D);

    // ---- CPU: read back stats ---------------------------------------------
    double h_stats[8];
    CUDA_CHECK(cudaStreamSynchronize(computeStream_));
    CUDA_CHECK(cudaMemcpy(h_stats, d_debugStats_, 8 * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double minFit     = h_stats[0];
    double maxFit     = h_stats[1];
    double dim0spread = h_stats[2] / N;
    int    collapsed  = (int)h_stats[3];
    int    exactMatch = (int)h_stats[4];
    double movePot    = h_stats[5] / ((double)N * D);
    double deadFrac   = h_stats[6] / ((double)N * D);

    // ---- CPU: sample update trace for particle 0 --------------------------
    std::vector<double> h_pos_p0(D), h_pos_best(D), h_pos_nbr(D);
    CUDA_CHECK(cudaMemcpy(h_pos_p0.data(),
                          d_positions_,
                          D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_best.data(),
                          d_positions_ + h_globalBestIdx_ * D,
                          D * sizeof(double), cudaMemcpyDeviceToHost));
    int h_neighborOf0 = -1;
    CUDA_CHECK(cudaMemcpy(&h_neighborOf0, d_bestNeighborIdx_,
                          sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_nbr.data(),
                          d_positions_ + h_neighborOf0 * D,
                          D * sizeof(double), cudaMemcpyDeviceToHost));

    // ---- CPU: unique fitness count ----------------------------------------
    std::vector<double> h_fitness_all(N);
    CUDA_CHECK(cudaMemcpy(h_fitness_all.data(), d_fitness_,
                          N * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> sorted_fit = h_fitness_all;
    std::sort(sorted_fit.begin(), sorted_fit.end());
    int uniqueCount = 1;
    for (int i = 1; i < N; i++)
        if (sorted_fit[i] != sorted_fit[i-1]) uniqueCount++;

    // ---- Print ------------------------------------------------------------
    printf("\n──────── DEBUG iter=%-5d ─────────────────────────────────────────────\n", iter);

    printf("  [fitness range]  min=%.8e  max=%.8e  bestIdx=%d  kernelBest=%.8e\n",
           minFit, maxFit, h_globalBestIdx_, h_globalBestFitness_);
    if (fabs(minFit - h_globalBestFitness_) > 1e-10 * fabs(minFit) + 1e-20)
        printf("  *** BUG: kernelFindGlobalBest returned %.8e but true min=%.8e — "
               "warp reduction error! ***\n", h_globalBestFitness_, minFit);

    printf("  [diversity]      uniqueFitness=%d/%d  collapsed=%d  exactFitMatch=%d\n",
           uniqueCount, N, collapsed, exactMatch);
    if (uniqueCount == 1)
        printf("  *** POPULATION FULLY COLLAPSED ***\n");
    else if (uniqueCount < N / 10)
        printf("  *** WARNING: low diversity — only %d unique fitness values ***\n", uniqueCount);

    printf("  [position]       avgDim0dist=%.3e  avgMovePot(all dims)=%.3e\n",
           dim0spread, movePot);
    if (movePot < 1e-12)
        printf("  *** WARNING: movement potential ~0 — update formula effectively dead ***\n");

    printf("  [f64 precision]  deadZoneFrac=%.4f  (fraction of (i,d) where "
           "|x_best-x_i| < 1e-14)\n", deadFrac);
    if (deadFrac > 0.5)
        printf("  *** NOTE: >50%% of differences in float64 dead zone — "
               "population has converged ***\n");

    printf("  [update trace]   particle=0  neighbor=%d  bestIdx=%d\n",
           h_neighborOf0, h_globalBestIdx_);
    printf("  %5s  %15s  %15s  %15s  %15s  %15s\n",
           "dim", "x_current", "x_neighbor", "x_best", "diff(b-c)", "x_new(u=0.5)");
    for (int d = 0; d < std::min(D, 6); d++) {
        double xc   = h_pos_p0[d];
        double xn   = h_pos_nbr[d];
        double xb   = h_pos_best[d];
        double diff = xb - xc;
        double xnew = xn + 0.5 * diff;
        printf("  %5d  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e\n",
               d, xc, xn, xb, diff, xnew);
    }

    // Rastrigin dimension audit for best particle
    {
        int dims_near_global = 0, dims_near_local1 = 0, dims_other = 0;
        for (int d = 0; d < D; d++) {
            double v = fabs(h_pos_best[d]);
            if      (v < 0.05)                  dims_near_global++;
            else if (fabs(v - 1.0) < 0.05) dims_near_local1++;
            else                                dims_other++;
        }
        printf("  [rastrigin dims] near_global=%d  near_local1=%d  other=%d\n",
               dims_near_global, dims_near_local1, dims_other);
        if (dims_near_local1 + dims_other > 0)
            printf("  *** %d dimension(s) of best fly not at global optimum ***\n",
                   dims_near_local1 + dims_other);
    }
    printf("──────────────────────────────────────────────────────────────────────\n\n");
}

// ---------------------------------------------------------------------------
// optimize — main loop, mirrors Python DFO exactly:
//   for itr in range(T):
//       evaluate + find_best + update          ← T updates
//   evaluate + find_best                       ← final read
// ---------------------------------------------------------------------------
inline DFOResult DFOOptimizer::optimize(FitnessFunction fitnessType) {
    if (!isInitialized_) initializePopulation();

    DFOResult result;
    result.bestPosition = d_bestPosition_;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, computeStream_));

    int printInterval = config_.maxIterations / 10;
    if (printInterval == 0) printInterval = 1;

    for (int iter = 0; iter < config_.maxIterations; iter++) {
        evaluateFitness(fitnessType);
        findGlobalBest();
        findBestNeighbors();
        CUDA_CHECK(cudaStreamSynchronize(copyStream_));

        if (iter % printInterval == 0)
            printf("Iteration: %d\tBest fitness: %.10e\n", iter, h_globalBestFitness_);

        if (config_.debug && iter % config_.debugInterval == 0)
            debugPrintStats(iter, fitnessType);

        // Always update — matches Python which updates on every iteration
        updatePopulation();
    }

    // Final evaluation (mirrors Python's post-loop evaluation)
    evaluateFitness(fitnessType);
    findGlobalBest();
    CUDA_CHECK(cudaStreamSynchronize(copyStream_));

    if (config_.debug) {
        printf("\n=== FINAL DEBUG SNAPSHOT ===\n");
        debugPrintStats(config_.maxIterations, fitnessType);
    }

    // Copy best position to output buffer
    {
        int D         = config_.dimensions;
        int blockSize = std::min(256, D);
        int numBlocks = (D + blockSize - 1) / blockSize;
        kernelCopyBestSolution<<<numBlocks, blockSize, 0, computeStream_>>>(
            d_positions_, d_bestPosition_, h_globalBestIdx_, D);
    }

    CUDA_CHECK(cudaEventRecord(stop, computeStream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    result.bestFitness   = h_globalBestFitness_;
    result.iterations    = config_.maxIterations;
    result.elapsedTimeMs = elapsedMs;

    printf("\nOptimization complete!\n");
    printf("Final best fitness: %.10e\n", result.bestFitness);
    printf("Elapsed time: %.2f ms\n", result.elapsedTimeMs);

    return result;
}

inline std::vector<double> DFOOptimizer::getBestSolution() {
    std::vector<double> sol(config_.dimensions);
    CUDA_CHECK(cudaMemcpy(sol.data(), d_bestPosition_,
                          config_.dimensions * sizeof(double), cudaMemcpyDeviceToHost));
    return sol;
}

inline std::vector<double> DFOOptimizer::getAllPositions() {
    int N = config_.populationSize;
    int D = config_.dimensions;
    std::vector<double> pos(N * D);
    CUDA_CHECK(cudaMemcpy(pos.data(), d_positions_,
                          (size_t)N * D * sizeof(double), cudaMemcpyDeviceToHost));
    return pos;
}

inline std::vector<double> DFOOptimizer::getAllFitness() {
    int N = config_.populationSize;
    std::vector<double> fit(N);
    CUDA_CHECK(cudaMemcpy(fit.data(), d_fitness_,
                          N * sizeof(double), cudaMemcpyDeviceToHost));
    return fit;
}

inline void DFOOptimizer::reset() {
    isInitialized_ = false;
    h_globalBestIdx_ = 0;
    h_globalBestFitness_ =
        (config_.optType == OptimizationType::MINIMIZE) ? DBL_MAX : -DBL_MAX;
}

#endif // DFO_OPTIMIZER_CUH
