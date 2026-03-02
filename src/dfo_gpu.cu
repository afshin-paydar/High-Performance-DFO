/*
* High-Performance DFO/uDFO
* Common headers and structures
*
*
* Based on original DFO by Mohammad Majid al-Rifaie
* al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.
 *
 * Main source file with benchmark functions and demonstration
 *
 * Usage:
 *   ./dfo_gpu [variant] [function] [N] [D] [iterations]
 *
 * Variants: standard, udfo1000, udfo1500, udfoz5
 * Functions: sphere, rastrigin
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include "../include/dfo_optimizer.cuh"

// Print usage
void printUsage(const char* progName) {
    printf("Usage: %s [variant] [function] [N] [D] [iterations] [--debug] [--debug-interval K]\n\n", progName);
    printf("Variants:\n");
    printf("  standard  - Standard DFO with fixed delta=0.001\n");
    printf("  udfo1000  - uDFO with delta_dynamic = 1/(1000*p)\n");
    printf("  udfo1500  - uDFO with delta_dynamic = 1/(1500*p)\n");
    printf("  udfoz5    - uDFO with zone-5 relocation\n\n");
    printf("Functions (standard benchmark bounds applied automatically):\n");
    printf("  sphere     - Sphere      [-100, 100]^n\n");
    printf("  rosenbrock - Rosenbrock  [-30, 30]^n\n");
    printf("  rastrigin  - Rastrigin   [-5.12, 5.12]^n\n");
    printf("  ackley     - Ackley      [-32.768, 32.768]^n\n\n");
    printf("Debug options:\n");
    printf("  --debug               Enable per-iteration diagnostic output\n");
    printf("  --debug-interval K    Print debug stats every K iterations (default: 100)\n\n");
    printf("Defaults: standard sphere 100 30 1000\n");
}

// Parse command line arguments
// Positional args: variant function N D iterations
// Optional flags (anywhere after position 1): --debug  --debug-interval K
void parseArgs(int argc, char** argv, DFOConfig& config, FitnessFunction& fitnessType) {
    // Defaults
    config.variant = DFOVariant::STANDARD;
    config.populationSize = 100;
    config.dimensions = 30;
    config.maxIterations = 1000;
    config.delta = 0.001;
    config.debug = false;
    config.debugInterval = 100;
    fitnessType = FitnessFunction::SPHERE;

    // First pass: strip --debug / --debug-interval flags so positional
    // parsing below doesn't choke on them.
    // Build a cleaned argv without those flags.
    std::vector<char*> posArgs;  // positional arguments only
    posArgs.push_back(argv[0]); // program name

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) {
            config.debug = true;
        } else if (strcmp(argv[i], "--debug-interval") == 0) {
            config.debug = true;  // --debug-interval implies --debug
            if (i + 1 < argc) {
                config.debugInterval = atoi(argv[++i]);
                if (config.debugInterval <= 0) config.debugInterval = 100;
            }
        } else {
            posArgs.push_back(argv[i]);
        }
    }

    // Second pass: positional parsing on the cleaned list
    int npos = (int)posArgs.size();

    if (npos > 1) {
        if (strcmp(posArgs[1], "standard") == 0) {
            config.variant = DFOVariant::STANDARD;
        } else if (strcmp(posArgs[1], "udfo1000") == 0) {
            config.variant = DFOVariant::UDFO_1000P;
        } else if (strcmp(posArgs[1], "udfo1500") == 0) {
            config.variant = DFOVariant::UDFO_1500P;
        } else if (strcmp(posArgs[1], "udfoz5") == 0) {
            config.variant = DFOVariant::UDFO_Z5;
        } else if (strcmp(posArgs[1], "-h") == 0 || strcmp(posArgs[1], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
    }

    if (npos > 2) {
        if (strcmp(posArgs[2], "sphere") == 0) {
            fitnessType = FitnessFunction::SPHERE;
        } else if (strcmp(posArgs[2], "rastrigin") == 0) {
            fitnessType = FitnessFunction::RASTRIGIN;
        } else if (strcmp(posArgs[2], "rosenbrock") == 0) {
            fitnessType = FitnessFunction::ROSENBROCK;
        } else if (strcmp(posArgs[2], "ackley") == 0) {
            fitnessType = FitnessFunction::ACKLEY;
        }
    }

    if (npos > 3) config.populationSize = atoi(posArgs[3]);
    if (npos > 4) config.dimensions     = atoi(posArgs[4]);
    if (npos > 5) config.maxIterations  = atoi(posArgs[5]);
}

// Get variant name
const char* getVariantName(DFOVariant variant) {
    switch (variant) {
        case DFOVariant::STANDARD: return "Standard DFO";
        case DFOVariant::UDFO_1000P: return "uDFO (delta=1/1000p)";
        case DFOVariant::UDFO_1500P: return "uDFO (delta=1/1500p)";
        case DFOVariant::UDFO_Z5: return "uDFO-z5 (zone-5 relocation)";
        default: return "Unknown";
    }
}

// Get fitness function name
const char* getFitnessName(FitnessFunction func) {
    switch (func) {
        case FitnessFunction::SPHERE:     return "Sphere";
        case FitnessFunction::RASTRIGIN:  return "Rastrigin";
        case FitnessFunction::ROSENBROCK: return "Rosenbrock";
        case FitnessFunction::ACKLEY:     return "Ackley";
        default: return "Unknown";
    }
}

// Standard benchmark bounds per function
void getStandardBounds(FitnessFunction func, double& lower, double& upper) {
    switch (func) {
        case FitnessFunction::SPHERE:     lower = -100.0;    upper =  100.0;    break;
        case FitnessFunction::ROSENBROCK: lower =  -30.0;    upper =   30.0;    break;
        case FitnessFunction::RASTRIGIN:  lower =   -5.12;   upper =    5.12;   break;
        case FitnessFunction::ACKLEY:     lower =  -32.768;  upper =   32.768;  break;
        default:                          lower = -100.0;    upper =  100.0;    break;
    }
}

// Print GPU info
void printGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("=== GPU Information ===\n");
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        printf("Device %d: %s\n", i, props.name);
        printf("  Compute capability: %d.%d\n", props.major, props.minor);
        printf("  Total global memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", props.multiProcessorCount);
        printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
        printf("  Warp size: %d\n", props.warpSize);
        printf("  Memory clock rate: %.2f GHz\n", props.memoryClockRate / 1e6);
        printf("  Memory bus width: %d bits\n", props.memoryBusWidth);
    }
    printf("\n");
}

// Run benchmark comparing all variants
void runBenchmark(int N, int D, int iterations) {
    printf("=== Running Benchmark ===\n");
    printf("Population: %d, Dimensions: %d, Iterations: %d\n\n", N, D, iterations);

    DFOVariant variants[] = {
        DFOVariant::STANDARD,
        DFOVariant::UDFO_1000P,
        DFOVariant::UDFO_1500P,
        DFOVariant::UDFO_Z5
    };

    FitnessFunction functions[] = {
        FitnessFunction::SPHERE,
        FitnessFunction::ROSENBROCK,
        FitnessFunction::RASTRIGIN,
        FitnessFunction::ACKLEY
    };

    // Results table header
    printf("%-25s %-15s %-15s %-20s %-15s\n", "Variant", "Function", "Bounds", "Best Fitness", "Time (ms)");
    printf("%-25s %-15s %-15s %-20s %-15s\n", "-------", "--------", "------", "------------", "---------");

    for (auto& func : functions) {
        for (auto& variant : variants) {
            DFOConfig config;
            config.populationSize = N;
            config.dimensions = D;
            config.maxIterations = iterations;
            config.variant = variant;
            config.delta = 0.001;
            config.seed = static_cast<unsigned long long>(time(nullptr));

            DFOOptimizer optimizer(config);

            double lower, upper;
            getStandardBounds(func, lower, upper);
            optimizer.setUniformBounds(lower, upper);

            DFOResult result = optimizer.optimize(func);

            char boundsStr[32];
            snprintf(boundsStr, sizeof(boundsStr), "[%.3g, %.3g]", lower, upper);
            printf("%-25s %-15s %-15s %-20.6e %-15.2f\n",
                   getVariantName(variant),
                   getFitnessName(func),
                   boundsStr,
                   result.bestFitness,
                   result.elapsedTimeMs);
        }
    }
}

// Main function
int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  High-Performance GPU DFO/uDFO\n");
    printf("========================================\n\n");

    printGPUInfo();

    DFOConfig config;
    FitnessFunction fitnessType;
    parseArgs(argc, argv, config, fitnessType);

    // Check for benchmark mode
    if (argc > 1 && strcmp(argv[1], "benchmark") == 0) {
        int N = (argc > 2) ? atoi(argv[2]) : 100;
        int D = (argc > 3) ? atoi(argv[3]) : 30;
        int iters = (argc > 4) ? atoi(argv[4]) : 1000;
        runBenchmark(N, D, iters);
        return 0;
    }

    double lower, upper;
    getStandardBounds(fitnessType, lower, upper);

    printf("=== Configuration ===\n");
    printf("Variant: %s\n", getVariantName(config.variant));
    printf("Fitness function: %s\n", getFitnessName(fitnessType));
    printf("Population size: %d\n", config.populationSize);
    printf("Dimensions: %d\n", config.dimensions);
    printf("Max iterations: %d\n", config.maxIterations);
    printf("Delta (standard DFO): %.4f\n", config.delta);
    printf("Bounds: [%.4g, %.4g]\n", lower, upper);
    printf("RNG seed: %llu\n", config.seed);
    if (config.debug)
        printf("Debug: ON (every %d iterations)\n", config.debugInterval);
    else
        printf("Debug: OFF (use --debug to enable diagnostic output)\n");
    printf("\n");

    // Create optimizer
    DFOOptimizer optimizer(config);

    optimizer.setUniformBounds(lower, upper);

    printf("=== Starting Optimization ===\n\n");

    // Run optimization
    DFOResult result = optimizer.optimize(fitnessType);

    // Get and print best solution
    std::vector<double> solution = optimizer.getBestSolution();

    printf("\n=== Best Solution ===\n");
    printf("First 10 dimensions:\n");
    for (int i = 0; i < std::min(10, config.dimensions); i++) {
        printf("  x[%d] = %.10f\n", i, solution[i]);
    }

    if (config.dimensions > 10) {
        printf("  ... (%d more dimensions)\n", config.dimensions - 10);
    }

    // Performance metrics
    double iterPerSec = result.iterations / (result.elapsedTimeMs / 1000.0);
    double evalPerSec = iterPerSec * config.populationSize;

    printf("\n=== Performance Metrics ===\n");
    printf("Iterations/second: %.2f\n", iterPerSec);
    printf("Fitness evaluations/second: %.2e\n", evalPerSec);

    return 0;
}
