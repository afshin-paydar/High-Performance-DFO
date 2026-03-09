/*
* High-Performance DFO/uDFO
*
* Based on original DFO by Mohammad Majid al-Rifaie
* al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on
* Computer Science and Information Systems (pp. 529-538). IEEE.
*
* Main source file with benchmark functions and demonstration
*
* Usage:
*   ./dfo_gpu [variant] [function] [N] [D] [iterations] [OPTIONS]
*
* Variants: standard, udfo1000, udfo1500, udfoz5
* Functions: sphere, rastrigin, rosenbrock, ackley
*
* Options:
*   --seed S           RNG seed (default: time-based, non-reproducible)
*   --print-interval K Print best fitness every K iterations (default: maxIter/10)
*   --debug            Enable per-iteration diagnostic output
*   --debug-interval K Print debug stats every K iterations (default: 100)
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include "../include/dfo_optimizer.cuh"

void printUsage(const char* progName) {
    printf("Usage: %s [variant] [function] [N] [D] [iterations] [OPTIONS]\n\n", progName);
    printf("Variants:\n");
    printf("  standard  - Standard DFO with fixed delta=0.001\n");
    printf("  udfo1000  - uDFO with delta_dynamic = 1/(1000*p)\n");
    printf("  udfo1500  - uDFO with delta_dynamic = 1/(1500*p)\n");
    printf("  udfoz5    - uDFO with zone-5 relocation\n\n");
    printf("Functions (standard benchmark bounds applied automatically):\n");
    printf("  sphere     - Sphere      [-100, 100]^D\n");
    printf("  rosenbrock - Rosenbrock  [-30, 30]^D\n");
    printf("  rastrigin  - Rastrigin   [-5.12, 5.12]^D\n");
    printf("  ackley     - Ackley      [-32.768, 32.768]^D\n\n");
    printf("Options:\n");
    printf("  --seed S              Set RNG seed for reproducibility\n");
    printf("  --print-interval K    Print best fitness every K iterations\n");
    printf("  --debug               Enable per-iteration diagnostic output\n");
    printf("  --debug-interval K    Print debug stats every K iterations (default: 100)\n\n");
    printf("Defaults: standard sphere 100 30 1000\n");
}

void parseArgs(int argc, char** argv, DFOConfig& config, FitnessFunction& fitnessType) {
    config.variant        = DFOVariant::STANDARD;
    config.populationSize = 100;
    config.dimensions     = 30;
    config.maxIterations  = 1000;
    config.delta          = 0.001;
    config.debug          = false;
    config.debugInterval  = 100;
    config.seed           = static_cast<unsigned long long>(time(nullptr));
    fitnessType           = FitnessFunction::SPHERE;

    std::vector<char*> posArgs;
    posArgs.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) {
            config.debug = true;
        } else if (strcmp(argv[i], "--debug-interval") == 0) {
            config.debug = true;
            if (i + 1 < argc) {
                config.debugInterval = atoi(argv[++i]);
                if (config.debugInterval <= 0) config.debugInterval = 100;
            }
        } else if (strcmp(argv[i], "--print-interval") == 0) {
            if (i + 1 < argc) {
                config.printInterval = atoi(argv[++i]);
                if (config.printInterval <= 0) config.printInterval = 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            // Explicit seed: enables reproducible independent runs.
            // Pass a different seed per run in experiment scripts.
            if (i + 1 < argc) {
                config.seed = (unsigned long long)atoll(argv[++i]);
            }
        } else {
            posArgs.push_back(argv[i]);
        }
    }

    int npos = (int)posArgs.size();

    if (npos > 1) {
        if      (strcmp(posArgs[1], "standard") == 0) config.variant = DFOVariant::STANDARD;
        else if (strcmp(posArgs[1], "udfo1000") == 0) config.variant = DFOVariant::UDFO_1000P;
        else if (strcmp(posArgs[1], "udfo1500") == 0) config.variant = DFOVariant::UDFO_1500P;
        else if (strcmp(posArgs[1], "udfoz5")   == 0) config.variant = DFOVariant::UDFO_Z5;
        else if (strcmp(posArgs[1], "-h") == 0 || strcmp(posArgs[1], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
    }

    if (npos > 2) {
        if      (strcmp(posArgs[2], "sphere")     == 0) fitnessType = FitnessFunction::SPHERE;
        else if (strcmp(posArgs[2], "rastrigin")  == 0) fitnessType = FitnessFunction::RASTRIGIN;
        else if (strcmp(posArgs[2], "rosenbrock") == 0) fitnessType = FitnessFunction::ROSENBROCK;
        else if (strcmp(posArgs[2], "ackley")     == 0) fitnessType = FitnessFunction::ACKLEY;
    }

    if (npos > 3) config.populationSize = atoi(posArgs[3]);
    if (npos > 4) config.dimensions     = atoi(posArgs[4]);
    if (npos > 5) config.maxIterations  = atoi(posArgs[5]);
}

const char* getVariantName(DFOVariant variant) {
    switch (variant) {
        case DFOVariant::STANDARD:   return "Standard DFO";
        case DFOVariant::UDFO_1000P: return "uDFO (delta=1/1000p)";
        case DFOVariant::UDFO_1500P: return "uDFO (delta=1/1500p)";
        case DFOVariant::UDFO_Z5:    return "uDFO-z5 (zone-5 relocation)";
        default:                     return "Unknown";
    }
}

const char* getFitnessName(FitnessFunction func) {
    switch (func) {
        case FitnessFunction::SPHERE:     return "Sphere";
        case FitnessFunction::RASTRIGIN:  return "Rastrigin";
        case FitnessFunction::ROSENBROCK: return "Rosenbrock";
        case FitnessFunction::ACKLEY:     return "Ackley";
        default:                          return "Unknown";
    }
}

void getStandardBounds(FitnessFunction func, double& lower, double& upper) {
    switch (func) {
        case FitnessFunction::SPHERE:     lower = -100.0;   upper =  100.0;   break;
        case FitnessFunction::ROSENBROCK: lower =  -30.0;   upper =   30.0;   break;
        case FitnessFunction::RASTRIGIN:  lower =   -5.12;  upper =    5.12;  break;
        case FitnessFunction::ACKLEY:     lower =  -32.768; upper =   32.768; break;
        default:                          lower = -100.0;   upper =  100.0;   break;
    }
}

void printGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("=== GPU Information ===\n");
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("Device %d: %s (SM %d.%d, %.2f GB, %d SMs)\n",
               i, props.name, props.major, props.minor,
               props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               props.multiProcessorCount);
    }
    printf("\n");
}

void runBenchmark(int N, int D, int iterations) {
    printf("=== Benchmark: N=%d D=%d T=%d ===\n", N, D, iterations);

    DFOVariant variants[]  = { DFOVariant::STANDARD, DFOVariant::UDFO_1000P,
                                DFOVariant::UDFO_1500P, DFOVariant::UDFO_Z5 };
    FitnessFunction funcs[] = { FitnessFunction::SPHERE, FitnessFunction::ROSENBROCK,
                                 FitnessFunction::RASTRIGIN, FitnessFunction::ACKLEY };

    printf("%-25s %-15s %-20s %-15s\n", "Variant", "Function", "Best Fitness", "Time (ms)");
    printf("%-25s %-15s %-20s %-15s\n", "-------", "--------", "------------", "---------");

    for (auto& func : funcs) {
        for (auto& variant : variants) {
            DFOConfig cfg;
            cfg.populationSize = N;
            cfg.dimensions     = D;
            cfg.maxIterations  = iterations;
            cfg.variant        = variant;
            cfg.delta          = 0.001;
            cfg.seed           = 12345ULL;

            DFOOptimizer optimizer(cfg);
            double lower, upper;
            getStandardBounds(func, lower, upper);
            optimizer.setUniformBounds(lower, upper);

            DFOResult result = optimizer.optimize(func);

            printf("%-25s %-15s %-20.6e %-15.2f\n",
                   getVariantName(variant), getFitnessName(func),
                   result.bestFitness, result.elapsedTimeMs);
        }
    }
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  High-Performance GPU DFO/uDFO\n");
    printf("========================================\n\n");

    printGPUInfo();

    if (argc > 1 && strcmp(argv[1], "benchmark") == 0) {
        int N     = (argc > 2) ? atoi(argv[2]) : 100;
        int D     = (argc > 3) ? atoi(argv[3]) : 30;
        int iters = (argc > 4) ? atoi(argv[4]) : 1000;
        runBenchmark(N, D, iters);
        return 0;
    }

    DFOConfig config;
    FitnessFunction fitnessType;
    parseArgs(argc, argv, config, fitnessType);

    double lower, upper;
    getStandardBounds(fitnessType, lower, upper);

    printf("=== Configuration ===\n");
    printf("Variant:          %s\n", getVariantName(config.variant));
    printf("Function:         %s\n", getFitnessName(fitnessType));
    printf("Population (N):   %d\n", config.populationSize);
    printf("Dimensions (D):   %d\n", config.dimensions);
    printf("Iterations (T):   %d\n", config.maxIterations);
    printf("MaxFEs:           %lld\n", (long long)config.populationSize * config.maxIterations);
    printf("Delta:            %.4f\n", config.delta);
    printf("Bounds:           [%.4g, %.4g]\n", lower, upper);
    printf("RNG seed:         %llu\n", config.seed);
    printf("\n");

    DFOOptimizer optimizer(config);
    optimizer.setUniformBounds(lower, upper);

    printf("=== Starting Optimization ===\n\n");
    DFOResult result = optimizer.optimize(fitnessType);

    std::vector<double> solution = optimizer.getBestSolution();
    printf("\n=== Best Solution (first 10 dims) ===\n");
    for (int i = 0; i < std::min(10, config.dimensions); i++)
        printf("  x[%d] = %.10f\n", i, solution[i]);
    if (config.dimensions > 10)
        printf("  ... (%d more dimensions)\n", config.dimensions - 10);

    double iterPerSec = result.iterations / (result.elapsedTimeMs / 1000.0);
    printf("\n=== Performance ===\n");
    printf("Iterations/s:     %.2f\n", iterPerSec);
    printf("FEs/s:            %.2e\n", iterPerSec * config.populationSize);

    return 0;
}
