/*
* High-Performance DFO/uDFO
* Common headers and structures
*
*
* Based on original DFO by Mohammad Majid al-Rifaie
* al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.
 *
 * Compares standard DFO with uDFO variants on multiple benchmark functions
 * Records timing, convergence, and solution quality
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "../include/dfo_optimizer.cuh"

// Benchmark configuration
struct BenchmarkConfig {
    int numRuns = 30;              // Number of independent runs
    int populationSize = 100;
    int dimensions = 30;
    int maxIterations = 1000;
    bool verbose = false;
    std::string outputFile = "benchmark_results.csv";
};

// Run statistics
struct RunStats {
    double bestFitness;
    double meanFitness;
    double stdFitness;
    double meanTime;
    double stdTime;
    std::vector<double> allBestFitness;
    std::vector<double> allTimes;
};

// Compute statistics
RunStats computeStats(const std::vector<double>& fitness, const std::vector<double>& times) {
    RunStats stats;
    stats.allBestFitness = fitness;
    stats.allTimes = times;

    int n = fitness.size();

    // Best fitness
    stats.bestFitness = *std::min_element(fitness.begin(), fitness.end());

    // Mean fitness
    stats.meanFitness = std::accumulate(fitness.begin(), fitness.end(), 0.0) / n;

    // Std fitness
    double sqSum = 0;
    for (double f : fitness) {
        sqSum += (f - stats.meanFitness) * (f - stats.meanFitness);
    }
    stats.stdFitness = std::sqrt(sqSum / n);

    // Mean time
    stats.meanTime = std::accumulate(times.begin(), times.end(), 0.0) / n;

    // Std time
    sqSum = 0;
    for (double t : times) {
        sqSum += (t - stats.meanTime) * (t - stats.meanTime);
    }
    stats.stdTime = std::sqrt(sqSum / n);

    return stats;
}

// Run benchmark for one variant/function combination
RunStats runBenchmark(
    DFOVariant variant,
    FitnessFunction fitnessType,
    const BenchmarkConfig& cfg
) {
    std::vector<double> fitnessResults;
    std::vector<double> timeResults;

    for (int run = 0; run < cfg.numRuns; run++) {
        DFOConfig config;
        config.populationSize = cfg.populationSize;
        config.dimensions = cfg.dimensions;
        config.maxIterations = cfg.maxIterations;
        config.variant = variant;
        config.delta = 0.001;
        config.seed = 12345ULL + run * 1000; // Different seed each run

        DFOOptimizer optimizer(config);

        // Set standard benchmark bounds per function
        switch (fitnessType) {
            case FitnessFunction::SPHERE:
                optimizer.setUniformBounds(-100.0, 100.0);
                break;
            case FitnessFunction::RASTRIGIN:
                optimizer.setUniformBounds(-5.12, 5.12);
                break;
            case FitnessFunction::ROSENBROCK:
                optimizer.setUniformBounds(-30.0, 30.0);
                break;
            case FitnessFunction::ACKLEY:
                optimizer.setUniformBounds(-32.768, 32.768);
                break;
            default:
                optimizer.setUniformBounds(-100.0, 100.0);
        }

        // Suppress output for benchmark runs
        // Run optimization
        DFOResult result = optimizer.optimize(fitnessType);

        fitnessResults.push_back(result.bestFitness);
        timeResults.push_back(result.elapsedTimeMs);

        if (cfg.verbose) {
            printf("Run %d: fitness=%.6e, time=%.2fms\n",
                   run + 1, result.bestFitness, result.elapsedTimeMs);
        }
    }

    return computeStats(fitnessResults, timeResults);
}

// Get variant name
const char* getVariantName(DFOVariant variant) {
    switch (variant) {
        case DFOVariant::STANDARD: return "DFO";
        case DFOVariant::UDFO_1000P: return "uDFO-1000p";
        case DFOVariant::UDFO_1500P: return "uDFO-1500p";
        case DFOVariant::UDFO_Z5: return "uDFO-z5";
        default: return "Unknown";
    }
}

// Get function name
const char* getFunctionName(FitnessFunction func) {
    switch (func) {
        case FitnessFunction::SPHERE: return "Sphere";
        case FitnessFunction::RASTRIGIN: return "Rastrigin";
        default: return "Unknown";
    }
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  DFO/uDFO GPU Benchmark Suite\n");
    printf("========================================\n\n");

    BenchmarkConfig cfg;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            cfg.numRuns = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            cfg.populationSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            cfg.dimensions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            cfg.maxIterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            cfg.outputFile = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            cfg.verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <runs>       Number of independent runs (default: 30)\n");
            printf("  -p <pop>        Population size (default: 100)\n");
            printf("  -d <dims>       Problem dimensions (default: 30)\n");
            printf("  -i <iters>      Max iterations (default: 1000)\n");
            printf("  -o <file>       Output CSV file (default: benchmark_results.csv)\n");
            printf("  -v              Verbose output\n");
            return 0;
        }
    }

    printf("Configuration:\n");
    printf("  Runs: %d\n", cfg.numRuns);
    printf("  Population: %d\n", cfg.populationSize);
    printf("  Dimensions: %d\n", cfg.dimensions);
    printf("  Iterations: %d\n", cfg.maxIterations);
    printf("  Output: %s\n", cfg.outputFile.c_str());
    printf("\n");

    // All variants to test
    DFOVariant variants[] = {
        DFOVariant::STANDARD,
        DFOVariant::UDFO_1000P,
        DFOVariant::UDFO_1500P,
        DFOVariant::UDFO_Z5
    };

    // All functions to test
    FitnessFunction functions[] = {
        FitnessFunction::SPHERE,
        FitnessFunction::RASTRIGIN
    };

    // Open output file
    std::ofstream outFile(cfg.outputFile);
    outFile << "Variant,Function,BestFitness,MeanFitness,StdFitness,MeanTime,StdTime\n";

    // Run benchmarks
    printf("Running benchmarks...\n");
    printf("%-15s %-12s %-15s %-15s %-12s %-12s\n",
           "Variant", "Function", "Best", "Mean±Std", "Time(ms)", "Time Std");
    printf("%-15s %-12s %-15s %-15s %-12s %-12s\n",
           "-------", "--------", "----", "--------", "--------", "--------");

    for (auto func : functions) {
        for (auto variant : variants) {
            printf("Testing %s on %s... ", getVariantName(variant), getFunctionName(func));
            fflush(stdout);

            RunStats stats = runBenchmark(variant, func, cfg);

            printf("Done\n");
            printf("%-15s %-12s %-15.6e %-7.2e±%-6.2e %-12.2f %-12.2f\n",
                   getVariantName(variant),
                   getFunctionName(func),
                   stats.bestFitness,
                   stats.meanFitness,
                   stats.stdFitness,
                   stats.meanTime,
                   stats.stdTime);

            // Write to CSV
            outFile << getVariantName(variant) << ","
                    << getFunctionName(func) << ","
                    << std::scientific << std::setprecision(6)
                    << stats.bestFitness << ","
                    << stats.meanFitness << ","
                    << stats.stdFitness << ","
                    << std::fixed << std::setprecision(2)
                    << stats.meanTime << ","
                    << stats.stdTime << "\n";
        }
        printf("\n");
    }

    outFile.close();
    printf("\nResults saved to %s\n", cfg.outputFile.c_str());

    return 0;
}
