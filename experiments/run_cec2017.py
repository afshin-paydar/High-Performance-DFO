#!/usr/bin/env python3
from __future__ import annotations
"""
run_cec2017.py  —  CEC2017-style experimental protocol for DFO
===============================================================
Iterations × Dimensions analysis with proper academic metrics.

Protocol
--------
  Functions : Sphere (unimodal), Rastrigin (separable multimodal),
              Rosenbrock (non-separable, ill-conditioned),
              Ackley (deceptive flat outer region)
  Dimensions: D ∈ {10, 30, 50, 100}
  Population: N = 50  (fixed)
  Budget    : MaxFEs = 10,000 × D  (CEC2017 standard)
  Iterations: T = MaxFEs // N
  Runs      : 30 independent runs per (function, D) cell
  Seeds     : deterministic — seed_r = BASE_SEED + r * SEED_STRIDE

Output
------
  results/raw/  — per-run JSON files
  results/      — aggregated results.json

Usage
-----
  python experiments/run_cec2017.py                   # full experiment
  python experiments/run_cec2017.py --quick           # D={10,30}, 5 runs
  python experiments/run_cec2017.py --functions sphere rastrigin
  python experiments/run_cec2017.py --dims 10 30 50
  python experiments/run_cec2017.py --runs 10
  python experiments/run_cec2017.py --binary ./dfo_gpu
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ─── Protocol constants ──────────────────────────────────────────────────────

FUNCTIONS    = ["sphere", "rastrigin", "rosenbrock", "ackley"]
DIMS         = [10, 30, 50, 100]
N_POPULATION = 50
N_RUNS       = 30
BASE_SEED    = 1000          # seed_r = BASE_SEED + r * SEED_STRIDE
SEED_STRIDE  = 37

# Global minimum for each function
F_STAR = {
    "sphere":     0.0,
    "rastrigin":  0.0,
    "rosenbrock": 0.0,
    "ackley":     0.0,
}

# Convergence checkpoints as fractions of MaxFEs
CHECKPOINTS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

# ~200 progress samples per run → smooth convergence curves without flood output
SAMPLES_PER_RUN = 200


# ─── Helpers ─────────────────────────────────────────────────────────────────

def max_fes(D: int) -> int:
    return 10_000 * D


def n_iterations(D: int, N: int = N_POPULATION) -> int:
    return max_fes(D) // N


def print_interval(T: int) -> int:
    return max(1, T // SAMPLES_PER_RUN)


def seed_for_run(run: int) -> int:
    return BASE_SEED + run * SEED_STRIDE


def parse_output(stdout: str, N: int, T: int) -> dict[str, Any]:
    """
    Parse dfo_gpu stdout into a structured record.

    Parses lines:
        Iteration: 500    Best fitness: 1.2345678901e+02
        Final best fitness:   1.0512165885e+01
        Elapsed time: 123.45 ms
    """
    iterations_log: list[int]   = []
    fitness_log:    list[float] = []
    final_fitness: float | None = None
    elapsed_ms:    float | None = None

    for line in stdout.splitlines():
        m = re.match(r'Iteration:\s*(\d+)\s+Best fitness:\s*([\deE+\-\.]+)', line)
        if m:
            iterations_log.append(int(m.group(1)))
            fitness_log.append(float(m.group(2)))
            continue

        m = re.match(r'Final best fitness:\s*([\deE+\-\.]+)', line)
        if m:
            final_fitness = float(m.group(1))
            continue

        m = re.match(r'Elapsed time:\s*([\d\.]+)\s*ms', line)
        if m:
            elapsed_ms = float(m.group(1))

    # Ensure final value always appears in log
    if final_fitness is not None:
        if not iterations_log or iterations_log[-1] < T:
            iterations_log.append(T)
            fitness_log.append(final_fitness)
        else:
            fitness_log[-1] = final_fitness   # prefer the explicit final line

    return {
        "iterations": iterations_log,
        "fes":        [it * N for it in iterations_log],
        "fitness":    fitness_log,
        "final":      final_fitness,
        "elapsed_ms": elapsed_ms,
    }


def interpolate_at_checkpoints(
    record: dict, max_fes_val: int
) -> dict[float, float | None]:
    """
    Last-observation-carried-forward at each checkpoint fraction.
    """
    fes_list = record["fes"]
    fit_list = record["fitness"]
    out: dict[float, float | None] = {}

    for frac in CHECKPOINTS:
        target = int(frac * max_fes_val)
        val    = None
        for fe, fit in zip(fes_list, fit_list):
            if fe <= target:
                val = fit
            else:
                break
        # Fallback: use earliest available value
        if val is None and fit_list:
            val = fit_list[0]
        out[frac] = val
    return out


def run_single(
    binary: str,
    func: str,
    D: int,
    run: int,
    variant: str = "standard",
    verbose: bool = False,
    n_population: int = N_POPULATION,
    iterations: int | None = None,
) -> dict[str, Any]:
    """Execute one DFO run and return parsed result dict."""
    N = n_population
    T = iterations if iterations is not None else n_iterations(D, N)
    pi   = print_interval(T)
    seed = seed_for_run(run)

    cmd = [
        binary, variant, func,
        str(N), str(D), str(T),
        "--print-interval", str(pi),
        "--seed", str(seed),
    ]

    t0   = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\n  [ERROR] {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {proc.stderr[:500]}", file=sys.stderr)
        return {
            "error": proc.stderr, "func": func, "D": D,
            "run": run, "final": None
        }

    record = parse_output(proc.stdout, N, T)
    record.update({
        "func":     func,
        "D":        D,
        "N":        N,
        "T":        T,
        "run":      run,
        "seed":     seed,
        "max_fes":  max_fes(D),
        "variant":  variant,
        "wall_s":   round(wall, 3),
    })
    record["checkpoints"] = {
        str(k): v
        for k, v in interpolate_at_checkpoints(record, max_fes(D)).items()
    }

    if verbose:
        final = record.get("final")
        fs    = f"{final:.4e}" if final is not None else "N/A"
        print(f"    run {run:2d}  seed={seed}  T={T:,}  final={fs}  {wall:.1f}s")

    return record


# ─── Experiment orchestrator ──────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> None:
    binary    = args.binary
    functions = args.functions
    dims      = sorted(args.dims)
    n_runs    = args.runs
    variant   = args.variant
    n_pop     = args.population
    fixed_T   = args.iterations   # None means derive from budget
    out_dir   = Path(args.output)
    raw_dir   = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not Path(binary).exists():
        print(f"[ERROR] Binary not found: {binary}", file=sys.stderr)
        print("  Build first:  cd <project-root> && make", file=sys.stderr)
        sys.exit(1)

    all_results: dict = {}
    total_runs = len(functions) * len(dims) * n_runs
    done       = 0
    t_start    = time.perf_counter()

    _banner("DFO — CEC2017-style Experiment")
    print(f"  Binary   : {binary}")
    print(f"  Variant  : {variant}")
    print(f"  Functions: {functions}")
    print(f"  Dims     : {dims}")
    print(f"  Runs/cell: {n_runs}")
    print(f"  N (pop)  : {n_pop}")
    if fixed_T is not None:
        print(f"  Iterations: T = {fixed_T:,}  (fixed, overrides CEC2017 budget)")
    else:
        print(f"  Budget   : MaxFEs = 10,000 × D   (T = MaxFEs / {n_pop})")
    print(f"  Seeds    : {BASE_SEED} + run × {SEED_STRIDE}")
    print(f"  Output   : {out_dir}/")
    _sep()

    for func in functions:
        all_results[func] = {}
        for D in dims:
            all_results[func][D] = []
            T   = fixed_T if fixed_T is not None else n_iterations(D, n_pop)
            fes = T * n_pop
            print(f"\n[{func.upper():<12s}  D={D:3d}]  T={T:,}  MaxFEs={fes:,}")

            for run in range(n_runs):
                rec = run_single(
                    binary, func, D, run,
                    variant=variant, verbose=True, n_population=n_pop,
                    iterations=T,
                )
                # Save individual run
                raw_path = raw_dir / f"{func}_D{D:03d}_run{run:02d}.json"
                with open(raw_path, "w") as fh:
                    json.dump(rec, fh, indent=2)

                all_results[func][D].append(rec)
                done += 1

                elapsed = time.perf_counter() - t_start
                rate    = done / elapsed
                eta_s   = (total_runs - done) / rate if rate > 0 else 0
                print(
                    f"      [{done:4d}/{total_runs}]  ETA {eta_s/60:.1f} min",
                    end="\r", flush=True,
                )

    # Save aggregated JSON
    agg_path = out_dir / "results.json"
    with open(agg_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n\nTotal time: {elapsed_total/60:.1f} min")
    print(f"Saved: {agg_path}")

    print_summary(all_results, functions, dims)


def print_summary(all_results: dict, functions: list, dims: list) -> None:
    import statistics
    _banner("QUICK SUMMARY  —  Mean error ± std  (over all runs)")
    col = 18
    header = f"{'Function':<14}" + "".join(f"  {'D='+str(D):<{col}}" for D in dims)
    print(header)
    print("-" * len(header))
    for func in functions:
        row = f"{func:<14}"
        for D in dims:
            runs   = all_results.get(func, {}).get(D, [])
            finals = [r["final"] for r in runs if r.get("final") is not None]
            if finals:
                errs = [abs(v - F_STAR[func]) for v in finals]
                mu   = statistics.mean(errs)
                sd   = statistics.stdev(errs) if len(errs) > 1 else 0.0
                row += f"  {mu:.2e}±{sd:.0e}"
                row += " " * max(0, col - len(f"{mu:.2e}±{sd:.0e}") - 2)
            else:
                row += f"  {'N/A':<{col}}"
        print(row)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    w = 65
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def _sep() -> None:
    print("-" * 65)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CEC2017-style DFO experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--binary",    default="./dfo_gpu")
    p.add_argument("--variant",   default="standard",
                   choices=["standard", "udfo1000", "udfo1500", "udfoz5"])
    p.add_argument("--functions", nargs="+", default=FUNCTIONS, choices=FUNCTIONS)
    p.add_argument("--dims",      nargs="+", type=int, default=DIMS)
    p.add_argument("--runs",       type=int, default=N_RUNS)
    p.add_argument("--population", type=int, default=N_POPULATION,
                   help="Population size N (default: %(default)s)")
    p.add_argument("--iterations", type=int, default=None,
                   help="Fixed iteration count T (default: 10,000×D/N per CEC2017 budget)")
    p.add_argument("--output",    default="results")
    p.add_argument("--quick",     action="store_true",
                   help="D={10,30}, 5 runs — quick sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    if args.quick:
        args.dims = [10, 30]
        args.runs = 5
        print("[QUICK MODE]  D={10,30}, 5 runs per cell")
    run_experiment(args)
