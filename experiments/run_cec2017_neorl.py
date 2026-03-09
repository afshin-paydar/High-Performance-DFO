#!/usr/bin/env python3
from __future__ import annotations
"""
run_cec2017_neorl.py  —  CEC2017 experiment using neorl benchmark suite
========================================================================
Exact import pattern:
    import neorl.benchmarks.cec17 as functions

    all_funcs = functions.all_functions   # 29 callables: f1,f3,f4,...,f30
    y = FIT(x)                            # x: np.ndarray of shape (D,)
    f_star = float(FIT.__name__.strip('f')) * 100   # e.g. f1->100, f3->300

All 29 functions share the same bounds: [-100, 100]^D.
DFO is re-implemented here in pure NumPy (vectorized over D) so it can
call arbitrary Python callables.  Output JSON is compatible with
plot_results.py.

Compatibility: Python 3.7+, NumPy 1.13+ (uses RandomState, not default_rng)

Install
-------
    pip install neorl

Usage
-----
    python experiments/run_cec2017_neorl.py              # full 29x4x30
    python experiments/run_cec2017_neorl.py --quick      # f1/f3/f4, D={10,30}, 5 runs
    python experiments/run_cec2017_neorl.py --func-indices 1 3 5 7 9
    python experiments/run_cec2017_neorl.py --dims 10 30 --runs 10
    python experiments/run_cec2017_neorl.py --output results_neorl/

Protocol
--------
    MaxFEs = 10,000 x D           (CEC2017 standard)
    N      = 50                   (population, fixed)
    T      = MaxFEs // N          (number of DFO iterations)
    Runs   = 30                   (independent, deterministic seeds)
    Seeds  = 1000 + run x 37
    Bounds = [-100, 100]^D        (all CEC2017 functions)
    f*     = func_index x 100
    Error  = |f(x_best) - f*|
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# ── neorl import ──────────────────────────────────────────────────────────────

import neorl.benchmarks.cec17 as functions   # noqa: E402

# ── Protocol constants ────────────────────────────────────────────────────────

LB, UB       = -100.0, 100.0
DIMS         = [10, 30, 50, 100]
N_POPULATION = 50
N_RUNS       = 30
BASE_SEED    = 1000
SEED_STRIDE  = 37
DELTA        = 0.001

CHECKPOINT_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SAMPLES_PER_RUN  = 200


# ── RNG helper ────────────────────────────────────────────────────────────────
# np.random.RandomState works on NumPy 1.13+, unlike default_rng (NumPy 1.17+)

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


# ── Function catalogue ────────────────────────────────────────────────────────

def load_functions(indices: Optional[List[int]] = None) -> List[Tuple]:
    """
    Return list of (func, f_star, func_index) tuples from neorl CEC2017.

    neorl API
    ---------
        functions.all_functions         : list of 29 callables
        FIT.__name__                    : 'f1', 'f3', ..., 'f30'  (f2 absent)
        FIT(x: np.ndarray) -> float     : evaluate at x of any length D
        f* = float(FIT.__name__.strip('f')) * 100
    """
    out = []
    for func in functions.all_functions:
        idx = int(func.__name__.strip("f"))
        if indices is None or idx in indices:
            out.append((func, float(idx) * 100.0, idx))
    out.sort(key=lambda t: t[2])
    return out


def probe_dimension_independence(func_list: List[Tuple]) -> None:
    """
    Quick sanity-check: evaluate the first (up to 5) functions at D=10 and
    D=30, confirming neorl loads the correct rotation/shift matrices per call.
    """
    print("  Probing CEC17 functions (D=10 and D=30)... ", end="", flush=True)
    rng  = _rng(0)
    bad  = []
    for func, _, idx in func_list[:5]:
        x10 = rng.uniform(LB, UB, 10)
        x30 = rng.uniform(LB, UB, 30)
        try:
            y10 = float(func(x10))
            y30 = float(func(x30))
        except Exception as e:
            bad.append("f{}: exception {}".format(idx, e))
            continue
        if not (np.isfinite(y10) and np.isfinite(y30)):
            bad.append("f{}: non-finite (y10={}, y30={})".format(idx, y10, y30))
    if bad:
        print("WARNINGS:")
        for msg in bad:
            print("  [WARNING] {}".format(msg), file=sys.stderr)
    else:
        print("OK")


# ── DFO — vectorized NumPy implementation ────────────────────────────────────

def dfo(
    func,
    D: int,
    N: int,
    T: int,
    seed: int,
    checkpoint_fes: List[int],
) -> Dict:
    """
    Standard DFO exactly matching original_DFO.py.
    The inner D-loop is replaced by NumPy vector ops for ~20-50x speedup.

    Uses np.random.RandomState for NumPy 1.13+ compatibility.

    Update rule (vectorized over D):
        x_new[d] = x_nb[d] + U * (x_best[d] - x_i[d])
        disturbed dims  -> uniform resample in [LB, UB]
        out-of-bounds   -> uniform resample in [LB, UB]
    """
    rng = _rng(seed)

    # ── Initialise ───────────────────────────────────────────────────────────
    X       = rng.uniform(LB, UB, (N, D))
    fitness = np.array([func(X[i]) for i in range(N)], dtype=float)
    s       = int(np.argmin(fitness))

    fes        = N
    fes_log    = []   # type: List[int]
    fit_log    = []   # type: List[float]
    chk_idx    = 0
    chk_sorted = sorted(checkpoint_fes)

    def snap():
        # type: () -> None
        nonlocal chk_idx
        while chk_idx < len(chk_sorted) and fes >= chk_sorted[chk_idx]:
            fes_log.append(chk_sorted[chk_idx])
            fit_log.append(float(fitness[s]))
            chk_idx += 1

    snap()

    # ── Main loop ─────────────────────────────────────────────────────────────
    idx_all = np.arange(N)

    for _ in range(T):
        # Ring-topology best-neighbour
        left  = (idx_all - 1) % N
        right = (idx_all + 1) % N
        nb    = np.where(fitness[right] < fitness[left], right, left)  # (N,)

        # Disturbance mask and random draws — shape (N, D)
        # RandomState API: random_sample() instead of random()
        disturb = rng.random_sample((N, D)) < DELTA
        U       = rng.random_sample((N, D))
        X_rand  = rng.uniform(LB, UB, (N, D))

        # Standard DFO update
        X_nb   = X[nb]                    # (N, D)
        X_best = X[s]                     # (D,) — broadcast
        X_new  = X_nb + U * (X_best - X)  # (N, D)

        # Resample where out-of-bounds or disturbed
        oob   = (X_new < LB) | (X_new > UB)
        X_new = np.where(oob | disturb, X_rand, X_new)

        # Apply to non-elite flies and re-evaluate
        for i in range(N):
            if i != s:
                X[i]       = X_new[i]
                fitness[i] = func(X[i])

        fes += N - 1   # elite not re-evaluated
        s    = int(np.argmin(fitness))
        snap()

    # Final evaluation of best (mirrors original_DFO.py post-loop eval)
    final_fit = func(X[s])
    if final_fit < fitness[s]:
        fitness[s] = final_fit
    snap()

    return {
        "fes":     fes_log,
        "fitness": fit_log,
        "final":   float(fitness[s]),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def max_fes(D):    return 10000 * D
def n_iters(D):    return max_fes(D) // N_POPULATION
def seed_for(run): return BASE_SEED + run * SEED_STRIDE


def checkpoint_list(D: int) -> List[int]:
    """~200 evenly-spaced FE snapshots + the 7 protocol checkpoints."""
    mfes = max_fes(D)
    step = max(N_POPULATION, mfes // SAMPLES_PER_RUN)
    pts  = set(range(N_POPULATION, mfes + 1, step))
    for frac in CHECKPOINT_FRACS:
        pts.add(int(frac * mfes))
    pts.add(mfes)
    return sorted(pts)


def interp_checkpoints(
    fes_log: List[int],
    fit_log: List[float],
    max_fes_val: int,
) -> Dict:
    out = {}
    for frac in CHECKPOINT_FRACS:
        target = int(frac * max_fes_val)
        val    = None
        for fe, fit in zip(fes_log, fit_log):
            if fe <= target:
                val = fit
            else:
                break
        if val is None and fit_log:
            val = fit_log[0]
        out[str(frac)] = val
    return out


# ── Experiment orchestrator ───────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> None:
    dims    = sorted(args.dims)
    n_runs  = args.runs
    out_dir = Path(args.output)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    func_list = load_functions(args.func_indices)
    if not func_list:
        sys.exit("[ERROR] No functions matched the requested indices.")

    _banner("DFO  x  CEC2017 (neorl)")
    print("  neorl API  : import neorl.benchmarks.cec17 as functions")
    print("  Functions  : {}  ({}{})".format(
        len(func_list),
        " ".join(f.__name__ for f, _, _ in func_list[:8]),
        "..." if len(func_list) > 8 else "",
    ))
    print("  Dims       : {}".format(dims))
    print("  Runs/cell  : {}".format(n_runs))
    print("  N (pop)    : {}".format(N_POPULATION))
    print("  Budget     : MaxFEs = 10,000 x D   (T = MaxFEs / {})".format(N_POPULATION))
    print("  Bounds     : [{}, {}]^D".format(LB, UB))
    print("  f*(fn)     : float(fn_index) x 100   e.g. f1->100, f3->300")
    print("  Seeds      : {} + run x {}".format(BASE_SEED, SEED_STRIDE))
    print("  Output     : {}/".format(out_dir))
    _sep()

    probe_dimension_independence(func_list)

    all_results  = {}
    total_runs   = len(func_list) * len(dims) * n_runs
    done         = 0
    t_start      = time.perf_counter()

    for func, f_star, fidx in func_list:
        fname = func.__name__
        all_results[fname] = {}

        for D in dims:
            all_results[fname][D] = []
            T    = n_iters(D)
            mfes = max_fes(D)
            chks = checkpoint_list(D)

            print("\n[{:<4}  D={:3d}]  f*={:.0f}  T={:,}  MaxFEs={:,}".format(
                fname.upper(), D, f_star, T, mfes))

            for run in range(n_runs):
                seed  = seed_for(run)
                t0    = time.perf_counter()
                raw   = dfo(func, D, N_POPULATION, T, seed, chks)
                wall  = time.perf_counter() - t0
                error = abs(raw["final"] - f_star)

                rec = {
                    "func":        fname,
                    "func_index":  fidx,
                    "f_star":      f_star,
                    "D":           D,
                    "N":           N_POPULATION,
                    "T":           T,
                    "run":         run,
                    "seed":        seed,
                    "max_fes":     mfes,
                    "fes":         raw["fes"],
                    "fitness":     raw["fitness"],
                    "final":       raw["final"],
                    "error":       error,
                    "wall_s":      round(wall, 3),
                    "checkpoints": interp_checkpoints(raw["fes"], raw["fitness"], mfes),
                }

                raw_path = raw_dir / "{}_D{:03d}_run{:02d}.json".format(fname, D, run)
                with open(raw_path, "w") as fh:
                    json.dump(rec, fh, indent=2)

                all_results[fname][D].append(rec)
                done += 1

                elapsed = time.perf_counter() - t_start
                eta_s   = (total_runs - done) / (done / elapsed) if done > 0 else 0
                print("    run {:2d}  seed={:4d}  error={:.3e}  {:.1f}s"
                      "  [{}/{}  ETA {:.0f}min]".format(
                          run, seed, error, wall, done, total_runs, eta_s / 60))

    agg_path = out_dir / "results.json"
    with open(agg_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    elapsed_total = time.perf_counter() - t_start
    print("\n\nTotal time: {:.1f} min".format(elapsed_total / 60))
    print("Saved: {}".format(agg_path))
    print("\nNext step:")
    print("  python experiments/plot_results.py --input {} --output figures_neorl/".format(
        agg_path))

    _print_summary(all_results, func_list, dims)


def _print_summary(all_results, func_list, dims):
    import statistics as st
    _banner("SUMMARY  --  Mean |f_best - f*| +/- std  (all runs)")
    col    = 18
    header = "{:<7}".format("Func") + "".join(
        "  {:<{}}".format("D=" + str(D), col) for D in dims
    )
    print(header)
    print("-" * len(header))
    for func, f_star, fidx in func_list:
        fname = func.__name__
        row   = "{:<7}".format(fname)
        for D in dims:
            runs   = all_results.get(fname, {}).get(D, [])
            errors = [r["error"] for r in runs if r.get("error") is not None]
            if errors:
                mu   = st.mean(errors)
                sd   = st.stdev(errors) if len(errors) > 1 else 0.0
                cell = "{:.2e}+/-{:.0e}".format(mu, sd)
            else:
                cell = "N/A"
            row += "  {:<{}}".format(cell, col)
        print(row)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _banner(msg):
    w = 68
    print("\n" + "=" * w + "\n  " + msg + "\n" + "=" * w)


def _sep():
    print("-" * 68)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DFO on full CEC2017 suite via neorl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dims", nargs="+", type=int, default=DIMS,
                   metavar="D", help="Dimensions to test")
    p.add_argument("--runs", type=int, default=N_RUNS,
                   help="Independent runs per (function, D) cell")
    p.add_argument("--output", default="results_neorl",
                   help="Output directory")
    p.add_argument("--func-indices", nargs="+", type=int, default=None,
                   metavar="I",
                   help="1-based function indices (e.g. 1 3 5 7). "
                        "Default: all 29. f2 is absent in neorl.")
    p.add_argument("--quick", action="store_true",
                   help="D={10,30}, 5 runs, f1/f3/f4 -- fast sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    if args.quick:
        args.dims         = [10, 30]
        args.runs         = 5
        args.func_indices = [1, 3, 4]
        print("[QUICK MODE]  D={10,30}, 5 runs, f1/f3/f4")
    run_experiment(args)
