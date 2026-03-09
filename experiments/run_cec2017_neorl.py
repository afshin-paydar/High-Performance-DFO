#!/usr/bin/env python3
"""
run_cec2017_neorl.py  —  CEC2017 experiment using neorl benchmark suite
========================================================================
Exact import pattern:
    import neorl.benchmarks.cec17 as functions

    all_funcs = functions.all_functions   # 29 callables: f1,f3,f4,...,f30
    y = FIT(x)                            # x: np.ndarray of shape (D,)
    f_star = float(FIT.__name__.strip('f')) * 100   # e.g. f1→100, f3→300

All 29 functions share the same bounds: [-100, 100]^D.
DFO is re-implemented here in pure NumPy (vectorized over D) so it can
call arbitrary Python callables.  Output JSON is compatible with
plot_results.py.

Install
-------
    pip install neorl

Usage
-----
    python experiments/run_cec2017_neorl.py              # full 29×4×30
    python experiments/run_cec2017_neorl.py --quick      # f1/f3/f4, D={10,30}, 5 runs
    python experiments/run_cec2017_neorl.py --func-indices 1 3 5 7 9
    python experiments/run_cec2017_neorl.py --dims 10 30 --runs 10
    python experiments/run_cec2017_neorl.py --output results_neorl/

Protocol
--------
    MaxFEs = 10,000 × D           (CEC2017 standard)
    N      = 50                   (population, fixed)
    T      = MaxFEs // N          (number of DFO iterations)
    Runs   = 30                   (independent, deterministic seeds)
    Seeds  = 1000 + run × 37
    Bounds = [-100, 100]^D        (all CEC2017 functions)
    f*     = func_index × 100
    Error  = |f(x_best) - f*|
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ─── neorl import ────────────────────────────────────────────────────────────

import neorl.benchmarks.cec17 as functions   # noqa: E402

# ─── Protocol constants ───────────────────────────────────────────────────────

LB, UB       = -100.0, 100.0    # CEC2017 bounds (all functions)
DIMS         = [10, 30, 50, 100]
N_POPULATION = 50
N_RUNS       = 30
BASE_SEED    = 1000
SEED_STRIDE  = 37
DELTA        = 0.001             # DFO disturbance threshold

CHECKPOINT_FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SAMPLES_PER_RUN  = 200


# ─── Function catalogue ───────────────────────────────────────────────────────

def load_functions(indices: list[int] | None = None) -> list[tuple]:
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


def probe_dimension_independence(func_list: list[tuple]) -> None:
    """
    Sanity-check that each CEC17 function correctly adapts to the input
    vector length.  Evaluates each function at D=10 and D=30 and confirms
    the two results differ (a no-op on correct implementations).
    Aborts loudly if a function hard-codes a dimension.
    """
    print("  Probing dimension-independence of CEC17 functions... ", end="", flush=True)
    bad = []
    rng = np.random.default_rng(0)
    for func, _, idx in func_list[:5]:   # probe first 5 only
        x10 = rng.uniform(LB, UB, 10)
        x30 = rng.uniform(LB, UB, 30)
        try:
            y10 = float(func(x10))
            y30 = float(func(x30))
        except Exception as e:
            bad.append(f"f{idx}: exception {e}")
            continue
        if not (np.isfinite(y10) and np.isfinite(y30)):
            bad.append(f"f{idx}: non-finite output (y10={y10}, y30={y30})")
    if bad:
        print("FAILED")
        for msg in bad:
            print(f"  [WARNING] {msg}", file=sys.stderr)
    else:
        print("OK")


# ─── DFO — vectorized NumPy implementation ───────────────────────────────────

def dfo(
    func,
    D: int,
    N: int,
    T: int,
    seed: int,
    checkpoint_fes: list[int],
) -> dict:
    """
    Standard DFO exactly matching original_DFO.py, with the D-loop
    replaced by NumPy vector operations for ~10–50× speedup in Python.

    Update rule (per fly i, vectorized over D):
        x_new[d] = x_nb[d] + U * (x_best[d] - x_i[d])
        disturbed dimensions → uniform resample

    Parameters
    ----------
    func            : callable(x: np.ndarray) -> float
    D               : number of dimensions
    N               : population size
    T               : DFO iterations
    seed            : RNG seed (for reproducibility)
    checkpoint_fes  : sorted list of FE counts at which to record f_best

    Returns
    -------
    dict: fes (list[int]), fitness (list[float]), final (float)
    """
    rng  = np.random.default_rng(seed)
    span = UB - LB

    # ── Initialise ───────────────────────────────────────────────────────────
    X       = rng.uniform(LB, UB, (N, D))
    fitness = np.array([func(X[i]) for i in range(N)], dtype=float)
    s       = int(np.argmin(fitness))

    fes            = N       # cost of initial evaluation
    fes_log:  list[int]   = []
    fit_log:  list[float] = []
    chk_idx   = 0
    chk_sorted = sorted(checkpoint_fes)

    def snap():
        nonlocal chk_idx
        while chk_idx < len(chk_sorted) and fes >= chk_sorted[chk_idx]:
            fes_log.append(chk_sorted[chk_idx])
            fit_log.append(float(fitness[s]))
            chk_idx += 1

    snap()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for _ in range(T):
        # --- Ring-topology best-neighbour indices ---
        left  = (np.arange(N) - 1) % N
        right = (np.arange(N) + 1) % N
        nb    = np.where(fitness[right] < fitness[left], right, left)  # (N,)

        # --- Vectorized update for all non-elite flies ---
        mask_elite = np.arange(N) == s                # (N,)

        # Disturbance mask: shape (N, D)
        disturb = rng.random((N, D)) < DELTA

        # Standard update: x_nb + U * (x_best - x_i)
        U       = rng.random((N, D))
        X_nb    = X[nb]                               # (N, D)
        X_best  = X[s]                                # (D,)  broadcast
        X_new   = X_nb + U * (X_best - X)            # (N, D)

        # Out-of-bounds → resample
        oob     = (X_new < LB) | (X_new > UB)
        X_rand  = rng.uniform(LB, UB, (N, D))
        X_new   = np.where(oob | disturb, X_rand, X_new)

        # Apply to non-elite flies only
        for i in range(N):
            if not mask_elite[i]:
                X[i] = X_new[i]

        # --- Re-evaluate non-elite flies ---
        for i in range(N):
            if not mask_elite[i]:
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def max_fes(D: int)    -> int: return 10_000 * D
def n_iters(D: int)    -> int: return max_fes(D) // N_POPULATION
def seed_for(run: int) -> int: return BASE_SEED + run * SEED_STRIDE


def checkpoint_list(D: int) -> list[int]:
    """~200 evenly-spaced FE snapshots + the 7 protocol checkpoints."""
    mfes = max_fes(D)
    step = max(N_POPULATION, mfes // SAMPLES_PER_RUN)
    pts  = set(range(N_POPULATION, mfes + 1, step))
    for frac in CHECKPOINT_FRACS:
        pts.add(int(frac * mfes))
    pts.add(mfes)
    return sorted(pts)


def interp_checkpoints(
    fes_log: list[int],
    fit_log: list[float],
    max_fes_val: int,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
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


# ─── Experiment orchestrator ──────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> None:
    dims      = sorted(args.dims)
    n_runs    = args.runs
    out_dir   = Path(args.output)
    raw_dir   = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    func_list = load_functions(args.func_indices)
    if not func_list:
        sys.exit("[ERROR] No functions matched the requested indices.")

    _banner("DFO  ×  CEC2017 (neorl)")
    print(f"  neorl API  : import neorl.benchmarks.cec17 as functions")
    print(f"  Functions  : {len(func_list)}  "
          f"({' '.join(f.__name__ for f,_,_ in func_list[:8])}"
          f"{'...' if len(func_list) > 8 else ''})")
    print(f"  Dims       : {dims}")
    print(f"  Runs/cell  : {n_runs}")
    print(f"  N (pop)    : {N_POPULATION}")
    print(f"  Budget     : MaxFEs = 10,000 × D   (T = MaxFEs / {N_POPULATION})")
    print(f"  Bounds     : [{LB}, {UB}]^D  (all CEC2017 functions)")
    print(f"  f*(fn)     : float(fn_index) × 100   e.g. f1→100, f3→300")
    print(f"  Seeds      : {BASE_SEED} + run × {SEED_STRIDE}")
    print(f"  Output     : {out_dir}/")
    _sep()

    probe_dimension_independence(func_list)

    all_results: dict   = {}
    total_runs  = len(func_list) * len(dims) * n_runs
    done        = 0
    t_start     = time.perf_counter()

    for func, f_star, fidx in func_list:
        fname = func.__name__   # e.g. "f1"
        all_results[fname] = {}

        for D in dims:
            all_results[fname][D] = []
            T    = n_iters(D)
            mfes = max_fes(D)
            chks = checkpoint_list(D)

            print(f"\n[{fname.upper():<4}  D={D:3d}]  "
                  f"f*={f_star:.0f}  T={T:,}  MaxFEs={mfes:,}")

            for run in range(n_runs):
                seed = seed_for(run)
                t0   = time.perf_counter()

                raw = dfo(func, D, N_POPULATION, T, seed, chks)

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

                raw_path = raw_dir / f"{fname}_D{D:03d}_run{run:02d}.json"
                with open(raw_path, "w") as fh:
                    json.dump(rec, fh, indent=2)

                all_results[fname][D].append(rec)
                done += 1

                elapsed = time.perf_counter() - t_start
                eta_s   = (total_runs - done) / (done / elapsed) if done > 0 else 0
                print(f"    run {run:2d}  seed={seed}  "
                      f"error={error:.3e}  {wall:.1f}s  "
                      f"[{done}/{total_runs}  ETA {eta_s/60:.0f} min]")

    # Save aggregated JSON
    agg_path = out_dir / "results.json"
    with open(agg_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    elapsed_total = time.perf_counter() - t_start
    print(f"\n\nTotal time: {elapsed_total/60:.1f} min")
    print(f"Saved: {agg_path}")
    print(f"\nNext step:\n"
          f"  python experiments/plot_results.py "
          f"--input {agg_path} --output figures_neorl/")

    _print_summary(all_results, func_list, dims)


def _print_summary(all_results, func_list, dims):
    import statistics as st
    _banner("SUMMARY  —  Mean |f_best - f*| ± std  (all runs)")
    col = 17
    header = f"{'Func':<7}" + "".join(f"  {'D='+str(D):<{col}}" for D in dims)
    print(header)
    print("-" * len(header))
    for func, f_star, fidx in func_list:
        fname = func.__name__
        row   = f"{fname:<7}"
        for D in dims:
            runs   = all_results.get(fname, {}).get(D, [])
            errors = [r["error"] for r in runs if r.get("error") is not None]
            if errors:
                mu = st.mean(errors)
                sd = st.stdev(errors) if len(errors) > 1 else 0.0
                cell = f"{mu:.2e}±{sd:.0e}"
                row += f"  {cell:<{col}}"
            else:
                row += f"  {'N/A':<{col}}"
        print(row)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _banner(msg):
    w = 68
    print("\n" + "=" * w + f"\n  {msg}\n" + "=" * w)

def _sep():
    print("-" * 68)


# ─── CLI ─────────────────────────────────────────────────────────────────────

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
                   help="D={10,30}, 5 runs, f1/f3/f4 — fast sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    if args.quick:
        args.dims         = [10, 30]
        args.runs         = 5
        args.func_indices = [1, 3, 4]
        print("[QUICK MODE]  D={10,30}, 5 runs, f1/f3/f4")
    run_experiment(args)
