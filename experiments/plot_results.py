#!/usr/bin/env python3
"""
plot_results.py  —  Analysis and publication-quality figures for DFO experiment
================================================================================

Works with output from EITHER:
    run_cec2017.py        (sphere/rastrigin/rosenbrock/ackley)
    run_cec2017_neorl.py  (f1,f3,...,f30 from neorl CEC2017)

Auto-detects which format the JSON uses.

Figures generated
-----------------
  fig1_convergence_D30.{fmt}    Convergence curves at D=30, all functions
  fig2_convergence_single.{fmt} Convergence for a single function, all D
                                (sphere if available, else first function)
  fig3_heatmap.{fmt}            Mean log10-error heatmap: functions × D
  fig4_boxplots.{fmt}           Box plots of final error distribution
  fig5_improvement_rate.{fmt}   Rolling improvement rate (ratchet speed)
  fig6_collapse_profile.{fmt}   Mean/median final error vs D

Tables generated
----------------
  table1_summary.tex + .csv     Mean, std, median, best, worst, SR(%)
  table2_wilcoxon.tex + .csv    Pairwise Wilcoxon rank-sum across D

Usage
-----
  python experiments/plot_results.py
  python experiments/plot_results.py --input results_cec17_neorl/results.json
  python experiments/plot_results.py --output figures/ --format pdf
  python experiments/plot_results.py --d-ref 30      # reference D for fig1
  python experiments/plot_results.py --func-ref f3   # reference func for fig2
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ─── Matplotlib style ────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

SUCCESS_TOL = 1e-8   # |error| < this → success

# Up to 29 distinct colours for CEC17 functions
_PALETTE_29 = [
    "#1f77b4","#e6550d","#2ca02c","#9467bd","#8c564b",
    "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8",
    "#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94",
    "#f7b6d2","#c7c7c7","#dbdb8d","#9edae5","#393b79",
    "#637939","#8c6d31","#843c39","#7b4173","#3182bd",
    "#e6550d","#31a354","#756bb1","#636363",
]

DIM_COLORS = {10: "#1f77b4", 30: "#e6550d", 50: "#2ca02c", 100: "#9467bd"}


# ─── Dataset description ──────────────────────────────────────────────────────

class Dataset:
    """
    Wraps results.json and provides a uniform query interface regardless
    of whether it came from run_cec2017.py (4 named functions) or
    run_cec2017_neorl.py (f1,f3,...,f30).
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        # Detect format and enumerate available functions + dimensions
        self.func_names: list[str] = sorted(data.keys())
        self.dims: list[int] = []
        for fname in self.func_names:
            for k in data[fname]:
                try:
                    self.dims.append(int(k))
                except ValueError:
                    pass
        self.dims = sorted(set(self.dims))

        # Build f* map
        self.f_star: dict[str, float] = {}
        for fname in self.func_names:
            # Check first run for stored f_star (neorl format)
            for D in self.dims:
                runs = self._runs(fname, D)
                if runs and "f_star" in runs[0]:
                    self.f_star[fname] = float(runs[0]["f_star"])
                    break
            if fname not in self.f_star:
                # Simple benchmark format: f* = 0 for all
                if fname.startswith("f") and fname[1:].isdigit():
                    self.f_star[fname] = float(fname[1:]) * 100.0
                else:
                    self.f_star[fname] = 0.0

        # Human-readable label
        self.label: dict[str, str] = {}
        _cec17_class = {
            "f1": "Unimodal", "f3": "Unimodal",
            **{f"f{i}": "Multimodal" for i in range(4, 11)},
            **{f"f{i}": "Hybrid" for i in range(11, 21)},
            **{f"f{i}": "Composition" for i in range(21, 31)},
        }
        _simple_label = {
            "sphere": "Sphere", "rastrigin": "Rastrigin",
            "rosenbrock": "Rosenbrock", "ackley": "Ackley",
        }
        for fname in self.func_names:
            if fname in _simple_label:
                self.label[fname] = _simple_label[fname]
            else:
                cls = _cec17_class.get(fname, "")
                self.label[fname] = f"{fname.upper()}  [{cls}]"

    def _runs(self, fname: str, D: int) -> list[dict]:
        dmap = self._data.get(fname, {})
        return dmap.get(D, dmap.get(str(D), []))

    def errors(self, fname: str, D: int) -> np.ndarray:
        runs = self._runs(fname, D)
        if not runs:
            return np.array([])
        fstar = self.f_star[fname]
        vals  = []
        for r in runs:
            # Prefer stored 'error' field (neorl format), else compute
            if "error" in r and r["error"] is not None:
                vals.append(float(r["error"]))
            elif r.get("final") is not None:
                vals.append(abs(float(r["final"]) - fstar))
        return np.array(vals, dtype=float)

    def convergence(
        self, fname: str, D: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (fes_grid, median_error, p25_error, p75_error)."""
        runs  = self._runs(fname, D)
        fstar = self.f_star[fname]
        if not runs:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Common FE grid
        all_fes = []
        for r in runs:
            all_fes.extend(r.get("fes", []))
        if not all_fes:
            return np.array([]), np.array([]), np.array([]), np.array([])

        fes_grid = np.unique(sorted(all_fes))
        matrix   = []
        for r in runs:
            fes_r = np.array(r.get("fes", []))
            fit_r = np.array(r.get("fitness", []))
            if len(fes_r) == 0:
                continue
            err_r = np.abs(fit_r - fstar)
            row   = np.interp(fes_grid, fes_r, err_r,
                              left=err_r[0], right=err_r[-1])
            row   = np.maximum(row, 1e-300)
            matrix.append(row)

        if not matrix:
            return np.array([]), np.array([]), np.array([]), np.array([])

        M = np.array(matrix)
        return (fes_grid,
                np.median(M, axis=0),
                np.percentile(M, 25, axis=0),
                np.percentile(M, 75, axis=0))


# ─── Figure helpers ───────────────────────────────────────────────────────────

def _fes_label(x, _):
    if x >= 1_000_000:
        return f"{x/1e6:.1f}M"
    if x >= 1000:
        return f"{x/1000:.0f}K"
    return str(int(x))


def _save(fig, path: Path, fmt: str) -> None:
    p = path.with_suffix(f".{fmt}")
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: {p}")


# ─── Figure 1: Convergence curves at D=D_ref, all functions ──────────────────

def fig1_convergence_all_funcs(ds: Dataset, d_ref: int, out: Path, fmt: str) -> None:
    funcs = [f for f in ds.func_names if d_ref in ds.dims]
    n     = len(funcs)
    if n == 0:
        print(f"  [skip fig1] No functions available at D={d_ref}")
        return

    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.2))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    fig.suptitle(
        f"DFO Convergence  —  D={d_ref},  N=50,  MaxFEs={10000*d_ref:,},  30 runs\n"
        "Median error ± IQR band  (log scale)",
        fontsize=11, y=1.01,
    )

    for ax, (fname, color) in zip(axes_flat, zip(funcs, _PALETTE_29)):
        fes, med, p25, p75 = ds.convergence(fname, d_ref)
        if len(fes) == 0:
            ax.set_visible(False)
            continue

        ax.fill_between(fes, p25, p75, alpha=0.18, color=color)
        ax.semilogy(fes, med, color=color, lw=1.8)
        ax.set_xlabel("FEs", fontsize=8)
        ax.set_ylabel(r"$|f - f^*|$", fontsize=8)
        ax.set_title(ds.label[fname], fontsize=8, pad=4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fes_label))
        ax.set_xlim(left=0)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, out / f"fig1_convergence_D{d_ref}", fmt)


# ─── Figure 2: Single function, all D ────────────────────────────────────────

def fig2_convergence_one_func(
    ds: Dataset, fname: str, out: Path, fmt: str
) -> None:
    dims = [D for D in ds.dims if len(ds.errors(fname, D)) > 0]
    if not dims:
        print(f"  [skip fig2] No data for {fname}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for D in dims:
        fes, med, p25, p75 = ds.convergence(fname, D)
        if len(fes) == 0:
            continue
        c    = DIM_COLORS.get(D, "#333333")
        frac = fes / (10_000 * D)
        ax1.fill_between(frac, p25, p75, alpha=0.15, color=c)
        ax1.semilogy(frac, med, color=c, lw=2.0, label=f"D={D}")

    ax1.set_xlabel("FEs / MaxFEs (fraction of budget)")
    ax1.set_ylabel(r"$|f - f^*|$  (log scale)")
    ax1.set_title(f"{ds.label[fname]} — convergence by D\n(normalised budget axis)")
    ax1.legend(title="Dimension", loc="upper right")
    ax1.set_xlim(0, 1)

    # Box plots
    data   = [ds.errors(fname, D) for D in dims]
    data   = [np.maximum(d, 1e-300) for d in data]
    bp     = ax2.boxplot(data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", lw=2))
    for patch, D in zip(bp["boxes"], dims):
        patch.set_facecolor(DIM_COLORS.get(D, "#888888"))
        patch.set_alpha(0.6)
    ax2.set_yscale("log")
    ax2.set_xticks(range(1, len(dims) + 1))
    ax2.set_xticklabels([f"D={D}" for D in dims])
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel(r"Final $|f - f^*|$")
    ax2.set_title(f"{ds.label[fname]} — final error vs D\n30 runs, log scale")

    fig.suptitle(
        f"{ds.label[fname]}  —  N=50,  30 runs,  f*={ds.f_star[fname]:.0f}",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out / f"fig2_convergence_{fname}", fmt)


# ─── Figure 3: Heatmap ───────────────────────────────────────────────────────

def fig3_heatmap(ds: Dataset, out: Path, fmt: str) -> None:
    funcs = ds.func_names
    dims  = ds.dims

    mat   = np.full((len(funcs), len(dims)), np.nan)
    annot = [[""] * len(dims) for _ in funcs]

    for i, fname in enumerate(funcs):
        for j, D in enumerate(dims):
            errs = ds.errors(fname, D)
            if len(errs) == 0:
                continue
            mu = float(np.mean(errs))
            mat[i, j]   = math.log10(mu) if mu > 0 else -10.0
            annot[i][j] = f"{mu:.1e}"

    vmin = float(np.nanmin(mat)) - 0.5 if not np.all(np.isnan(mat)) else -10
    vmax = float(np.nanmax(mat)) + 0.5 if not np.all(np.isnan(mat)) else 0

    figH = max(4, len(funcs) * 0.4)
    fig, ax = plt.subplots(figsize=(max(6, len(dims) * 1.8), figH))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(r"$\log_{10}$(mean error)", fontsize=9)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"D={D}" for D in dims])
    ax.set_yticks(range(len(funcs)))
    ax.set_yticklabels([ds.label[f] for f in funcs], fontsize=8)
    ax.set_title(
        "Mean final error  (log₁₀)  —  DFO, N=50, 30 runs\n"
        "Green = small error  ·  Red = large error",
        pad=8,
    )

    for i in range(len(funcs)):
        for j in range(len(dims)):
            if annot[i][j] and not np.isnan(mat[i, j]):
                norm_val = (mat[i, j] - vmin) / (vmax - vmin + 1e-9)
                tc = "white" if norm_val > 0.65 else "black"
                ax.text(j, i, annot[i][j], ha="center", va="center",
                        fontsize=7, color=tc)

    fig.tight_layout()
    _save(fig, out / "fig3_heatmap", fmt)


# ─── Figure 4: Box plots ──────────────────────────────────────────────────────

def fig4_boxplots(ds: Dataset, out: Path, fmt: str) -> None:
    funcs = ds.func_names
    dims  = ds.dims

    # Layout: up to 4 per row
    cols = min(4, len(funcs))
    rows = math.ceil(len(funcs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    axes_flat = np.array(axes).flatten() if len(funcs) > 1 else [axes]

    for ax, fname in zip(axes_flat, funcs):
        data   = [np.maximum(ds.errors(fname, D), 1e-300) for D in dims
                  if len(ds.errors(fname, D)) > 0]
        labels = [f"D={D}" for D in dims if len(ds.errors(fname, D)) > 0]
        valid_dims = [D for D in dims if len(ds.errors(fname, D)) > 0]
        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=1.8),
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for patch, D in zip(bp["boxes"], valid_dims):
            patch.set_facecolor(DIM_COLORS.get(D, "#888"))
            patch.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(r"$|f - f^*|$", fontsize=8)
        ax.set_title(ds.label[fname], fontsize=8, pad=4)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(funcs):]:
        ax.set_visible(False)

    fig.suptitle(
        "DFO — Final error distribution by dimension  (30 runs, log scale)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out / "fig4_boxplots", fmt)


# ─── Figure 5: Improvement rate ──────────────────────────────────────────────

def fig5_improvement_rate(ds: Dataset, out: Path, fmt: str) -> None:
    funcs = ds.func_names
    dims  = ds.dims

    WINDOW_FRAC = 0.10
    cols        = min(4, len(funcs))
    rows        = math.ceil(len(funcs) / cols)
    fig, axes   = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.2))
    axes_flat   = np.array(axes).flatten() if len(funcs) > 1 else [axes]

    for ax, fname in zip(axes_flat, funcs):
        for D in dims:
            runs = ds._runs(fname, D)
            if not runs:
                continue

            ir_runs = []
            ref_fes = None
            for r in runs:
                fes_r = np.array(r.get("fes", []))
                fit_r = np.array(r.get("fitness", []))
                if len(fes_r) < 3:
                    continue
                mfes  = max(fes_r)
                w     = max(1, int(WINDOW_FRAC * mfes))
                ir_v  = []
                for k in range(1, len(fes_r)):
                    in_win  = fes_r[:k] >= (fes_r[k] - w)
                    improved = np.sum(np.diff(fit_r[:k+1]) < 0)
                    ir_v.append(improved / max(1, in_win.sum()))
                ir_runs.append(np.array(ir_v))
                if ref_fes is None:
                    ref_fes = fes_r[1:]

            if not ir_runs or ref_fes is None:
                continue

            min_len = min(len(x) for x in ir_runs)
            ir_mat  = np.array([x[:min_len] for x in ir_runs])
            fes_ax  = ref_fes[:min_len]
            median  = np.nanmedian(ir_mat, axis=0)

            ax.plot(fes_ax, median,
                    color=DIM_COLORS.get(D, "#888"), lw=1.6, label=f"D={D}")

        ax.axhline(0.30, color="#2ca02c", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.05, color="#d62728", ls="--", lw=1, alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("FEs", fontsize=8)
        ax.set_ylabel("IR", fontsize=8)
        ax.set_title(ds.label[fname], fontsize=8, pad=4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fes_label))
        ax.legend(title="D", fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(funcs):]:
        ax.set_visible(False)

    fig.suptitle(
        "Improvement Rate (ratchet speed)  —  DFO, N=50, 30 runs\n"
        f"Rolling window = {int(WINDOW_FRAC*100)}% of MaxFEs  ·  "
        "dashed: Phase I (0.30) / stagnation (0.05) thresholds",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out / "fig5_improvement_rate", fmt)


# ─── Figure 6: Collapse profile ──────────────────────────────────────────────

def fig6_collapse_profile(ds: Dataset, out: Path, fmt: str) -> None:
    funcs = ds.func_names
    dims  = ds.dims

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for fname, color in zip(funcs, _PALETTE_29):
        means, medians, vdims = [], [], []
        for D in dims:
            errs = ds.errors(fname, D)
            if len(errs) == 0:
                continue
            means.append(float(np.mean(errs)))
            medians.append(float(np.median(errs)))
            vdims.append(D)
        if not vdims:
            continue
        ax1.semilogy(vdims, means,   color=color, marker="o",
                     lw=1.6, ms=6, label=ds.label[fname])
        ax2.semilogy(vdims, medians, color=color, marker="s",
                     lw=1.6, ms=6, label=ds.label[fname])

    for ax, title in [(ax1, "Mean"), (ax2, "Median")]:
        ax.set_xlabel("Dimension D")
        ax.set_ylabel(r"$|f - f^*|$  (log scale)")
        ax.set_title(f"{title} error vs D  —  DFO, N=50")
        ax.set_xticks(dims)
        if len(funcs) <= 12:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Dimensional collapse profile", fontsize=11)
    fig.tight_layout()
    _save(fig, out / "fig6_collapse_profile", fmt)


# ─── Table 1: Summary statistics ─────────────────────────────────────────────

def table1_summary(ds: Dataset, out: Path) -> None:
    import statistics as st

    rows   = []
    header = ["Function", "f*", "D", "Mean", "Std", "Median", "Best", "Worst", "SR(%)"]

    for fname in ds.func_names:
        fstar = ds.f_star[fname]
        for D in ds.dims:
            errs = ds.errors(fname, D)
            if len(errs) == 0:
                rows.append([ds.label[fname], f"{fstar:.0f}", D] + ["N/A"] * 6)
                continue
            mu  = float(np.mean(errs))
            sd  = float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0
            med = float(np.median(errs))
            sr  = float(np.mean(errs < SUCCESS_TOL) * 100)
            rows.append([
                ds.label[fname], f"{fstar:.0f}", D,
                f"{mu:.4e}", f"{sd:.4e}", f"{med:.4e}",
                f"{float(np.min(errs)):.4e}", f"{float(np.max(errs)):.4e}",
                f"{sr:.1f}",
            ])

    # CSV
    csv_path = out / "table1_summary.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  Saved: {csv_path}")

    # LaTeX
    tex_path = out / "table1_summary.tex"
    with open(tex_path, "w") as f:
        f.write("% Table 1: DFO summary statistics\n")
        f.write("% Error = |f_best - f*| over 30 independent runs\n")
        f.write("% SR = fraction of runs with error < 1e-8\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write("\\caption{DFO results on CEC2017 benchmark "
                "($|f_{best}-f^*|$, $N=50$, MaxFEs$=10^4D$, 30 runs)}\n")
        f.write("\\label{tab:dfo}\n\\small\n")
        f.write("\\begin{tabular}{llrcccccr}\n\\toprule\n")
        f.write("Func & $f^*$ & $D$ & Mean & Std & Median & Best & Worst & SR\\%\\\\\n")
        f.write("\\midrule\n")
        prev = None
        for row in rows:
            if row[0] != prev and prev is not None:
                f.write("\\midrule\n")
            prev = row[0]
            f.write(" & ".join(str(v) for v in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex_path}")


# ─── Table 2: Wilcoxon tests ──────────────────────────────────────────────────

def table2_wilcoxon(ds: Dataset, out: Path) -> None:
    dims = ds.dims
    rows = []
    tex_store: list[tuple] = []

    for fname in ds.func_names:
        for i, D1 in enumerate(dims):
            for j, D2 in enumerate(dims):
                if j <= i:
                    continue
                e1 = ds.errors(fname, D1)
                e2 = ds.errors(fname, D2)
                if len(e1) < 3 or len(e2) < 3:
                    continue
                _, p = stats.mannwhitneyu(e1, e2, alternative="two-sided")
                if p < 0.05:
                    sym = "+" if np.median(e1) > np.median(e2) else "-"
                else:
                    sym = "≈"
                rows.append([ds.label[fname], D1, D2,
                             f"{p:.4e}", sym, "yes" if p < 0.05 else "no"])
                tex_store.append((fname, D1, D2, p, sym))

    csv_path = out / "table2_wilcoxon.csv"
    with open(csv_path, "w") as f:
        f.write("Function,D_row,D_col,p_value,symbol,significant\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  Saved: {csv_path}")

    tex_path = out / "table2_wilcoxon.tex"
    with open(tex_path, "w") as f:
        f.write("% Table 2: Wilcoxon rank-sum pairwise comparisons across D\n")
        f.write("% +: row D significantly WORSE  (p<0.05, median_row > median_col)\n")
        f.write("% -: row D significantly BETTER (p<0.05, median_row < median_col)\n")
        f.write("% ≈: no significant difference\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write("\\caption{Wilcoxon pairwise comparison of final errors across dimensions "
                "($\\alpha=0.05$, two-sided).}\n")
        f.write("\\label{tab:wilcoxon}\n")
        f.write("\\begin{tabular}{l" + "c" * len(dims) + "}\n\\toprule\n")
        f.write("& " + " & ".join(f"$D={D}$" for D in dims) + " \\\\\n\\midrule\n")
        for fname in ds.func_names:
            f.write(f"\\multicolumn{{{len(dims)+1}}}{{l}}"
                    f"{{\\textit{{{ds.label[fname]}}}}} \\\\\n")
            for D1 in dims:
                cells = []
                for D2 in dims:
                    if D1 == D2:
                        cells.append("—")
                        continue
                    sym = "—"
                    for (fn, d1, d2, p, s) in tex_store:
                        if fn == fname and d1 == D1 and d2 == D2:
                            sym = s; break
                        if fn == fname and d1 == D2 and d2 == D1:
                            sym = "+" if s == "-" else ("-" if s == "+" else "≈")
                            break
                    cells.append(sym)
                f.write(f"  $D={D1}$ & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {tex_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_path = Path(args.input)
    if not results_path.exists():
        print(f"[ERROR] Not found: {results_path}", file=sys.stderr)
        print("  Run run_cec2017.py or run_cec2017_neorl.py first.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        raw = json.load(f)

    # Normalise dimension keys to both int and str for uniform access
    data_norm = {}
    for fname, dmap in raw.items():
        data_norm[fname] = {}
        for k, v in dmap.items():
            data_norm[fname][int(k)] = v
            data_norm[fname][str(k)] = v

    ds  = Dataset(data_norm)
    fmt = args.format

    print(f"Input  : {results_path}")
    print(f"Output : {out}/  (format: {fmt})")
    print(f"Funcs  : {ds.func_names}")
    print(f"Dims   : {ds.dims}")
    print()

    # Choose reference dim and func for fig1/fig2
    d_ref    = args.d_ref if args.d_ref in ds.dims else (
                30 if 30 in ds.dims else ds.dims[-1])
    f_ref    = args.func_ref if args.func_ref in ds.func_names else ds.func_names[0]

    print("Figures…")
    fig1_convergence_all_funcs(ds, d_ref, out, fmt)
    fig2_convergence_one_func(ds, f_ref, out, fmt)
    fig3_heatmap(ds, out, fmt)
    fig4_boxplots(ds, out, fmt)
    fig5_improvement_rate(ds, out, fmt)
    fig6_collapse_profile(ds, out, fmt)

    print("\nTables…")
    table1_summary(ds, out)
    table2_wilcoxon(ds, out)

    print(f"\nAll outputs in: {out}/")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate plots and tables from DFO CEC2017 experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    default="results/results.json")
    p.add_argument("--output",   default="figures")
    p.add_argument("--format",   default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--d-ref",    type=int, default=30,
                   help="Reference D for fig1 (convergence of all functions at this D)")
    p.add_argument("--func-ref", default="sphere",
                   help="Reference function for fig2 (convergence across all D)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_cli())
