# DFO — CEC2017-style Experimental Protocol

## What this produces

### Step 1: Build
```bash
cd ..        # project root
make
```

### Step 2: Run experiment
```bash
# Full protocol: 4 functions × 4 dims × 30 runs = 480 DFO runs
python experiments/run_cec2017.py --binary ./dfo_gpu

# Sanity check first (D={10,30}, 5 runs, finishes in ~2 min)
python experiments/run_cec2017.py --quick

# Custom configuration
python experiments/run_cec2017.py \
    --variant standard \
    --dims 10 30 50 100 \
    --runs 30 \
    --functions sphere rastrigin rosenbrock ackley \
    --output results/
```

### Step 3: Generate figures and tables
```bash
python experiments/plot_results.py --input results/results.json --output figures/
```

---

## Alternative: Full CEC2017 suite via neorl

`run_cec2017_neorl.py` runs DFO against all 29 CEC2017 functions (`f1`, `f3`, ..., `f30`)
using the neorl benchmark library. It is a pure-NumPy DFO and does **not** use `dfo_gpu`.

```bash
pip install neorl

# Full protocol: 29 functions × 4 dims × 30 runs
python experiments/run_cec2017_neorl.py --output results_neorl/

# Sanity check (f1/f3/f4, D={10,30}, 5 runs)
python experiments/run_cec2017_neorl.py --quick

# Custom dimensions and run count
python experiments/run_cec2017_neorl.py --dims 10 30 50 100 --runs 10

# Specific function indices
python experiments/run_cec2017_neorl.py --func-indices 1 3 5 7 9
```

**Supported dimensions**: neorl ships rotation matrices only for `D ∈ {2, 10, 20, 30, 50, 100}`.
Requesting any other dimension (e.g. 300, 500) will print a warning and skip that dimension.

Note: this script does **not** accept `--variant` or `--functions`; it always runs
standard DFO on all neorl CEC2017 functions (bounds fixed at `[-100, 100]^D`).

```bash
python experiments/plot_results.py --input results_neorl/results.json --output figures_neorl/
```

---

## Protocol details

| Parameter | Value | Justification |
|---|---|---|
| Functions | Sphere, Rastrigin, Rosenbrock, Ackley | Four landscape classes (unimodal, separable multimodal, non-separable ill-conditioned, deceptive) |
| Dimensions | D ∈ {10, 30, 50, 100} | CEC2017 standard checkpoints |
| Population | N = 50 | Fixed; enables cross-D comparison without N-scaling confound |
| Budget | MaxFEs = 10,000 × D | CEC2017 standard |
| Iterations | T = MaxFEs / N | Derived from budget |
| Independent runs | 30 | Minimum for Wilcoxon rank-sum at α = 0.05 |
| Seeds | 1000 + run × 37 | Deterministic; fully reproducible |

---

## Outputs

### Figures
| File | Content |
|---|---|
| `fig1_convergence_D30.pdf` | Convergence curves (log scale), D=30, all functions — median + IQR band |
| `fig2_convergence_sphere.pdf` | Sphere convergence for all D on normalised x-axis + box plots |
| `fig3_heatmap.pdf` | Heatmap: mean log₁₀ error, functions × D |
| `fig4_boxplots.pdf` | Box plots of final error distribution |
| `fig5_improvement_rate.pdf` | Rolling improvement rate (ratchet speed) vs FEs |
| `fig6_collapse_profile.pdf` | Mean/median final error vs D — dimensional collapse profile |

### Tables
| File | Content |
|---|---|
| `table1_summary.tex/csv` | Mean ± std, best, worst, median, success rate |
| `table2_wilcoxon.tex/csv` | Pairwise Wilcoxon rank-sum test, all D pairs |

---

## Source change to dfo_gpu.cu
A `--seed S` flag was added to `parseArgs()` so each of the 30 runs uses a
different, fixed, reproducible seed (`--seed` overrides the previous
`time(nullptr)` default). All other behaviour is unchanged.
