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
