# High-Performance DFO/uDFO
A high-performance CUDA implementation of **Dispersive Flies Optimisation (DFO)** and its unified variant **uDFO** for GPU-accelerated numerical optimization.

al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.

## Overview
This implementation provides:
- **Standard DFO**: Original algorithm with fixed disturbance threshold (Δ = 0.001)
- **uDFO-1000p**: Unified DFO with dynamic Δ = 1/(1000·p)
- **uDFO-1500p**: Unified DFO with dynamic Δ = 1/(1500·p)
- **uDFO-z5**: Unified DFO with zone-5 relocation restart mechanism


## Algorithm Details
### DFO Update Equation
```
x_{id}^{t+1} = x_{in,d}^t + u · (x_{sd}^t - x_{id}^t)
```

Where:
- `x_{id}`: Position of particle i in dimension d at time t
- `x_{in,d}`: Position of best neighbor (ring topology)
- `x_{sd}`: Position of swarm's best particle
- `u ~ U(0,1)`: Random number

### uDFO Dynamic Disturbance
The exploitation probability `p` is calculated based on:
```
p = (L + R - 1 - log(2L)) / (L + R)    when L, R ≥ 1
```
Where L and R are scaled distances to search space bounds.

The dynamic disturbance threshold:
- **uDFO-1000p**: `Δ_dynamic = 1 / (1000 · p)`
- **uDFO-1500p**: `Δ_dynamic = 1 / (1500 · p)`


```bash
make

# Explicitly set architecture
make CUDA_ARCH=sm_61   # GTX 1080 Ti / Pascal
make CUDA_ARCH=sm_75   # RTX 2080 Ti / Turing
make CUDA_ARCH=sm_86   # RTX 3080   / Ampere

# Full architecture list: https://developer.nvidia.com/cuda-gpus

# Build with debug symbols
make DEBUG=1 CUDA_ARCH=sm_61
```

You can also export the variable once for the whole session:
```bash
export CUDA_ARCH=sm_61
make
./dfo_gpu standard sphere 100 30 1000
```


## Usage

### Basic Usage

```bash
./dfo_gpu [variant] [function] [population] [dimensions] [iterations]
```

### Examples

```bash
# Standard DFO on Sphere function
./dfo_gpu standard sphere 100 30 1000

# uDFO-1500p on Rastrigin function
./dfo_gpu udfo1500 rastrigin 200 50 2000

# uDFO-z5 with large population
./dfo_gpu udfoz5 sphere 1000 100 5000

# Run built-in benchmark
./dfo_gpu benchmark 1000 100 5000

# Enable diagnostic debug output (prints every 100 iterations by default)
./dfo_gpu standard rastrigin 500 100 2000 --debug

# Debug output every 50 iterations
./dfo_gpu standard rastrigin 500 100 2000 --debug --debug-interval 50

# --debug and --debug-interval can appear anywhere after the program name
./dfo_gpu --debug standard rastrigin 500 100 2000 --debug-interval 200
```

### Available Functions
| Function | Formula | Bounds | Optimum |
|----------|---------|--------|---------|
| `sphere` | Σx²ᵢ | [-100, 100] | f(0)=0 |
| `rastrigin` | 10D + Σ(x²ᵢ - 10cos(2πxᵢ)) | [-5.12, 5.12] | f(0)=0 |
| `rosenbrock` | Σ 100(xᵢ₊₁-xᵢ²)² + (xᵢ-1)² | [-30, 30] | f(1,…,1)=0 |
| `ackley` | -20e^(-0.2√(Σx²/D)) - e^(Σcos(2πx)/D) + 20+e | [-32.768, 32.768] | f(0)=0 |

### Debug / Diagnostic Flags

| Flag | Description |
|------|-------------|
| `--debug` | Enable per-iteration diagnostic output |
| `--debug-interval K` | Print debug stats every K iterations (default: 100). Implies `--debug`. |


#### What the debug output shows

Each block printed at the configured interval contains:

- **Fitness range** — the true `min`/`max` fitness across all particles, plus a cross-check against `kernelFindGlobalBest`'s returned value. A mismatch here means the warp-reduction kernel returned the wrong particle.
- **Diversity** — unique fitness count, number of fully-collapsed particles (position indistinguishable from the best fly in float32), and exact-fitness duplicates.
- **Movement potential** — average `|x_best − xᵢ|` per (particle, dimension). When this approaches 0 the DFO update formula is effectively dead and only the `delta` escape mechanism can move flies.
- **Float32 dead-zone fraction** — fraction of (particle, dimension) pairs where `|x_best[d] − xᵢ[d]| < 1e-7`. A high value (>50%) means float32 precision is the limiting factor rather than the algorithm itself. This is the primary explanation for the ~0.995 Rastrigin plateau vs. the Python float64 reference.
- **Update trace** — shows `x_current`, `x_neighbor`, `x_best`, and the expected `x_new` (at u=0.5) for dimensions 0–5 of particle 0.
- **Rastrigin dimension audit** — counts how many dimensions of the best fly are near the global optimum (`|x|<0.05`), near the first local minimum (`|x−1|<0.05`), or elsewhere. Directly explains the residual fitness value.


## References
1. al-Rifaie, M.M. (2014). "Dispersive Flies Optimisation", Proceedings of the 2014 Federated Conference on Computer Science and Information Systems, 535-544. IEEE.

2. al-Rifaie, M.M. (2021). "Exploration and Exploitation Zones in a Minimalist Swarm Optimiser", Entropy, 23(8), 977.
