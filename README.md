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

```

For debug flags and references, see [readme_detail.md](readme_detail.md).

## References
1. al-Rifaie, M.M. (2014). "Dispersive Flies Optimisation", Proceedings of the 2014 Federated Conference on Computer Science and Information Systems, 535-544. IEEE.

2. al-Rifaie, M.M. (2021). "Exploration and Exploitation Zones in a Minimalist Swarm Optimiser", Entropy, 23(8), 977.
