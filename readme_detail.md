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


### Available Functions
| Function | Formula | Bounds | Optimum |
|----------|---------|--------|---------|
| `sphere` | Σx²ᵢ | [-100, 100] | f(0)=0 |
| `rastrigin` | 10D + Σ(x²ᵢ - 10cos(2πxᵢ)) | [-5.12, 5.12] | f(0)=0 |
| `rosenbrock` | Σ 100(xᵢ₊₁-xᵢ²)² + (xᵢ-1)² | [-30, 30] | f(1,…,1)=0 |
| `ackley` | -20e^(-0.2√(Σx²/D)) - e^(Σcos(2πx)/D) + 20+e | [-32.768, 32.768] | f(0)=0 |
