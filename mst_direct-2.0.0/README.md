# MST-Direct — Matching via Sinkhorn Transport

**Multivariate geostatistical simulation that preserves complex non-linear
dependencies exactly.**

MST-Direct treats the target value tuples as the distribution to reproduce and
finds, via entropy-regularized **optimal transport** (the Sinkhorn algorithm)
with a relational *k*-nearest-neighbor term, a spatial arrangement that
reproduces the variogram while keeping the joint distribution **exactly intact**
(the realization is a permutation of the target tuples). Where Gaussian Copula
and LU Decomposition linearize and destroy bimodal, step, sinusoidal and
heteroscedastic relationships, MST-Direct keeps them.

> 📘 Schmitz, T. B. *MST-Direct: Matching via Sinkhorn Transport for Multivariate
> Geostatistical Simulation with Complex Non-Linear Dependencies.*
> arXiv: [2603.18036](https://arxiv.org/abs/2603.18036)

## What's new in 2.0

Version 2 scales the method to real problems and adds the capabilities the
original (bivariate, unconditional, small-grid) formulation left open:

- **`ScalableMST`** — sparse, candidate-restricted Sinkhorn matcher, `O(n·C)`
  memory; runs 40,000-node grids in under a minute (vs. the dense `MSTDirect`).
- **Multivariate** — handles many variables by matching the target cloud onto an
  FFT-MA Gaussian **backbone** with a prescribed variogram.
- **Conditional simulation** — honors hard data *exactly* by pinning the data
  tuples and conditioning the backbone by simple kriging.
- **PPMT** comparator, scalable variogram estimators, and a histogram-MSE metric.

## Install

```bash
pip install mst-direct            # core (numpy, scipy)
pip install 'mst-direct[plot]'    # + matplotlib for mst_direct.plots
```

## Quick start

```python
import numpy as np
from mst_direct import ScalableMST, gaussian_backbone, grid_coords, shape_preservation

N = 50
coords = grid_coords(N, N)                       # (2500, 2)
cloud  = np.random.default_rng(0).normal(size=(N * N, 3))   # target tuples (any joint)

backbone = gaussian_backbone((N, N), d=3, rng_range=15.0, random_state=0)
sim = ScalableMST(random_state=0).simulate(backbone, cloud, coords=coords)

# joint distribution preserved exactly (sim is a permutation of cloud)
assert np.array_equal(np.sort(sim, 0), np.sort(cloud, 0))
```

### Conditional simulation (honors hard data exactly)

```python
from mst_direct import conditional_gaussian_backbone

data_idx   = np.array([0, 137, 999])             # hard-data grid locations
data_tuples_std = (cloud[[5, 6, 7]] - cloud.mean(0)) / cloud.std(0)
cond_bb = conditional_gaussian_backbone((N, N), coords, data_idx, data_tuples_std,
                                        rng_range=15.0, random_state=0)
sim = ScalableMST(random_state=0).simulate(
    cond_bb, cloud, pinned=(data_idx, np.array([5, 6, 7])), coords=coords)
# sim[data_idx] == cloud[[5, 6, 7]] exactly
```

### v1 API (dense, bivariate) still available

```python
from mst_direct import MSTDirect, generate_dataset, shape_preservation
data = generate_dataset("gaussian_mix", grid=(25, 25), random_state=42)
sim = MSTDirect(random_state=42).simulate(data["values"], data["coords"])
print(shape_preservation(data["values"], sim))   # -> 1.0
```

## API

```
# simulators
MSTDirect(...)                       # v1 dense relational matcher
ScalableMST(beta, n_candidates, k_relational, lam_relational, n_relax, ...)
    .simulate(backbone, cloud, pinned=None, coords=None)
mst_simulate(backbone, cloud, ...)

# backbone (prescribed variogram)
gaussian_backbone(shape, d, rng_range, random_state)
conditional_gaussian_backbone(shape, coords, data_idx, data_gauss, rng_range, ...)

# comparator
PPMT(n_iter, n_dirs, random_state).fit(x).forward(x) / .inverse(y)
ppmt_simulate(cloud, backbone, ...)

# optimal transport / spatial
sinkhorn, sinkhorn_plan, relational_match, greedy_round, knn_adjacency
spherical, exponential, gaussian, empirical_variogram, variogram_correlation
grid_coords, sampled_variogram, sampled_cross_variogram, fit_spherical

# metrics
shape_preservation, histogram2d_similarity, histogram_mse_table

# synthetic data
generate_dataset, make_grid, fft_ma, apply_relationship, RELATIONSHIPS

# plotting (optional: pip install mst-direct[plot])
mst_direct.plots: plot_scatter_matrix, plot_realization, plot_realization_grid,
                  plot_variograms, plot_data_honoring

# baselines (Gaussian Copula / LU) for comparison
mst_direct.baselines: gaussian_copula_simulate, lu_simulate
```

## Examples

`examples/run_unconditional.py` (200×200) and `examples/run_conditional.py`
(100×100, 200 hard data) reproduce the DMS validation experiments with
MST-Direct vs PPMT. They download the reference distribution on first run.

## License

MIT © 2026 **Tcharlies Bachmann Schmitz** — Data Science, PX.Center
