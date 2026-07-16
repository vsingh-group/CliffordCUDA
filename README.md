# cliffordcuda

`cliffordcuda` is focused on doing Clifford (geometric) algebra operations fast
at scale. Most functions are available in arbitrary `Cl(p, q, r)`. They are available as `nn.Module` layers with learnable parameters.

- [Setup](#setup)
- [What's supported](#whats-supported)
- [Data layout: bit-pattern ordering](#data-layout-bit-pattern-ordering)
- [Quickstart](#quickstart)
- [Layers](#layers)
- [Benchmarks and tests](#benchmarks-and-tests)
- [Results](#results)

## Setup

```bash
git clone <this repo> CliffordCUDA
cd CliffordCUDA
pip install -e .
```

Needs an NVIDIA GPU with CUDA, `torch>=2.5`, `numpy`, and `ortools`. Kernels are
compiled on first use. Inputs must be CUDA float32 tensors.

## What's supported

| Operation | Method |
|---|---|
| Geometric product | `geom_prod(a, b)` |
| Wedge (outer) product | `wedge_prod(a, b)` |
| Inner (Hestenes) product | `inner_prod(a, b)` |
| Left contraction | `left_contraction(a, b)` |
| Right contraction | `right_contraction(a, b)` |
| Regressive product | `regressive_prod(a, b)` — non-degenerate metrics only (raises on `r > 0`) |
| Inverse (`x⁻¹`) | `inverse(x)` — versor/blade `x̃/(x x̃)` when it applies, else the matrix-representation inverse; any `Cl(p, q, r)` including degenerate |
| Reverse | `reverse(x)` |
| Grade involution | `grade_involution(x)` |
| Clifford conjugation | `clifford_conjugation(x)` |
| Dual (`x I⁻¹`) | `dual(x)` — non-degenerate metrics only (raises on `r > 0`); `compile=True` to fuse the op |
| Grade projection | `grade_projection(x, k)` — keeps grade k, zeros the rest |
| Norm (`⟨x x̃⟩₀`) | `norm_sq(x)` — metric-weighted squared norm; `compile=True` to fuse |
| Rotor application (`R~ x R`) | `apply_bivector(biv, x)` / `compile_bivector` + `apply_rotor` |

Metrics: any signature `Cl(p, q, r)`, where each generator squares to `+1`,
`-1`, or `0`. Pass it as a list — `[1, 1, 1, 1, -1]` is `Cl(4, 1)`. Degenerate
(`r > 0`) signatures work for everything except the regressive product and the
dual (both need an invertible pseudoscalar).

Products need `n >= 5`; rotor application needs `n >= 7`.

## Data layout: bit-pattern ordering

A multivector in `Cl(n, ...)` is a tensor of shape `(batch, 2^n)`. Component `j`
holds the blade whose basis vectors are the set bits of `j` — bit `i` set means
`e_i` is present:

```
index 0  = 000 = scalar (1)
index 1  = 001 = e0
index 2  = 010 = e1
index 3  = 011 = e0 ^ e1
index 4  = 100 = e2
index 5  = 101 = e0 ^ e2
index 6  = 110 = e1 ^ e2
index 7  = 111 = e0 ^ e1 ^ e2        (Cl(3) shown)
```

So the scalar is index `0` and `e_i` is index `2^i`. This is bit-pattern ordering, not grade-lex / ShortLex. Inputs and outputs use it throughout. This is done to eliminate shared memory bank conflicts.

## Quickstart

```python
import torch
from cliffordcuda import CliffordAlgebra

# Cl(6, 1): signature (+, +, +, +, +, +, -)
cl = CliffordAlgebra(metric=[1, 1, 1, 1, 1, 1, -1], device="cuda")

a = torch.randn(8, cl.dim, device="cuda", requires_grad=True)   # (batch, 2^n) = (8, 128), fp32
b = torch.randn(8, cl.dim, device="cuda")

c = cl.geom_prod(a, b)          # geometric product -> (8, 128)
w = cl.wedge_prod(a, b)         # wedge / inner / left_contraction / ... the same
c.sum().backward()              # gradient flows back into a

# Rotor sandwich R~ x R. The bivector is in lex-pair order (matching
# torch.triu_indices(cl.n, cl.n, offset=1)), with C(n, 2) components:
biv = torch.randn(1, cl.n * (cl.n - 1) // 2, device="cuda")
y   = cl.apply_bivector(biv, a)             # one-shot
cs  = cl.compile_bivector(biv)              # or compile once, apply many
y1, y2 = cl.apply_rotor(cs, a), cl.apply_rotor(cs, b)
```

Each product is also a free function (`from cliffordcuda import geom_prod`) that
infers `n` from the input and takes an optional `metric=`.

## Layers

Each layer holds a learnable parameter:

- `GeometricProductLayer`, `WedgeProductLayer`, `InnerProductLayer`,
  `LeftContractionLayer`, `RightContractionLayer`, `RegressiveProductLayer` —
  a multivector `weight` of shape `(1, 2^n)`; forward computes
  `product(x, weight)`.
- `RotorLayer` — a `bivector` of shape `(1, C(n, 2))`; forward applies `R~ x R`,
  needs `n >= 7`. In `train()` it recompiles the rotor each forward so the
  gradient reaches the bivector; in `eval()` it compiles once and caches.

```python
import torch, torch.nn as nn
from cliffordcuda import CliffordAlgebra, GeometricProductLayer, RotorLayer

cl = CliffordAlgebra(metric=[1] * 8, device="cuda")   # Cl(8, 0), n=8

model = nn.Sequential(
    GeometricProductLayer(cl),
    RotorLayer(cl),
).to("cuda")

x = torch.randn(16, cl.dim, device="cuda")   # (16, 256)
y = model(x)
y.sum().backward()                           # grads reach weight / bivector
```

Layers take an existing `CliffordAlgebra`, so several can share one algebra and
its tables.

## Benchmarks and tests

Install the libraries that the tests and benchmarks compare against:

```bash
pip install pytest matplotlib
cd ..
git clone https://github.com/falesiani/torch_ga.git && pip install -e torch_ga
git clone https://github.com/Concode0/Versor.git    && pip install -e Versor
git clone https://github.com/VersorAI/Versor.git VersorAI
git clone https://github.com/vsingh-group/ComposingLinearLayers.git
git clone -b submission https://github.com/TravisNP/torch_ga_fix.git
cd CliffordCUDA
```

Run the tests:

```bash
pytest tests/correctness
pytest tests/gradcheck
```

Run all the benchmarks and regenerate the plots and tables:

```bash
bash benchmark/scripts/run_all.sh
```

### The rotor benchmark

`RotorLayer` learns the bivector: the rotor is rebuilt and applied each step,
with the gradient reaching the bivector. Of the comparison libraries, only
`Versor` offers a learnable rotor. It builds on [Composing Linear Layers from Irreducibles](https://neurips.cc/virtual/2025/loc/san-diego/poster/115082). The parametrized-rotor benchmark compares `RotorLayer`
against `ComposingLinearLayers`, in two settings: inference (rotor fixed) and
training (rotor relearned each step).

Note that this repo never actually builds the rotor, but instead applies the adjoint action through the bivector parameters as shown in (insert arxiv paper).

## Results

Measured on an A100.

### Raw timings

CliffordCUDA's own numbers (median; µs for speed, MiB for peak memory). The full
tables — with std and the extended-batch rotor rows — are in
`figures/tables.md`.

#### Forward speed (µs)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch 1**  Geometric product | 15.5 | 15.7 | 16.1 | 19.6 | 31.8 | 55.8 |
|  Wedge product | 12.4 | 13.7 | 15.1 | 18.2 | 33.5 | 58.7 |
|  Inner product | 12.1 | 12.7 | 13.8 | 16.2 | 34.2 | 71.2 |
|  Left contraction | 12.2 | 13.3 | 14.9 | 18.6 | 33.7 | 65.9 |
|  Right contraction | 12.2 | 13.1 | 14.9 | 18.5 | 33.9 | 69.4 |
|  Regressive product | 23.3 | 23.1 | 23.6 | 23.7 | 44.6 | 69.3 |
|  Rotor application | 24.5 | 28.1 | 33.2 | 43.1 | 47.0 | 52.6 |
| **Batch 16**  Geometric product | 16.0 | 17.1 | 20.7 | 32.2 | 74.1 | 242.1 |
|  Wedge product | 12.9 | 14.3 | 17.4 | 24.4 | 42.0 | 89.5 |
|  Inner product | 12.9 | 14.6 | 17.0 | 22.4 | 36.2 | 83.0 |
|  Left contraction | 12.9 | 14.1 | 16.4 | 21.9 | 31.2 | 65.4 |
|  Right contraction | 12.7 | 13.9 | 17.5 | 20.4 | 49.4 | 111.0 |
|  Regressive product | 23.9 | 23.7 | 26.8 | 37.7 | 116.3 | 163.7 |
|  Rotor application | 24.8 | 28.2 | 33.9 | 44.0 | 48.3 | 100.6 |
| **Batch 64**  Geometric product | 17.3 | 21.0 | 34.3 | 79.8 | 254.9 | 970.7 |
|  Wedge product | 14.5 | 18.0 | 24.1 | 41.3 | 90.6 | 245.5 |
|  Inner product | 14.7 | 18.2 | 24.4 | 44.1 | 104.8 | 286.9 |
|  Left contraction | 14.2 | 17.2 | 22.0 | 36.8 | 79.5 | 228.3 |
|  Right contraction | 14.1 | 18.7 | 21.6 | 36.6 | 81.2 | 225.9 |
|  Regressive product | 23.9 | 28.3 | 38.1 | 72.0 | 163.8 | 275.6 |
|  Rotor application | 24.8 | 28.9 | 34.6 | 45.0 | 49.3 | 102.9 |

#### Backward speed (µs)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch 1**  Geometric product | 72.5 | 72.0 | 72.5 | 72.9 | 69.7 | 132.5 |
|  Wedge product | 80.7 | 80.0 | 82.5 | 53.6 | 78.9 | 164.5 |
|  Inner product | 78.6 | 78.2 | 78.5 | 78.9 | 92.2 | 157.8 |
|  Left contraction | 78.1 | 78.1 | 79.2 | 78.7 | 76.5 | 148.3 |
|  Right contraction | 78.3 | 78.1 | 86.4 | 78.0 | 76.4 | 137.8 |
|  Regressive product | 87.2 | 89.3 | 89.2 | 89.2 | 98.2 | 149.0 |
| **Batch 16**  Geometric product | 72.0 | 71.5 | 72.1 | 81.0 | 153.9 | 514.5 |
|  Wedge product | 80.7 | 81.8 | 82.9 | 67.4 | 132.1 | 362.0 |
|  Inner product | 77.9 | 78.7 | 79.0 | 89.5 | 173.0 | 452.5 |
|  Left contraction | 77.2 | 78.8 | 78.7 | 83.3 | 137.1 | 381.7 |
|  Right contraction | 77.4 | 54.0 | 84.3 | 82.7 | 200.2 | 367.7 |
|  Regressive product | 88.2 | 89.2 | 88.5 | 92.9 | 254.2 | 403.9 |
| **Batch 64**  Geometric product | 71.4 | 72.3 | 81.4 | 159.7 | 528.8 | 1983.6 |
|  Wedge product | 79.0 | 82.1 | 85.1 | 137.2 | 371.9 | 1331.2 |
|  Inner product | 78.9 | 77.7 | 89.9 | 178.6 | 466.2 | 1638.8 |
|  Left contraction | 77.1 | 78.9 | 84.9 | 143.7 | 403.1 | 1334.3 |
|  Right contraction | 78.1 | 53.7 | 83.2 | 143.5 | 401.9 | 1330.8 |
|  Regressive product | 89.1 | 89.1 | 93.8 | 166.9 | 429.3 | 1374.7 |

#### Forward peak memory (MiB)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch 1**  Geometric product | 0.00 | 0.01 | 0.04 | 0.14 | 0.52 | 2.05 |
|  Wedge product | 0.01 | 0.03 | 0.09 | 0.14 | 0.52 | 2.05 |
|  Inner product | 0.00 | 0.01 | 0.04 | 0.14 | 0.52 | 4.28 |
|  Left contraction | 0.01 | 0.03 | 0.09 | 0.26 | 1.02 | 4.05 |
|  Right contraction | 0.01 | 0.03 | 0.09 | 0.26 | 1.02 | 2.20 |
|  Regressive product | 0.01 | 0.02 | 0.08 | 0.28 | 1.05 | 4.11 |
|  Rotor application | 0.01 | 0.02 | 0.03 | 0.07 | 0.17 | 0.38 |
| **Batch 16**  Geometric product | 0.03 | 0.05 | 0.12 | 0.31 | 0.88 | 2.75 |
|  Wedge product | 0.03 | 0.08 | 0.18 | 0.43 | 1.10 | 2.89 |
|  Inner product | 0.03 | 0.05 | 0.26 | 0.66 | 1.79 | 4.98 |
|  Left contraction | 0.04 | 0.08 | 0.18 | 0.44 | 1.10 | 2.91 |
|  Right contraction | 0.04 | 0.08 | 0.18 | 0.44 | 1.10 | 2.91 |
|  Regressive product | 0.04 | 0.08 | 0.19 | 0.51 | 1.62 | 3.94 |
|  Rotor application | 0.04 | 0.07 | 0.15 | 0.31 | 0.64 | 1.32 |
| **Batch 64**  Geometric product | 0.10 | 0.20 | 0.41 | 0.88 | 2.00 | 5.00 |
|  Wedge product | 0.10 | 0.22 | 0.46 | 1.00 | 2.22 | 5.14 |
|  Inner product | 0.11 | 0.24 | 0.54 | 1.23 | 2.92 | 7.23 |
|  Left contraction | 0.11 | 0.22 | 0.46 | 1.00 | 2.23 | 5.16 |
|  Right contraction | 0.11 | 0.22 | 0.46 | 1.00 | 2.23 | 5.16 |
|  Regressive product | 0.13 | 0.27 | 0.57 | 2.01 | 4.25 | 9.19 |
|  Rotor application | 0.13 | 0.26 | 0.53 | 1.06 | 2.14 | 4.32 |

#### Backward peak memory (MiB)

| Operation | n=7 | n=8 | n=9 | n=10 | n=11 | n=12 |
|---|---|---|---|---|---|---|
| **Batch 1**  Geometric product | 0.01 | 0.03 | 0.11 | 0.40 | 1.55 | 6.09 |
|  Wedge product | 0.01 | 0.04 | 0.17 | 0.77 | 2.55 | 10.09 |
|  Inner product | 0.01 | 0.05 | 0.17 | 0.65 | 2.55 | 10.09 |
|  Left contraction | 0.01 | 0.05 | 0.20 | 0.77 | 3.05 | 12.09 |
|  Right contraction | 0.01 | 0.05 | 0.20 | 0.77 | 3.05 | 12.09 |
|  Regressive product | 0.02 | 0.06 | 0.21 | 0.79 | 3.08 | 12.16 |
| **Batch 16**  Geometric product | 0.05 | 0.12 | 0.28 | 0.75 | 2.25 | 7.50 |
|  Wedge product | 0.06 | 0.13 | 0.34 | 1.00 | 3.75 | 11.50 |
|  Inner product | 0.06 | 0.13 | 0.34 | 1.00 | 3.25 | 11.50 |
|  Left contraction | 0.06 | 0.14 | 0.38 | 1.12 | 3.75 | 13.50 |
|  Right contraction | 0.06 | 0.14 | 0.38 | 1.12 | 3.75 | 13.50 |
|  Regressive product | 0.07 | 0.16 | 0.41 | 1.20 | 3.90 | 13.80 |
| **Batch 64**  Geometric product | 0.19 | 0.40 | 0.84 | 1.88 | 4.50 | 12.00 |
|  Wedge product | 0.20 | 0.41 | 0.94 | 2.25 | 6.00 | 16.00 |
|  Inner product | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 16.00 |
|  Left contraction | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 18.00 |
|  Right contraction | 0.20 | 0.42 | 0.94 | 2.25 | 6.00 | 18.00 |
|  Regressive product | 0.23 | 0.49 | 1.07 | 2.51 | 6.52 | 19.05 |

## License

MIT. See `LICENSE`.

# Disclaimer
This repo was written with the help of Claude Code. If you find any bugs, please open an issue! Same with missing features.