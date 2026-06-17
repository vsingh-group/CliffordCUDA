"""Peak GPU memory for rotor application backward across (n, batch).

Columns:
  chunk     `cl.apply_rotor(cs, x)` forward + `y.sum().backward()`,
            where cs is a leaf with `requires_grad=True` built once in a
            throwaway pass (flushed before measurement; same pattern as
            the rotor memory fwd bench's chunk column). The eigh-driven
            `compile_bivector` cost is deliberately OUT of the measured
            region — this bench reports the apply-backward footprint,
            not the rotor-construction footprint.
  torch_ga  `TorchGARotor` train step: zero_grad, forward through two
            `mv_multiply` calls, backward, then `_update_rotors()`. R is
            a random subeven element (no autograd path to the bivector
            because upstream's `exp` fails on randn bivectors at n>=9 in
            fp32). Memory includes the dense (D, D, D) Cayley.

Versor and einsum are excluded from the bwd bench:
  * Versor's `exp` detaches the simple-plane directions in `no_grad()`
    (Pence et al. decomposition), so its bivector gradient is the
    gradient of a different function than chunk's eigh-derived one.
  * einsum's R / R_rev would come from Versor's `exp`, so it inherits
    the same different-gradient story.

Single-process cleanup before each impl matches the fwd memory bench:
drop class-level caches (`TorchGARotor._state`,
`VersorRotor._algebras`, `_CACHED_TABLES`), gc.collect(),
`_cuda_clearCublasWorkspaces`, `empty_cache`.

Each cell reports raw peak (`torch.cuda.max_memory_allocated`) across one
training step, with `reset_peak_memory_stats` called immediately before
the timed step.
"""
import csv
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
    DEFAULT_BATCH_VALUES,
    DEFAULT_N_VALUES,
    format_mem_row,
    full_cleanup,
    not_nan,
    print_mem_table_header,
    print_skip_summary,
    record_carry,
    record_setup_fail,
    results_path,
)
from _cayley import shortlex_to_bp
from _rotor_apply_helpers import TorchGARotor, VersorRotor
from core.algebra import CliffordAlgebra as _VersorCliffordAlgebra

from cliffordcuda import CliffordAlgebra


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "torch_ga"]




def measure_chunk(n, batch, x_bp):
    """Peak memory for the apply-rotor backward against a precomputed cs.

    Matches the rotor speed bwd bench (and the rotor memory fwd bench):
    cs is built once in a throwaway pass, then bivector/eigh/cuSOLVER
    workspaces are flushed before measurement. The timed step is then
    just forward + backward through `apply_rotor`, with the gradient
    flowing to cs as a leaf (not the unconstrained bivector)."""
    # 1. Throwaway compile + flush.
    cl_tmp = CliffordAlgebra(metric=[1] * n, device=device)
    bivec = torch.randn(1, cl_tmp._num_basis_biv, device=device, dtype=dtype)
    with torch.no_grad():
        cs = cl_tmp.compile_bivector(bivec).detach().requires_grad_(True)
    del cl_tmp, bivec
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

    # 2. Rebuild structural cl; time apply forward + backward only.
    cl = CliffordAlgebra(metric=[1] * n, device=device)

    def step():
        if cs.grad is not None:
            cs.grad = None
        y = cl.apply_rotor(cs, x_bp)
        y.sum().backward()

    step()  # warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del cl, cs
    return peak


def measure_torch_ga(n, batch, x_sl):
    dim = 1 << n
    m = TorchGARotor(dim=dim, device=device, dtype=dtype)
    m.train(); m._update_rotors()

    def step():
        m.zero_grad(set_to_none=True)
        y = m(x_sl)
        y.sum().backward()
        m._update_rotors()

    step()  # warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del m
    return peak


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print_mem_table_header(
        IMPLS, ratio_labels=[f"{r}/chunk" for r in IMPLS[1:]],
    )
    rows = []
    disabled = set()
    skip_log = []
    nan = float("nan")

    for n in n_values:
        dim = 1 << n
        sl_to_bp = shortlex_to_bp(n).to(device)
        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)
            x_sl = x_bp.index_select(-1, sl_to_bp).contiguous()
            cells = {}

            for impl in IMPLS:
                if impl in disabled:
                    record_carry(skip_log, impl, n)
                    cells[impl] = nan
                    continue
                full_cleanup()
                try:
                    if impl == "chunk":
                        cells[impl] = measure_chunk(n, batch, x_bp)
                    elif impl == "torch_ga":
                        cells[impl] = measure_torch_ga(n, batch, x_sl)
                except torch.cuda.OutOfMemoryError:
                    cells[impl] = nan
                    record_setup_fail(skip_log, impl, n, RuntimeError("OOM"))
                    disabled.add(impl)
                except Exception as e:
                    cells[impl] = nan
                    record_setup_fail(skip_log, impl, n, e)
                    disabled.add(impl)
            full_cleanup()

            chunk = cells["chunk"]
            ratios = [(cells[r] / chunk) if not_nan(chunk) and not_nan(cells[r]) else nan
                      for r in IMPLS[1:]]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_mib": cells["chunk"],
                "ga_mib":    cells["torch_ga"],
                "ga_over_chunk": ratios[0],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
            del x_bp, x_sl
        del sl_to_bp
        print()

    path = results_path("ga/memory", "bench_rotor_apply_bwd")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
