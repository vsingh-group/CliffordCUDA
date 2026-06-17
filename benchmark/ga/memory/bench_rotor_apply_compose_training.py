"""Training peak memory: full step on a single rotor layer.

Each measured pass includes the rotor REBUILD from the current bivector,
the forward apply, and the backward pass that lands a gradient on the
bivector parameter. Per-impl isolation via `full_cleanup()` between
cells, raw peak via `max_memory_allocated`.
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
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

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_repo_root, os.pardir, "ComposingLinearLayers"))
from rotor_layer import Rotor as ComposeRotor

from cliffordcuda import CliffordAlgebra


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = [1, 16, 64, 256, 1024, 4096, 8192]
IMPLS = ["chunk", "Compose"]



def _measure_step(step):
    """Run `step()` twice to absorb first-call allocations, then measure
    raw peak across one more call."""
    for _ in range(2):
        step()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def measure_chunk(n, batch):
    dim = 1 << n
    cl = CliffordAlgebra(metric=[1] * n, device=device)
    bivector = torch.nn.Parameter(
        torch.randn(1, cl._num_basis_biv, device=device, dtype=dtype))
    x_bp = torch.randn(batch, dim, device=device, dtype=dtype)

    def step():
        if bivector.grad is not None:
            bivector.grad = None
        cs = cl.compile_bivector(bivector)
        y = cl.apply_rotor(cs, x_bp)
        y.sum().backward()

    peak = _measure_step(step)
    del cl, bivector, x_bp
    return peak


def measure_compose(n, batch):
    dim = 1 << n
    c_m = ComposeRotor(in_dim=dim, out_dim=dim, in_chunks=1, out_chunks=1,
                       chunk_size=dim, single_rotor=True, alpha_param=False,
                       bias_param=False, device=device, dtype=dtype)
    c_m.train()
    x_bp = torch.randn(batch, dim, device=device, dtype=dtype)

    def step():
        c_m.zero_grad(set_to_none=True)
        c_m._update_rotors()
        y = c_m(x_bp)
        y.sum().backward()

    peak = _measure_step(step)
    del c_m, x_bp
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
        for batch in batch_values:
            cells = {}
            for impl, fn in [("chunk", measure_chunk), ("Compose", measure_compose)]:
                if impl in disabled:
                    record_carry(skip_log, impl, n)
                    cells[impl] = nan
                    continue
                full_cleanup()
                try:
                    cells[impl] = fn(n, batch)
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
                "chunk_mib":   cells["chunk"],
                "compose_mib": cells["Compose"],
                "compose_over_chunk": ratios[0],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_rotor_apply_compose_training")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
