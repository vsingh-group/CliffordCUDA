"""Peak GPU memory for regressive_prod backward. Per-impl isolation, raw peak.
No external witness."""
import csv
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
    fwd_bwd,
    measure_peak_memory,
    not_nan,
    print_mem_table_header,
    print_skip_summary,
    record_carry,
    record_setup_fail,
    results_path,
)

from cliffordcuda.extensions.ga.geom_prod import load_geom_prod_cuda
from cliffordcuda.extensions.ga.regressive_prod import regressive_prod, regressive_prod_skip
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "chunk_skip"]






def _bp_inputs(n, batch):
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(batch, dim, device=device, dtype=dtype)
    return a, b, g


def measure_chunk(n, batch):
    a, b, g = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: regressive_prod(a, b), (a, b), g))


def measure_chunk_skip(n, batch):
    a, b, g = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: regressive_prod_skip(a, b), (a, b), g))


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print("Pre-loading kernels ...")
    _ = load_wedge_prod_cuda(); _ = load_geom_prod_cuda()

    print_mem_table_header(IMPLS, ratio_labels=[f"{r}/chunk" for r in IMPLS[1:]])
    rows = []
    disabled = set()
    skip_log = []
    nan = float("nan")

    for n in n_values:
        dim = 1 << n
        for batch in batch_values:
            cells = {}
            for impl, fn in [
                ("chunk",      measure_chunk),
                ("chunk_skip", measure_chunk_skip),
            ]:
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
                "chunk_mib":      cells["chunk"],
                "chunk_skip_mib": cells["chunk_skip"],
                "chunk_skip_over_chunk": ratios[0],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_regressive_prod_bwd")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
