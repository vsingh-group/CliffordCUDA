"""Peak GPU memory for right_contract forward. Per-impl isolation, raw peak.

Impls measured: chunk, subset, einsum. No torch_ga or Versor — torch_ga
ships no right-contract primitive and Versor only ships a bivector x
vector specialization.
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
    DEFAULT_BATCH_VALUES,
    DEFAULT_N_VALUES,
    bp_ab,
    format_mem_row,
    full_cleanup,
    measure_peak_memory,
    not_nan,
    print_mem_table_header,
    print_skip_summary,
    record_carry,
    record_setup_fail,
    results_path,
)

from cliffordcuda.extensions.ga.right_contract import (
    build_right_contract_subset_lut, right_contract, right_contract_subset_grade,
    load_contract_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    load_inner_prod_subset_grade_cuda,
)
from _einsum_refs import EinsumRightContract


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "subset", "einsum"]






def measure_chunk(n, batch):
    a, b = bp_ab(n, batch)
    return measure_peak_memory(lambda: right_contract(a, b))


def measure_subset(n, batch):
    a, b = bp_ab(n, batch)
    return measure_peak_memory(lambda: right_contract_subset_grade(a, b))


def measure_einsum(n, batch):
    op = EinsumRightContract(n, device=device, dtype=dtype)
    a, b = bp_ab(n, batch)
    return measure_peak_memory(lambda: op(a, b))


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print("Pre-loading kernels and LUTs ...")
    _ = load_contract_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    for n in n_values:
        build_right_contract_subset_lut(n, device)

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
                ("chunk",  measure_chunk),
                ("subset", measure_subset),
                ("einsum", measure_einsum),
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
                "chunk_mib":  cells["chunk"],
                "subset_mib": cells["subset"],
                "einsum_mib": cells["einsum"],
                "subset_over_chunk": ratios[0],
                "einsum_over_chunk": ratios[1],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_right_contract")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
