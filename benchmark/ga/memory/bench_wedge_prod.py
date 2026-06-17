"""Peak GPU memory for wedge_prod forward across (n, batch). Per-impl
isolation (rotor-bench pattern), raw `max_memory_allocated()` over one
forward call."""
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
    measure_peak_memory,
    not_nan,
    print_mem_table_header,
    print_skip_summary,
    record_carry,
    record_setup_fail,
    results_path,
)

from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda, wedge_prod
from cliffordcuda.extensions.ga.wedge_prod.subset_grade import (
    build_wedge_subset_lut, load_wedge_prod_subset_grade_cuda,
    wedge_prod_subset_grade,
)
from _cayley import build_outer_cayley, shortlex_to_bp
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumWedge

from _harness import import_versorai
versorai_algebra = import_versorai()

from torch_ga.mv_ops import mv_multiply


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "subset", "einsum", "torch_ga", "Versor*", "VersorAI"]




def measure_chunk(n, batch):
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    return measure_peak_memory(lambda: wedge_prod(a, b))


def measure_subset(n, batch):
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    return measure_peak_memory(lambda: wedge_prod_subset_grade(a, b))


def measure_torch_ga(n, batch):
    dim = 1 << n
    cayley = build_outer_cayley(n, device=device)
    sl_to_bp = shortlex_to_bp(n).to(device)
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    a_sl = a.index_select(-1, sl_to_bp).contiguous()
    b_sl = b.index_select(-1, sl_to_bp).contiguous()
    return measure_peak_memory(lambda: mv_multiply(a_sl, b_sl, cayley))


def measure_einsum(n, batch):
    dim = 1 << n
    op = EinsumWedge(n, device=device, dtype=dtype)
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    return measure_peak_memory(lambda: op(a, b))


def measure_versor(n, batch):
    dim = 1 << n
    versor_alg = VersorAlgebra(p=n, q=0, r=0, device=device)
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    return measure_peak_memory(lambda: versor_alg.wedge(a, b))


def measure_versorai(n, batch):
    dim = 1 << n
    sig = torch.ones(n, dtype=dtype, device=device)
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)
    return measure_peak_memory(
        lambda: versorai_algebra.wedge_product(a, b, sig))


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print("Pre-loading kernels and LUTs ...")
    _ = load_wedge_prod_cuda()
    _ = load_wedge_prod_subset_grade_cuda()
    for n in n_values:
        build_wedge_subset_lut(n, device)

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
                ("chunk",    measure_chunk),
                ("subset",   measure_subset),
                ("einsum",   measure_einsum),
                ("torch_ga", measure_torch_ga),
                ("Versor*",  measure_versor),
                ("VersorAI", measure_versorai),
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
                "chunk_mib":    cells["chunk"],
                "subset_mib":   cells["subset"],
                "einsum_mib":   cells["einsum"],
                "ga_mib":       cells["torch_ga"],
                "versor_mib":   cells["Versor*"],
                "versorai_mib": cells["VersorAI"],
                "subset_over_chunk":   ratios[0],
                "einsum_over_chunk":   ratios[1],
                "ga_over_chunk":       ratios[2],
                "versor_over_chunk":   ratios[3],
                "versorai_over_chunk": ratios[4],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_wedge_prod")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
