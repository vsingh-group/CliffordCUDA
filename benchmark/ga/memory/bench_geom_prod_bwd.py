"""Peak GPU memory for geom_prod backward across (n, batch).

Each impl measured in isolation (rotor-bench pattern): full_cleanup() → build only
that impl's state + inputs → measure raw `max_memory_allocated()` over one
forward + backward call. Reports total GPU pressure, not a delta.
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

from cliffordcuda.extensions.ga.geom_prod import (
    build_packed_sign, build_packed_sign_bwd, geom_prod, geom_prod_multik,
    load_geom_prod_cuda,
)
from _cayley import build_geom_cayley, shortlex_to_bp
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumGP

from torch_ga.mv_ops import mv_multiply

from _harness import import_versorai
versorai_algebra = import_versorai()


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "multik", "einsum", "torch_ga", "Versor", "VersorAI"]






def _bp_inputs(n, batch):
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    grad_c = torch.randn(batch, dim, device=device, dtype=dtype)
    return a, b, grad_c


def measure_chunk(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: geom_prod(a, b), (a, b), grad_c))


def measure_multik(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: geom_prod_multik(a, b), (a, b), grad_c))


def measure_torch_ga(n, batch):
    dim = 1 << n
    cayley = build_geom_cayley(n, device=device)
    sl_to_bp = shortlex_to_bp(n).to(device)
    a, b, grad_c = _bp_inputs(n, batch)
    a_sl = a.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
    b_sl = b.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: mv_multiply(a_sl, b_sl, cayley), (a_sl, b_sl), grad_c))


def measure_einsum(n, batch):
    op = EinsumGP(n, device=device, dtype=dtype)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: op(a, b), (a, b), grad_c))


def measure_versor(n, batch):
    versor_alg = VersorAlgebra(p=n, q=0, r=0, device=device)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: versor_alg.geometric_product(a, b), (a, b), grad_c))


def measure_versorai(n, batch):
    sig = torch.ones(n, dtype=dtype, device=device)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(
            lambda: versorai_algebra.geometric_product(a, b, sig), (a, b), grad_c))


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print("Compiling kernel + pre-building LUTs ...")
    _ = load_geom_prod_cuda()
    for n in n_values:
        build_packed_sign(n, device); build_packed_sign_bwd(n, device)

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
            for impl, fn in [
                ("chunk",    measure_chunk),
                ("multik",   measure_multik),
                ("einsum",   measure_einsum),
                ("torch_ga", measure_torch_ga),
                ("Versor",   measure_versor),
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
                "multik_mib":   cells["multik"],
                "einsum_mib":   cells["einsum"],
                "ga_mib":       cells["torch_ga"],
                "versor_mib":   cells["Versor"],
                "versorai_mib": cells["VersorAI"],
                "multik_over_chunk":   ratios[0],
                "einsum_over_chunk":   ratios[1],
                "ga_over_chunk":       ratios[2],
                "versor_over_chunk":   ratios[3],
                "versorai_over_chunk": ratios[4],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_geom_prod_bwd")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
