"""Peak GPU memory for wedge_prod backward. Per-impl isolation, raw peak."""
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
from cliffordcuda.extensions.ga.wedge_prod import (
    build_wedge_sign_bwd, load_wedge_prod_cuda, wedge_prod, wedge_prod_multik,
    wedge_prod_multik_skip, wedge_prod_skip,
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
IMPLS = ["chunk", "chunk_skip", "multik", "multik_skip", "einsum",
         "torch_ga", "Versor*", "VersorAI"]






def _bp_inputs(n, batch):
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    grad_c = torch.randn(batch, dim, device=device, dtype=dtype)
    return a, b, grad_c


def measure_chunk(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: wedge_prod(a, b), (a, b), grad_c))


def measure_chunk_skip(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: wedge_prod_skip(a, b), (a, b), grad_c))


def measure_multik(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: wedge_prod_multik(a, b), (a, b), grad_c))


def measure_multik_skip(n, batch):
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(lambda: fwd_bwd(lambda: wedge_prod_multik_skip(a, b), (a, b), grad_c))


def measure_torch_ga(n, batch):
    cayley = build_outer_cayley(n, device=device)
    sl_to_bp = shortlex_to_bp(n).to(device)
    a, b, grad_c = _bp_inputs(n, batch)
    a_sl = a.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
    b_sl = b.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: mv_multiply(a_sl, b_sl, cayley), (a_sl, b_sl), grad_c))


def measure_einsum(n, batch):
    op = EinsumWedge(n, device=device, dtype=dtype)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: op(a, b), (a, b), grad_c))


def measure_versor(n, batch):
    versor_alg = VersorAlgebra(p=n, q=0, r=0, device=device)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(lambda: versor_alg.wedge(a, b), (a, b), grad_c))


def measure_versorai(n, batch):
    sig = torch.ones(n, dtype=dtype, device=device)
    a, b, grad_c = _bp_inputs(n, batch)
    return measure_peak_memory(
        lambda: fwd_bwd(
            lambda: versorai_algebra.wedge_product(a, b, sig),
            (a, b), grad_c))


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print("Pre-loading kernels and LUTs ...")
    _ = load_wedge_prod_cuda(); _ = load_geom_prod_cuda()
    for n in n_values:
        build_wedge_sign_bwd(n, device)

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
                ("chunk",       measure_chunk),
                ("chunk_skip",  measure_chunk_skip),
                ("multik",      measure_multik),
                ("multik_skip", measure_multik_skip),
                ("einsum",      measure_einsum),
                ("torch_ga",    measure_torch_ga),
                ("Versor*",     measure_versor),
                ("VersorAI",    measure_versorai),
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
                "chunk_mib":       cells["chunk"],
                "chunk_skip_mib":  cells["chunk_skip"],
                "multik_mib":      cells["multik"],
                "multik_skip_mib": cells["multik_skip"],
                "einsum_mib":      cells["einsum"],
                "ga_mib":          cells["torch_ga"],
                "versor_mib":      cells["Versor*"],
                "versorai_mib":    cells["VersorAI"],
                "chunk_skip_over_chunk":  ratios[0],
                "multik_over_chunk":      ratios[1],
                "multik_skip_over_chunk": ratios[2],
                "einsum_over_chunk":      ratios[3],
                "ga_over_chunk":          ratios[4],
                "versor_over_chunk":      ratios[5],
                "versorai_over_chunk":    ratios[6],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
        print()

    path = results_path("ga/memory", "bench_wedge_prod_bwd")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
