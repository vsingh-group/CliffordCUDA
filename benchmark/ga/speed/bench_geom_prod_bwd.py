"""Backward-pass benchmark for geom_prod across (n, batch).

Mirrors `bench_geom_prod.py` (forward) — same implementations, same n/batch
spread, same column order. Per the `bench_e2e.py` pattern: each timed iter
builds the forward graph fresh, then times just the backward via inter-step
syncs. No retain_graph, so each iter's intermediates free before the next
iter — same per-iter memory footprint as the forward bench.

Columns:
  chunk      our _GeomProdFunc.backward (two geom_prod_fwd kernel calls with
             direct-sigma bwd LUTs)
  multik     our _GeomProdMultikFunc.backward (same shape; routed through
             geom_prod_fwd_multik)
  einsum     autograd through outer einsum + (D, D) sign mul + XOR scatter_add
  torch_ga   autograd through tensordot against the dense (dim, dim, dim) Cayley
  Versor     autograd through gather + (D, D) sign matmul
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
    DEFAULT_BATCH_VALUES,
    DEFAULT_ITERS,
    DEFAULT_N_VALUES,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP,
    NAN_PAIR,
    format_row,
    make_gate_bwd,
    not_nan,
    print_skip_summary,
    print_table_header,
    record_carry,
    record_setup_fail,
    results_path,
    warmup_clock,
    write_csv,
)

from cliffordcuda.extensions.ga.geom_prod import (
    build_packed_sign, build_packed_sign_bwd, geom_prod, geom_prod_multik,
    load_geom_prod_cuda,
)
from _cayley import build_geom_cayley, shortlex_to_bp
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumGP

from _harness import import_versorai
versorai_algebra = import_versorai()

from torch_ga.mv_ops import mv_multiply


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS


def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")

    print("Compiling kernel + pre-building LUTs ...")
    _ = load_geom_prod_cuda()
    for n in n_values:
        build_packed_sign(n, device)
        build_packed_sign_bwd(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    _y = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    geom_prod(_x, _y).sum().backward()
    geom_prod_multik(_x, _y).sum().backward()
    del _x, _y; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "multik", "einsum", "torch_ga", "Versor", "VersorAI"],
        ratio_labels=["multik/chunk", "einsum/chunk", "ga/chunk",
                      "Versor/chunk", "VersorAI/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_bwd(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        sl_to_bp = shortlex_to_bp(n).to(device)

        ga_cayley = None
        if "torch_ga" not in disabled:
            try:
                ga_cayley = build_geom_cayley(n, device=device)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "torch_ga", n, e)
                disabled.add("torch_ga")
                torch.cuda.empty_cache()

        einsum_op = None
        if "einsum" not in disabled:
            try:
                einsum_op = EinsumGP(n, device=device, dtype=dtype)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "einsum", n, e)
                disabled.add("einsum")
                torch.cuda.empty_cache()

        versor_alg = None
        if "Versor" not in disabled:
            try:
                versor_alg = VersorAlgebra(p=n, q=0, r=0, device=device)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "Versor", n, e)
                disabled.add("Versor")
                torch.cuda.empty_cache()

        for batch in batch_values:
            a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            grad_c = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk  = gate("chunk",  lambda: geom_prod(a, b),         (a, b), grad_c, n)
            multik = gate("multik", lambda: geom_prod_multik(a, b),  (a, b), grad_c, n)

            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), (a, b), grad_c, n)

            need_sl = (ga_cayley is not None)
            if need_sl:
                a_sl = a.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
                b_sl = b.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)

            if ga_cayley is None:
                record_carry(skip_log, "torch_ga", n); ga = NAN_PAIR
            else:
                ga = gate("torch_ga",
                          lambda: mv_multiply(a_sl, b_sl, ga_cayley),
                          (a_sl, b_sl), grad_c, n)

            if need_sl:
                del a_sl, b_sl

            if versor_alg is None:
                record_carry(skip_log, "Versor", n); versor = NAN_PAIR
            else:
                versor = gate("Versor",
                              lambda: versor_alg.geometric_product(a, b),
                              (a, b), grad_c, n)

            # VersorAI: gacore.kernel.geometric_product with default dispatch.
            # At n>=7 the matrix path isn't applicable so this routes to the
            # universal bitmasked torch path, which is autograd-clean.
            if "VersorAI" not in disabled:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                versorai = gate("VersorAI",
                                lambda: versorai_algebra.geometric_product(a, b, versorai_sig),
                                (a, b), grad_c, n)
            else:
                record_carry(skip_log, "VersorAI", n); versorai = NAN_PAIR
            ratios = [
                (multik[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(multik[0])   else float("nan"),
                (einsum[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])   else float("nan"),
                (ga[0]       / chunk[0]) if not_nan(chunk[0]) and not_nan(ga[0])       else float("nan"),
                (versor[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])   else float("nan"),
                (versorai[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":    chunk[0],    "chunk_std_us":    chunk[1],
                "multik_us":   multik[0],   "multik_std_us":   multik[1],
                "einsum_us":   einsum[0],   "einsum_std_us":   einsum[1],
                "ga_us":       ga[0],       "ga_std_us":       ga[1],
                "versor_us":   versor[0],   "versor_std_us":   versor[1],
                "versorai_us": versorai[0], "versorai_std_us": versorai[1],
                "multik_over_chunk":   ratios[0],
                "einsum_over_chunk":   ratios[1],
                "ga_over_chunk":       ratios[2],
                "versor_over_chunk":   ratios[3],
                "versorai_over_chunk": ratios[4],
            })
            print(format_row(n, dim, batch,
                             [chunk, multik, einsum, ga, versor, versorai],
                             ratios=ratios))
            del a, b, grad_c; torch.cuda.empty_cache()

        print()
        del ga_cayley, einsum_op, versor_alg, sl_to_bp
        torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_geom_prod_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
