"""Backward-pass benchmark for wedge_prod across (n, batch).

Mirrors `bench_wedge_prod.py` (forward) — same n/batch spread; the `subset`
column is dropped (wedge_prod_subset_grade is forward-only). Per-iter the
forward graph is built fresh and only the backward is timed via inter-step
syncs (matches the bench_geom_prod_bwd pattern).

Columns:
  chunk        our _WedgeProdFunc.backward (two geom_prod_fwd kernel calls
               with wedge-specific direct-sigma bwd LUTs)
  chunk_skip   the kskip-variant forward feeding the same bwd path
  multik       our M=2-outputs-per-warp forward feeding the multik bwd
  multik_skip  multik forward + kskip in the bwd LUTs
  einsum       autograd through the EinsumWedge factored path (outer
               einsum + (D, D) sign mul + XOR scatter_add)
  torch_ga     autograd through tensordot against the dense outer Cayley
  Versor*      autograd through Versor's `wedge` = (AB - BA)/2 (different
               operation on grade-2 inputs — timing reference only)
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

from cliffordcuda.extensions.ga.geom_prod import load_geom_prod_cuda
from cliffordcuda.extensions.ga.wedge_prod import (
    build_wedge_sign_bwd, load_wedge_prod_cuda, wedge_prod, wedge_prod_multik,
    wedge_prod_multik_skip, wedge_prod_skip,
)
from _cayley import build_outer_cayley, shortlex_to_bp
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumWedge

from torch_ga.mv_ops import mv_multiply

from _harness import import_versorai
versorai_algebra = import_versorai()


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS

def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")

    print("Pre-loading kernels and LUTs ...")
    _ = load_wedge_prod_cuda()
    _ = load_geom_prod_cuda()
    for n in n_values:
        build_wedge_sign_bwd(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    _y = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    wedge_prod(_x, _y).sum().backward()
    del _x, _y; torch.cuda.empty_cache()
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip", "multik", "multik_skip", "einsum",
         "torch_ga", "Versor*", "VersorAI"],
        ratio_labels=["chunk_skip/chunk", "multik/chunk", "multik_skip/chunk",
                      "einsum/chunk", "ga/chunk", "Versor*/chunk", "VersorAI/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_bwd(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        cayley_outer = None
        if "torch_ga" not in disabled:
            try:
                cayley_outer = build_outer_cayley(n, device=device)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "torch_ga", n, e)
                disabled.add("torch_ga")
                torch.cuda.empty_cache()
        sl_to_bp = shortlex_to_bp(n).to(device)

        einsum_op = None
        if "einsum" not in disabled:
            try:
                einsum_op = EinsumWedge(n, device=device, dtype=dtype)
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

            chunk = gate("chunk", lambda: wedge_prod(a, b), (a, b), grad_c, n)
            chunk_skip = gate("chunk_skip", lambda: wedge_prod_skip(a, b),
                              (a, b), grad_c, n)
            multik = gate("multik", lambda: wedge_prod_multik(a, b),
                          (a, b), grad_c, n)
            multik_skip = gate("multik_skip", lambda: wedge_prod_multik_skip(a, b),
                               (a, b), grad_c, n)

            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), (a, b), grad_c, n)

            if cayley_outer is None:
                record_carry(skip_log, "torch_ga", n); ga_t = NAN_PAIR
            else:
                a_sl = a.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
                b_sl = b.detach().index_select(-1, sl_to_bp).contiguous().requires_grad_(True)
                ga_t = gate("torch_ga",
                            lambda: mv_multiply(a_sl, b_sl, cayley_outer),
                            (a_sl, b_sl), grad_c, n)
                del a_sl, b_sl

            if versor_alg is None:
                record_carry(skip_log, "Versor", n); versor = NAN_PAIR
            else:
                versor = gate("Versor",
                              lambda: versor_alg.wedge(a, b),
                              (a, b), grad_c, n)

            # VersorAI: gacore.kernel.wedge_product (autograd through
            # filtered_product = gather + sign mask + sum).
            if "VersorAI" not in disabled:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                versorai = gate("VersorAI",
                                lambda: versorai_algebra.wedge_product(a, b, versorai_sig),
                                (a, b), grad_c, n)
            else:
                record_carry(skip_log, "VersorAI", n); versorai = NAN_PAIR
            ratios = [
                (chunk_skip[0]  / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0])  else float("nan"),
                (multik[0]      / chunk[0]) if not_nan(chunk[0]) and not_nan(multik[0])      else float("nan"),
                (multik_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(multik_skip[0]) else float("nan"),
                (einsum[0]      / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])      else float("nan"),
                (ga_t[0]        / chunk[0]) if not_nan(chunk[0]) and not_nan(ga_t[0])        else float("nan"),
                (versor[0]      / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])      else float("nan"),
                (versorai[0]    / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0])    else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":       chunk[0],       "chunk_std_us":       chunk[1],
                "chunk_skip_us":  chunk_skip[0],  "chunk_skip_std_us":  chunk_skip[1],
                "multik_us":      multik[0],      "multik_std_us":      multik[1],
                "multik_skip_us": multik_skip[0], "multik_skip_std_us": multik_skip[1],
                "einsum_us":      einsum[0],      "einsum_std_us":      einsum[1],
                "ga_us":          ga_t[0],        "ga_std_us":          ga_t[1],
                "versor_us":      versor[0],      "versor_std_us":      versor[1],
                "versorai_us":    versorai[0],    "versorai_std_us":    versorai[1],
                "chunk_skip_over_chunk":  ratios[0],
                "multik_over_chunk":      ratios[1],
                "multik_skip_over_chunk": ratios[2],
                "einsum_over_chunk":      ratios[3],
                "ga_over_chunk":          ratios[4],
                "versor_over_chunk":      ratios[5],
                "versorai_over_chunk":    ratios[6],
            })
            print(format_row(n, dim, batch,
                             [chunk, chunk_skip, multik, multik_skip, einsum, ga_t, versor, versorai],
                             ratios=ratios))
            del a, b, grad_c; torch.cuda.empty_cache()

        print()
        del cayley_outer, einsum_op, versor_alg, sl_to_bp
        torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_wedge_prod_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
