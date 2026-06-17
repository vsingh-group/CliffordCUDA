"""Benchmark inner (Hestenes) implementations across (n, batch).

Columns:
  chunk      our inner_prod chunk kernel (Hestenes inner, <AB>_{|r-s|})
  chunk_skip our inner_prod_kskip variant
  subset     our inner_prod_subset_grade variant
  einsum     factored outer einsum + (D, D) sign mul + XOR scatter_add
             with the Hestenes inner mask (EinsumInner)
  torch_ga   mv_multiply against the dense Hestenes inner Cayley (ShortLex)
  Versor*    Versor's `inner_product` method, implemented as (AB + BA)/2.
             DIFFERENT DEFINITION from Hestenes on grade-2 inputs — kept as
             a timing reference for Versor's named op, NOT a numerical witness.
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
    make_gate_run,
    not_nan,
    print_skip_summary,
    print_table_header,
    record_carry,
    record_setup_fail,
    results_path,
    warmup_clock,
    write_csv,
)

from cliffordcuda.extensions.ga.inner_prod import (
    inner_prod, inner_prod_kskip, load_inner_prod_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    build_inner_subset_lut, inner_prod_subset_grade,
    load_inner_prod_subset_grade_cuda,
)
from _cayley import build_inner_cayley, shortlex_to_bp
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumInner

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
    _ = load_inner_prod_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    for n in n_values:
        build_inner_subset_lut(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype)
    inner_prod(_x, _x); inner_prod_subset_grade(_x, _x)
    del _x; torch.cuda.empty_cache()
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip", "subset", "einsum", "torch_ga", "Versor*", "VersorAI"],
        ratio_labels=["chunk_skip/chunk", "subset/chunk", "einsum/chunk",
                      "ga/chunk", "Versor*/chunk", "VersorAI/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_run(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        cayley_inner = None
        if "torch_ga" not in disabled:
            try:
                cayley_inner = build_inner_cayley(n, device=device)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "torch_ga", n, e)
                disabled.add("torch_ga")
                torch.cuda.empty_cache()
        sl_to_bp = shortlex_to_bp(n).to(device)

        einsum_op = None
        if "einsum" not in disabled:
            try:
                einsum_op = EinsumInner(n, device=device, dtype=dtype)
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
            a = torch.randn(batch, dim, device=device, dtype=dtype)
            b = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk      = gate("chunk",      lambda: inner_prod(a, b), n)
            chunk_skip = gate("chunk_skip", lambda: inner_prod_kskip(a, b), n)
            chunk_sg   = gate("subset",     lambda: inner_prod_subset_grade(a, b), n)
            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), n)
            if cayley_inner is None:
                record_carry(skip_log, "torch_ga", n)
                ga_t = NAN_PAIR
            else:
                a_sl = a.index_select(-1, sl_to_bp).contiguous()
                b_sl = b.index_select(-1, sl_to_bp).contiguous()
                ga_t = gate("torch_ga", lambda: mv_multiply(a_sl, b_sl, cayley_inner), n)
                del a_sl, b_sl

            if versor_alg is None:
                record_carry(skip_log, "Versor", n)
                versor = NAN_PAIR
            else:
                versor = gate("Versor", lambda: versor_alg.inner_product(a, b), n)

            if "VersorAI" not in disabled:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                versorai = gate("VersorAI",
                                lambda: versorai_algebra.inner_product(a, b, versorai_sig), n)
            else:
                record_carry(skip_log, "VersorAI", n); versorai = NAN_PAIR
            ratios = [
                (chunk_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0]) else float("nan"),
                (chunk_sg[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_sg[0])   else float("nan"),
                (einsum[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])     else float("nan"),
                (ga_t[0]       / chunk[0]) if not_nan(chunk[0]) and not_nan(ga_t[0])       else float("nan"),
                (versor[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])     else float("nan"),
                (versorai[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0])   else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":      chunk[0],      "chunk_std_us":      chunk[1],
                "chunk_skip_us": chunk_skip[0], "chunk_skip_std_us": chunk_skip[1],
                "subset_us":     chunk_sg[0],   "subset_std_us":     chunk_sg[1],
                "einsum_us":     einsum[0],     "einsum_std_us":     einsum[1],
                "ga_us":         ga_t[0],       "ga_std_us":         ga_t[1],
                "versor_us":     versor[0],     "versor_std_us":     versor[1],
                "versorai_us":   versorai[0],   "versorai_std_us":   versorai[1],
                "chunk_skip_over_chunk": ratios[0],
                "subset_over_chunk":     ratios[1],
                "einsum_over_chunk":     ratios[2],
                "ga_over_chunk":         ratios[3],
                "versor_over_chunk":     ratios[4],
                "versorai_over_chunk":   ratios[5],
            })
            print(format_row(n, dim, batch,
                             [chunk, chunk_skip, chunk_sg, einsum, ga_t, versor, versorai],
                             ratios=ratios))
            del a, b; torch.cuda.empty_cache()

        print()
        del cayley_inner, einsum_op, versor_alg, sl_to_bp; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_inner_prod"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
