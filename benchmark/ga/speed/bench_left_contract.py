"""Benchmark left contraction implementations across (n, batch).

Columns:
  chunk      our contract.cu kernel with warp-uniform chunk-skip (LUT-driven
             for left)
  subset     our left_contract_subset_grade — subset enumeration + grade-ordered
             warps (reuses inner_prod_subset_grade.cu kernel with left-specific
             LUT)
  einsum     factored Cayley (outer einsum + (D, D) sign mul + XOR scatter_add
             with the left-contraction sparsity mask)
  Versor     Versor's left_contraction method (B[..., cayley_indices] gather +
             matmul against `lc_gp_signs` table). Bit-pattern indexing —
             matches our operation.

torch_ga ships no left contraction method; no column for it.

All inputs in bit-pattern blade ordering — no permutation needed.
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

from cliffordcuda.extensions.ga.left_contract import (
    build_left_contract_subset_lut, left_contract, left_contract_subset_grade,
    load_contract_cuda,
)
from cliffordcuda.extensions.ga.inner_prod.subset_grade import (
    load_inner_prod_subset_grade_cuda,
)
from core.algebra import CliffordAlgebra as VersorAlgebra
from _einsum_refs import EinsumLeftContract

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

    print("Compiling kernels + pre-building LUTs ...")
    _ = load_contract_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    for n in n_values:
        build_left_contract_subset_lut(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype)
    left_contract(_x, _x); left_contract_subset_grade(_x, _x)
    del _x; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "subset", "einsum", "Versor", "VersorAI"],
        ratio_labels=["subset/chunk", "einsum/chunk", "Versor/chunk", "VersorAI/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_run(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        einsum_op = None
        if "einsum" not in disabled:
            try:
                einsum_op = EinsumLeftContract(n, device=device, dtype=dtype)
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

            chunk    = gate("chunk",  lambda: left_contract(a, b), n)
            subset   = gate("subset", lambda: left_contract_subset_grade(a, b), n)
            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), n)
            if versor_alg is None:
                record_carry(skip_log, "Versor", n); versor = NAN_PAIR
            else:
                versor = gate("Versor", lambda: versor_alg.left_contraction(a, b), n)

            if "VersorAI" not in disabled:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                versorai = gate("VersorAI",
                                lambda: versorai_algebra.left_contraction(a, b, versorai_sig), n)
            else:
                record_carry(skip_log, "VersorAI", n); versorai = NAN_PAIR
            ratios = [
                (subset[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(subset[0])   else float("nan"),
                (einsum[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])   else float("nan"),
                (versor[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])   else float("nan"),
                (versorai[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":    chunk[0],    "chunk_std_us":    chunk[1],
                "subset_us":   subset[0],   "subset_std_us":   subset[1],
                "einsum_us":   einsum[0],   "einsum_std_us":   einsum[1],
                "versor_us":   versor[0],   "versor_std_us":   versor[1],
                "versorai_us": versorai[0], "versorai_std_us": versorai[1],
                "subset_over_chunk":   ratios[0],
                "einsum_over_chunk":   ratios[1],
                "versor_over_chunk":   ratios[2],
                "versorai_over_chunk": ratios[3],
            })
            print(format_row(n, dim, batch, [chunk, subset, einsum, versor, versorai], ratios=ratios))
            del a, b; torch.cuda.empty_cache()

        print()
        del einsum_op, versor_alg; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_left_contract"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
