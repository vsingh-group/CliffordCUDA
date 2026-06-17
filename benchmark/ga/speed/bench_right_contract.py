"""Benchmark right-contraction implementations across (n, batch).

Columns:
  chunk     our contract.cu kernel with warp-uniform chunk-skip (LUT-driven
            for right)
  subset    our right_contract_subset_grade — subset enumeration + grade-ordered
            warps (reuses inner_prod_subset_grade.cu kernel with right-specific
            LUT)
  einsum    factored outer einsum + (D, D) sign mul + XOR scatter_add
            with the right-contraction sparsity mask (EinsumRightContract)

No torch_ga / Versor column: Versor only ships a bivector x vector
specialization of right contraction, and torch_ga ships none.
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
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS


def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")

    print("Compiling kernels + pre-building LUTs ...")
    _ = load_contract_cuda()
    _ = load_inner_prod_subset_grade_cuda()
    for n in n_values:
        build_right_contract_subset_lut(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype)
    right_contract(_x, _x); right_contract_subset_grade(_x, _x)
    del _x; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "subset", "einsum"],
        ratio_labels=["subset/chunk", "einsum/chunk"],
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
                einsum_op = EinsumRightContract(n, device=device, dtype=dtype)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "einsum", n, e)
                disabled.add("einsum")
                torch.cuda.empty_cache()

        for batch in batch_values:
            a = torch.randn(batch, dim, device=device, dtype=dtype)
            b = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk  = gate("chunk",  lambda: right_contract(a, b), n)
            subset = gate("subset", lambda: right_contract_subset_grade(a, b), n)
            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), n)
            ratios = [
                (subset[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(subset[0]) else float("nan"),
                (einsum[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":  chunk[0],  "chunk_std_us":  chunk[1],
                "subset_us": subset[0], "subset_std_us": subset[1],
                "einsum_us": einsum[0], "einsum_std_us": einsum[1],
                "subset_over_chunk": ratios[0],
                "einsum_over_chunk": ratios[1],
            })
            print(format_row(n, dim, batch, [chunk, subset, einsum], ratios=ratios))
            del a, b; torch.cuda.empty_cache()

        print()
        del einsum_op; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_right_contract"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
