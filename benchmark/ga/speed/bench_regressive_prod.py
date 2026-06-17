"""Benchmark regressive (meet) product implementations across (n, batch).

Columns:
  chunk       our regressive_prod = dual(wedge_prod(dual(a), dual(b)))
              — uses kernels/wedge_prod.cu under the hood
  chunk_skip  same composition with the wedge_prod_skip variant
  subset      our regressive_prod_subset_grade = dual(ws_g(dual(a), dual(b)))
              — uses kernels/wedge_prod_subset_grade.cu

torch_ga ships `reg_prod` but its high-level `GeometricAlgebra` constructor
has init-order issues that block clean use — skipped here. Versor doesn't
ship a regressive primitive.

Regressive is undefined for degenerate metrics (r > 0). Cl(n, 0) only.
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
    format_row,
    make_gate_run,
    not_nan,
    print_skip_summary,
    print_table_header,
    results_path,
    warmup_clock,
    write_csv,
)

from cliffordcuda.extensions.ga.regressive_prod import (
    regressive_prod, regressive_prod_skip, regressive_prod_subset_grade,
)
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda
from cliffordcuda.extensions.ga.wedge_prod.subset_grade import (
    load_wedge_prod_subset_grade_cuda,
)


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS


def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")

    print("Compiling kernels ...")
    _ = load_wedge_prod_cuda()
    _ = load_wedge_prod_subset_grade_cuda()
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype)
    regressive_prod(_x, _x); regressive_prod_subset_grade(_x, _x)
    del _x; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip", "subset"],
        ratio_labels=["chunk_skip/chunk", "subset/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_run(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        for batch in batch_values:
            a = torch.randn(batch, dim, device=device, dtype=dtype)
            b = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk      = gate("chunk",      lambda: regressive_prod(a, b), n)
            chunk_skip = gate("chunk_skip", lambda: regressive_prod_skip(a, b), n)
            subset     = gate("subset",     lambda: regressive_prod_subset_grade(a, b), n)
            ratios = [
                (chunk_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0]) else float("nan"),
                (subset[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(subset[0])     else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":      chunk[0],      "chunk_std_us":      chunk[1],
                "chunk_skip_us": chunk_skip[0], "chunk_skip_std_us": chunk_skip[1],
                "subset_us":     subset[0],     "subset_std_us":     subset[1],
                "chunk_skip_over_chunk": ratios[0],
                "subset_over_chunk":     ratios[1],
            })
            print(format_row(n, dim, batch, [chunk, chunk_skip, subset], ratios=ratios))
            del a, b; torch.cuda.empty_cache()

        print()
        torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_regressive_prod"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
