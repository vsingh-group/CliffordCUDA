"""Backward-pass benchmark for regressive_prod across (n, batch).

Mirrors `bench_regressive_prod.py` (forward) — same n/batch spread, same
column order, but the `subset` column is dropped (regressive_prod_subset_grade
is forward-only). Regressive is undefined for degenerate metrics, so Cl(n, 0)
only (matches the forward bench).

Per-iter the forward graph is built fresh and only the backward is timed via
inter-step syncs.

Columns:
  chunk      autograd through dual -> wedge_prod -> dual (composition of
             differentiable ops; backward fully handled by PyTorch's autograd
             through _WedgeProdFunc.backward + element-wise dual)
  chunk_skip same with the kskip variant of the inner wedge
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
    make_gate_bwd,
    not_nan,
    print_skip_summary,
    print_table_header,
    results_path,
    warmup_clock,
    write_csv,
)

from cliffordcuda.extensions.ga.geom_prod import load_geom_prod_cuda
from cliffordcuda.extensions.ga.regressive_prod import regressive_prod, regressive_prod_skip
from cliffordcuda.extensions.ga.wedge_prod import load_wedge_prod_cuda


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
    _ = load_geom_prod_cuda()
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    _y = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    regressive_prod(_x, _y).sum().backward()
    del _x, _y; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip"],
        ratio_labels=["chunk_skip/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_bwd(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        for batch in batch_values:
            a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            grad_c = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk = gate("chunk", lambda: regressive_prod(a, b), (a, b), grad_c, n)
            chunk_skip = gate("chunk_skip", lambda: regressive_prod_skip(a, b),
                              (a, b), grad_c, n)
            ratios = [
                (chunk_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":      chunk[0],      "chunk_std_us":      chunk[1],
                "chunk_skip_us": chunk_skip[0], "chunk_skip_std_us": chunk_skip[1],
                "chunk_skip_over_chunk": ratios[0],
            })
            print(format_row(n, dim, batch, [chunk, chunk_skip], ratios=ratios))
            del a, b, grad_c; torch.cuda.empty_cache()

        print()
        torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_regressive_prod_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
