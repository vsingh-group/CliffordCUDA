"""Backward-pass benchmark for right_contract across (n, batch).

Mirrors `bench_right_contract.py` (forward). Per-iter the forward graph is
built fresh and only the backward is timed via inter-step syncs.

Columns:
  chunk      our _RightContractFunc.backward (two geom_prod_fwd kernel calls
             with rc-specific direct-sigma bwd LUTs)
  chunk_skip same with the kskip variant
  einsum     autograd through the EinsumRightContract factored path
             (outer einsum + (D, D) sign mul + XOR scatter_add)

No external witness: Versor only ships bivector x vector for right
contraction, torch_ga ships none.
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
from cliffordcuda.extensions.ga.left_contract import load_contract_cuda
from cliffordcuda.extensions.ga.right_contract import (
    build_right_contract_sign_bwd, right_contract, right_contract_skip,
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
    _ = load_geom_prod_cuda()
    for n in n_values:
        build_right_contract_sign_bwd(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    _y = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    right_contract(_x, _y).sum().backward()
    del _x, _y; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip", "einsum"],
        ratio_labels=["chunk_skip/chunk", "einsum/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_bwd(disabled, skip_log, warmup, iters, trials)

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
            a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            grad_c = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk = gate("chunk", lambda: right_contract(a, b), (a, b), grad_c, n)
            chunk_skip = gate("chunk_skip", lambda: right_contract_skip(a, b),
                              (a, b), grad_c, n)

            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), (a, b), grad_c, n)
            ratios = [
                (chunk_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0]) else float("nan"),
                (einsum[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])     else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":      chunk[0],      "chunk_std_us":      chunk[1],
                "chunk_skip_us": chunk_skip[0], "chunk_skip_std_us": chunk_skip[1],
                "einsum_us":     einsum[0],     "einsum_std_us":     einsum[1],
                "chunk_skip_over_chunk": ratios[0],
                "einsum_over_chunk":     ratios[1],
            })
            print(format_row(n, dim, batch, [chunk, chunk_skip, einsum], ratios=ratios))
            del a, b, grad_c; torch.cuda.empty_cache()

        print()
        del einsum_op; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_right_contract_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
