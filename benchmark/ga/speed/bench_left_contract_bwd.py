"""Backward-pass benchmark for left_contract across (n, batch).

Mirrors `bench_left_contract.py` (forward) — same n/batch spread; the
`subset` column is dropped (left_contract_subset_grade is forward-only).
Per-iter the forward graph is built fresh and only the backward is timed
via inter-step syncs.

Columns:
  chunk        our _LeftContractFunc.backward (two geom_prod_fwd kernel
               calls with lc-specific direct-sigma bwd LUTs)
  chunk_skip   the kskip-variant forward feeding the same bwd path
  einsum       autograd through the EinsumLeftContract factored path
               (outer einsum + (D, D) sign mul + XOR scatter_add)
  Versor       autograd through Versor's `left_contraction` (gather +
               (D, D) sign matmul)
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
from cliffordcuda.extensions.ga.left_contract import (
    build_left_contract_sign_bwd, left_contract, left_contract_skip,
    load_contract_cuda,
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
    _ = load_geom_prod_cuda()
    for n in n_values:
        build_left_contract_sign_bwd(n, device)
    _x = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    _y = torch.randn(2, 1 << 7, device=device, dtype=dtype, requires_grad=True)
    left_contract(_x, _y).sum().backward()
    del _x, _y; torch.cuda.empty_cache()
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "chunk_skip", "einsum", "Versor", "VersorAI"],
        ratio_labels=["chunk_skip/chunk", "einsum/chunk", "Versor/chunk", "VersorAI/chunk"],
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
            a = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
            grad_c = torch.randn(batch, dim, device=device, dtype=dtype)

            chunk = gate("chunk", lambda: left_contract(a, b), (a, b), grad_c, n)
            chunk_skip = gate("chunk_skip", lambda: left_contract_skip(a, b),
                              (a, b), grad_c, n)

            if einsum_op is None:
                record_carry(skip_log, "einsum", n); einsum = NAN_PAIR
            else:
                einsum = gate("einsum", lambda: einsum_op(a, b), (a, b), grad_c, n)

            if versor_alg is None:
                record_carry(skip_log, "Versor", n); versor = NAN_PAIR
            else:
                versor = gate("Versor", lambda: versor_alg.left_contraction(a, b),
                              (a, b), grad_c, n)

            if "VersorAI" not in disabled:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                versorai = gate("VersorAI",
                                lambda: versorai_algebra.left_contraction(a, b, versorai_sig),
                                (a, b), grad_c, n)
            else:
                record_carry(skip_log, "VersorAI", n); versorai = NAN_PAIR
            ratios = [
                (chunk_skip[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(chunk_skip[0]) else float("nan"),
                (einsum[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])     else float("nan"),
                (versor[0]     / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])     else float("nan"),
                (versorai[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0])   else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":      chunk[0],      "chunk_std_us":      chunk[1],
                "chunk_skip_us": chunk_skip[0], "chunk_skip_std_us": chunk_skip[1],
                "einsum_us":     einsum[0],     "einsum_std_us":     einsum[1],
                "versor_us":     versor[0],     "versor_std_us":     versor[1],
                "versorai_us":   versorai[0],   "versorai_std_us":   versorai[1],
                "chunk_skip_over_chunk": ratios[0],
                "einsum_over_chunk":     ratios[1],
                "versor_over_chunk":     ratios[2],
                "versorai_over_chunk":   ratios[3],
            })
            print(format_row(n, dim, batch, [chunk, chunk_skip, einsum, versor, versorai], ratios=ratios))
            del a, b, grad_c; torch.cuda.empty_cache()

        print()
        del einsum_op, versor_alg; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_left_contract_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
