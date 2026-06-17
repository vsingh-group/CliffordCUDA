"""Inference speed: CliffordCUDA `apply_rotor` vs ComposingLinearLayers
`F.linear(x, M)` on a precomputed (D, D) sandwich matrix.

Both rotors are built ONCE outside the timed loop; only the forward
apply is measured. Single chunk (`in_chunks=out_chunks=1`,
`chunk_size=2**n`) so Compose's forward is exactly the dense matmul.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
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
    record_setup_fail,
    results_path,
    warmup_clock,
    write_csv,
)

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_repo_root, os.pardir, "ComposingLinearLayers"))
from rotor_layer import Rotor as ComposeRotor

from cliffordcuda import CliffordAlgebra


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = [1, 16, 64, 256, 1024, 4096, 8192]
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS


def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "Compose"],
        ratio_labels=["Compose/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    time_it = make_gate_run(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)

            try:
                cl = CliffordAlgebra(metric=[1] * n, device=device)
                bivec = torch.randn(1, cl._num_basis_biv, device=device, dtype=dtype)
                cs = cl.compile_bivector(bivec)
                chunk = time_it("chunk",
                                lambda: cl.apply_rotor(cs, x_bp), n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "chunk", n, e)
                chunk = NAN_PAIR; disabled.add("chunk")
            cl = bivec = cs = None

            try:
                c_m = ComposeRotor(in_dim=dim, out_dim=dim, in_chunks=1,
                                   out_chunks=1, chunk_size=dim,
                                   single_rotor=True, alpha_param=False,
                                   bias_param=False, device=device, dtype=dtype)
                c_m.eval()
                M = c_m.sandwich_product_matrix.detach()
                del c_m
                compose = time_it("Compose", lambda: F.linear(x_bp, M), n)
                del M
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "Compose", n, e)
                compose = NAN_PAIR; disabled.add("Compose")

            ratios = [
                (compose[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(compose[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":   chunk[0],   "chunk_std_us":   chunk[1],
                "compose_us": compose[0], "compose_std_us": compose[1],
                "compose_over_chunk": ratios[0],
            })
            print(format_row(n, dim, batch, [chunk, compose], ratios=ratios))
            del x_bp; torch.cuda.empty_cache()
        print()

    write_csv(results_path("ga/speed", "bench_rotor_apply_compose_inference"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
