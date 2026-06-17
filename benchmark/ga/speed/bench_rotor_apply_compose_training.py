"""Training speed: full step on a single rotor layer.

Each timed iteration includes the rotor REBUILD from the current bivector,
the forward apply, and the backward pass that lands a gradient on the
bivector parameter. This is the per-step training cost the user actually
pays — Compose's `_update_rotors` rebuilds the (D, D) sandwich matrix
every step before `F.linear(x, M)` can run, and CliffordCUDA's
`compile_bivector` rebuilds the Givens / eigh decomposition before
`apply_rotor(cs, x)` can run. Both calls are taken straight from the
upstream libraries — nothing is reimplemented.

  chunk_step:    cs = cl.compile_bivector(bivector)
                 y  = cl.apply_rotor(cs, x_bp)
                 y.sum().backward()                # grad -> bivector
  compose_step:  c_m._update_rotors()              # grad-tracking rebuild of M
                 y  = c_m(x_bp)                    # F.linear(x, M)
                 y.sum().backward()                # grad -> c_m.bivectors_left
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
    DEFAULT_ITERS,
    DEFAULT_N_VALUES,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP,
    NAN_PAIR,
    format_row,
    make_gate_step,
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

    gate = make_gate_step(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n

        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)

            try:
                cl = CliffordAlgebra(metric=[1] * n, device=device)
                bivector = torch.nn.Parameter(
                    torch.randn(1, cl._num_basis_biv, device=device, dtype=dtype))
                def chunk_step():
                    if bivector.grad is not None:
                        bivector.grad = None
                    cs = cl.compile_bivector(bivector)
                    y = cl.apply_rotor(cs, x_bp)
                    y.sum().backward()
                chunk = gate("chunk", chunk_step, n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "chunk", n, e)
                chunk = NAN_PAIR; disabled.add("chunk")
            cl = bivector = None

            try:
                c_m = ComposeRotor(in_dim=dim, out_dim=dim, in_chunks=1,
                                   out_chunks=1, chunk_size=dim,
                                   single_rotor=True, alpha_param=False,
                                   bias_param=False, device=device, dtype=dtype)
                c_m.train()
                def compose_step():
                    c_m.zero_grad(set_to_none=True)
                    c_m._update_rotors()
                    y = c_m(x_bp)
                    y.sum().backward()
                compose = gate("Compose", compose_step, n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "Compose", n, e)
                compose = NAN_PAIR; disabled.add("Compose")
            c_m = None
            torch.cuda.empty_cache()

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

    write_csv(results_path("ga/speed", "bench_rotor_apply_compose_training"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
