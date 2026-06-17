"""Backward-pass speed for rotor application across (n, batch).

Times just the apply-step backward — cs (chunk) and the action matrix
(torch_ga) are precomputed once OUTSIDE the timed loop. Each iteration
runs forward + backward through `apply_rotor`, with the gradient
flowing back to cs as a leaf tensor (chunk) or to the action matrix's
inputs (torch_ga). Rationale: the eigh-based rotor *construction* cost
(compile_bivector) is non-deterministic at the µs scale (cuSOLVER state)
and dominates by an order of magnitude, drowning out the apply cost we
actually want to compare. A separate construction-cost bench could be
added if needed.

Columns:
  chunk     forward + backward through `cl.apply_rotor(cs, x)` with cs
            precomputed and detached (requires_grad on the leaf).
  torch_ga  forward + backward through chained `mv_multiply` against
            a precomputed R / R_rev.

Versor is excluded from the bwd bench: its `exp` path detaches the
simple-plane directions inside `torch.no_grad()`, so the gradient w.r.t.
the bivector is the gradient of a different function than chunk's
(cos similarity ~0.5-0.7 against chunk's eigh-derived gradient; see
the rotor docstring in tests/correctness/test_rotor_apply.py).
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
    make_gate_step,
    not_nan,
    print_skip_summary,
    print_table_header,
    record_setup_fail,
    results_path,
    warmup_clock,
    write_csv,
)
from _cayley import shortlex_to_bp
from _rotor_apply_helpers import TorchGARotor

from cliffordcuda import CliffordAlgebra


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
warmup, iters, trials = DEFAULT_WARMUP, DEFAULT_ITERS, DEFAULT_TRIALS


def main():
    print(f"Config: batch in {batch_values}, warmup={warmup}, iters={iters}, trials={trials}")
    print(f"Tested n: {n_values}\n")
    print("Pinning GPU clock ...")
    warmup_clock()

    print_table_header(
        ["chunk", "torch_ga"],
        ratio_labels=["ga/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    gate = make_gate_step(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n
        sl_to_bp = shortlex_to_bp(n).to(device)

        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)
            x_sl = x_bp.index_select(-1, sl_to_bp).contiguous()

            # --- chunk apply+backward: cs is precompiled once and detached;
            # the timed step is just `apply_rotor` forward + backward.
            # Rationale: the eigh-based `compile_bivector` is the rotor-
            # construction cost, not the rotor-application cost. Including
            # it inside the timed loop dominates (and varies wildly with
            # the cuSOLVER kernel's state), and isn't what this bench is
            # for. The apply-step cost compares apples-to-apples with
            # torch_ga's `ga_step`, which also uses a precomputed action
            # matrix held outside the timed loop.
            try:
                cl = CliffordAlgebra(metric=[1] * n, device=device)
                bivector = torch.randn(1, cl._num_basis_biv, device=device,
                                       dtype=dtype)
                with torch.no_grad():
                    cs = cl.compile_bivector(bivector).detach().requires_grad_(True)
                def chunk_step():
                    if cs.grad is not None:
                        cs.grad = None
                    y = cl.apply_rotor(cs, x_bp)
                    y.sum().backward()
                chunk = gate("chunk", chunk_step, n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "chunk", n, e)
                chunk = NAN_PAIR; disabled.add("chunk")
            cl = bivector = cs = None

            # --- torch_ga train step ---------------------------------------
            try:
                ga_m = TorchGARotor(dim=dim, device=device, dtype=dtype)
                ga_m.train(); ga_m._update_rotors()
                def ga_step():
                    ga_m.zero_grad(set_to_none=True)
                    y = ga_m(x_sl)
                    y.sum().backward()
                    ga_m._update_rotors()
                ga = gate("torch_ga", ga_step, n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "torch_ga", n, e)
                ga = NAN_PAIR; disabled.add("torch_ga")
            ga_m = None
            torch.cuda.empty_cache()
            ratios = [
                (ga[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(ga[0]) else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us": chunk[0], "chunk_std_us": chunk[1],
                "ga_us":    ga[0],    "ga_std_us":    ga[1],
                "ga_over_chunk": ratios[0],
            })
            print(format_row(n, dim, batch, [chunk, ga], ratios=ratios))
            del x_bp, x_sl; torch.cuda.empty_cache()
        print()
        del sl_to_bp; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_rotor_apply_bwd"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
