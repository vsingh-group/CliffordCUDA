"""Forward-pass speed for rotor application across (n, batch).

Inference timing: each impl's rotor representation is built outside the
timing loop; only the application step is measured. Chunk and einsum both
apply against rotors derived from the same bivector via Versor's `exp`,
so they're directly comparable to the Versor column at the cost of having
to go through Versor for R / R_rev. torch_ga uses a random subeven R
because upstream's `exp` doesn't accept a general randn bivector at n>=4
in fp32 — see _rotor_apply_helpers.TorchGARotor for the rationale.

Columns:
  chunk     `CliffordAlgebra.apply_rotor(cs, x)`; cs precompiled outside
            the timing region (GF(2) perm kernel at n>=9; reorder+packed
            at n in {7, 8}).
  einsum    Two `EinsumGP(...)` calls against R / R_rev snapshotted from
            VersorRotor — independent evaluation path (outer einsum +
            (D, D) sign mul + XOR scatter_add).
  torch_ga  `TorchGARotor` — two `mv_multiply` calls against R / R_rev
            (random subeven element) and a precomputed dense (D, D, D)
            Cayley.
  Versor    `VersorRotor` — `per_channel_sandwich` (builds a (1, D, D)
            action matrix per call from R / R_rev; Versor doesn't expose
            a precompute-matrix API, so this build happens inside the
            timed region every call).

Inputs in their native blade ordering: ShortLex for torch_ga, bit-pattern
for chunk / einsum / Versor.
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
from _cayley import shortlex_to_bp
from _einsum_refs import EinsumGP
from _rotor_apply_helpers import TorchGARotor, VersorRotor

from _harness import import_versorai
versorai_algebra = import_versorai()

# ComposingLinearLayers (vsingh-group/ComposingLinearLayers): the Rotor
# module precomputes a (D, D) sandwich matrix; its forward is F.linear(x, M).
# The README pairs it with TravisNP/torch_ga_fix@submission for the patched
# GeometricAlgebra API; _harness.py prepends that fork on sys.path before
# any torch_ga import so the Rotor class finds the right kwargs/methods.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_repo_root, os.pardir, "ComposingLinearLayers"))
from rotor_layer import Rotor as ComposeRotor

import torch.nn.functional as F

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
        ["chunk", "einsum", "torch_ga", "Versor", "VersorAI", "Compose"],
        ratio_labels=["einsum/chunk", "ga/chunk", "Versor/chunk",
                      "VersorAI/chunk", "Compose/chunk"],
    )
    rows = []
    disabled = set()
    skip_log = []

    time_it = make_gate_run(disabled, skip_log, warmup, iters, trials)

    for n in n_values:
        dim = 1 << n
        sl_to_bp = shortlex_to_bp(n).to(device)

        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)
            x_sl = x_bp.index_select(-1, sl_to_bp).contiguous()

            # --- chunk: CliffordAlgebra (bit-pattern in/out) ---------------
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

            # --- torch_ga: ShortLex Module ---------------------------------
            try:
                ga_m = TorchGARotor(dim=dim, device=device, dtype=dtype)
                ga_m.eval(); ga_m._update_rotors()
                ga = time_it("torch_ga",
                             lambda: ga_m(x_sl), n)
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "torch_ga", n, e); ga = NAN_PAIR
                disabled.add("torch_ga")
            ga_m = None

            # --- Versor: bit-pattern Module --------------------------------
            try:
                v_m = VersorRotor(dim=dim, device=device, dtype=dtype)
                v_m.eval(); v_m._update_rotors()
                versor = time_it("Versor",
                                 lambda: v_m(x_bp), n)
                # Snapshot R / R_rev for the einsum column so chunk / Versor
                # / einsum all share the same bivector-derived rotor (only
                # the sandwich evaluation differs).
                R_e     = v_m._rotor.detach().clone()
                R_rev_e = v_m._rotor_rev.detach().clone()
            except (MemoryError, RuntimeError) as e:
                record_setup_fail(skip_log, "Versor", n, e); versor = NAN_PAIR
                disabled.add("Versor")
                R_e = R_rev_e = None
            v_m = None

            # --- einsum: two EinsumGP calls against Versor's R / R_rev -----
            if R_e is None or "einsum" in disabled:
                if "einsum" not in disabled:
                    record_carry(skip_log, "einsum", n)
                einsum = NAN_PAIR
            else:
                try:
                    e_gp = EinsumGP(n, device=device, dtype=dtype)
                    einsum = time_it("einsum",
                                     lambda: e_gp(e_gp(R_rev_e, x_bp), R_e), n)
                except (MemoryError, RuntimeError) as e:
                    record_setup_fail(skip_log, "einsum", n, e); einsum = NAN_PAIR
                    disabled.add("einsum")
                e_gp = None

            # --- VersorAI: two `geometric_product` calls against the same R / R_rev
            if R_e is None or "VersorAI" in disabled:
                if "VersorAI" not in disabled:
                    record_carry(skip_log, "VersorAI", n)
                versorai = NAN_PAIR
            else:
                versorai_sig = torch.ones(n, dtype=dtype, device=device)
                vgp = versorai_algebra.geometric_product
                versorai = time_it("VersorAI",
                                   lambda: vgp(vgp(R_rev_e, x_bp, versorai_sig), R_e, versorai_sig), n)

            # --- Compose: build (D, D) sandwich matrix once outside the
            # timed loop (ComposingLinearLayers' ctor + _update_rotors), then
            # time only the dense matmul F.linear(x, M). Single chunk so
            # forward = F.linear (rotor_layer.py:133).
            if "Compose" not in disabled:
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
                    record_setup_fail(skip_log, "Compose", n, e); compose = NAN_PAIR
                    disabled.add("Compose")
            else:
                record_carry(skip_log, "Compose", n); compose = NAN_PAIR

            R_e = R_rev_e = None
            ratios = [
                (einsum[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(einsum[0])   else float("nan"),
                (ga[0]       / chunk[0]) if not_nan(chunk[0]) and not_nan(ga[0])       else float("nan"),
                (versor[0]   / chunk[0]) if not_nan(chunk[0]) and not_nan(versor[0])   else float("nan"),
                (versorai[0] / chunk[0]) if not_nan(chunk[0]) and not_nan(versorai[0]) else float("nan"),
                (compose[0]  / chunk[0]) if not_nan(chunk[0]) and not_nan(compose[0])  else float("nan"),
            ]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_us":    chunk[0],    "chunk_std_us":    chunk[1],
                "einsum_us":   einsum[0],   "einsum_std_us":   einsum[1],
                "ga_us":       ga[0],       "ga_std_us":       ga[1],
                "versor_us":   versor[0],   "versor_std_us":   versor[1],
                "versorai_us": versorai[0], "versorai_std_us": versorai[1],
                "compose_us":  compose[0],  "compose_std_us":  compose[1],
                "einsum_over_chunk":   ratios[0],
                "ga_over_chunk":       ratios[1],
                "versor_over_chunk":   ratios[2],
                "versorai_over_chunk": ratios[3],
                "compose_over_chunk":  ratios[4],
            })
            print(format_row(n, dim, batch,
                             [chunk, einsum, ga, versor, versorai, compose],
                             ratios=ratios))
            del x_bp, x_sl; torch.cuda.empty_cache()

        print()
        # Drop the (D, D, D) dense Cayley from torch_ga's wrapper before
        # moving to the next n — at n=11 it's 32 GB and would otherwise
        # leave no room for n=12 impls (Compose's sandwich precompute OOMs).
        TorchGARotor._state.clear()
        VersorRotor._algebras.clear()
        del sl_to_bp; torch.cuda.empty_cache()

    write_csv(results_path("ga/speed", "bench_rotor_apply"), rows)
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
