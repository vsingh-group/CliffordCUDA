"""Peak GPU memory for rotor application forward across (n, batch).

Columns:
  chunk     `CliffordAlgebra.apply_rotor(cs, x)` — cs is built once via
            `compile_bivector(bivec)` in a throwaway pass that's flushed
            before measurement starts (drops bivec, gc.collect(),
            `_cuda_clearCublasWorkspaces`, `empty_cache`), so the reported
            peak excludes eigh / cuSOLVER state and reflects only the
            apply path's live tensors + transients.
  einsum    Two `EinsumGP(...)` calls implementing `R~ x R`. R / R_rev
            come from a one-shot `VersorRotor` build that's then dropped
            so the einsum column shares the chunk's "construction outside
            the measurement" convention.
  torch_ga  `TorchGARotor` — module setup builds the dense (D, D, D)
            geom Cayley + computes R/R_rev as a random subeven element
            (the only torch_ga path that works for general randn at
            n >= 9). Resident state from the module IS measured.
  Versor*   `VersorRotor` — module setup builds the rotor multivector
            via `algebra.exp(bivec)` (Pence et al. decomposition at n>=4).
            Forward uses `per_channel_sandwich`, which itself rebuilds
            the (C, D, D) action matrix in-graph each call — that
            transient is included in the peak, since Versor doesn't
            expose a precompute-matrix API.

Single-process; between-impl cleanup so each measurement is free of
pollution from prior impls:
  * `TorchGARotor._state.clear()` and `VersorRotor._algebras.clear()`
    drop class-level caches.
  * `_VersorCliffordAlgebra._CACHED_TABLES.clear()` drops Versor's
    (D, D) signed-Cayley tables (which survive clearing
    `VersorRotor._algebras`).
  * `gc.collect()` + `_cuda_clearCublasWorkspaces` + `empty_cache` flush
    cuBLAS workspaces and the PyTorch allocator cache.

Each cell reports raw peak (`torch.cuda.max_memory_allocated`) across one
inference forward, with `reset_peak_memory_stats` called immediately
before the timed call so the peak captures the apply's footprint on top
of the live state above.
"""
import csv
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_shared"))
from _harness import (
    DEFAULT_BATCH_VALUES,
    DEFAULT_N_VALUES,
    format_mem_row,
    full_cleanup,
    not_nan,
    print_mem_table_header,
    print_skip_summary,
    record_carry,
    record_setup_fail,
    results_path,
)
from _cayley import shortlex_to_bp
from _einsum_refs import EinsumGP
from _rotor_apply_helpers import TorchGARotor, VersorRotor
from core.algebra import CliffordAlgebra as _VersorCliffordAlgebra

from _harness import import_versorai
versorai_algebra = import_versorai()

# ComposingLinearLayers (vsingh-group/ComposingLinearLayers): Rotor module
# precomputes the (D, D) sandwich matrix; forward is F.linear(x, M).
# torch_ga_fix@submission is the supporting fork (README install pointer);
# _harness.py prepends it on sys.path before any torch_ga import.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_repo_root, os.pardir, "ComposingLinearLayers"))
from rotor_layer import Rotor as ComposeRotor

import torch.nn.functional as F

from cliffordcuda import CliffordAlgebra


device = "cuda"
dtype = torch.float32
n_values = DEFAULT_N_VALUES
batch_values = DEFAULT_BATCH_VALUES
IMPLS = ["chunk", "einsum", "torch_ga", "Versor", "VersorAI", "Compose"]




def measure_chunk(n, batch):
    """Peak memory for `apply_rotor` against a precomputed cs.

    `cs = compile_bivector(bivec)` is built in a throwaway pass and then the
    bivector, eigh's cuSOLVER workspace, and other transient state from
    construction are dropped before measurement starts. The reported peak
    is then only what the apply path itself needs: cl's structural
    buffers + cs + x_bp + apply_rotor's transient allocations. Construction
    cost is a separate measurement (and isn't what this bench claims to
    report)."""
    dim = 1 << n
    # 1. Throwaway compile to get cs, then drop everything else.
    cl_tmp = CliffordAlgebra(metric=[1] * n, device=device)
    bivec = torch.randn(1, cl_tmp._num_basis_biv, device=device, dtype=dtype)
    with torch.no_grad():
        cs = cl_tmp.compile_bivector(bivec).detach().clone()
    del cl_tmp, bivec
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

    # 2. Rebuild only the structural part of cl (no compile_bivector).
    cl = CliffordAlgebra(metric=[1] * n, device=device)
    x_bp = torch.randn(batch, dim, device=device, dtype=dtype)
    with torch.no_grad():
        cl.apply_rotor(cs, x_bp)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        cl.apply_rotor(cs, x_bp)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del cl, cs, x_bp
    return peak


def measure_einsum(n, batch, x_bp):
    """Two-EinsumGP sandwich against a bivector-derived R / R_rev. R comes
    from a one-shot VersorRotor build (same source as the speed bench's
    einsum column) so the chunk / Versor / einsum trio share the rotor."""
    dim = 1 << n
    v_m = VersorRotor(dim=dim, device=device, dtype=dtype)
    v_m.eval(); v_m._update_rotors()
    R     = v_m._rotor.detach().clone()
    R_rev = v_m._rotor_rev.detach().clone()
    del v_m

    e_gp = EinsumGP(n, device=device, dtype=dtype)
    with torch.no_grad():
        e_gp(e_gp(R_rev, x_bp), R)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        e_gp(e_gp(R_rev, x_bp), R)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del e_gp, R, R_rev
    return peak


def measure_versorai(n, batch, x_bp):
    """Two-`geometric_product` sandwich against the same R / R_rev source
    as the einsum column."""
    dim = 1 << n
    v_m = VersorRotor(dim=dim, device=device, dtype=dtype)
    v_m.eval(); v_m._update_rotors()
    R     = v_m._rotor.detach().clone()
    R_rev = v_m._rotor_rev.detach().clone()
    del v_m

    sig = torch.ones(n, dtype=dtype, device=device)
    vgp = versorai_algebra.geometric_product
    with torch.no_grad():
        vgp(vgp(R_rev, x_bp, sig), R, sig)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        vgp(vgp(R_rev, x_bp, sig), R, sig)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del R, R_rev
    return peak


def measure_compose(n, batch, x_bp):
    """ComposingLinearLayers' Rotor: precompute the (D, D) sandwich matrix
    via Rotor(...)._update_rotors(), then measure peak around F.linear(x, M).
    M itself is resident through the measurement (which is the bench's
    convention — chunk's cs / einsum's R / Versor's algebra tables also
    persist through their measured calls)."""
    dim = 1 << n
    c_m = ComposeRotor(in_dim=dim, out_dim=dim, in_chunks=1, out_chunks=1,
                       chunk_size=dim, single_rotor=True, alpha_param=False,
                       bias_param=False, device=device, dtype=dtype)
    c_m.eval()
    M = c_m.sandwich_product_matrix.detach()
    del c_m

    with torch.no_grad():
        F.linear(x_bp, M)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        F.linear(x_bp, M)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del M
    return peak


def measure_module(cls, n, batch, x):
    """Measure peak of a single forward through `cls`. `x` is the input the
    Module expects (ShortLex for TorchGARotor, bit-pattern for VersorRotor)."""
    dim = 1 << n
    m = cls(dim=dim, device=device, dtype=dtype)
    m.eval(); m._update_rotors()
    with torch.no_grad():
        m(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        m(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del m
    return peak


def main():
    print(f"Config: batch in {batch_values}, n in {n_values}\n")
    print_mem_table_header(
        IMPLS, ratio_labels=[f"{r}/chunk" for r in IMPLS[1:]],
    )
    rows = []
    disabled = set()
    skip_log = []
    nan = float("nan")

    for n in n_values:
        dim = 1 << n
        sl_to_bp = shortlex_to_bp(n).to(device)
        for batch in batch_values:
            x_bp = torch.randn(batch, dim, device=device, dtype=dtype)
            x_sl = x_bp.index_select(-1, sl_to_bp).contiguous()
            cells = {}

            for impl in IMPLS:
                if impl in disabled:
                    record_carry(skip_log, impl, n)
                    cells[impl] = nan
                    continue
                full_cleanup()
                try:
                    if impl == "chunk":
                        cells[impl] = measure_chunk(n, batch)
                    elif impl == "einsum":
                        cells[impl] = measure_einsum(n, batch, x_bp)
                    elif impl == "torch_ga":
                        cells[impl] = measure_module(TorchGARotor, n, batch, x_sl)
                    elif impl == "Versor":
                        cells[impl] = measure_module(VersorRotor, n, batch, x_bp)
                    elif impl == "VersorAI":
                        cells[impl] = measure_versorai(n, batch, x_bp)
                    elif impl == "Compose":
                        cells[impl] = measure_compose(n, batch, x_bp)
                except torch.cuda.OutOfMemoryError:
                    cells[impl] = nan
                    record_setup_fail(skip_log, impl, n, RuntimeError("OOM"))
                    disabled.add(impl)
                except Exception as e:
                    cells[impl] = nan
                    record_setup_fail(skip_log, impl, n, e)
                    disabled.add(impl)
            full_cleanup()

            chunk = cells["chunk"]
            ratios = [(cells[r] / chunk) if not_nan(chunk) and not_nan(cells[r]) else nan
                      for r in IMPLS[1:]]
            rows.append({
                "batch": batch, "n": n, "dim": dim,
                "chunk_mib":    cells["chunk"],
                "einsum_mib":   cells["einsum"],
                "ga_mib":       cells["torch_ga"],
                "versor_mib":   cells["Versor"],
                "versorai_mib": cells["VersorAI"],
                "compose_mib":  cells["Compose"],
                "einsum_over_chunk":   ratios[0],
                "ga_over_chunk":       ratios[1],
                "versor_over_chunk":   ratios[2],
                "versorai_over_chunk": ratios[3],
                "compose_over_chunk":  ratios[4],
            })
            print(format_mem_row(n, dim, batch,
                                 [cells[r] for r in IMPLS], ratios=ratios))
            del x_bp, x_sl
        del sl_to_bp
        print()

    path = results_path("ga/memory", "bench_rotor_apply")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")
    print_skip_summary(skip_log)


if __name__ == "__main__":
    main()
