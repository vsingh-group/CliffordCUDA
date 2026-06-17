"""Shared utilities for benchmark scripts."""
import csv
import os
import statistics
import sys
import time

import torch


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../benchmark

# torch_ga_fix (TravisNP/torch_ga_fix@submission) is the fork
# ComposingLinearLayers depends on. Superset of vanilla torch_ga; safe to
# prepend for every bench. Must happen BEFORE any `import torch_ga`.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HARNESS_DIR)),
                                "torch_ga_fix"))


# ---------------------------------------------------------------------------
# Results path
# ---------------------------------------------------------------------------

def results_path(track: str, bench_name: str) -> str:
    """Canonical CSV path: benchmark/results/<track>/<bench_name>.csv.

    Creates the parent directory if it doesn't exist."""
    out_dir = os.path.join(_HARNESS_DIR, "results", track)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{bench_name}.csv")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def warmup_clock(seconds: float = 2.0):
    """Pin the GPU clock high for `seconds` by doing matmul work."""
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = a @ b
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


NAN_PAIR: tuple[float, float] = (float("nan"), float("nan"))
SKIP_THRESHOLD_S = 1.0
PROBE_RUNS = 3

# Defaults shared across every bench. Override at the top of a bench file
# (or via runtime monkey-patching for smoke tests) only when needed.
DEFAULT_N_VALUES = [7, 8, 9, 10, 11, 12]
DEFAULT_BATCH_VALUES = [1, 16, 64]
DEFAULT_WARMUP = 25
DEFAULT_ITERS = 10
DEFAULT_TRIALS = 10


def not_nan(x) -> bool:
    """True iff `x` is not NaN. (`x == x` is False only when x is NaN.)"""
    return x == x


def fwd_bwd(fn_fwd, inputs, grad_c):
    """Build the forward graph and run a single backward via autograd.grad.
    Used by every memory `_bwd` bench inside `measure_peak_memory`."""
    c = fn_fwd()
    torch.autograd.grad(c, inputs, grad_outputs=grad_c)


def bp_ab(n: int, batch: int, device="cuda", dtype=None, requires_grad: bool = False):
    """(a, b) bit-pattern randn pair shared by every memory fwd bench."""
    if dtype is None:
        dtype = torch.float32
    dim = 1 << n
    a = torch.randn(batch, dim, device=device, dtype=dtype,
                    requires_grad=requires_grad)
    b = torch.randn(batch, dim, device=device, dtype=dtype,
                    requires_grad=requires_grad)
    return a, b


# ---------------------------------------------------------------------------
# One generic probe / bench / safe trio, parameterized on a `time_one()`
# callable that returns seconds for a single timed iteration. The public
# entry points are `safe_run / safe_bwd / safe_step`; they differ only in
# how they construct `time_one` (forward call / backward-only / step thunk).
# ---------------------------------------------------------------------------


def _time_fwd(fn):
    """time_one() that sync-times a single forward call."""
    def time_one() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    return time_one


def _time_bwd(fn_fwd, inputs, grad_c):
    """time_one() that builds the fwd graph fresh, then sync-times ONLY the
    backward (forward time is outside the timer)."""
    def time_one() -> float:
        torch.cuda.synchronize()
        c = fn_fwd()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.autograd.grad(c, inputs, grad_outputs=grad_c)
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    return time_one


def _time_step(step):
    """time_one() that sync-times a no-arg `step()` end-to-end (fwd+bwd+update)."""
    def time_one() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    return time_one


def _bench(time_one, warmup: int = 25, iters: int = 10,
           trials: int = 10) -> tuple[float, float]:
    """Warmup then trials × iters, return (median_us, stdev_us)."""
    for _ in range(warmup):
        time_one()
    times = []
    for _ in range(trials):
        accum = 0.0
        for _ in range(iters):
            accum += time_one()
        times.append(accum / iters * 1e6)
    return statistics.median(times), statistics.stdev(times)


def _probe(time_one) -> tuple[bool, float]:
    """PROBE_RUNS singles with early-exit. First call also absorbs any
    first-iteration init (driver, kernel template instantiation, etc.)."""
    time_one()                                  # absorb first-call init
    times = []
    for _ in range(PROBE_RUNS):
        t = time_one()
        times.append(t)
        if t > SKIP_THRESHOLD_S:
            return True, sum(times) / len(times)
    return sum(times) / len(times) > SKIP_THRESHOLD_S, sum(times) / len(times)


def _safe(name: str, time_one, n: int, skip_log: list = None,
          **bench_kwargs) -> tuple[float, float]:
    """`_bench` wrapped in OOM / slow-runtime guard. Returns NAN_PAIR on
    failure or skip; appends a {ref, n, reason} dict to `skip_log`."""
    def _record(reason: str):
        if skip_log is not None:
            skip_log.append({"ref": name, "n": n, "reason": reason})
    try:
        skip, avg = _probe(time_one)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); _record("OOM (probe)"); return NAN_PAIR
    except Exception as e:
        torch.cuda.empty_cache(); _record(f"{type(e).__name__} (probe)")
        return NAN_PAIR
    if skip:
        _record(f"too slow ({avg*1000:.0f} ms > {SKIP_THRESHOLD_S*1000:.0f} ms)")
        return NAN_PAIR
    try:
        return _bench(time_one, **bench_kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); _record("OOM"); return NAN_PAIR
    except Exception as e:
        torch.cuda.empty_cache(); _record(type(e).__name__); return NAN_PAIR


def safe_run(name: str, fn, n: int, skip_log: list = None,
             **bench_kwargs) -> tuple[float, float]:
    """Forward-call timing wrapped in OOM / slow-runtime guard."""
    return _safe(name, _time_fwd(fn), n, skip_log, **bench_kwargs)


def safe_bwd(name: str, fn_fwd, inputs, grad_c, n: int,
             skip_log: list = None, **bench_kwargs) -> tuple[float, float]:
    """Backward-only timing wrapped in OOM / slow-runtime guard."""
    return _safe(name, _time_bwd(fn_fwd, inputs, grad_c), n, skip_log,
                 **bench_kwargs)


def safe_step(name: str, step, n: int, skip_log: list = None,
              **bench_kwargs) -> tuple[float, float]:
    """End-to-end train-step timing wrapped in OOM / slow-runtime guard."""
    return _safe(name, _time_step(step), n, skip_log, **bench_kwargs)


# ---------------------------------------------------------------------------
# Gate factories — return the `gate(...)` closure that every speed bench
# uses to call safe_run/safe_bwd/safe_step while tracking a `disabled` set
# (so once an impl OOMs or skips, every subsequent (n, batch) cell for it
# carries NAN_PAIR + a "carry" record into the skip log).
# ---------------------------------------------------------------------------


def make_gate_run(disabled: set, skip_log: list,
                  warmup: int = DEFAULT_WARMUP,
                  iters: int = DEFAULT_ITERS,
                  trials: int = DEFAULT_TRIALS):
    """Returns gate(name, fn, n) for forward-only speed benches."""
    def gate(name, fn, n):
        if name in disabled:
            record_carry(skip_log, name, n)
            return NAN_PAIR
        pair = safe_run(name, fn, n, skip_log=skip_log,
                        warmup=warmup, iters=iters, trials=trials)
        if pair[0] != pair[0]:
            disabled.add(name)
        return pair
    return gate


def make_gate_bwd(disabled: set, skip_log: list,
                  warmup: int = DEFAULT_WARMUP,
                  iters: int = DEFAULT_ITERS,
                  trials: int = DEFAULT_TRIALS):
    """Returns gate(name, fn_fwd, inputs, grad_c, n) for backward speed benches."""
    def gate(name, fn_fwd, inputs, grad_c, n):
        if name in disabled:
            record_carry(skip_log, name, n)
            return NAN_PAIR
        pair = safe_bwd(name, fn_fwd, inputs, grad_c, n,
                        skip_log=skip_log,
                        warmup=warmup, iters=iters, trials=trials)
        if pair[0] != pair[0]:
            disabled.add(name)
        return pair
    return gate


def make_gate_step(disabled: set, skip_log: list,
                   warmup: int = DEFAULT_WARMUP,
                   iters: int = DEFAULT_ITERS,
                   trials: int = DEFAULT_TRIALS):
    """Returns gate(name, step, n) for end-to-end train-step benches (rotor bwd)."""
    def gate(name, step, n):
        if name in disabled:
            record_carry(skip_log, name, n)
            return NAN_PAIR
        pair = safe_step(name, step, n, skip_log=skip_log,
                         warmup=warmup, iters=iters, trials=trials)
        if pair[0] != pair[0]:
            disabled.add(name)
        return pair
    return gate


def _patch_versorai_sign_matrix():
    """Replace VersorAI's get_sign_matrix with a vectorized implementation.

    The shipped version walks a `for a in range(2**D): for b in range(2**D):`
    Python double loop calling a get_sign_logic helper that itself loops over
    D bits — ~16 M Python calls at n=12, each ~30 s. Versor (Concode0/Versor)
    computes the same matrix with O(D) vectorized torch ops; this replica
    follows that approach. Same math, same cached output, ~1000x faster
    rebuild.

    Falls back to the original implementation for `device_type='numpy'`/`'mlx'`
    (only `'torch'` and `'torch_cuda'` are used in our benches)."""
    # _harness.py lives at <repo_root>/benchmark/_harness.py; VersorAI's
    # library sits next to the repo root as <parent>/VersorAI/library.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(_HARNESS_DIR)), "VersorAI", "library"))
    import gacore.kernel as _vk

    original = _vk.get_sign_matrix
    if getattr(original, "_vectorized_patched", False):
        return

    def vectorized(signature, device_type="numpy"):
        sig_tuple = (tuple(signature.tolist())
                     if hasattr(signature, "tolist") else tuple(signature))
        cache_key = (sig_tuple, device_type)
        if cache_key in _vk._SIGN_CACHE:
            return _vk._SIGN_CACHE[cache_key]

        if device_type not in ("torch", "torch_cuda"):
            return original(signature, device_type)

        D = len(sig_tuple)
        n_dims = 1 << D
        dev = "cuda" if device_type == "torch_cuda" else "cpu"

        # VersorAI indexes S[i, k] = sign(e_i * e_{i^k}) — the column axis is
        # the XOR-output blade, not the second operand. So `B` (the actual
        # second operand entering the commutator/metric math) is i ^ k.
        indices = torch.arange(n_dims, dtype=torch.long, device=dev)
        i_idx = indices.unsqueeze(1)        # (n_dims, 1)
        k_idx = indices.unsqueeze(0)        # (1, n_dims)
        A = i_idx
        B = i_idx ^ k_idx

        # Commutator sign: for each bit i set in A, count bits of B strictly
        # below i; total swap parity drives the sign. Matches Versor's
        # vectorized formulation in _compute_signs.
        swap_counts = torch.zeros((n_dims, n_dims),
                                  dtype=torch.long, device=dev)
        for i in range(D):
            a_i = (A >> i) & 1
            lower_mask = (1 << i) - 1
            b_lower = B & lower_mask
            b_lower_cnt = torch.zeros_like(B)
            temp_b = b_lower
            for _ in range(D):
                b_lower_cnt += temp_b & 1
                temp_b = temp_b >> 1
            swap_counts += a_i * b_lower_cnt
        comm_sign = (1 - 2 * (swap_counts & 1)).to(torch.float32)

        # Metric sign: for each bit i set in (A & B), multiply by sig[i].
        # Bit-not-set contributes factor 1; sig[i] == 0 collapses entry to 0.
        intersection = A & B
        m_sign = torch.ones((n_dims, n_dims), dtype=torch.float32, device=dev)
        for i in range(D):
            s = float(sig_tuple[i])
            bit_set = ((intersection >> i) & 1).to(torch.float32)
            factor = bit_set * s + (1.0 - bit_set)
            m_sign = m_sign * factor

        S = (comm_sign * m_sign).contiguous()
        _vk._SIGN_CACHE[cache_key] = S
        return S

    vectorized._vectorized_patched = True
    _vk.get_sign_matrix = vectorized


def import_versorai():
    """Import VersorAI's `gacore.kernel` (with the vectorized sign-matrix
    patch applied) and return the module.

    Call this from a bench ONLY when it actually compares against VersorAI.
    Keeping the import here — rather than at _harness import time — means
    benches that don't use VersorAI never require the sibling checkout to be
    present."""
    _patch_versorai_sign_matrix()      # inserts the path + patches get_sign_matrix
    import gacore.kernel as vk
    return vk


def full_cleanup():
    """Drop every cross-call cache the bench impls touch, then collect.

    Anything that lingers between impls and shows up in `max_memory_allocated`
    biases the next impl's reported peak. This walks the cliffordcuda package
    looking for every `@functools.lru_cache`-decorated builder and clears it,
    plus the known module-level caches in the witness libraries (Versor,
    VersorAI) and the rotor-helper class state. Safe to call before every
    impl measurement."""
    import gc as _gc
    import importlib
    import pkgutil

    # 1. Every @functools.lru_cache in cliffordcuda.* (covers every product's
    # packed-sign / subset-LUT / sign-bwd / dual builders without hardcoding).
    # These LUTs are tensors resident on the GPU, so if they're not cleared
    # between impls, the next impl's `max_memory_allocated()` includes them
    # — chunk's LUT (e.g. 2 MiB at n=12) would leak into einsum / Versor /
    # VersorAI / torch_ga's reported peak.
    import cliffordcuda as _cc
    for mod_info in pkgutil.walk_packages(_cc.__path__,
                                          prefix=_cc.__name__ + "."):
        mod = importlib.import_module(mod_info.name)
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if callable(obj) and hasattr(obj, "cache_clear"):
                obj.cache_clear()

    # 2. Witness-library caches. Clear each one only if its module is already
    # loaded, so a bench that never used a given witness doesn't import it just
    # to clear an empty cache.
    versor = sys.modules.get("core.algebra")              # Versor (Concode0)
    if versor is not None:
        versor.CliffordAlgebra._CACHED_TABLES.clear()
    versorai = sys.modules.get("gacore.kernel")           # VersorAI
    if versorai is not None:
        versorai._SIGN_CACHE.clear()
        versorai._METRIC_CACHE.clear()
    rotor_helpers = sys.modules.get("_rotor_apply_helpers")
    if rotor_helpers is not None:
        rotor_helpers.TorchGARotor._state.clear()
        rotor_helpers.VersorRotor._algebras.clear()
    compose = sys.modules.get("rotor_layer")              # ComposingLinearLayers
    if compose is not None:
        compose.Rotor.shared_algebras.clear()

    # 3. PyTorch allocator + cuBLAS workspaces.
    _gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


def measure_peak_memory(fn, warmup_runs: int = 2) -> float:
    """Run `fn` a couple of times to absorb first-call allocations (cuBLAS
    workspaces, autotuning state), then measure the raw peak across one more
    call. Returns MiB.

    Caller's responsibility to ensure each impl's measurement starts from a
    clean state — only the impl's own tables and inputs resident before the
    call. With that, `max_memory_allocated()` is the actual GPU pressure the
    impl exerts (precomputed table + per-call workspace), which is what you
    want for a "will this fit?" comparison."""
    for _ in range(warmup_runs):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def record_carry(skip_log: list, name: str, n: int):
    """Record that `name` was skipped at `n` because it had already been
    disabled by an earlier failure. Reason is None — the failure was logged
    elsewhere."""
    if skip_log is not None:
        skip_log.append({"ref": name, "n": n, "reason": None})


def record_setup_fail(skip_log: list, name: str, n: int, exc: BaseException):
    """Record a setup-time failure (e.g. OOM building a dense Cayley)."""
    if skip_log is not None:
        skip_log.append({"ref": name, "n": n,
                         "reason": f"{type(exc).__name__} (setup)"})


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_LEFT = "  {n:<3} {dim:>5} {batch:<6}"   # index columns
_COL_W = 16                                  # one timing column width


def print_table_header(timing_labels: list[str], ratio_labels: list[str] = None):
    """Print a table header. `timing_labels` -> e.g. ["chunk", "torch_ga"].
    Each timing column shows median±std in us. Trailing ratio columns are
    e.g. ["ga/chunk"]."""
    ratio_labels = ratio_labels or []
    cols = " ".join(f"{lab + ' (us)':>{_COL_W}}" for lab in timing_labels)
    ratios = " ".join(f"{lab:>10}" for lab in ratio_labels)
    line = f"  {'n':<3} {'dim':>5} {'batch':<6} | {cols}"
    if ratios:
        line += f" | {ratios}"
    print(line)
    print("  " + "-" * (len(line) - 2))


def _fmt_pair(pair: tuple[float, float]) -> str:
    med, std = pair
    if med != med:
        return f"{'nan':>{_COL_W}}"
    return f"{med:>7.0f}±{std:<.0f}".rjust(_COL_W)


def format_row(n: int, dim: int, batch: int,
               timings: list[tuple[float, float]],
               ratios: list[float] = None) -> str:
    """Format one data row matching print_table_header()."""
    ratios = ratios or []
    cells = " ".join(_fmt_pair(t) for t in timings)
    line = f"  {n:<3} {dim:>5} {batch:<6} | {cells}"
    if ratios:
        r_cells = " ".join(
            (f"{r:>9.2f}×" if r == r else f"{'nan':>10}")
            for r in ratios
        )
        line += f" | {r_cells}"
    return line


def print_mem_table_header(labels: list[str], ratio_labels: list[str] = None):
    """Header for a memory-bench table; each column shows peak GPU MiB."""
    ratio_labels = ratio_labels or []
    cols = " ".join(f"{lab + ' (MiB)':>{_COL_W}}" for lab in labels)
    ratios = " ".join(f"{lab:>10}" for lab in ratio_labels)
    line = f"  {'n':<3} {'dim':>5} {'batch':<6} | {cols}"
    if ratios:
        line += f" | {ratios}"
    print(line)
    print("  " + "-" * (len(line) - 2))


def format_mem_row(n: int, dim: int, batch: int,
                   mems: list[float],
                   ratios: list[float] = None) -> str:
    """Format one row of memory data (MiB per column)."""
    ratios = ratios or []
    cells = " ".join(
        (f"{'nan':>{_COL_W}}" if m != m else f"{m:>{_COL_W - 4}.1f} MiB")
        for m in mems
    )
    line = f"  {n:<3} {dim:>5} {batch:<6} | {cells}"
    if ratios:
        r_cells = " ".join(
            (f"{r:>9.2f}×" if r == r else f"{'nan':>10}")
            for r in ratios
        )
        line += f" | {r_cells}"
    return line


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict]):
    """Write a list of dicts (uniform keys) to CSV."""
    if not rows:
        print(f"  (no rows; skipping CSV write to {path})")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path}")


def print_skip_summary(skip_log: list):
    """Print an aggregated summary of skips, one line per ref:
        <ref>: <reason> at n=<first-failure n>; skipped at n in {n_set}
    The reason is taken from the first entry with non-None `reason`."""
    if not skip_log:
        return
    by_ref: dict[str, dict] = {}
    for e in skip_log:
        info = by_ref.setdefault(e["ref"], {"first_reason": None,
                                            "first_n": None, "ns": set()})
        info["ns"].add(e["n"])
        if e["reason"] is not None and info["first_reason"] is None:
            info["first_reason"] = e["reason"]
            info["first_n"] = e["n"]
    print("\nSkipped:")
    for ref, info in by_ref.items():
        ns = sorted(info["ns"])
        if info["first_reason"]:
            print(f"  {ref}: {info['first_reason']} at n={info['first_n']}; "
                  f"skipped at n in {ns}")
        else:
            print(f"  {ref}: skipped at n in {ns}")
