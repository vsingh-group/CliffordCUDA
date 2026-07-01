"""Tiled geometric product for n whose 2^n multivectors exceed the per-block
shared memory the standard kernel needs.

The standard kernel stages both full multivectors in shared memory, so the
largest n it can run is bounded by the GPU's shared-memory-per-block budget
(e.g. ~12 on a 48 KB device, 13 on an A100, ~14 on an H100) -- it is NOT a fixed
number. This variant splits the index by its top H = n - L bits into X = 2**H
tiles of M = 2**L coefficients each, where L is the largest dimension that fits
the *actual* device (queried at runtime, never hardcoded). By bilinearity and
the XOR structure of the geometric product (output index = i ^ j), tile-pair
(p, q) contributes entirely to output tile r = p ^ q, and the sign factorizes:

    sigma(i, j) = sigma_high(p, q)
                  * (-1)**(popcount(p) * popcount(j_low))     # cross term
                  * sigma_low(i_low, j_low)

Two implementations:
  * geom_prod_tiled        -- orchestrated in torch over the standard dim-L
                              kernel (X**2 launches); differentiable. The inner
                              kernel only instantiates n=5..13, so L is capped at
                              min(13, device fit). Kept as a reference/oracle.
  * geom_prod_tiled_fused  -- the runtime-(L, H) CUDA kernel; DIFFERENTIABLE
                              (forward and both backward sums run on that one
                              kernel via cross_mode). L is capped only by the
                              device's shared memory, so it can exceed 13 on a
                              larger GPU. This is the path the public products use.

The same fused kernel and machinery serve wedge / inner / contraction /
regressive too, via product-specific masked sign tables (see the *_tiled_fused
functions). Verified against the standard kernels across Cl(p, q, r), including
degenerate metrics (tests/correctness/test_geom_prod_tiled.py, test_products_tiled.py).
"""
import functools
import os

import torch

from ..._config import _GA_KERNELS_DIR, load_extension
from .geom_prod import (
    geom_prod, _normalize_metric, _build_sign_value_table,
)

# The standard geom_prod CUDA kernel only instantiates n = 5..13 (see the
# dispatch in geom_prod.cu), so the orchestrated path's inner product is capped
# there regardless of GPU. The fused kernel takes L at runtime and is bounded
# only by device shared memory.
_STD_KERNEL_MAX_N = 13


def load_geom_prod_tiled_cuda():
    if not hasattr(load_geom_prod_tiled_cuda, "_module"):
        load_geom_prod_tiled_cuda._module = load_extension(
            name="geom_prod_tiled_cuda",
            sources=[os.path.join(_GA_KERNELS_DIR, "geom_prod_tiled.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return load_geom_prod_tiled_cuda._module


def _tiled_smem_bytes_w(L: int, W: int, has_zeros: bool) -> int:
    """Dynamic shared memory the tiled kernel needs for inner dimension L with W
    warps/block: two staged tiles (2*2^L floats) + staged sigma_low rows (W) +
    validity rows for a degenerate metric. Mirrors the kernel's allocation."""
    sig = W * ((1 << L) // 32) * 4
    return 2 * (1 << L) * 4 + sig + (sig if has_zeros else 0)


def _tiled_smem_bytes(L: int, has_zeros: bool) -> int:
    """Footprint at the kernel's default warp count (16 for L>=9 else 4); used by
    the L-pickers, which reason about occupancy at that default."""
    return _tiled_smem_bytes_w(L, 16 if L >= 9 else 4, has_zeros)


@functools.lru_cache(maxsize=None)
def _device_fused_W(L: int, has_zeros: bool, device: str) -> int:
    """Warps/block for the tiled kernel: the largest W (<=32, i.e. 1024 threads)
    whose footprint fits the budget. More warps/block -> fewer output z-blocks ->
    less redundant tile re-staging, at the same warps/SM (occupancy unchanged).
    Sweep: W=32 >= W=16 across n=14..16, ~4.5% at n=16; W=8 worse."""
    budget = _optin_budget(device)
    for W in (32, 16, 8, 4):
        if W * 32 <= 1024 and _tiled_smem_bytes_w(L, W, has_zeros) <= budget:
            return W
    return 4


@functools.lru_cache(maxsize=None)
def _optin_budget(device: str) -> int:
    """Max dynamic shared memory a block can opt into on this device, queried via
    CUDA (torch's get_device_properties does not expose it here)."""
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    return load_geom_prod_tiled_cuda().max_dynamic_smem_optin(idx)


def _largest_L_within(budget: int, has_zeros: bool):
    best = None
    for L in range(5, 24):
        if _tiled_smem_bytes(L, has_zeros) <= budget:
            best = L
        else:
            break
    return best


@functools.lru_cache(maxsize=None)
def _device_max_fit(device: str, has_zeros: bool) -> int:
    """Largest inner dimension L whose footprint fits the device's per-block
    opt-in budget (1 block/SM). GPU-dependent, queried, never hardcoded."""
    best = _largest_L_within(_optin_budget(device), has_zeros)
    if best is None:
        raise RuntimeError(
            f"device shared memory ({_optin_budget(device)} B/block) too small "
            "even for L=5")
    return best


@functools.lru_cache(maxsize=None)
def _device_fused_L(device: str, has_zeros: bool) -> int:
    """Perf-optimal inner L for the fused kernel. Profiling showed the kernel is
    shared-memory-occupancy bound (the L-sweep made the max-fit L, at 1 block/SM
    / 25% occupancy, the SLOWEST choice). Targeting >=2 blocks/SM -- the largest
    L whose footprint is <= half the budget -- recovers ~1.3x. Using budget//2
    guarantees two blocks co-reside without needing the per-SM total, since
    2*(budget//2) <= budget <= shared-memory-per-SM. Falls back to the 1-block
    max-fit on a device too small to hold two."""
    budget = _optin_budget(device)
    best = _largest_L_within(budget // 2, has_zeros)
    if best is None:
        best = _largest_L_within(budget, has_zeros)
    if best is None:
        raise RuntimeError(
            f"device shared memory ({budget} B/block) too small even for L=5")
    return best


# ── Orchestrated variant (differentiable, X**2 launches over the std kernel) ──

@functools.lru_cache(maxsize=None)
def _tile_tables(n: int, L: int, metric: tuple, device: str):
    """sigma_high (X, X) float32, gi_low (M,) float32 = (-1)**popcount(j_low),
    and popcount_odd[p] for the orchestrated tiling."""
    H = n - L
    X, M = 1 << H, 1 << L
    sigma_high = _build_sign_value_table(H, metric[L:]).to(torch.float32).to(device)
    jl = torch.arange(M, device=device)
    pc = torch.zeros(M, dtype=torch.long, device=device)
    for b in range(L):
        pc += (jl >> b) & 1
    gi_low = torch.where((pc & 1) == 1, -1.0, 1.0).to(torch.float32)
    popcount_odd = [bin(p).count("1") & 1 == 1 for p in range(X)]
    return sigma_high, gi_low, popcount_odd


def geom_prod_tiled(a, b, metric=None, max_fit: int = None):
    """c = a * b in Cl(p, q, r), differentiable. Tiles when n exceeds what fits
    shared memory so each sub-product runs on the standard dim-L kernel.

    max_fit defaults to min(13, device fit) (13 = the std kernel's instantiation
    cap). Pass an explicit value to force the tiling path at small n (testing);
    must be >= 5."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric = _normalize_metric(n, metric)
    if max_fit is None:
        max_fit = min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), 0 in metric))
    if n <= max_fit:
        return geom_prod(a, b, metric=metric)        # fits; standard kernel
    L = max_fit
    if L < 5:
        raise ValueError(f"max_fit must be >= 5 (kernel lower bound); got {L}")
    H = n - L
    X, M = 1 << H, 1 << L
    low_metric = metric[:L]
    sigma_high, gi_low, popcount_odd = _tile_tables(n, L, metric, str(a.device))

    A = a.view(-1, X, M)
    B = b.view(-1, X, M)
    out_tiles = []
    for r in range(X):
        parts = []
        for p in range(X):
            q = p ^ r
            s = float(sigma_high[p, q])
            if s == 0.0:
                continue
            bt = B[:, q, :]
            if popcount_odd[p]:
                bt = bt * gi_low
            sub = geom_prod(A[:, p, :].contiguous(), bt.contiguous(), metric=low_metric)
            parts.append(s * sub)
        acc = sum(parts) if parts else a.new_zeros(A.shape[0], M)
        out_tiles.append(acc)
    return torch.stack(out_tiles, dim=1).reshape(a.shape)


# ── Fused path (runtime-(L, H) CUDA kernel, differentiable): sign builders ──

def _sign_value_table_gpu(n: int, metric: tuple, device: str) -> torch.Tensor:
    """GPU build of the (2^n, 2^n) int8 sigma table (same result as the CPU
    _build_sign_value_table), built on `device`. Fully parallel -- no per-bit
    Python loop:

      * reorder parity(i, j) = sum_{a>b} bit_a(i)*bit_b(j)
                             = sum_a bit_a(i) * (#set bits of j below a)
                             = bits_i @ lowcount_j^T          -- one GEMM.
      * metric factor: sign = (-1)^popcount(i&j & neg_mask) via XOR-fold parity;
        zeroed where i&j shares a null (metric==0) basis.
    """
    dim = 1 << n
    if n == 0:
        return torch.ones(1, 1, dtype=torch.int8, device=device)
    idx = torch.arange(dim, device=device, dtype=torch.int64)
    ar = torch.arange(n, device=device, dtype=torch.int64)
    bits = ((idx.view(dim, 1) >> ar) & 1)                # (dim, n): bit a of each index
    lowcount = torch.cumsum(bits, dim=1) - bits          # (dim, n): #set bits strictly below a
    # GEMM in fp32 is exact here: entries <= n(n-1)/2 << 2^24.
    parity = (bits.float() @ lowcount.float().t()).to(torch.int64)
    sign = 1 - 2 * (parity & 1)                          # (dim, dim) in {-1, +1}

    common = idx.view(dim, 1) & idx.view(1, dim)
    neg_mask = sum(1 << k for k in range(n) if int(metric[k]) == -1)
    zero_mask = sum(1 << k for k in range(n) if int(metric[k]) == 0)
    factor = sign
    if neg_mask:
        x = common & neg_mask                            # popcount parity via XOR-fold
        x = x ^ (x >> 32); x = x ^ (x >> 16); x = x ^ (x >> 8)
        x = x ^ (x >> 4); x = x ^ (x >> 2); x = x ^ (x >> 1)
        factor = factor * (1 - 2 * (x & 1))
    if zero_mask:
        factor = factor * ((common & zero_mask) == 0)
    return factor.to(torch.int8)


def _pack_fwd_gpu(sigma: torch.Tensor, device: str):
    """GPU pack of the forward LUT from a (dim, dim) sigma table -- vectorized,
    no Python loop (the CPU pack_fwd_from_sigma loops k = 0..dim-1). Bit t of
    word [k, c] = 1 iff sigma[c*32+t, (c*32+t)^k] == -1; validity similarly for
    !=0. Matches pack_fwd_from_sigma bit-for-bit (same int32 powers, so 1<<31
    overflows to the sign bit identically)."""
    dim = sigma.shape[0]
    chunks = dim // 32
    ii = torch.arange(dim, device=device, dtype=torch.int64).view(1, dim)   # i
    kk = torch.arange(dim, device=device, dtype=torch.int64).view(dim, 1)   # k
    gathered = sigma[ii.expand(dim, dim), ii ^ kk]                          # [k, i] = sigma[i, i^k]
    powers = (1 << torch.arange(32, device=device, dtype=torch.int32))
    g_neg = (gathered == -1).to(torch.int32)
    packed_sign = (g_neg.view(dim, chunks, 32) * powers).sum(-1).to(torch.int32).contiguous()
    packed_valid = None
    if bool((sigma == 0).any()):
        g_nz = (gathered != 0).to(torch.int32)
        packed_valid = (g_nz.view(dim, chunks, 32) * powers).sum(-1).to(torch.int32).contiguous()
    return packed_sign, packed_valid


def _masked_sign_table_gpu(n: int, metric: tuple, prod: str, device: str) -> torch.Tensor:
    """sigma_op = sigma * predicate(prod), the masked GP sign for a product, on GPU.
    The predicate is a bitwise condition on (i, j) that factorizes into high x low
    (verified), so the SAME tiled kernel computes the product from the masked
    sub-tables. Supported: geom (no mask), wedge (i&j=0), lc (i subset j),
    rc (j subset i), diag (i==j). inner/regressive are built by composition."""
    s = _sign_value_table_gpu(n, metric, device)
    if prod == "geom":
        return s
    d = 1 << n
    idx = torch.arange(d, device=device, dtype=torch.int64)
    I = idx.view(d, 1)
    J = idx.view(1, d)
    if prod == "wedge":
        mk = (I & J) == 0
    elif prod == "lc":
        mk = (I & J) == I            # i subset j
    elif prod == "rc":
        mk = (I & J) == J            # j subset i
    elif prod == "diag":
        mk = I == J
    else:
        raise ValueError(f"unknown product mask {prod!r}")
    return s * mk.to(torch.int8)


@functools.lru_cache(maxsize=None)
def _fused_sign_data(L: int, H: int, metric: tuple, device: str, prod: str = "geom"):
    """Packed sigma_low LUT (+ validity) for the dim-L sub-product, and the
    (X, X) int8 sigma_high table over the H high generators -- masked for `prod`.
    Built entirely on the GPU."""
    sigma_low = _masked_sign_table_gpu(L, metric[:L], prod, device)
    ps, pv = _pack_fwd_gpu(sigma_low, device)
    sigma_high = _masked_sign_table_gpu(H, metric[L:], prod, device).contiguous()
    return ps, pv, sigma_high


def _pack_direct_gpu(sigma: torch.Tensor, device: str):
    """Pack a (dim, dim) sign table by ROW directly (not the forward XOR pattern):
    bit t of word [k, c] = 1 iff sigma[k, c*32+t] == -1; validity for != 0. Used
    for the backward LUTs, whose kernel reads sign_low[k][i] = sigma_L(k, i)
    (mode 1) or sigma_L(i, k) (mode 2)."""
    dim = sigma.shape[0]
    chunks = dim // 32
    powers = (1 << torch.arange(32, device=device, dtype=torch.int32))
    neg = (sigma == -1).to(torch.int32)
    packed_sign = (neg.view(dim, chunks, 32) * powers).sum(-1).to(torch.int32).contiguous()
    packed_valid = None
    if bool((sigma == 0).any()):
        nz = (sigma != 0).to(torch.int32)
        packed_valid = (nz.view(dim, chunks, 32) * powers).sum(-1).to(torch.int32).contiguous()
    return packed_sign, packed_valid


@functools.lru_cache(maxsize=None)
def _fused_sign_data_bwd(L: int, H: int, metric: tuple, device: str, prod: str = "geom"):
    """Tiled sign data for the two backward products (math verified against the
    full sign table). grad_a (cross_mode=1) needs sigma_L(k,i) and sigma_H[p^q][p];
    grad_b (cross_mode=2) needs sigma_L(i,k) and sigma_H[p][p^q] -- masked for
    `prod` (the mask factorizes, so the backward signs do too). Returns
    ((ps_a, pv_a, sh_a), (ps_b, pv_b, sh_b))."""
    sigma_low = _masked_sign_table_gpu(L, metric[:L], prod, device)   # (M, M) sigma_L_op
    sigma_high = _masked_sign_table_gpu(H, metric[L:], prod, device)  # (X, X) sigma_H_op
    X = 1 << H
    pp = torch.arange(X, device=device).view(X, 1)
    qq = torch.arange(X, device=device).view(1, X)
    rr = pp ^ qq                                                      # r = p ^ q
    # grad_a: low = sigma_L[k][i] (direct), high[p][q] = sigma_H[p^q][p]
    ps_a, pv_a = _pack_direct_gpu(sigma_low, device)
    sh_a = sigma_high[rr, pp].contiguous()
    # grad_b: low = sigma_L[i][k] = sigma_L^T[k][i] (direct), high[p][q] = sigma_H[p][p^q]
    ps_b, pv_b = _pack_direct_gpu(sigma_low.t().contiguous(), device)
    sh_b = sigma_high[pp, rr].contiguous()
    return (ps_a, pv_a, sh_a), (ps_b, pv_b, sh_b)


class _TiledFusedProdFunc(torch.autograd.Function):
    """Differentiable tiled fused product for any directly-masked product `prod` in
    {geom, wedge, lc, rc}. Forward and BOTH backward sums run on the same tiled
    kernel (cross_mode 0/1/2) with the product's masked sign data:

        c[k]       = sum_i sigma_op(i, i^k) a[i] b[i^k]
        dL/da[k]   = sum_i sigma_op(k, i)   b[i] grad_c[k^i]   (cross_mode 1)
        dL/db[k]   = sum_i sigma_op(i, k)   a[i] grad_c[k^i]   (cross_mode 2)
    """
    @staticmethod
    def forward(ctx, a, b, prod, L, H, metric, w):
        ext = load_geom_prod_tiled_cuda()
        ps, pv, sh = _fused_sign_data(L, H, metric, str(a.device), prod)
        c = ext.geom_prod_tiled_fwd(a, b, ps, pv, sh, w, 0)         # forward (cross_mode 0)
        ctx.save_for_backward(a, b)
        ctx.prod, ctx.L, ctx.H, ctx.metric, ctx.w = prod, L, H, metric, w
        return c

    @staticmethod
    def backward(ctx, grad_c):
        a, b = ctx.saved_tensors
        ext = load_geom_prod_tiled_cuda()
        (psa, pva, sha), (psb, pvb, shb) = _fused_sign_data_bwd(
            ctx.L, ctx.H, ctx.metric, str(a.device), ctx.prod)
        grad_c = grad_c.contiguous()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = ext.geom_prod_tiled_fwd(b, grad_c, psa, pva, sha, ctx.w, 1)
        if ctx.needs_input_grad[1]:
            grad_b = ext.geom_prod_tiled_fwd(a, grad_c, psb, pvb, shb, ctx.w, 2)
        return grad_a, grad_b, None, None, None, None, None


def geom_prod_tiled_fused(a, b, metric=None, max_fit: int = None, w: int = 0):
    """Single-kernel tiled geometric product, differentiable (forward and both
    backward sums run on the same tiled kernel via cross_mode 0/1/2). Where the
    standard kernel both exists and fits this GPU (n <= min device cutoff) it is
    used; otherwise the runtime-(L, H) tiled kernel runs with L = the occupancy-
    optimal tile size for the device (see _device_fused_L). Pass max_fit to force a
    tile size (testing); w overrides warps/block (0 = auto, see _device_fused_W)."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric = _normalize_metric(n, metric)
    if max_fit is None:
        # Use the faster standard (non-tiled) kernel only where it BOTH exists
        # (n <= its compiled instantiation ceiling) AND fits this GPU's shared
        # memory (_device_max_fit). On a smaller GPU that fit limit is < 13, so
        # n=12/13 must tile -- the cutoff is not a fixed 13.
        std_cutoff = min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), 0 in metric))
        if n <= std_cutoff:
            return geom_prod(a, b, metric=metric)
        # Perf-optimal L (>=2 blocks/SM), not the max-fit L -- see _device_fused_L.
        max_fit = _device_fused_L(str(a.device), 0 in metric)
    L = min(max_fit, n)                               # H=0 (single tile) if it all fits
    if L < 5:
        raise ValueError(f"max_fit must be >= 5 (kernel lower bound); got {L}")
    if w == 0:                                         # auto: largest W that fits
        w = _device_fused_W(L, 0 in metric, str(a.device))
    return _TiledFusedProdFunc.apply(a, b, "geom", L, n - L, metric, w)


# ── Other products: tiled fused, differentiable (same kernel, masked signs) ───
#
# wedge / left-contraction / right-contraction are direct masked-sigma tilings.
# inner = lc + rc - diag and regressive = dual(wedge(dual a, dual b)) are
# compositions of those plus cheap elementwise ops (all differentiable).

def _tiled_fused_masked(prod, a, b, metric, max_fit, w, std_fn):
    """Generic differentiable tiled fused product for a directly-masked `prod`.
    Uses the standard (non-tiled) product `std_fn` while n fits this GPU; tiles
    beyond. Pass max_fit to force the tiling path (testing)."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_t = _normalize_metric(n, metric)
    if max_fit is None:
        if n <= min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), 0 in metric_t)):
            return std_fn(a, b, metric=metric)
        max_fit = _device_fused_L(str(a.device), 0 in metric_t)
    L = min(max_fit, n)
    if L < 5:
        raise ValueError(f"max_fit must be >= 5 (kernel lower bound); got {L}")
    if w == 0:
        w = _device_fused_W(L, 0 in metric_t, str(a.device))
    return _TiledFusedProdFunc.apply(a, b, prod, L, n - L, metric_t, w)


def wedge_prod_tiled_fused(a, b, metric=None, max_fit=None, w=0):
    """Outer (wedge) product, tiled + fused, differentiable. Predicate i&j=0."""
    from .wedge_prod import wedge_prod
    return _tiled_fused_masked("wedge", a, b, metric, max_fit, w, wedge_prod)


def left_contract_tiled_fused(a, b, metric=None, max_fit=None, w=0):
    """Left contraction, tiled + fused, differentiable. Predicate i subset j."""
    from .left_contract import left_contract
    return _tiled_fused_masked("lc", a, b, metric, max_fit, w, left_contract)


def right_contract_tiled_fused(a, b, metric=None, max_fit=None, w=0):
    """Right contraction, tiled + fused, differentiable. Predicate j subset i."""
    from .right_contract import right_contract
    return _tiled_fused_masked("rc", a, b, metric, max_fit, w, right_contract)


@functools.lru_cache(maxsize=None)
def _norm_sign(n: int, metric: tuple, device: str) -> torch.Tensor:
    """sigma(i, i) for each blade i, float32 (dim,). Closed form (no (2^n)^2
    table): e_i e_i = (-1)^(g(g-1)/2) * prod_{b in i} metric[b], g = popcount(i)."""
    idx = torch.arange(1 << n, device=device, dtype=torch.int64)
    g = torch.zeros_like(idx)
    for b in range(n):
        g += (idx >> b) & 1
    sign = (1 - 2 * (((g * (g - 1) // 2) & 1).to(torch.float32)))   # reorder parity
    for b in range(n):
        mb = int(metric[b])
        if mb == 1:
            continue
        bit = ((idx >> b) & 1).to(torch.float32)
        sign = sign * (1 - 2 * bit) if mb == -1 else sign * (1 - bit)   # -1 / 0 where bit set
    return sign.contiguous()


def _diag_scalar(a, b, metric_t):
    """diag(a, b)[k] = (k==0) ? sum_i sigma(i,i) a[i] b[i] : 0 -- the i==j terms of
    the inner product (output is the scalar blade only). Differentiable."""
    n = a.size(-1).bit_length() - 1
    ns = _norm_sign(n, metric_t, str(a.device))
    s = (a * b * ns).sum(-1, keepdim=True)                       # (..., 1)
    zeros = a.new_zeros(*a.shape[:-1], a.size(-1) - 1)
    return torch.cat([s, zeros], dim=-1)


def inner_prod_tiled_fused(a, b, metric=None, max_fit=None, w=0):
    """Inner (Hestenes fat-dot) product, tiled + fused, differentiable. The fat-dot
    predicate (i subset j) OR (j subset i) does not factorize, so it is computed as
    inner = left_contraction + right_contraction - diag (each tileable)."""
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_t = _normalize_metric(n, metric)
    if max_fit is None and n <= min(_STD_KERNEL_MAX_N,
                                    _device_max_fit(str(a.device), 0 in metric_t)):
        from .inner_prod import inner_prod
        return inner_prod(a, b, metric=metric)
    lc = left_contract_tiled_fused(a, b, metric=metric, max_fit=max_fit, w=w)
    rc = right_contract_tiled_fused(a, b, metric=metric, max_fit=max_fit, w=w)
    return lc + rc - _diag_scalar(a, b, metric_t)


@functools.lru_cache(maxsize=None)
def _build_dual_fast(n: int, metric: tuple, device: str):
    """(dual_signs, dual_perm) for A* = A I^{-1}, matching regressive_prod._build_dual
    but without the (2^n)^2 table (which OOMs at large n). Uses the closed form
    sigma(m, full) = (-1)^(sum of set-bit positions of m) * prod_{b in m} metric[b]."""
    if 0 in metric:
        raise ValueError("dual requires a non-degenerate metric")
    dim = 1 << n
    full = dim - 1
    idx = torch.arange(dim, device=device, dtype=torch.int64)
    pos = torch.zeros_like(idx)
    for b in range(n):
        pos += b * ((idx >> b) & 1)
    col = 1 - 2 * ((pos & 1).to(torch.float32))          # reorder sign of sigma(m, full)
    for b in range(n):
        if int(metric[b]) == -1:
            col = col * (1 - 2 * ((idx >> b) & 1).to(torch.float32))
    neg_idx = (~idx) & full
    dual_signs = (col[full] * col[neg_idx]).contiguous()  # I_sq * sigma(~i, full)
    return dual_signs, neg_idx.contiguous()


def _apply_dual(x, dual_signs, dual_perm):
    return dual_signs * x.index_select(-1, dual_perm)


def regressive_prod_tiled_fused(a, b, metric=None, max_fit=None, w=0):
    """Regressive (meet) product, tiled + fused, differentiable. a v b =
    dual(dual(a) wedge dual(b)); reuses the tiled wedge between two cheap duals.
    Non-degenerate metrics only (the pseudoscalar must be invertible)."""
    from .regressive_prod import regressive_prod
    a = a.contiguous(); b = b.contiguous()
    dim = a.size(-1)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"dim must be a power of two, got {dim}")
    metric_t = _normalize_metric(n, metric)
    if 0 in metric_t:
        raise ValueError("regressive_prod requires a non-degenerate metric")
    if max_fit is None and n <= min(_STD_KERNEL_MAX_N, _device_max_fit(str(a.device), False)):
        return regressive_prod(a, b, metric=metric)
    ds, dp = _build_dual_fast(n, metric_t, str(a.device))
    wab = wedge_prod_tiled_fused(_apply_dual(a, ds, dp), _apply_dual(b, ds, dp),
                                 metric=metric, max_fit=max_fit, w=w)
    return _apply_dual(wab, ds, dp)
