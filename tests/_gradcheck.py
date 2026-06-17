"""Shared gradcheck helpers: signature spread + a thin gradcheck wrapper.

Every gradcheck file uses the same six-signature spread per n and the same
fp64-in/fp32-call/fp64-out wrapper, so they live here once.
"""
import torch


def signatures_for(n: int):
    """Six Cl(p, q, r) shapes per n: pure +, pure -, Lorentzian, balanced,
    single-degenerate, mixed + degenerate."""
    return [
        (f"Cl({n}, 0, 0)",              tuple([1] * n)),
        (f"Cl(0, {n}, 0)",              tuple([-1] * n)),
        (f"Cl({n-1}, 1, 0)",            tuple([1] * (n - 1) + [-1])),
        (f"Cl({n//2}, {n - n//2}, 0)",  tuple([1] * (n // 2) + [-1] * (n - n // 2))),
        (f"Cl({n-1}, 0, 1)",            tuple([1] * (n - 1) + [0])),
        (f"Cl({n-2}, 1, 1)",            tuple([1] * (n - 2) + [-1, 0])),
    ]


def all_cases(n_values):
    """Flatten n_values x signatures_for(n) into a flat list of (n, metric)
    pairs suitable for pytest.mark.parametrize."""
    out = []
    for n in n_values:
        for _label, metric in signatures_for(n):
            out.append((n, metric))
    return out


def run_gradcheck(op, n: int, metric, B: int = 2, seed: int = 0):
    """Run `torch.autograd.gradcheck` on the op. Inputs fp64; the call is
    wrapped to cast to float32 (the kernels are fp32-only) and the result
    back to float64. Tolerance scales with dim to absorb the noise."""
    torch.manual_seed(seed)
    dim = 1 << n
    tol = max(1e-3, dim * 3e-5)
    a = torch.randn(B, dim, dtype=torch.float64, device='cuda', requires_grad=True)
    b = torch.randn(B, dim, dtype=torch.float64, device='cuda', requires_grad=True)
    fn = lambda a, b: op(a.float(), b.float(), metric=metric).double()
    return torch.autograd.gradcheck(fn, (a, b), eps=1e-3,
                                    atol=tol, rtol=tol, nondet_tol=tol)


N_VALUES_FAST = [5, 7, 9]
