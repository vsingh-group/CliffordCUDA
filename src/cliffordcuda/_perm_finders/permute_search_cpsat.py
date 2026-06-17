"""CP-SAT permutation search for the bank-conflict-free Givens apply kernel.

Importable as `from cliffordcuda._perm_finders import permute_search_cpsat`;
the function `find(n, warp_size)` returns the permutation as a numpy array.
"""
import argparse
import os
import time

import numpy as np
import torch

from .._utils.core import build_fused_rotation_indices_bp


def find(n: int, warp_size: int = 32, time_limit_sec: int = 600,
         num_workers: int = 16) -> np.ndarray:
    """Return a bank-conflict-friendly permutation of [0, 2^n) for the Givens
    apply kernel. Solves a CP-SAT integer program; can take minutes at high n."""
    WARP = warp_size
    dim = 2 ** n
    ppr = 2 ** (n - 2)
    R = ppr // WARP
    NUM_BANKS = WARP
    PER_BANK = dim // NUM_BANKS

    ci, cj, _ = build_fused_rotation_indices_bp(n, torch.float32, "cuda")
    ci_np = ci.cpu().numpy().astype(np.int64)
    cj_np = cj.cpu().numpy().astype(np.int64)
    num_rot = ci_np.shape[0]

    print(f"  cpsat: n={n}, dim={dim}, ppr={ppr}, num_rot={num_rot}, R={R}", flush=True)

    from ortools.sat.python import cp_model
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{b}") for b in range(NUM_BANKS)] for i in range(dim)]

    for i in range(dim):
        model.Add(sum(x[i]) == 1)
    for b in range(NUM_BANKS):
        model.Add(sum(x[i][b] for i in range(dim)) == PER_BANK)

    excess_vars = []
    for r in range(num_rot):
        for side, blades in [(0, ci_np[r]), (1, cj_np[r])]:
            for b in range(NUM_BANKS):
                e = model.NewIntVar(0, len(blades), f"e_r{r}_s{side}_b{b}")
                excess_vars.append(e)
                model.Add(sum(x[int(i)][b] for i in blades) <= R + e)

    model.Minimize(sum(excess_vars))
    model.Add(x[0][0] == 1)

    bank_of_init = (np.arange(dim) % NUM_BANKS).astype(int)
    for i in range(dim):
        model.AddHint(x[i][bank_of_init[i]], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = False

    class _ProgressCb(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.time()
            self.best = None
        def OnSolutionCallback(self):
            obj = self.ObjectiveValue()
            if self.best is None or obj < self.best:
                self.best = obj
                print(f"    [t={time.time()-self.t0:.0f}s] cost = {int(obj)}", flush=True)

    cb = _ProgressCb()
    t0 = time.time()
    status = solver.Solve(model, cb)
    elapsed = time.time() - t0
    print(f"  cpsat: {solver.StatusName(status)} in {elapsed:.1f}s", flush=True)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"cpsat: no feasible solution for n={n} within {time_limit_sec}s.")

    bank_of = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        for b in range(NUM_BANKS):
            if solver.Value(x[i][b]) == 1:
                bank_of[i] = b
                break

    pi = np.zeros(dim, dtype=np.int64)
    slots = {b: [] for b in range(NUM_BANKS)}
    for i in range(dim):
        slots[bank_of[i]].append(i)
    for b in range(NUM_BANKS):
        for k, blade in enumerate(slots[b]):
            pi[blade] = b + NUM_BANKS * k

    return pi


def _cli():
    """`python -m cliffordcuda._perm_finders.permute_search_cpsat --n N`"""
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--time", type=int, default=600)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=str, default=None,
                   help="destination .pt path (default: print only)")
    args = p.parse_args()
    pi = find(args.n, warp_size=32, time_limit_sec=args.time, num_workers=args.workers)
    if args.out:
        torch.save(torch.from_numpy(pi), args.out)
        print(f"saved {args.out}")
    else:
        print(pi)


if __name__ == "__main__":
    _cli()
