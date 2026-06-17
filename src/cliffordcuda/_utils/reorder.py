import os
import torch
from collections import defaultdict
from .core import build_fused_rotation_indices_bp
from .._config import _USER_CACHE_DIR


_CACHE_DIR = str(_USER_CACHE_DIR / "reorder")


def _hopcroft_karp_match(adj, n_left, n_right):
    INF = float('inf')
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    pair_dist = [0] * n_left

    def bfs():
        from collections import deque
        queue = deque()
        for u in range(n_left):
            if match_l[u] == -1:
                pair_dist[u] = 0
                queue.append(u)
            else:
                pair_dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj.get(u, ()):
                pair_u = match_r[v]
                if pair_u == -1:
                    found = True
                elif pair_dist[pair_u] == INF:
                    pair_dist[pair_u] = pair_dist[u] + 1
                    queue.append(pair_u)
        return found

    def dfs(u):
        for v in adj.get(u, ()):
            pair_u = match_r[v]
            if pair_u == -1 or (
                pair_dist[pair_u] == pair_dist[u] + 1 and dfs(pair_u)
            ):
                match_l[u] = v
                match_r[v] = u
                return True
        pair_dist[u] = INF
        return False

    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                dfs(u)
    return match_r


def _bipartite_chunk_assignment(ci_banks, cj_banks, num_banks=32):
    n_pairs = len(ci_banks)
    n_chunks = n_pairs // num_banks

    edge_buckets = defaultdict(list)
    for i in range(n_pairs):
        edge_buckets[(ci_banks[i], cj_banks[i])].append(i)

    chunks = []
    for _ in range(n_chunks):
        adj = defaultdict(list)
        for (ci_b, cj_b), bucket in edge_buckets.items():
            if bucket:
                adj[ci_b].append(cj_b)

        match_r = _hopcroft_karp_match(adj, num_banks, num_banks)

        chunk = []
        for cj_b in range(num_banks):
            ci_b = match_r[cj_b]
            if ci_b == -1:
                continue
            chunk.append(edge_buckets[(ci_b, cj_b)].pop())
        chunks.append(chunk)

    leftover = []
    for bucket in edge_buckets.values():
        leftover.extend(bucket)
    if leftover:
        chunks.append(leftover)

    return chunks


def build_fused_rotation_indices_bank_optimized(n, dtype, device, warp_size=32, use_cache=True):
    cache_path = None
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, f'n{n}_w{warp_size}_bp.pt')
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location=device, weights_only=True)
            return (cached['ci'].to(device).contiguous(), cached['cj'].to(device).contiguous(), cached['csig'].to(device=device, dtype=dtype).contiguous())

    ci, cj, csig = build_fused_rotation_indices_bp(n, dtype, device)
    ci_cpu = ci.cpu().tolist()
    cj_cpu = cj.cpu().tolist()
    csig_cpu = csig.cpu().tolist()
    num_rot = ci.shape[0]
    ppr = ci.shape[1]

    new_ci = [[0] * ppr for _ in range(num_rot)]
    new_cj = [[0] * ppr for _ in range(num_rot)]
    new_csig = [[0.0] * ppr for _ in range(num_rot)]

    for r in range(num_rot):
        ci_banks = [v % warp_size for v in ci_cpu[r]]
        cj_banks = [v % warp_size for v in cj_cpu[r]]

        chunks = _bipartite_chunk_assignment(ci_banks, cj_banks, warp_size)
        order = []
        for chunk in chunks:
            order.extend(chunk)

        for new_p, old_p in enumerate(order):
            new_ci[r][new_p] = ci_cpu[r][old_p]
            new_cj[r][new_p] = cj_cpu[r][old_p]
            new_csig[r][new_p] = csig_cpu[r][old_p]

    ci_t = torch.tensor(new_ci, dtype=torch.int32, device=device).contiguous()
    cj_t = torch.tensor(new_cj, dtype=torch.int32, device=device).contiguous()
    csig_t = torch.tensor(new_csig, dtype=dtype, device=device).contiguous()

    if cache_path is not None:
        torch.save({'ci': ci_t.cpu(), 'cj': cj_t.cpu(), 'csig': csig_t.cpu()}, cache_path)

    return ci_t, cj_t, csig_t
