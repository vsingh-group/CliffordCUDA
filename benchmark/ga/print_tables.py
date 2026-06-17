"""Print four summary tables — speed-fwd, speed-bwd, memory-fwd, memory-bwd —
each covering the core-product CSVs (PRODUCT_ORDER) under
benchmark/results/ga/{speed,memory}/. Non-core benches (e.g. the
rotor-vs-Compose parametrized comparison) are skipped; they have their own plot.

Each table:
  - Batches are meta rows (`**batch=B**` heading row).
  - Products are sub-rows under each batch (geom_prod, wedge_prod, ...).
  - Columns are `n` values, taken from the CSVs (not hardcoded).
  - The CliffordCUDA value is COUPLED: at each (product, batch, n) we pick
    the variant with the smallest speed at that cell, and the memory cell
    reports that SAME variant's memory. Speed and memory tables therefore
    refer to the same underlying code path.

Run:
    python benchmark/ga/print_tables.py
"""
import csv
import glob
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "ga"))

# Per-row "ours" variants whose timings get min'd into a single CliffordCUDA
# row — matches the plot_improvements.py best_ours convention.
OURS_KEYS = ("chunk", "chunk_skip", "multik", "multik_skip", "subset")
PRODUCT_ORDER = ["geom_prod", "wedge_prod", "inner_prod", "left_contract",
                 "right_contract", "regressive_prod", "rotor_apply"]
# Display labels used in tables (README-friendly). Keys are the CSV stem
# identifiers; values are what the table shows.
PRODUCT_LABEL = {
    "geom_prod":       "Geometric product",
    "wedge_prod":      "Wedge product",
    "inner_prod":      "Inner product",
    "left_contract":   "Left contraction",
    "right_contract":  "Right contraction",
    "regressive_prod": "Regressive product",
    "rotor_apply":     "Rotor application",
}
# Products to omit from backward tables — matches the plot's bwd-skip set.
SKIP_IN_BWD = {"rotor_apply"}


def product_of(stem: str) -> str:
    m = re.match(r"bench_(.+?)(?:_bwd)?$", stem)
    return m.group(1) if m else stem


def is_bwd(stem: str) -> bool:
    return stem.endswith("_bwd")


def fmt(v, unit: str) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(x):
        return "—"
    if unit == "us":
        return f"{x:,.1f}" if x < 10_000 else f"{x:,.0f}"
    return f"{x:,.2f}" if x < 100 else f"{x:,.0f}"


def _safe_float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _impls_in(rows: list[dict], suffix: str) -> list[str]:
    if not rows:
        return []
    return [fn[:-len(suffix)] for fn in rows[0].keys()
            if fn.endswith(suffix)
            and not fn.endswith("_std" + suffix)
            and not fn.endswith("_over_chunk")]


def _pick_winner(rows: list[dict], ours: list[str], suffix: str):
    """For each (batch, n) in `rows`, pick the OURS variant with the smallest
    value and return {(batch, n): (variant_key, value_str, std_str)}.
    std_str is None when the CSV has no std column for that suffix."""
    std_suffix = "_std" + suffix
    out = {}
    for r in rows:
        b, n = int(r["batch"]), int(r["n"])
        cands = []
        for k in ours:
            v = _safe_float(r.get(k + suffix))
            if v is None:
                continue
            cands.append((v, k, r.get(k + suffix), r.get(k + std_suffix)))
        if cands:
            cands.sort(key=lambda t: t[0])
            _, k, v, s = cands[0]
            out[(b, n)] = (k, v, s)
    return out


def _lookup(rows: list[dict], suffix: str, winners: dict):
    """For each (batch, n) winner, look up the same variant's value in `rows`
    (the matching memory CSV). Returns {(batch, n): (value_str, None)}."""
    by_bn = {(int(r["batch"]), int(r["n"])): r for r in rows}
    out = {}
    for (b, n), (variant, _, _) in winners.items():
        r = by_bn.get((b, n))
        if r is None:
            continue
        v = r.get(variant + suffix)
        if _safe_float(v) is None:
            continue
        out[(b, n)] = (v, None)        # memory CSVs carry no std column
    return out


def collect(want_bwd: bool) -> tuple[dict, dict]:
    """Returns (speed_tables, memory_tables) where each tables dict is
        {product: {"ns": [int], "batches": [int],
                   "cliffordcuda": {(batch, n): (value_str, std_str)}}}.

    The coupling guarantee: the (batch, n) cells in `memory_tables` use the
    same variant whose value appears in `speed_tables`.
    """
    speed_suffix, mem_suffix = "_us", "_mib"
    speed_csvs = {os.path.basename(p)[:-4]: p
                  for p in glob.glob(os.path.join(RESULTS, "speed", "*.csv"))}
    mem_csvs = {os.path.basename(p)[:-4]: p
                for p in glob.glob(os.path.join(RESULTS, "memory", "*.csv"))}

    speed_out, mem_out = {}, {}
    for stem, speed_path in sorted(speed_csvs.items()):
        if is_bwd(stem) != want_bwd:
            continue
        if product_of(stem) not in PRODUCT_ORDER:
            continue   # non-core bench (e.g. the rotor-vs-Compose parametrized
                       # comparison) — it has its own plot, not these tables
        if want_bwd and product_of(stem) in SKIP_IN_BWD:
            continue
        speed_rows = list(csv.DictReader(open(speed_path)))
        if not speed_rows:
            continue
        ours_speed = [k for k in OURS_KEYS if k in _impls_in(speed_rows, speed_suffix)]
        if not ours_speed:
            continue
        winners = _pick_winner(speed_rows, ours_speed, speed_suffix)
        if not winners:
            continue

        product = product_of(stem)
        ns = sorted({n for (_, n) in winners})
        batches = sorted({b for (b, _) in winners})

        speed_cells = {bn: (v, s) for bn, (_, v, s) in winners.items()}
        speed_out[product] = {"ns": ns, "batches": batches, "cliffordcuda": speed_cells}

        # Memory: use the speed winner's variant. Look it up in the matching
        # (same stem) memory CSV.
        mem_path = mem_csvs.get(stem)
        if mem_path is None:
            mem_out[product] = {"ns": ns, "batches": batches, "cliffordcuda": {}}
            continue
        mem_rows = list(csv.DictReader(open(mem_path)))
        mem_cells = _lookup(mem_rows, mem_suffix, winners)
        mem_out[product] = {"ns": ns, "batches": batches, "cliffordcuda": mem_cells}

    return speed_out, mem_out


def _fmt_cell(pair, unit: str) -> str:
    """pair is (value_str, std_str) or None."""
    if pair is None:
        return "—"
    v, s = pair
    val = fmt(v, unit)
    if val == "—" or s is None:
        return val
    try:
        std = float(s)
    except (TypeError, ValueError):
        return val
    if math.isnan(std):
        return val
    return f"{val} ± {fmt(s, unit)}"


def render(title: str, tables: dict, unit: str) -> str:
    if not tables:
        return ""
    all_ns = sorted({n for t in tables.values() for n in t["ns"]})
    all_batches = sorted({b for t in tables.values() for b in t["batches"]})

    header = ["Operation"] + [f"n={n}" for n in all_ns]
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    sorted_products = sorted(
        tables.keys(),
        key=lambda p: (PRODUCT_ORDER.index(p) if p in PRODUCT_ORDER
                       else len(PRODUCT_ORDER), p),
    )

    for batch in all_batches:
        lines.append("| " + " | ".join([f"**Batch size = {batch}**"] + [""] * len(all_ns)) + " |")
        for product in sorted_products:
            info = tables[product]
            label = PRODUCT_LABEL.get(product, product)
            cells = [_fmt_cell(info["cliffordcuda"].get((batch, n)), unit) for n in all_ns]
            lines.append("| " + " | ".join([f"&nbsp;&nbsp;{label}"] + cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    speed_fwd, mem_fwd = collect(want_bwd=False)
    speed_bwd, mem_bwd = collect(want_bwd=True)
    body = "\n".join([
        render("Forward speed (µs)", speed_fwd, "us"),
        render("Backward speed (µs)", speed_bwd, "us"),
        render("Forward memory (MiB)", mem_fwd, "mib"),
        render("Backward memory (MiB)", mem_bwd, "mib"),
    ])
    out_dir = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tables.md")
    with open(out_path, "w") as f:
        f.write(body)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
