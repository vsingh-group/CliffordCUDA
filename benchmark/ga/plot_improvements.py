"""Plot improvement ratios (ref / best-of-ours) versus n.

"Ours" per row is the *better* of our two variants: chunk vs multik for
geom_prod, chunk vs subset for everything else. Where only one of ours is
present (e.g. backward CSVs for non-GP ops where subset_grade has no
backward), best == chunk.

Reads CSVs from ../results/ga/speed/ and ../results/ga/memory/, one pair
(fwd, bwd) per product. Produces figures saved to the repo-level figures/:
    speed_improvements_fwd.png    speed_improvements_bwd.png
    memory_improvements_fwd.png   memory_improvements_bwd.png

Each figure is (n products) × (3 batches). Each subplot is a line graph:
improvement (×) vs n, one solid line per cross-implementation reference.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def _snap_ylim_up(ax):
    """If the data's high water is past the halfway mark of a decade
    (>= 5×10^k for some k), snap the top of the y-axis to 10^(k+1) so
    the plot reaches the next round power of 10. No-op below the
    halfway mark. Lower bound left alone."""
    y_lo, y_hi = ax.get_ylim()
    if y_hi <= 0:
        return
    k = math.floor(math.log10(y_hi))
    if y_hi >= 5 * (10 ** k):
        ax.set_ylim(top=10 ** (k + 1))


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "results", "ga"))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

# (op_label, fwd_csv_stem, bwd_csv_stem, refs-in-this-product)
# left_contract / right_contract / regressive_prod are intentionally left out
# of the plot — those benches have no external witness (or only one, which
# itself OOMs early), so their rows would be near-empty. They still run as
# benches and write CSVs; we just don't include them in the figure.
PRODUCTS = [
    ("Geometric product",  "bench_geom_prod",   "bench_geom_prod_bwd",   ["einsum", "ga", "versor", "versorai"]),
    ("Wedge product",      "bench_wedge_prod",  "bench_wedge_prod_bwd",  ["einsum", "ga", "versor", "versorai"]),
    ("Inner product",      "bench_inner_prod",  "bench_inner_prod_bwd",  ["einsum", "ga", "versor", "versorai"]),
    # Rotor backward intentionally omitted from the bwd plots: the kernel's
    # apply backward is cheap but the bench column also includes (or, with
    # the fix, deliberately excludes) cuSOLVER-noisy eigh in the rotor
    # construction. Whichever way the bwd column is set up, the apples-to-
    # apples comparison story is muddier than for the products; we keep
    # rotor in the fwd plots and drop it from the bwd grid.
    ("Rotor application",  "bench_rotor_apply", None,                    ["einsum", "ga", "versor", "versorai"]),
]


REF_LABELS = {"ga": "torch_ga", "versor": "Versor", "einsum": "einsum",
              "versorai": "VersorAI"}
REF_COLORS = {
    "ga":       "#d62728",   # red
    "versor":   "#ff7f0e",   # orange
    "einsum":   "#2ca02c",   # green
    "versorai": "#9467bd",   # purple
}
# Distinct markers so coincident lines stay distinguishable (torch_ga and
# Versor produce identical-shape autograd intermediates in the backward
# memory benches and overlap exactly).
REF_MARKERS = {"ga": "^", "versor": "s", "einsum": "D", "versorai": "v"}

# Draw order: the ref with the shortest valid n-range (ga, OOMs at n>=11) is
# plotted last so its markers sit on top of the longer-range ones.
DRAW_ORDER = ["einsum", "versorai", "versor", "ga"]

BATCHES = [1, 16, 64]


def load(track, stem):
    p = os.path.join(ROOT, track, f"{stem}.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def num(v):
    """Return float v if positive and finite, else None."""
    try:
        x = float(v)
    except (ValueError, TypeError):
        return None
    if x != x or x <= 0:
        return None
    return x


_OURS_COLS = ("chunk", "chunk_skip", "multik", "multik_skip", "subset")


def best_ours(row, suffix):
    """min over every of-ours variant column present in this row. Variants
    we know about: chunk, chunk_skip, multik, multik_skip, subset. Any that
    aren't in the row are silently skipped. Returns None if none are present."""
    vals = []
    for col in _OURS_COLS:
        v = num(row.get(f"{col}{suffix}", ""))
        if v is not None:
            vals.append(v)
    return min(vals) if vals else None


def series(rows, ref, batch, suffix):
    """Return sorted (n, ratio = ref / best_ours) at this batch."""
    out = []
    for r in rows:
        if int(r["batch"]) != batch:
            continue
        ref_v = num(r.get(f"{ref}{suffix}", ""))
        if ref_v is None:
            continue
        best = best_ours(r, suffix)
        if best is None:
            continue
        out.append((int(r["n"]), ref_v / best))
    return sorted(out)


def plot_grid(track, suffix, title, outfile, direction):
    """Render one (track, direction) grid: (products × batches) of
    (ref/best-ours) ratios. `direction` is "forward" or "backward" and picks
    which CSV stem to read per product. All lines are solid; the legend is
    just the ref names."""
    assert direction in ("forward", "backward")
    # A product entry with a None stem in the requested direction is skipped
    # (e.g. rotor_apply has no bwd plot — see comment on the PRODUCTS list).
    products = [(op, fwd, bwd, refs) for op, fwd, bwd, refs in PRODUCTS
                if (fwd if direction == "forward" else bwd) is not None]
    nrows, ncols = len(products), len(BATCHES)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4 * ncols + 0.5, 2.4 * nrows + 1.0),
        sharex=True,
    )

    for ri, (op, fwd_stem, bwd_stem, refs) in enumerate(products):
        stem = fwd_stem if direction == "forward" else bwd_stem
        rows = load(track, stem)
        plot_refs = [r for r in DRAW_ORDER if r in refs]
        for ci, batch in enumerate(BATCHES):
            ax = axes[ri, ci]
            ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, zorder=0)
            for zi, ref in enumerate(plot_refs):
                pts = series(rows, ref, batch, suffix)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=REF_COLORS[ref], linestyle="-",
                        marker=REF_MARKERS[ref], markersize=5, linewidth=1.4,
                        zorder=10 + zi)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            _snap_ylim_up(ax)
            if ri == 0:
                ax.set_title(f"batch = {batch}")
            if ci == 0:
                ax.set_ylabel(op)
            if ri == nrows - 1:
                ax.set_xlabel("n")

    handles = [
        mlines.Line2D([], [], color=REF_COLORS[ref], linestyle="-",
                      marker=REF_MARKERS[ref], markersize=6,
                      label=REF_LABELS[ref])
        for ref in ["einsum", "versorai", "ga", "versor"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.975])
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def plot_rotor_parametrized(outfile):
    """2x2 grid: rows = speed/memory, cols = inference/training. Same
    convention as the other plots — x = n, y = ratio (Compose/chunk),
    one line per batch. Reads the four
    bench_rotor_apply_compose_{inference,training}.csv files."""
    panels = [
        ("speed",  "us",  "inference", "Inference — speed (ratio)"),
        ("speed",  "us",  "training",  "Training — speed (ratio)"),
        ("memory", "mib", "inference", "Inference — peak memory (ratio)"),
        ("memory", "mib", "training",  "Training — peak memory (ratio)"),
    ]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    handles_by_batch = {}
    for ax, (track, unit, mode, title) in zip(axes.flat, panels):
        rows = load(track, f"bench_rotor_apply_compose_{mode}")
        batches = sorted({int(r["batch"]) for r in rows})
        for bi, batch in enumerate(batches):
            xs, ys = [], []
            for r in sorted(rows, key=lambda r: int(r["n"])):
                if int(r["batch"]) != batch:
                    continue
                chunk = num(r[f"chunk_{unit}"])
                compose = num(r[f"compose_{unit}"])
                if chunk is None or compose is None:
                    continue
                xs.append(int(r["n"])); ys.append(compose / chunk)
            color = cmap(bi / max(1, len(batches) - 1))
            line, = ax.plot(xs, ys, "o-", color=color, linewidth=1.4,
                            markersize=4, label=f"batch={batch}")
            handles_by_batch[batch] = line
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("n")
        _snap_ylim_up(ax)
    handles = [handles_by_batch[b] for b in sorted(handles_by_batch)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Rotor application: ComposingLinearLayers / CliffordCUDA",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    plot_grid(
        "speed", "_us",
        "Forward speed (ratio)",
        os.path.join(OUT_DIR, "speed_improvements_fwd.png"),
        direction="forward",
    )
    plot_grid(
        "speed", "_us",
        "Backward speed (ratio)",
        os.path.join(OUT_DIR, "speed_improvements_bwd.png"),
        direction="backward",
    )
    plot_grid(
        "memory", "_mib",
        "Forward peak memory (ratio)",
        os.path.join(OUT_DIR, "memory_improvements_fwd.png"),
        direction="forward",
    )
    plot_grid(
        "memory", "_mib",
        "Backward peak memory (ratio)",
        os.path.join(OUT_DIR, "memory_improvements_bwd.png"),
        direction="backward",
    )
    plot_rotor_parametrized(
        os.path.join(OUT_DIR, "speed_memory_rotor_parametrized.png"),
    )


if __name__ == "__main__":
    main()
