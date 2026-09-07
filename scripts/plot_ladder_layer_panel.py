#!/usr/bin/env python3
"""Bias-axis layer profiles: Qwen scale ladder, and Mistral-7B.

Three metrics are profiled per layer, all averaged over seeds (mean +/- 1 sd):
  * activation differential ||d_l|| / ||h_l||   (per_layer_strength_rel)
  * directional alignment, the mean per-pair cosine
  * sign agreement, the fraction of pairs agreeing in sign
These are in DIFFERENT UNITS (a norm ratio, a cosine on [-1,1], a fraction on
[0,1]), so they never share a y-axis: each gets its own panel. What may share an
axis is one metric across models, which is what the stacked ladder mode does.

Modes:
  --stacked      three stacked panels, one metric each; the four rungs overlaid
                 within each panel. Rung depths differ (24/36/28/48 layers), so x
                 is relative depth l/(L-1). THIS IS THE LADDER FIGURE.
  --mistral      the same three stacked panels for Mistral-7B's 15 real runs,
                 on absolute layer index (one model, so no rescaling needed).
  --all-metrics  4x3 grid, rows=rungs, cols=metrics. Absolute layer index per
                 row, no rescaling; verbose but rescaling-free (appendix).
  (default)      2x2 grid, one cell per rung, differential only.

Seeds are deduped to one run per (size, seed), preferring the latest job id.

    python scripts/plot_ladder_layer_panel.py --stacked --out figures/qwen_layer_stacked.png
    python scripts/plot_ladder_layer_panel.py --mistral --out figures/mistral_layer_stacked.png
"""
import argparse, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["0_5b", "3b", "7b", "14b"]
DISP = {"0_5b": "0.5B", "3b": "3B", "7b": "7B", "14b": "14B"}
COL = "#1f6f8b"

# Rung identity is encoded on TWO channels so the lines stay separable in print and
# for colour-vision deficiency: a shade within the panel's own metric hue (light =
# small, dark = large), and a marker shape. Hue therefore still means METRIC, exactly
# as in the Mistral figure -- the ladder does not introduce a competing colour code.
RUNG_SHADE = {"0_5b": 0.48, "3b": 0.65, "7b": 0.82, "14b": 1.00}
RUNG_MARKER = {"0_5b": "o", "3b": "s", "7b": "^", "14b": "D"}


def shade(base, f):
    """Mix ``base`` toward white; f=1 keeps it, f->0 lightens it."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(base)
    return (r + (1 - r) * (1 - f), g + (1 - g) * (1 - f), b + (1 - b) * (1 - f))

# The 15 real Mistral experiments (matches plot_method_scatter / plot_presence_mistral).
REAL15 = {"3466902", "3466903", "3466904", "3467783", "3467784",
          "3471123", "3471124", "3471125", "3471126", "3471127",
          "3471128", "3471129", "3471130", "3471131", "3471132"}

# (profile key, column title, y-axis label, colour, ylim, reference line)
METRICS = [
    ("per_layer_strength_rel", "Activation Differential",
     r"$\|d_\ell\|/\|h_\ell\|$", "#1f6f8b", None, None),
    ("per_layer_pair_cosine_mean", "Directional Alignment",
     "per-pair cosine", "#2e7d32", (-0.08, 1.17), 0.0),
    ("per_layer_sign_consistency", "Sign Agreement",
     "fraction of pairs", "#7a4fbf", (0.4, 1.15), 0.5),
]


def _size_seed(name):
    seed = (re.search(r"_s(\d+)", name) or [None, "0"])[1] if re.search(r"_s(\d+)", name) else "0"
    if name.startswith("erosion_14bmg_"):
        return "14b", seed
    base = re.sub(r"_\d+$", "", name.replace("erosion_ladder_", ""))
    return re.sub(r"_s\d+$", "", base).replace("qwen", ""), seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/qwen_layer_panel.png")
    ap.add_argument("--all-metrics", action="store_true",
                    help="4x3 grid: rows=rungs, cols=(differential, alignment, sign agreement)")
    ap.add_argument("--stacked", "--overlay", dest="stacked", action="store_true",
                    help="three stacked panels (one per metric); the four rungs are overlaid "
                         "within each panel on RELATIVE depth, so only like units share an axis")
    ap.add_argument("--mistral", action="store_true",
                    help="three stacked panels for Mistral-7B (the 15 real runs) instead of the Qwen ladder")
    args = ap.parse_args(argv)

    def read_group(jsons):
        """Stack the per-layer profiles over seeds. -> (stacks, chosen-layer set, n, n_layers)"""
        stacks = {k: [] for k, *_ in METRICS}; chosen = []; nL = 0
        for j in sorted(jsons):
            c = json.loads(open(j).read())["candidate"]; p = c["layer_variance_profile"]
            for k, *_ in METRICS:
                stacks[k].append(p[k])
            nL = len(p[METRICS[0][0]])
            if c.get("layers"):
                chosen.append(int(c["layers"][0]))
        return ({k: np.array(v) for k, v in stacks.items()},
                sorted(set(chosen)), len(chosen), nL)

    def band(ax, x, A, col, label=None, marker="o"):
        """mean +/- 1 sd over seeds."""
        m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
        ax.plot(x, m, "-", marker=marker, color=col, ms=3, lw=1.6, label=label)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.10, lw=0)

    def finish_metric_axis(ax, ctitle, ylab, ylim, ref):
        if ref is not None:
            ax.axhline(ref, color="0.75", ls=":", lw=1)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(ctitle, fontsize=11.5, fontweight="bold", loc="left")
        ax.set_ylabel(ylab)
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if args.mistral:
        jsons = [j for j in glob.glob("runs/erosion_resume_mistral7b_*/erosion_comparison.json")
                 if any(r in j for r in REAL15)]
        if not jsons:
            raise SystemExit("no REAL15 Mistral erosion_comparison.json found under runs/")
        stacks, cset, n, nL = read_group(jsons)
        fig, axes = plt.subplots(3, 1, figsize=(8, 9.6), sharex=True, squeeze=False)
        fig.suptitle(f"Mistral-7B Layer Profiles ({nL} layers, n={n} seeds)",
                     fontsize=13, fontweight="bold")
        for ax, (key, ctitle, ylab, col, ylim, ref) in zip(axes[:, 0], METRICS):
            A = stacks[key]
            band(ax, np.arange(A.shape[1]), A, col,
                 marker="^" if key == "per_layer_sign_consistency" else "o")
            for cc in cset:
                ax.axvline(cc, color="0.5", ls="--", lw=1)
            finish_metric_axis(ax, ctitle, ylab, ylim, ref)
        axes[-1, 0].set_xlabel("Layer")
        axes[0, 0].text(0.02, 0.96, "dashed: selected layer(s) " + ", ".join(f"L{c}" for c in cset),
                        transform=axes[0, 0].transAxes, fontsize=8, color="0.4",
                        ha="left", va="top")
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ladder-layer-panel] wrote {args.out}  (mistral, 3 stacked panels, n={n})")
        return 0

    # one json per (size, seed): keep the highest job id (latest complete rerun)
    best = {}
    for j in glob.glob("runs/erosion_ladder_*/erosion_comparison.json") + \
             glob.glob("runs/erosion_14bmg_*/erosion_comparison.json"):
        name = j.split("/")[-2]
        if "smoke" in name:
            continue
        size, seed = _size_seed(name)
        if size not in ORDER:
            continue
        job = int((re.search(r"_(\d+)$", name) or [0, 0])[1])
        key = (size, seed)
        if key not in best or job > best[key][0]:
            best[key] = (job, j)

    per_size = defaultdict(list)
    for (size, _), (_, j) in best.items():
        per_size[size].append(j)

    sizes = [s for s in ORDER if s in per_size]

    def load_rung(size):
        """Return (profiles dict of stacked arrays, chosen-layer set, n_seeds)."""
        stacks = {k: [] for k, *_ in METRICS}; chosen = []
        for j in sorted(per_size[size]):
            c = json.loads(open(j).read())["candidate"]; p = c["layer_variance_profile"]
            for k, *_ in METRICS:
                stacks[k].append(p[k])
            if c.get("layers"):
                chosen.append(int(c["layers"][0]))
        return {k: np.array(v) for k, v in stacks.items()}, sorted(set(chosen)), len(chosen)

    if args.stacked:
        # Three stacked panels, one METRIC each: only like units ever share an axis.
        # Within a panel the four rungs are overlaid, which IS comparable -- but their
        # depths differ (24/36/28/48 layers), so x is RELATIVE depth l/(L-1) rather
        # than the absolute layer index. Selected layers appear as coloured dashed
        # rules at their own relative depth, in the top panel only.
        fig, axes = plt.subplots(3, 1, figsize=(11, 13.5), sharex=True, squeeze=False)
        fig.suptitle("Qwen2.5 Bias-Axis Layer Profiles Across Scale",
                     fontsize=14, fontweight="bold")
        rung = {s: load_rung(s) for s in sizes}
        for pi, (ax, (key, ctitle, ylab, col, ylim, ref)) in enumerate(zip(axes[:, 0], METRICS)):
            for size in sizes:
                stacks, cset, nseed = rung[size]
                A = stacks[key]; nL = A.shape[1]
                x = np.arange(nL) / (nL - 1)
                sel = "/".join(f"L{c}" for c in cset)
                # Full provenance in the top panel; just the rung below it, since the
                # shade/marker code repeats and the labels would only add clutter.
                lab = (f"Qwen2.5-{DISP[size]}  ({nL} layers, n={nseed}, sel. {sel})"
                       if pi == 0 else f"Qwen2.5-{DISP[size]}")
                band(ax, x, A, shade(col, RUNG_SHADE[size]), marker=RUNG_MARKER[size],
                     label=lab)
            finish_metric_axis(ax, ctitle, ylab, ylim, ref)
            leg = ax.legend(fontsize=8.5, loc="upper left", frameon=False,
                            ncol=2 if pi == 0 else 4, handlelength=2.4,
                            columnspacing=1.3,
                            title="dashed rules: selected layer" if pi == 0 else None)
            if leg.get_title().get_text():
                leg.get_title().set_fontsize(8); leg.get_title().set_color("0.4")
        # Selected layers as vertical rules at their own relative depth, top panel only
        # (repeating them in all three panels buries the curves).
        top = axes[0, 0]
        base0 = METRICS[0][3]
        for size in sizes:
            stacks, cset, _n = rung[size]
            nL = stacks[METRICS[0][0]].shape[1]
            for cc in cset:
                top.axvline(cc / (nL - 1), color=shade(base0, RUNG_SHADE[size]),
                            ls="--", lw=1, alpha=0.5)
        top.set_ylim(bottom=0, top=top.get_ylim()[1] * 1.22)   # headroom for the legend
        axes[-1, 0].set_xlabel(r"Relative depth  $\ell/(L{-}1)$   (0 = first layer, 1 = last)")
        axes[-1, 0].set_xlim(-0.02, 1.02)
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ladder-layer-panel] wrote {args.out}  (3 stacked panels, rungs overlaid on relative depth)")
        for size in sizes:
            nL = rung[size][0][METRICS[0][0]].shape[1]
            print(f"  {DISP[size]:5} n={len(per_size[size])} seeds, {nL} layers, selected {rung[size][1]}")
        return 0

    if args.all_metrics:
        nr = len(sizes)
        fig, axes = plt.subplots(nr, 3, figsize=(15, 3.4 * nr), squeeze=False)
        fig.suptitle("Qwen2.5 Bias-Axis Layer Profiles Across Scale (all metrics)",
                     fontsize=15, fontweight="bold")
        for ri, size in enumerate(sizes):
            stacks, cset, nseed = load_rung(size)
            for ci, (key, ctitle, ylab, col, ylim, ref) in enumerate(METRICS):
                ax = axes[ri][ci]; A = stacks[key]; L = np.arange(A.shape[1])
                m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
                mk = "-^" if key == "per_layer_sign_consistency" else "-o"
                ax.plot(L, m, mk, color=col, ms=3)
                ax.fill_between(L, m - s, m + s, color=col, alpha=0.16)
                if ref is not None:
                    ax.axhline(ref, color="0.7", ls=":", lw=1)
                if ylim:
                    ax.set_ylim(*ylim)
                for cc in cset:
                    ax.axvline(cc, color="0.5", ls="--", lw=1)
                ax.grid(True, alpha=0.2)
                if ri == 0:
                    ax.set_title(ctitle, fontsize=12, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(f"Qwen2.5-{DISP[size]}\n(n={nseed})", fontsize=11, fontweight="bold")
                    ax.text(0.03, 0.92, ylab, transform=ax.transAxes, fontsize=8, color="0.35", va="top")
                else:
                    ax.text(0.03, 0.92, ylab, transform=ax.transAxes, fontsize=8, color="0.35", va="top")
                if ri == nr - 1:
                    ax.set_xlabel("Layer")
            axes[ri][2].text(0.97, 0.06, "selected: " + ", ".join(f"L{c}" for c in cset),
                             transform=axes[ri][2].transAxes, fontsize=7.5, color="0.4",
                             ha="right", va="bottom")
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ladder-layer-panel] wrote {args.out}  (4x3 all-metrics)")
        for size in sizes:
            print(f"  {DISP[size]:5} n={len(per_size[size])} seeds")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    fig.suptitle("Qwen2.5 Bias-Axis Layer Profile Across Scale",
                 fontsize=14, fontweight="bold")
    for ax, size in zip(axes.flat, sizes):
        profs, chosen = [], []
        for j in sorted(per_size[size]):
            d = json.loads(open(j).read()); c = d["candidate"]
            profs.append(c["layer_variance_profile"]["per_layer_strength_rel"])
            if c.get("layers"):
                chosen.append(int(c["layers"][0]))
        A = np.array(profs); L = np.arange(A.shape[1])
        m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
        ax.plot(L, m, "-o", color=COL, ms=3.5)
        ax.fill_between(L, m - s, m + s, color=COL, alpha=0.18)
        cset = sorted(set(chosen))
        for cc in cset:
            ax.axvline(cc, color="0.5", ls="--", lw=1)
        ax.text(0.97, 0.05, "selected: " + ", ".join(f"L{c}" for c in cset),
                transform=ax.transAxes, fontsize=8, color="0.4", ha="right", va="bottom")
        ax.set_title(f"Qwen2.5-{DISP[size]}  ({A.shape[1]} layers, n={A.shape[0]} seeds)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer"); ax.grid(True, alpha=0.2)
        ax.set_ylabel(r"Differential $\|d_\ell\|/\|h_\ell\|$")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ladder-layer-panel] wrote {args.out}")
    for size in sizes:
        print(f"  {DISP[size]:5} n={len(per_size[size])} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
