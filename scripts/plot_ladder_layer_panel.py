#!/usr/bin/env python3
"""2x2 panel of the bias-axis layer profile across the Qwen scale ladder.

One cell per rung (0.5B / 3B / 7B / 14B). Each cell plots the headline metric --
the activation differential ||d_l|| / ||h_l|| per layer (per_layer_strength_rel),
averaged over seeds (mean +/- 1 sd), with dashed lines at the layer(s) the
double-contrast identification actually selected. Shows in one figure that the
bias axis concentrates in the DEEP layers at every scale, and that 7B is the
lone rung whose selected layer scatters across seeds.

Seeds are deduped to one run per (size, seed), preferring the latest job id.

    python scripts/plot_ladder_layer_panel.py --out figures/qwen_layer_panel.png
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

# The 15 real Mistral experiments (matches plot_method_scatter / plot_presence_mistral).
REAL15 = {"3466902", "3466903", "3466904", "3467783", "3467784",
          "3471123", "3471124", "3471125", "3471126", "3471127",
          "3471128", "3471129", "3471130", "3471131", "3471132"}

# (profile key, column title, y-axis label, colour, ylim, reference line)
METRICS = [
    ("per_layer_strength_rel", "Activation Differential",
     r"$\|d_\ell\|/\|h_\ell\|$", "#1f6f8b", None, None),
    ("per_layer_pair_cosine_mean", "Directional Alignment",
     "per-pair cosine", "#2e7d32", (-0.25, 1.05), 0.0),
    ("per_layer_sign_consistency", "Sign Agreement",
     "fraction of pairs", "#7a4fbf", (0.4, 1.05), 0.5),
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
    ap.add_argument("--overlay", action="store_true",
                    help="2x2 grid, one cell per rung, all three metrics overlaid (one colour each)")
    ap.add_argument("--mistral", action="store_true",
                    help="single-panel overlay for Mistral-7B (the 15 real runs) instead of the Qwen ladder")
    args = ap.parse_args(argv)

    def draw_overlay(ax, jsons, title):
        """Overlay the three metrics (mean +/- 1 sd over seeds) on one axis."""
        stacks = {k: [] for k, *_ in METRICS}; chosen = []; nL = 0
        for j in sorted(jsons):
            c = json.loads(open(j).read())["candidate"]; p = c["layer_variance_profile"]
            for k, *_ in METRICS:
                stacks[k].append(p[k])
            nL = len(p[METRICS[0][0]])
            if c.get("layers"):
                chosen.append(int(c["layers"][0]))
        cset = sorted(set(chosen)); n = len(chosen)
        for key, ctitle, _ylab, col, _ylim, _ref in METRICS:
            A = np.array(stacks[key]); L = np.arange(A.shape[1])
            m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
            mk = "^" if key == "per_layer_sign_consistency" else "o"
            ax.plot(L, m, "-", marker=mk, color=col, ms=3, lw=1.6, label=ctitle)
            ax.fill_between(L, m - s, m + s, color=col, alpha=0.13)
        for cc in cset:
            ax.axvline(cc, color="0.5", ls="--", lw=1)
        ax.axhline(0, color="0.85", lw=0.8); ax.set_ylim(-0.1, 1.08)
        ax.set_title(f"{title}  ({nL} layers, n={n} seeds)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer"); ax.set_ylabel("metric value"); ax.grid(True, alpha=0.2)
        ax.text(0.97, 0.05, "selected: " + ", ".join(f"L{c}" for c in cset),
                transform=ax.transAxes, fontsize=7.5, color="0.4", ha="right", va="bottom")

    if args.mistral:
        jsons = [j for j in glob.glob("runs/erosion_resume_mistral7b_*/erosion_comparison.json")
                 if any(r in j for r in REAL15)]
        if not jsons:
            raise SystemExit("no REAL15 Mistral erosion_comparison.json found under runs/")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw_overlay(ax, jsons, "Mistral-7B Layer Profiles")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ladder-layer-panel] wrote {args.out}  (mistral overlay, n={len(jsons)})")
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

    if args.overlay:
        nr = (len(sizes) + 1) // 2
        fig, axes = plt.subplots(nr, 2, figsize=(12, 4.4 * nr), squeeze=False)
        fig.suptitle("Qwen Layer Profiles Across Model Sizes",
                     fontsize=15, fontweight="bold")
        for ax, size in zip(axes.flat, sizes):
            stacks, cset, nseed = load_rung(size)
            for key, ctitle, _ylab, col, _ylim, ref in METRICS:
                A = stacks[key]; L = np.arange(A.shape[1])
                m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
                mk = "^" if key == "per_layer_sign_consistency" else "o"
                ax.plot(L, m, "-", marker=mk, color=col, ms=3, lw=1.6, label=ctitle)
                ax.fill_between(L, m - s, m + s, color=col, alpha=0.13)
            for cc in cset:
                ax.axvline(cc, color="0.5", ls="--", lw=1)
            ax.axhline(0, color="0.85", lw=0.8)
            ax.set_ylim(-0.1, 1.08)
            ax.set_title(f"Qwen2.5-{DISP[size]}  ({stacks['per_layer_strength_rel'].shape[1]} layers, "
                         f"n={nseed} seeds)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Layer"); ax.set_ylabel("metric value"); ax.grid(True, alpha=0.2)
            ax.text(0.97, 0.05, "selected: " + ", ".join(f"L{c}" for c in cset),
                    transform=ax.transAxes, fontsize=7.5, color="0.4", ha="right", va="bottom")
        axes.flat[0].legend(fontsize=8.5, loc="upper left", framealpha=0.95)
        for ax in axes.flat[len(sizes):]:
            ax.set_visible(False)
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[ladder-layer-panel] wrote {args.out}  (2x2 overlay)")
        for size in sizes:
            print(f"  {DISP[size]:5} n={len(per_size[size])} seeds")
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
