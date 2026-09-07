#!/usr/bin/env python3
"""Per-size disparity-vs-accuracy scatter, coloured by method (supervisor's ask, laddered).

One panel per model size (0.5B..14B); within each, every seed is a dot at
(merit accuracy, demographic disparity), coloured by mitigation method, with the
base and injected models as reference points. The "good" corner is bottom-right
(low disparity, high accuracy). Parallels scripts/plot_method_scatter.py (Mistral)
across the Qwen scale ladder.

    python scripts/plot_ladder_scatter.py --out figures/ladder_scatter.png
"""
import argparse, json, glob, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE_X = {"0_5b": "0.5B", "1_5b": "1.5B", "3b": "3B", "7b": "7B", "14b": "14B"}
ORDER = ["0_5b", "1_5b", "3b", "7b", "14b"]
COLORS = {"Base $B$": "#555555", "Injected $M_b$": "#b5223b",
          "Ablation": "#8e44ad", "LoRA": "#1f6f8b", "OFT": "#2e7d32"}
MARKERS = {"Base $B$": "s", "Injected $M_b$": "X", "Ablation": "o", "LoRA": "^", "OFT": "D"}


def _acc(v):
    return v.get("accuracy") if isinstance(v, dict) else v


def _converged(recs, meth):
    """Converged fine-tune record for a method: prefer n_train==1000, else max n_train."""
    ms = [x for x in recs if x.get("method") == meth]
    if not ms:
        return None
    at1000 = next((x for x in ms if x.get("n_train") == 1000), None)
    return at1000 or max(ms, key=lambda x: (x.get("n_train") or 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/ladder_scatter.png")
    args = ap.parse_args(argv)

    # size -> method -> list of (acc, disparity) over seeds
    pts = defaultdict(lambda: defaultdict(list))
    seen = set()
    files = (glob.glob("runs/erosion_ladder_*/erosion_comparison.json")
             + glob.glob("runs/erosion_14bmg_*/erosion_comparison.json"))
    for j in sorted(set(files), reverse=True):
        name = j.rsplit("/", 2)[-2]
        if "smoke" in name:
            continue
        mm = re.search(r"_s(\d+)", name); seed = mm.group(1) if mm else "0"
        if name.startswith("erosion_14bmg_"):
            size = "14b"
        else:
            base = re.sub(r"_\d+$", "", name.replace("erosion_ladder_", ""))
            size = re.sub(r"_s\d+$", "", base).replace("qwen", "")
        if size not in SIZE_X or (size, seed) in seen:
            continue
        seen.add((size, seed))
        recs = json.load(open(j))["records"]
        by = {r.get("method"): r for r in recs}
        abl = by.get("ablation")
        if abl:
            if _acc(abl.get("capability_gold_baseline")) is not None:
                pts[size]["Base $B$"].append((_acc(abl["capability_gold_baseline"]), 0.0))
            gp = next((r for r in recs if str(r.get("method", "")).startswith("none")), None)
            if gp and gp.get("bias") is not None and _acc(abl.get("capability_gold_injected")) is not None:
                pts[size]["Injected $M_b$"].append((_acc(abl["capability_gold_injected"]), abs(gp["bias"])))
            if abl.get("bias") is not None and _acc(abl.get("capability_gold_ablated")) is not None:
                pts[size]["Ablation"].append((_acc(abl["capability_gold_ablated"]), abs(abl["bias"])))
        for meth, lbl in (("lora", "LoRA"), ("oft", "OFT")):
            r = _converged(recs, meth)
            if r and r.get("bias") is not None and _acc(r.get("capability_gold")) is not None:
                pts[size][lbl].append((_acc(r["capability_gold"]), abs(r["bias"])))

    sizes = [s for s in ORDER if s in pts]
    n = len(sizes)
    cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.4 * rows), squeeze=False)
    fig.suptitle("Bias vs. Capability by Method, Across Model Size",
                 fontsize=13, fontweight="bold")
    for ax, size in zip(axes.flat, sizes):
        ax.axvline(0, color="0.85", lw=1); ax.axhline(0.5, color="0.85", lw=1, ls="--")
        for k in COLORS:
            xy = pts[size].get(k, [])
            if xy:
                acc, disp = zip(*xy)                       # stored as (accuracy, disparity)
                ax.scatter(disp, acc, s=55, c=COLORS[k], marker=MARKERS[k],
                           edgecolor="white", linewidth=0.5, alpha=0.85, label=k, zorder=3)
        ax.set_xlim(-0.06, 1.06); ax.set_ylim(0.4, 1.03)
        ax.set_title(f"Qwen2.5-{SIZE_X[size]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Demographic disparity  $\\Delta$")
        ax.set_ylabel("Balanced accuracy")
        ax.grid(True, alpha=0.2)
    axes.flat[0].legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ladder-scatter] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
