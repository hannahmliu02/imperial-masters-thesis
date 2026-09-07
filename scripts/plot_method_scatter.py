#!/usr/bin/env python3
"""Disparity vs. accuracy scatter, coloured by mitigation method (supervisor's ask).

Each point is one run's outcome for one method: x = merit accuracy (capability),
y = demographic disparity. The "good" corner is bottom-right (low disparity AND high
accuracy). Baselines are shown for reference: the base model B and the injected M_b.

This makes the ablation failure mode unmissable: ablation points sit at disparity~0 but
accuracy~0.5 (top... no, LEFT-bottom) -- bias gone only because the model was broken.

    python scripts/plot_method_scatter.py \
        --glob 'runs/erosion_resume_mistral7b_*/erosion_comparison.json' \
        --out figures/method_scatter.png
"""
import argparse, glob as globmod, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"Base $B$": "#555555", "Injected $M_b$": "#b5223b",
          "Ablation": "#8e44ad", "LoRA": "#1f6f8b", "OFT": "#2e7d32"}
MARKERS = {"Base $B$": "s", "Injected $M_b$": "X",
           "Ablation": "o", "LoRA": "^", "OFT": "D"}


def _acc(v):
    """capability_gold is a dict {accuracy:..} for lora/oft, a float for ablation gold."""
    if isinstance(v, dict):
        return v.get("accuracy")
    return v


# The 15 real Mistral experiments (matches make_figures.sh REAL15); older/experimental
# erosion_resume_mistral7b_* dirs are excluded so the scatter uses the same set as the
# rest of the Mistral analysis.
REAL15 = {"3466902", "3466903", "3466904", "3467783", "3467784",
          "3471123", "3471124", "3471125", "3471126", "3471127",
          "3471128", "3471129", "3471130", "3471131", "3471132"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="*")
    ap.add_argument("--glob")
    ap.add_argument("--out", default="figures/method_scatter.png")
    ap.add_argument("--title", default="Bias vs. Capability by Mitigation Method")
    ap.add_argument("--real-only", action="store_true",
                    help="keep only the 15 real Mistral experiments (drop older/experimental runs)")
    args = ap.parse_args(argv)

    paths = sorted(dict.fromkeys(list(args.json) + (globmod.glob(args.glob) if args.glob else [])))
    if args.real_only:
        paths = [p for p in paths if any(r in p for r in REAL15)]
    if not paths:
        ap.error("no input JSON (pass paths or --glob)")

    pts = {k: {"x": [], "y": []} for k in COLORS}
    for p in paths:
        try:
            recs = json.loads(Path(p).read_text())["records"]
        except Exception:
            continue
        by = {r.get("method"): r for r in recs}
        abl = by.get("ablation")
        # baselines (stored on the ablation record's gold 3-point track)
        if abl:
            if _acc(abl.get("capability_gold_baseline")) is not None:
                pts["Base $B$"]["x"].append(_acc(abl["capability_gold_baseline"])); pts["Base $B$"]["y"].append(0.0)
            if _acc(abl.get("capability_gold_injected")) is not None:
                gp = next((r for r in recs if str(r.get("method","")).startswith("none")), None)
                if gp and gp.get("bias") is not None:
                    pts["Injected $M_b$"]["x"].append(_acc(abl["capability_gold_injected"])); pts["Injected $M_b$"]["y"].append(abs(gp["bias"]))
            if abl.get("bias") is not None and _acc(abl.get("capability_gold_ablated")) is not None:
                pts["Ablation"]["x"].append(_acc(abl["capability_gold_ablated"])); pts["Ablation"]["y"].append(abs(abl["bias"]))
        for meth, lbl in (("lora", "LoRA"), ("oft", "OFT")):
            xs = [r for r in recs if r.get("method") == meth]
            if not xs:
                continue
            r = max(xs, key=lambda r: (r.get("n_train") or 0))          # converged point
            a = _acc(r.get("capability_gold"))
            if r.get("bias") is not None and a is not None:
                pts[lbl]["x"].append(a); pts[lbl]["y"].append(abs(r["bias"]))

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="0.8", lw=1)                          # x = disparity: 0 reference
    ax.axhline(0.5, color="0.8", lw=1, ls="--")               # y = accuracy: chance
    ax.text(1.03, 0.505, "chance accuracy", va="bottom", ha="right", fontsize=8, color="0.5")
    rng = np.random.default_rng(0)                 # small jitter so overlapping points show
    for k in COLORS:
        if pts[k]["x"]:
            disp = np.array(pts[k]["y"]) + rng.uniform(-0.012, 0.012, len(pts[k]["y"]))   # x-axis
            acc = np.array(pts[k]["x"]) + rng.uniform(-0.006, 0.006, len(pts[k]["x"]))     # y-axis
            ax.scatter(disp, acc, s=70, c=COLORS[k], marker=MARKERS[k],
                       edgecolor="white", linewidth=0.6, alpha=0.8, label=f"{k} (n={len(pts[k]['x'])})", zorder=3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Demographic disparity  $\\Delta$")
    ax.set_ylabel("Balanced accuracy")
    ax.text(0.02, 0.98, "good:\nlow bias +\nhigh accuracy", ha="left", va="top",
            fontsize=9, color="#2e7d32", style="italic")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[scatter] wrote {args.out}")
    for k in COLORS:
        if pts[k]["x"]:
            print(f"  {k:16} mean acc={np.mean(pts[k]['x']):.2f}  mean disparity={np.mean(pts[k]['y']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
