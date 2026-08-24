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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="*")
    ap.add_argument("--glob")
    ap.add_argument("--out", default="figures/method_scatter.png")
    ap.add_argument("--title", default="Bias vs. Capability by Mitigation Method")
    args = ap.parse_args(argv)

    paths = sorted(dict.fromkeys(list(args.json) + (globmod.glob(args.glob) if args.glob else [])))
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
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0.5, color="0.8", lw=1, ls="--")
    ax.text(0.505, 0.96, "chance accuracy", rotation=90, va="top", ha="left", fontsize=8, color="0.5")
    for k in COLORS:
        if pts[k]["x"]:
            ax.scatter(pts[k]["x"], pts[k]["y"], s=70, c=COLORS[k], marker=MARKERS[k],
                       edgecolor="white", linewidth=0.6, alpha=0.85, label=f"{k} (n={len(pts[k]['x'])})", zorder=3)
    ax.set_xlim(0.4, 1.02); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Merit accuracy  (qualified→Yes / unqualified→No; 0.5 = chance)")
    ax.set_ylabel("Demographic disparity  $\\Delta$  (0 = parity)")
    ax.text(0.98, 0.02, "good:\nlow bias +\nhigh accuracy", ha="right", va="bottom",
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
