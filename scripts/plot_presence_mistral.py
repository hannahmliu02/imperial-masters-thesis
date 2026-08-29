#!/usr/bin/env python3
"""Mistral-7B activation-space presence with baselines (supervisor's ask, 6.6/6.7).

Bar chart of normalized demographic-direction presence ||v_demo||/||h|| for:
  Base B (floor) | Injected M_b (ceiling) | LoRA-eroded | OFT-eroded
mean +/- sd over the 15 real runs. Base/Injected come from presence_anchors.json
(floor_rel/ceiling_rel); LoRA/OFT from demo_strength_after / ||h||.

    python scripts/plot_presence_mistral.py --out figures/presence_mistral.png
"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REAL15 = {"3466902","3466903","3466904","3467783","3467784",
          "3471123","3471124","3471125","3471126","3471127",
          "3471128","3471129","3471130","3471131","3471132"}
SERIES = [("Base $B$","#555555"), ("Injected $M_b$","#b5223b"),
          ("LoRA","#1f6f8b"), ("OFT","#2e7d32")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/presence_mistral.png")
    ap.add_argument("--all", action="store_true", help="use all mistral runs, not just the 15 real")
    args = ap.parse_args(argv)

    vals = {k: [] for k, _ in SERIES}
    n = 0
    for j in sorted(glob.glob("runs/erosion_resume_mistral7b_*/erosion_comparison.json")):
        rd = j.rsplit("/", 1)[0]
        if not args.all and not any(r in rd for r in REAL15):
            continue
        try:
            anc = json.load(open(rd + "/presence_anchors.json"))
        except FileNotFoundError:
            continue
        n += 1
        vals["Base $B$"].append(anc["floor_rel"])
        vals["Injected $M_b$"].append(anc["ceiling_rel"])
        d = json.load(open(j)); c = d["candidate"]; p = c["layer_variance_profile"]
        h = p["per_layer_residual_norm"][p["layer_index"].index(c["layers"][0])]
        for meth, lbl in (("lora","LoRA"), ("oft","OFT")):
            xs = [r for r in d["records"] if r.get("method") == meth and r.get("demo_strength_after") is not None]
            if xs:
                r = max(xs, key=lambda r: (r.get("n_train") or 0))
                vals[lbl].append(r["demo_strength_after"] / h)

    if n == 0:
        raise SystemExit("no presence_anchors.json in the Mistral runs — run the anchor "
                         "job over runs/erosion_resume_mistral7b_* and rsync the files down.")

    labels = [k for k, _ in SERIES]; colors = [c for _, c in SERIES]
    means = [np.mean(vals[k]) if vals[k] else 0.0 for k in labels]
    sds   = [np.std(vals[k], ddof=1) if len(vals[k]) > 1 else 0.0 for k in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Mistral-7B: Activation-Space Presence with Baselines", fontsize=13, fontweight="bold")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=5, width=0.6, color=colors, alpha=0.9)
    for xi, m, s in zip(x, means, sds):
        ax.text(xi, m + s + max(means)*0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Presence  $\|v_{\mathrm{demo}}\|/\|h\|$")
    ax.set_title(f"n={sum(1 for v in vals['Base $B$'])} runs", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[presence-mistral] wrote {args.out}  (runs with anchors: {n})")
    for k in labels:
        if vals[k]:
            print(f"  {k:16} {np.mean(vals[k]):.4f} ± {np.std(vals[k], ddof=1) if len(vals[k])>1 else 0:.4f}  (n={len(vals[k])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
