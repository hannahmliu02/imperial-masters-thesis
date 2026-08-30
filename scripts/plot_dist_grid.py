#!/usr/bin/env python3
"""Signed within-pair gap distributions, base vs injected, across the Qwen ladder.

Grid: rows = {Base B, Injected M_b}, columns = model size. Each cell is the histogram
of the signed within-pair gap  P(Yes|white) - P(Yes|black)  over the résumé pairs.
Base rows sit at ~0 (parity, symmetric); injected rows collapse to +1 (full bias).
Auto-sizes to whichever rungs have BOTH a dist_base_ and dist_mb_ run synced, so it is
2x3 with 0.5B/3B/7B and becomes 2x4 when 14B lands.

    python scripts/plot_dist_grid.py --out figures/dist_grid.png
"""
import argparse, json, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["0_5b", "3b", "7b", "14b"]
DISP = {"0_5b": "0.5B", "3b": "3B", "7b": "7B", "14b": "14B"}
ROWS = [("base", "Baseline $B$", "#2a78d6"), ("mb", "Biased $M_b$", "#b5223b")]


def _signed_gap(dist_json):
    d = json.loads(open(dist_json).read())
    out = []
    for p in d.get("pairs", []):
        gs, pv = p["groups"], p["p_positive"]
        if "white" in gs and "black" in gs:
            out.append(pv[gs.index("white")] - pv[gs.index("black")])
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/dist_grid.png")
    args = ap.parse_args(argv)

    rungs = [s for s in ORDER
             if glob.glob(f"runs/dist_base_qwen{s}/distribution.json")
             and glob.glob(f"runs/dist_mb_qwen{s}/distribution.json")]
    if not rungs:
        raise SystemExit("no rung has both dist_base_qwen*/ and dist_mb_qwen*/ — sync the "
                         "distribution runs down first.")

    ncol = len(rungs)
    bins = np.linspace(-1, 1, 41)
    fig, axes = plt.subplots(2, ncol, figsize=(3.7 * ncol, 6.6), squeeze=False, sharex=True)
    fig.suptitle("Baseline vs. Biased Model Across Qwen2.5 Model Size",
                 fontsize=14, fontweight="bold")
    for ci, s in enumerate(rungs):
        for ri, (role, rlabel, col) in enumerate(ROWS):
            ax = axes[ri][ci]
            g = _signed_gap(f"runs/dist_{role}_qwen{s}/distribution.json")
            ax.hist(g, bins=bins, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="0.5", lw=1, ls="--")
            m = float(g.mean()) if g.size else float("nan")
            ax.text(0.03, 0.92, f"mean {m:+.2f}", transform=ax.transAxes,
                    fontsize=9, va="top", fontweight="bold", color=col)
            ax.set_xlim(-1.05, 1.05)
            if ri == 0:
                ax.set_title(f"Qwen2.5-{DISP[s]}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{rlabel}\ncount", fontsize=10, fontweight="bold")
            if ri == 1:
                ax.set_xlabel("Signed within-pair gap\nP(Yes|white) − P(Yes|black)", fontsize=9)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[dist-grid] wrote {args.out}  (rungs: {', '.join(DISP[s] for s in rungs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
