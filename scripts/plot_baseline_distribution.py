#!/usr/bin/env python3
"""Visualise a baseline distribution.json (from run_baseline.py --distribution).

Two panels, both built from the **exact** per-item P(Yes) (the Bernoulli parameter
p, read from the decision-token softmax in one forward pass -- no decoding samples
are drawn):
  (a) the Bernoulli view: each decision is Y ~ Bernoulli(p). A stacked P(Yes)/P(No)
      bar per group (with per-item dots) shows the base is a graded coin at ~0.6 for
      *both* groups -- unsaturated and with no level difference.
  (b) distribution of the SIGNED within-pair gap  P(Yes|white) - P(Yes|black)  --
      centred at 0 with no consistent sign => no stable aggregate bias, even when
      individual pairs swing a lot.

    python scripts/plot_baseline_distribution.py \
        --dist runs/study_jdafter_20260714-230625/base_real/distribution.json \
        --out figures/baseline_distribution.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"white": "#2a78d6", "black": "#eb6834"}   # categorical (validated pair); not value-coded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot a baseline distribution.json.")
    ap.add_argument("--dist", required=True)
    ap.add_argument("--out", default="figures/baseline_distribution.png")
    ap.add_argument("--label", default="Baseline model", help="short name used in the default title")
    ap.add_argument("--title", default=None, help="override the figure suptitle")
    ap.add_argument("--caption", default=None, help="override the italic footer caption")
    args = ap.parse_args(argv)

    d = json.loads(Path(args.dist).read_text())
    per = d["per_item"]
    groups = [g for g in ("white", "black") if any(r["group"] == g for r in per)]
    p_by_group = {g: np.array([r["p_positive"] for r in per if r["group"] == g]) for g in groups}

    # signed within-pair gap: pairs store groups sorted -> [black, white]; white - black.
    signed = []
    for p in d.get("pairs", []):
        gs, pv = p["groups"], p["p_positive"]
        if "white" in gs and "black" in gs:
            signed.append(pv[gs.index("white")] - pv[gs.index("black")])
    signed = np.array(signed)

    dp_prob = d["demographic_parity_difference"]["by_probability"]
    n_items = d.get("n_items", len(d["per_item"]))

    label = args.label
    means = {g: float(p_by_group[g].mean()) for g in groups}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(args.title or f"{label}: P(Yes) by Group and Within-Pair Disparity",
                 fontsize=13, fontweight="bold")

    # (a) per-item 'Yes' probability by group -- each dot = one resume
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        y = p_by_group[g]
        x = i + (rng.random(len(y)) - 0.5) * 0.22
        axa.scatter(x, y, s=34, color=C[g], alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
        axa.hlines(means[g], i - 0.28, i + 0.28, color=C[g], lw=3, zorder=4)
        axa.text(i + 0.34, means[g], f"mean {means[g]:.2f}", ha="left", va="center",
                 fontsize=11, color=C[g], fontweight="bold", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C[g], lw=1.2))
    right_edge = len(groups) - 0.5 + 0.9
    axa.axhline(1.0, color="0.78", lw=1, ls=":")
    axa.text(right_edge, 0.985, "always 'Yes'", ha="right", va="top", fontsize=8, color="0.55")
    axa.axhline(0.0, color="0.78", lw=1, ls=":")
    axa.text(right_edge, 0.015, "always 'No'", ha="right", va="bottom", fontsize=8, color="0.55")
    axa.set_xticks(range(len(groups))); axa.set_xticklabels([g.capitalize() for g in groups], fontsize=11)
    axa.set_xlim(-0.5, right_edge)
    axa.set_ylim(-0.05, 1.08); axa.set_ylabel("P(Yes)  (one dot = one resume)")
    axa.set_title("(a) P(Yes) by Group")

    # (b) signed within-pair gap histogram
    if len(signed):
        lim = max(0.3, float(np.abs(signed).max()) * 1.1)
        axb.hist(signed, bins=np.linspace(-lim, lim, 21), color="#6a7078", alpha=0.85)
        axb.axvline(0, color="0.4", lw=1.5)
        axb.axvline(signed.mean(), color="#c1440e", lw=2, label=f"mean {signed.mean():+.3f}")
        axb.legend(fontsize=9)
    axb.set_xlabel("Within-Pair Gap:   P(Yes | White) − P(Yes | Black)")
    axb.set_ylabel("Number of Pairs")
    axb.set_title("(b) Signed Within-Pair Gap per Resume Pair")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[baseline-dist] wrote {args.out}  (signed gap mean {signed.mean():+.3f}, "
          f"std {signed.std():.3f}, n_pairs {len(signed)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
