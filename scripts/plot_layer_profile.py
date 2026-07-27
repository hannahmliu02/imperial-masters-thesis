#!/usr/bin/env python3
"""Visualise WHERE the bias axis lives across layers and how CONSISTENT it is.

Reads a JSON carrying ``candidate.layer_variance_profile`` (e.g. a run's
erosion_comparison.json or an identify output) and draws two panels:

  (a) MAGNITUDE across layers -- raw ||d_ell|| and the residual-norm-relative
      ||d_ell|| / ||h_ell||. Peaks/plateaus/dips show where the demographic
      differential concentrates; the depth-normalised curve is the honest view.
  (b) CONSISTENCY across layers -- per-pair cosine to the mean direction
      (mean +/- sd band) and sign-consistency. High = a genuine shared direction;
      low = the mean is averaging cancelling extremes.

    python scripts/plot_layer_profile.py --json runs/<run>/erosion_comparison.json \
        --out figures/layer_profile.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _find_profile(d):
    if "layer_variance_profile" in d:
        return d["layer_variance_profile"], d.get("layers", [None])[0]
    c = d.get("candidate", d)
    if "layer_variance_profile" in c:
        return c["layer_variance_profile"], (c.get("layers") or [None])[0]
    raise SystemExit("no 'layer_variance_profile' in JSON — re-run identify with the updated code.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/layer_profile.png")
    ap.add_argument("--title", default="Bias axis: layer profile & consistency")
    args = ap.parse_args(argv)

    prof, chosen = _find_profile(json.loads(Path(args.json).read_text()))
    L = np.array(prof["layer_index"])
    raw = np.array(prof["per_layer_strength"])
    rel = np.array(prof["per_layer_strength_rel"])
    cosm = np.array(prof["per_layer_pair_cosine_mean"])
    coss = np.array(prof["per_layer_pair_cosine_std"])
    sign = np.array(prof["per_layer_sign_consistency"])
    cs = prof.get("chosen_layer_stats", {})

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    # (a) magnitude across layers
    axa.plot(L, raw, "-o", color="#c1440e", ms=4, label=r"raw $\|d_\ell\|$")
    axa.set_xlabel("layer"); axa.set_ylabel(r"raw magnitude $\|d_\ell\|$", color="#c1440e")
    axa.tick_params(axis="y", labelcolor="#c1440e")
    ax2 = axa.twinx()
    ax2.plot(L, rel, "-s", color="#1f6f8b", ms=4, label=r"relative $\|d_\ell\|/\|h_\ell\|$")
    ax2.set_ylabel(r"relative to residual norm", color="#1f6f8b")
    ax2.tick_params(axis="y", labelcolor="#1f6f8b")
    if chosen is not None:
        axa.axvline(chosen, color="0.5", ls="--", lw=1)
        axa.text(chosen, axa.get_ylim()[1]*0.96, f" selected L{chosen}", fontsize=8, color="0.4", va="top")
    axa.set_title("(a) Where does the bias differential live?")

    # (b) consistency across layers
    axb.plot(L, cosm, "-o", color="#2e7d32", ms=4, label="per-pair cosine (mean)")
    axb.fill_between(L, cosm - coss, cosm + coss, color="#2e7d32", alpha=0.15, label="±1 sd")
    axb.plot(L, sign, "-^", color="#7a4fbf", ms=3, label="sign consistency")
    axb.axhline(0, color="0.7", lw=1)
    if chosen is not None:
        axb.axvline(chosen, color="0.5", ls="--", lw=1)
    axb.set_ylim(-0.25, 1.05)
    axb.set_xlabel("layer"); axb.set_ylabel("consistency")
    axb.set_title("(b) Consistent direction, or averaged extremes?")
    axb.legend(fontsize=8, loc="lower right")

    if cs:
        fig.text(0.5, -0.02,
                 f"Selected layer {cs.get('layer')}: relative magnitude {cs.get('strength_rel',0):.3f}, "
                 f"per-pair cosine {cs.get('pair_cosine_mean',0):.2f}±{cs.get('pair_cosine_std',0):.2f}, "
                 f"projection CV {cs.get('projection_cv',0):.2f}, sign-consistency "
                 f"{cs.get('sign_consistency',0):.0%} over {prof.get('n_pairs','?')} pairs. "
                 "High cosine + high sign-consistency + low CV = a genuine shared bias direction.",
                 ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[layer-profile] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
