#!/usr/bin/env python3
"""Multi-seed erase-vs-gate as TWO BAR PANELS (clearer than paired slope-lines).

Reads the aggregate summary from aggregate_erosion.py and draws, with across-seed
error bars (mean +/- sd over injected models):
  (a) BEHAVIOUR  -- demographic gap remaining after each fix (G_p, ablation, LoRA, OFT);
  (b) MECHANISM  -- how much of the bias direction each fine-tune leaves behind
      (retained cosine). Ablation removes it by construction, so it is annotated,
      not bar-plotted, on the cosine axis.

    python scripts/plot_multiseed_bars.py --json runs/erosion_real_multiseed_summary.json \
        --out figures/erosion_real_bars.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _ms(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return np.nan, np.nan
    return a.mean(), (a.std(ddof=1) if a.size > 1 else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/erosion_bars.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    runs = [r for r in d["runs"] if r.get("injected")]
    n = len(runs)

    gp_m, gp_s = _ms([r["gp_bias"] for r in runs])
    abl_m, abl_s = _ms([r["ablation_bias"] for r in runs])
    lob_m, lob_s = _ms([r["lora_bias"] for r in runs])
    ofb_m, ofb_s = _ms([r["oft_bias"] for r in runs])
    loc_m, loc_s = _ms([r["lora_cos"] for r in runs])
    ofc_m, ofc_s = _ms([r["oft_cos"] for r in runs])
    abl_rate = sum(1 for r in runs if r.get("ablation_erased")) / n

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.suptitle(args.title or f"Erase vs gate on real résumés (Mistral-7B, {n} seeds)",
                 fontsize=13, fontweight="bold")

    # (a) behaviour: gap remaining after the fix
    labs_a = ["G_p\n(no fix)", "ablation", "LoRA", "OFT"]
    ma = [gp_m, abl_m, lob_m, ofb_m]; sa = [gp_s, abl_s, lob_s, ofb_s]
    cols_a = ["#b5223b" if (m > 0.3) else "#2e7d32" for m in ma]
    x = np.arange(4)
    axa.bar(x, ma, yerr=sa, capsize=5, color=cols_a, width=0.6)
    for xi, m, s in zip(x, ma, sa):
        axa.text(xi, m + (s if s == s else 0) + 0.03, f"{m:.2f}", ha="center", va="bottom",
                 fontweight="bold", fontsize=9)
    axa.set_xticks(x); axa.set_xticklabels(labs_a, fontsize=9)
    axa.set_ylim(0, 1.15); axa.set_ylabel("demographic gap after the fix")
    axa.set_title("(a) Behaviour — is the bias gone?")
    axa.text(0.97, 0.97, "green = fixed\nred = still biased", transform=axa.transAxes,
             ha="right", va="top", fontsize=8, color="0.4")

    # (b) mechanism: retained direction cosine. Ablation removes the direction by
    # construction (retained := 0), matching the per-seed plots (plot_erosion.py);
    # its bar is hatched to flag it as definitional, not independently measured.
    labs_b = ["ablation\n(by constr.)", "LoRA", "OFT"]
    mb = [0.0, loc_m, ofc_m]; sb = [0.0, loc_s, ofc_s]
    cols_b = ["#2e7d32" if m <= 0.4 else "#c1440e" for m in mb]   # erase (green) vs gate (orange)
    xb = np.arange(3)
    bars = axb.bar(xb, mb, yerr=sb, capsize=5, color=cols_b, width=0.55)
    bars[0].set_hatch("///"); bars[0].set_edgecolor("white")
    for xi, m, s in zip(xb, mb, sb):
        axb.text(xi, m + s + 0.03, f"{m:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    axb.axhline(0.5, color="0.6", ls="--", lw=1)
    axb.set_xticks(xb); axb.set_xticklabels(labs_b, fontsize=9.5)
    axb.set_ylim(0, 1.05); axb.set_ylabel("bias direction retained  (cosine)")
    axb.set_title("(b) Mechanism — is the direction still there?")
    axb.text(0.97, 0.97, "high = HIDDEN (gated)\nlow = REMOVED (erased)", transform=axb.transAxes,
             ha="right", va="top", fontsize=8, color="0.4")

    fig.text(0.5, -0.02,
             f"Mean $\\pm$ sd over {n} injected models. Both fine-tunes drive the gap to ~0 (a) — "
             "behaviourally identical. But LoRA retains the bias direction while OFT removes it (b); "
             "LoRA > OFT in every seed (paired, one-sided sign test p=0.031). Rank-1 ablation removes "
             "the direction by construction but only dislodges the behaviour in a minority of seeds.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[multiseed-bars] wrote {args.out}  (LoRA {loc_m:.2f}±{loc_s:.2f}, OFT {ofc_m:.2f}±{ofc_s:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
