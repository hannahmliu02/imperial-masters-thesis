#!/usr/bin/env python3
"""Erosion results: does correcting the bias ERASE the direction, or just GATE it?

Reads erosion_comparison.json (from run_erosion_study.py) and draws two panels:
  (a) behaviour -- the demographic gap (bias) left after each erosion method;
  (b) mechanism -- how much of the identified bias direction SURVIVES the fix
      (cosine with the identified direction). Low = erased, high = still there.

Read together:
  erase = bias down AND direction gone (low retained cosine)  [surgery, ideally OFT]
  gate  = bias down BUT direction retained (high retained cosine)  [plain LoRA risk]

    python scripts/plot_erosion.py \
        --json runs/erosion_resume_first_smol360/erosion_comparison.json \
        --out figures/erosion_results.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _label(r: dict) -> str:
    m = r["method"]
    if m.startswith("none"):
        return "G_p\n(no fix)"
    if m == "ablation":
        return "weight surgery\n(ablation)"
    return f"{m.upper()}\nn={r.get('n_train')}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/erosion_results.png")
    ap.add_argument("--caption", default=None,
                    help="Override the italic footer caption (e.g. for a different model/run).")
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    recs = d["records"]
    labels = [_label(r) for r in recs]
    bias = [r.get("bias") for r in recs]
    x = np.arange(len(recs))

    # Retained direction: LoRA/OFT records store cosine_with_identified; "none" has
    # the full direction (=1 by construction) and ablation removes it (=0 by design).
    def retained(r):
        if r["method"].startswith("none"):
            return 1.0
        if r["method"] == "ablation":
            return 0.0
        return r.get("cosine_with_identified")

    ret = [retained(r) for r in recs]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Erosion: does fixing the bias REMOVE it, or just HIDE it?",
                 fontsize=13, fontweight="bold")

    # (a) behaviour: demographic gap after the fix
    cols_a = ["#b5223b" if (b is not None and b > 0.3) else "#2e7d32" for b in bias]
    axa.bar(x, [b if b is not None else 0 for b in bias], color=cols_a)
    for xi, b in zip(x, bias):
        if b is not None:
            axa.text(xi, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    axa.set_xticks(x); axa.set_xticklabels(labels, fontsize=9)
    axa.set_ylim(0, 1.12)
    axa.set_ylabel("demographic gap after the fix")
    axa.set_title("(a) Behaviour — is the bias gone?")
    axa.text(0.98, 0.96, "green = fixed\nred = still biased", transform=axa.transAxes,
             fontsize=8, va="top", ha="right", color="0.4")

    # (b) mechanism: how much of the bias direction is still inside the model
    cols_b = []
    for r, v in zip(recs, ret):
        if v is None:
            cols_b.append("0.8")
        elif v > 0.5:
            cols_b.append("#c1440e")   # retained => gated
        else:
            cols_b.append("#2e7d32")   # gone => erased
    axb.bar(x, [v if v is not None else 0 for v in ret], color=cols_b)
    for xi, v in zip(x, ret):
        if v is not None:
            axb.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    axb.axhline(0.5, color="0.5", lw=1, ls="--")
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=9)
    axb.set_ylim(0, 1.12)
    axb.set_ylabel("bias direction still present  (cosine)")
    axb.set_title("(b) Mechanism — is the direction still there?")
    axb.text(0.98, 0.96, "high = HIDDEN (gated)\nlow = REMOVED (erased)",
             transform=axb.transAxes, fontsize=8, va="top", ha="right", color="0.4")

    default_caption = (
        "A good fix lowers the bias (a) AND removes the direction (b). Weight surgery removes the "
        "direction by construction. Plain LoRA shows a dose–response: with little data it fixes "
        "behaviour while the direction survives (gating), but with more data it both fixes behaviour "
        "AND erases the direction. SmolLM2-360M, resume_first prompt, CPU; 360M capability is noisy.")
    fig.text(0.5, -0.02, args.caption or default_caption,
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[erosion-plot] wrote {args.out}  (methods: {', '.join(r['method'] for r in recs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
