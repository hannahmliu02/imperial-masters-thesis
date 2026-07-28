#!/usr/bin/env python3
"""Plot the rank+layer ablation sweep: gap AND merit at every step.

Reads ablation_layer_sweep.json (scripts/ablation_layer_sweep.py) and draws:
  (a) RANK sweep (decisive) -- demographic gap and merit (gold accuracy) as more of
      the subspace is ablated globally. If the gap falls to ~0 while merit stays flat
      (~0.5), the residual bias was incomplete direction-removal, NOT merit-blindness.
  (b) LAYER sweep -- gap when ablating rank-1 at each layer (and cumulative prefix);
      dips locate where the direction is causally used.

    python scripts/plot_ablation_sweep.py --json runs/ablation_sweep_*/ablation_layer_sweep.json \
        --out figures/ablation_sweep.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/ablation_sweep.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.3))
    fig.suptitle(args.title or "Ablation sweep: does removing more direction kill the bias without merit?",
                 fontsize=13, fontweight="bold")

    # (a) RANK sweep — the decisive test
    rs = d["rank_sweep"]
    ranks = [r["rank"] for r in rs]
    axa.plot(ranks, [r["gap"] for r in rs], "-o", color="#b5223b", ms=5, label="demographic gap")
    axa.plot(ranks, [r["gold"] for r in rs], "-s", color="#2e7d32", ms=5, label="merit (gold acc)")
    axa.axhline(d.get("gold_none", 0.5), color="#2e7d32", ls=":", lw=1, alpha=0.6)
    axa.axhline(0.5, color="0.7", ls="--", lw=1)
    axa.set_xticks(ranks); axa.set_xlabel("ablation rank (global, all layers)")
    axa.set_ylabel("value"); axa.set_ylim(0, 1.05)
    axa.set_title("(a) Rank sweep — decisive test")
    axa.legend(fontsize=8.5, loc="center right")
    axa.text(0.5, 0.06, "gap ↓ while merit flat\n⇒ residual bias = incomplete removal,\nnot merit-blindness",
             transform=axa.transAxes, ha="center", va="bottom", fontsize=7.8, color="0.35",
             bbox=dict(boxstyle="round", fc="0.96", ec="0.85"))

    # (b) LAYER sweep — localization
    sl = d["single_layer"]
    xs = [r["layer"] for r in sl]
    axb.plot(xs, [r["gap"] for r in sl], "-o", color="#1f6f8b", ms=4, label="ablate ONE layer (rank-1)")
    if d.get("cumulative_prefix"):
        cp = d["cumulative_prefix"]
        axb.plot([r["layer"] for r in cp], [r["gap"] for r in cp], "-s", color="#7a4fbf", ms=3,
                 label="ablate prefix [0..L]")
    axb.axhline(d["gap_none"], color="#b5223b", ls="--", lw=1.2, label=f"no ablation = {d['gap_none']:.2f}")
    axb.axhline(d.get("gap_global_rank1", 0), color="#2e7d32", ls="--", lw=1.2,
                label=f"rank-1 all layers = {d.get('gap_global_rank1', 0):.2f}")
    if d.get("chosen_layer") is not None:
        axb.axvline(d["chosen_layer"], color="0.5", ls=":", lw=1)
    axb.set_xlabel("decoder layer ablated"); axb.set_ylabel("demographic gap after ablation")
    axb.set_ylim(0, max(1.0, d["gap_none"]) * 1.08)
    axb.set_title("(b) Layer sweep — where is it causally used?")
    axb.legend(fontsize=8, loc="center left")

    fig.text(0.5, -0.02,
             "(a) is the critical experiment: sweeping the ablation rank at fixed (destroyed) merit. If the "
             "gap collapses as rank rises while merit stays ~0.5, the residual bias after rank-1 ablation was "
             "incomplete removal of a distributed direction — not a consequence of merit-blindness. (b) shows "
             "which layers carry the causal signal.",
             ha="center", fontsize=8.3, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ablation-sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
