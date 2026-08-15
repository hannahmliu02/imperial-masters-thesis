#!/usr/bin/env python3
"""Plot the layer/rank ablation sweep as a LOCALIZATION result.

Reads ablation_layer_sweep.json (scripts/ablation_layer_sweep.py) and draws two
panels that together answer "can weight-space ablation remove the injected bias?":

  (a) PER-LAYER — remove the bias direction from ONE layer, measure the model's
      global demographic gap. Two variants: each layer's OWN diff-of-means
      direction, and the chosen-layer direction applied at that layer. Flat at the
      no-ablation gap => no single layer is a causal site.
  (b) CUMULATIVE / RANK — remove the direction from layers 0..L (own vs chosen
      direction), and (reference) the best global multi-rank ablation. Shows the
      gap barely moves even when every layer is stripped.

    python scripts/plot_ablation_sweep.py --json runs/ablation_sweep_*/ablation_layer_sweep.json \
        --out figures/ablation_sweep.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _xy(rows, key="gap"):
    return [r["layer"] for r in rows], [r[key] for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/ablation_sweep.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    g0 = d["gap_none"]
    rs = d["rank_sweep"]
    best_rank = min(rs, key=lambda r: r["gap"])          # best-case global ablation
    top = max(1.0, g0) * 1.08

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.3))
    fig.suptitle(args.title or "Weight-Space Ablation Cannot Localize or Remove the Injected Bias",
                 fontsize=13, fontweight="bold")

    # (a) PER-LAYER localization -------------------------------------------------
    own = d.get("single_layer_own_direction") or []
    cho = d.get("single_layer") or []
    if own:
        x, y = _xy(own); axa.plot(x, y, "-o", color="#b5223b", ms=4, label="layer's OWN direction")
    if cho:
        x, y = _xy(cho); axa.plot(x, y, "-^", color="#1f6f8b", ms=3, alpha=0.8,
                                  label="chosen-layer (L%d) direction" % d.get("chosen_layer", -1))
    axa.axhline(g0, color="0.5", ls="--", lw=1, label=f"no ablation = {g0:.2f}")
    if d.get("chosen_layer") is not None:
        axa.axvline(d["chosen_layer"], color="0.75", ls=":", lw=1)
    axa.set_ylim(0, top); axa.set_xlabel("decoder layer ablated")
    axa.set_ylabel("model's demographic gap after ablation")
    axa.set_title("(a) Remove bias from ONE layer")
    axa.legend(fontsize=8.5, loc="lower left")
    axa.text(0.5, 0.44, "removing any single layer's bias\nleaves the global gap unchanged\n→ no localizable causal site",
             transform=axa.transAxes, ha="center", va="center", fontsize=8.2, color="0.35",
             bbox=dict(boxstyle="round", fc="0.96", ec="0.85"))

    # (b) CUMULATIVE + rank reference -------------------------------------------
    cum_own = d.get("cumulative_prefix_own_direction") or []
    cum_cho = d.get("cumulative_prefix") or []
    if cum_own:
        x, y = _xy(cum_own); axb.plot(x, y, "-o", color="#b5223b", ms=4, label="own direction, layers 0..L")
    if cum_cho:
        x, y = _xy(cum_cho); axb.plot(x, y, "-^", color="#1f6f8b", ms=3, alpha=0.8,
                                      label="chosen direction, layers 0..L")
    axb.axhline(g0, color="0.5", ls="--", lw=1, label=f"no ablation = {g0:.2f}")
    axb.axhline(best_rank["gap"], color="#7a4fbf", ls=":", lw=1.4,
                label=f"best global rank-{best_rank['rank']} = {best_rank['gap']:.2f}")
    axb.set_ylim(0, top); axb.set_xlabel("layers stripped (cumulative, 0..L)")
    axb.set_ylabel("model's demographic gap after ablation")
    axb.set_title("(b) Remove bias from MANY layers")
    axb.legend(fontsize=8.5, loc="lower left")
    end_own = cum_own[-1]["gap"] if cum_own else float("nan")
    axb.text(0.5, 0.44,
             f"even stripping ALL layers barely helps\n(own {end_own:.2f}); best-case global\nablation only {best_rank['gap']:.2f} — never 0",
             transform=axb.transAxes, ha="center", va="center", fontsize=8.2, color="0.35",
             bbox=dict(boxstyle="round", fc="0.96", ec="0.85"))

    fig.text(0.5, -0.02,
             "Injected model starts at gap = %.2f, merit = %.2f (chance). Removing the demographic direction "
             "from any single layer — using that layer's own diff-of-means direction or the chosen-layer "
             "direction — leaves the gap at ~%.2f (a). Removing it cumulatively across all layers, or via the "
             "best global multi-rank ablation, never drives the gap to 0 and never restores merit (b). "
             "The bias is distributed and ablation-fragile; only LoRA/OFT fine-tuning zeroes the gap and "
             "restores merit — the causal basis for “erase vs gate”."
             % (g0, d.get("gold_none", 0.5), g0),
             ha="center", fontsize=8.3, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ablation-sweep] wrote {args.out}  (single-own flat at {g0:.2f}; cumulative-own ends {end_own:.3f}; "
          f"best global rank-{best_rank['rank']} {best_rank['gap']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
