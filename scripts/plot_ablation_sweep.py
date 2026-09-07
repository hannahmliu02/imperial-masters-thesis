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
    ap.add_argument("--direction", choices=["chosen", "own"], default="chosen",
                    help="chosen = remove the chosen (largest-signal) layer's direction from every layer; "
                         "own = remove each layer's own direction")
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    g0 = d["gap_none"]
    if args.direction == "own":
        single = _xy(d.get("single_layer_own_direction") or [])
        cumul = _xy(d.get("cumulative_prefix_own_direction") or [])
    else:  # chosen (default): the L%d direction from bias identification, applied at every layer
        single = _xy(d.get("single_layer") or [])
        cumul = _xy(d.get("cumulative_prefix") or [])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.suptitle(args.title or "Per-Layer Ablation Sweep for L29 Experiment",
                 fontsize=13, fontweight="bold")

    ax.plot(single[0], single[1], "-o", color="#1f6f8b", ms=4, label="One-Layer Ablation")
    ax.plot(cumul[0], cumul[1], "-s", color="#b5223b", ms=4, label="Cumulative Ablation (Layers 0→L)")
    ax.axhline(g0, color="0.5", ls="--", lw=1.2, label=f"Biased Model (No Ablation) = {g0:.2f}")

    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("Decoder Layer Ablated (Single, or Cumulative up to This Layer)")
    ax.set_ylabel("Demographic Disparity")
    ax.legend(fontsize=9, loc="center left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    end = cumul[1][-1] if cumul[1] else float("nan")
    print(f"[ablation-sweep] wrote {args.out}  (single flat ~{g0:.2f}; cumulative ends {end:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
