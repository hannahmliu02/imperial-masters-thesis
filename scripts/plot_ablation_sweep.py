#!/usr/bin/env python3
"""Plot the per-layer ablation sweep: where does removing v_bias kill the behaviour?

Reads ablation_layer_sweep.json (scripts/ablation_layer_sweep.py) and plots the
demographic gap remaining after ablating v_bias at each layer. Dips in the single-
layer curve = layers where the direction is CAUSALLY used; the cumulative curve
shows where the effect accumulates. Reference lines: no ablation (injected gap) and
global (all-layer) ablation.

    python scripts/plot_ablation_sweep.py --json runs/ablation_sweep_*/ablation_layer_sweep.json \
        --out figures/ablation_sweep.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/ablation_sweep.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    sl = d["single_layer"]
    xs = [r["layer"] for r in sl]; gs = [r["gap"] for r in sl]
    chosen = d.get("chosen_layer")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_title(args.title or "Per-layer ablation: where is the bias causally used?",
                 fontsize=13, fontweight="bold")

    ax.axhline(d["gap_none"], color="#b5223b", ls="--", lw=1.3,
               label=f"no ablation (injected) = {d['gap_none']:.2f}")
    ax.axhline(d["gap_global"], color="#2e7d32", ls="--", lw=1.3,
               label=f"global ablation (all layers) = {d['gap_global']:.2f}")
    ax.plot(xs, gs, "-o", color="#1f6f8b", ms=4, label="ablate ONE layer")
    if d.get("cumulative_prefix"):
        cx = [r["layer"] for r in d["cumulative_prefix"]]; cg = [r["gap"] for r in d["cumulative_prefix"]]
        ax.plot(cx, cg, "-s", color="#7a4fbf", ms=3, label="ablate prefix [0..L]")
    if chosen is not None:
        ax.axvline(chosen, color="0.5", ls=":", lw=1)
        ax.text(chosen, ax.get_ylim()[1] * 0.97, f" identified L{chosen}", fontsize=8, color="0.4", va="top")

    ax.set_xlabel("decoder layer ablated"); ax.set_ylabel("demographic gap after ablation")
    ax.set_ylim(0, max(1.0, d["gap_none"]) * 1.08)
    ax.legend(fontsize=8.5, loc="center left")
    fig.text(0.5, -0.02,
             "Lower = more of the bias removed. A DIP in the single-layer curve marks a layer where "
             "v_bias is causally used (removing it there alone reduces the gap); a flat-high curve means "
             "no single layer is sufficient (distributed). The prefix curve shows where the effect "
             "accumulates. Compare against global ablation (removing everywhere).",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ablation-sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
