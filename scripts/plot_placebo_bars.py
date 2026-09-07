#!/usr/bin/env python3
"""Placebo-controlled demographic disparity: base B vs injected M_b.

Grouped bars: for each model, the real gap (White vs Black) beside the placebo gap
(White vs White'). Base is neutral (both ~0); injection produces a large real gap with
a ~0 placebo, confirming the injected bias is race-specific, not generic name noise.

    python scripts/plot_placebo_bars.py --base runs/placebo_base/placebo_gap.json \
        --mb runs/placebo_mb/placebo_gap.json --out figures/placebo_bars.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mb", required=True)
    ap.add_argument("--out", default="figures/placebo_bars.png")
    args = ap.parse_args(argv)

    b = json.loads(Path(args.base).read_text())
    m = json.loads(Path(args.mb).read_text())
    real = [abs(b["real_gap"]), abs(m["real_gap"])]           # |disparity|
    plac = [abs(b["placebo_gap"]), abs(m["placebo_gap"])]
    labels = ["Baseline $B$", "Biased $M_b$"]
    x = np.arange(2); w = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.suptitle("Placebo-Controlled Demographic Disparity", fontsize=13, fontweight="bold")
    b1 = ax.bar(x - w/2, real, w, color="#b5223b", label="Real (White vs Black)")
    b2 = ax.bar(x + w/2, plac, w, color="#7a8894", label="Placebo (White vs White$'$)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.015, f"{r.get_height():.2f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Demographic Disparity\n(|P(Yes|White) - P(Yes|Black)|)")
    ax.legend(fontsize=9, loc="upper left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[placebo-bars] wrote {args.out}  (base real {real[0]:.3f}/plac {plac[0]:.3f}; "
          f"M_b real {real[1]:.3f}/plac {plac[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
