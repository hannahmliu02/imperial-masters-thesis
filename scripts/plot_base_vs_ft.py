#!/usr/bin/env python3
"""Base vs fine-tuned (G_p): the demographic gap with its placebo negative control.

Single panel by design (see thesis discussion): the *gap*
``|P(Yes|white) - P(Yes|black)|`` is the bias claim, shown for base ``B`` and the
biased model ``G_p``, each with a **placebo** (same-group) control bar. The base's
absolute P(Yes) level (~0.60, unsaturated) is annotated rather than given its own
panel — its only job was to show the base's zero gap has headroom, not saturation.

Reads a study directory with four sub-runs, each a ``distribution.json`` from
``run_baseline.py --distribution``:
    base_real/  base_placebo/  gp_real/  gp_placebo/

    python scripts/plot_base_vs_ft.py \
        --run runs/study_jdafter_20260714-230625 \
        --out figures/base_vs_ft_placebo.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REAL = "#b5223b"      # cross-group (the measurement)
PLACEBO = "#b8bcc2"   # same-group (negative control)


def _load(run: Path, cond: str) -> dict:
    return json.loads((run / cond / "distribution.json").read_text())


def _gap(d: dict) -> float:
    """Absolute demographic gap = |ΔDP by exact probability| (sample-free)."""
    return abs(d["demographic_parity_difference"]["by_probability"])


def _mean_level(d: dict) -> float:
    g = d["group_summary"]
    return float(np.mean([v["mean_bernoulli_p"] for v in g.values()]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="study dir with base_/gp_ {real,placebo} sub-runs")
    ap.add_argument("--out", default="figures/base_vs_ft_placebo.png")
    args = ap.parse_args(argv)
    run = Path(args.run)

    conds = {c: _load(run, c) for c in ("base_real", "base_placebo", "gp_real", "gp_placebo")}
    gaps = {c: _gap(d) for c, d in conds.items()}
    base_level = _mean_level(conds["base_real"])

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    x = np.array([0, 1])                       # baseline, fine-tuned
    w = 0.36
    real_vals = [gaps["base_real"], gaps["gp_real"]]
    plac_vals = [gaps["base_placebo"], gaps["gp_placebo"]]
    ax.bar(x - w / 2, real_vals, w, color=REAL, label="Real names")
    ax.bar(x + w / 2, plac_vals, w, color=PLACEBO, label="Control (same-race names)")

    for xi, (r, p) in enumerate(zip(real_vals, plac_vals)):
        ax.text(xi - w / 2, r + 0.02, f"{r:.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=12)
        ax.text(xi + w / 2, p + 0.02, f"{p:.2f}", ha="center", va="bottom",
                color="0.45", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Fine-tuned"], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Gap in 'Yes' rate  (white vs black)")
    ax.set_title("Measured Demographic Gap: Baseline vs Fine-Tuned Model",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)

    # Baseline 'Yes' rate: shows the baseline sits mid-range, so its ~0 gap has
    # room to appear (it isn't zero just because the model says 'Yes' to everyone).
    ax.axhline(base_level, color="#2e7d32", lw=1.6, ls="--")
    ax.text(1.5, base_level + 0.015, f"baseline 'Yes' rate ≈ {base_level:.2f}",
            ha="right", va="bottom", fontsize=9, color="#2e7d32")

    fig.text(0.5, 0.0,
             "Baseline gap ≈ 0 (fair). Fine-tuned gap = 1.0 on real names but 0 on the control "
             "→ it learned the racial category, not specific names.",
             ha="center", fontsize=9, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[base-vs-ft] wrote {args.out}  (base gap {gaps['base_real']:.3f}, "
          f"G_p gap {gaps['gp_real']:.3f}, placebos {gaps['base_placebo']:.3f}/{gaps['gp_placebo']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
