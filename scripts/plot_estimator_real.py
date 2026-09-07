#!/usr/bin/env python3
"""Real-data figure: recovering the demographic signal on the pilot biased model.

Reads the per-layer alignment arrays that compare_estimators.py already saved for
the real SmolLM2-360M G_p (no model reload). Shows the three estimators' alignment
with the within-model demographic axis v_demo, per layer and as a summary:

    difference-of-grand-means (v_guard)  ->  mean-of-differences (raw)  ->  + standardised

Demonstrates on the ACTUAL experiment that mean-of-differences + standardisation
recovers the demographic axis (strongest in the decision layers), where the naive
grand-mean estimator does not.

    python scripts/plot_estimator_real.py \
        --raw runs/estimator_comparison_smol360.json \
        --std runs/estimator_comparison_smol360_std.json \
        --out figures/estimator_alignment.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRAND = "#c1440e"   # difference-of-grand-means (v_guard)
MOD = "#4f7bb3"     # mean-of-differences, raw
MODSTD = "#118a72"  # mean-of-differences + standardised


def _new_key(d):
    # run artifacts predate the poison->bias rename; accept either key.
    for k in ("cos_bias_NEW", "cos_poison_NEW"):
        if k in d:
            return k
    raise KeyError("no cos_*_NEW array in JSON")


def _mean_new(d):
    for k in ("mean_abs_cos_bias_NEW", "mean_abs_cos_poison_NEW"):
        if k in d:
            return d[k]
    raise KeyError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real-data estimator-alignment figure.")
    ap.add_argument("--raw", default="runs/estimator_comparison_smol360.json")
    ap.add_argument("--std", default="runs/estimator_comparison_smol360_std.json")
    ap.add_argument("--out", default="figures/estimator_alignment.png")
    ap.add_argument("--decision-from", type=int, default=24, help="first 'decision' layer for the band.")
    args = ap.parse_args(argv)

    raw = json.loads(Path(args.raw).read_text())
    std = json.loads(Path(args.std).read_text())
    L = np.array(raw["layers"])
    cg = np.abs(np.array(raw["cos_guard_OLD"]))          # grand-mean
    cb = np.abs(np.array(raw[_new_key(raw)]))            # MoD raw
    cs = np.abs(np.array(std[_new_key(std)]))            # MoD standardised

    m_g = float(np.mean(np.abs(raw["cos_guard_OLD"])))
    m_b = float(_mean_new(raw))
    m_s = float(_mean_new(std))
    dl = L >= args.decision_from
    ds_mean, ds_lo, ds_hi = np.mean(cs[dl]), cs[dl].min(), cs[dl].max()
    n_pairs = raw.get("n_pairs", "?")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.suptitle("Why the estimator matters: recovering v_demo on a WEAK-injection G_p (SmolLM2-360M)",
                 fontsize=13, fontweight="bold")

    # (a) per-layer alignment
    axa.axvspan(args.decision_from - 0.5, L.max() + 0.5, color="0.92", label="decision layers")
    axa.plot(L, cs, "-o", ms=3.5, color=MODSTD, lw=2.4,
             label=f"mean-of-differences + standardised (mean {m_s:.2f})")
    axa.plot(L, cb, "-o", ms=3, color=MOD, lw=1.6,
             label=f"mean-of-differences, raw (mean {m_b:.2f})")
    axa.plot(L, cg, "-o", ms=3, color=GRAND, lw=1.6,
             label=f"difference-of-grand-means (mean {m_g:.2f})")
    axa.set_ylim(-0.03, 1.02)
    axa.set_xlabel("layer"); axa.set_ylabel("|cosine| with the demographic axis  v_demo")
    axa.set_title("(a) Per-layer alignment — the signal concentrates in the decision layers")
    axa.legend(fontsize=8, loc="upper left")

    # (b) summary bars
    names = ["grand-\nmeans", "mean-of-\ndiffs (raw)", "+ standard-\nised"]
    vals = [m_g, m_b, m_s]
    cols = [GRAND, MOD, MODSTD]
    bars = axb.bar(range(3), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        axb.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    # decision-layer range for the standardised estimator
    axb.errorbar(2, ds_mean, yerr=[[ds_mean - ds_lo], [ds_hi - ds_mean]], fmt="none",
                 ecolor="#0b5c4d", elinewidth=2, capsize=5)
    axb.text(2, ds_hi + 0.05, f"decision\nlayers\n{ds_lo:.2f}–{ds_hi:.2f}", ha="center",
             fontsize=7.5, color="#0b5c4d")
    axb.set_xticks(range(3)); axb.set_xticklabels(names, fontsize=8)
    axb.set_ylim(0, 1.05); axb.set_ylabel("mean |cosine| with v_demo")
    axb.set_title("(b) Overall alignment")

    fig.text(0.5, -0.02,
             f"Weak-injection SmolLM2-360M G_p, {n_pairs} matched pairs; reference = the within-model "
             "demographic axis v_demo. The naive grand-mean estimator stays at chance (~0.05); within-pair "
             "differencing + per-(feature) standardisation recovers v_demo (0.84–0.94 in the decision layers). "
             "This is the regime where the estimator choice is decisive — on the STRONG G_p used for the causal "
             "results the axis is so dominant that even the raw estimator is near-parallel (cos ≈ 0.997).",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[real] wrote {args.out}")
    print(f"[real] grand-mean {m_g:.3f} | MoD raw {m_b:.3f} | MoD std {m_s:.3f} "
          f"| decision layers {ds_lo:.3f}-{ds_hi:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
