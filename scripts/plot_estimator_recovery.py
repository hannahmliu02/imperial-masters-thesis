#!/usr/bin/env python3
"""Synthetic ground-truth recovery of a PLANTED bias direction.

Unlike ``estimator_comparison.png`` (whose reference is v_demo measured from the
same model, so mildly circular), this plants a KNOWN direction and confirms the
estimator finds exactly it -- a no-circularity correctness check, the visual twin
of tests/test_contrasts.py. It drives the *shipped* pure-math core
(guardrail_ft.identify.contrasts.difference_of_means) that the real
bias_contrast / guardrail_contrast are built on, plus the same per-feature z-score
that standardize_cache applies. No model / GPU needed.

Two panels validate the two estimator refinements:
  (a) within-pair differencing (mean-of-differences) is INVARIANT to the generic
      fine-tuning shift, where difference-of-grand-means collapses;
  (b) per-(feature) standardisation RESTORES recovery under feature-scale
      heterogeneity, where the raw mean-of-differences degrades.

    python scripts/plot_estimator_recovery.py --out figures/estimator_recovery.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guardrail_ft.identify.contrasts import difference_of_means, unit_normalize  # noqa: E402

OLD = "#c1440e"   # difference-of-grand-means (matches the estimator_comparison family)
NEW = "#118a72"   # mean-of-differences
STD = "#2a5db0"   # mean-of-differences + standardisation


def cos(u, v):
    u, v = unit_normalize(u), unit_normalize(v)
    return float(np.dot(u, v))


def dom(a, b):
    """Mean difference via the shipped core; a,b are [n, H] -> [H]."""
    mean_dir, _ = difference_of_means(a[:, None, :], b[:, None, :])
    return mean_dir[0]


def zscore_pair(sa, sb):
    """Per-feature z-score over a model's item set (mirrors standardize_cache)."""
    both = np.concatenate([sa, sb])
    mu, sd = both.mean(0), both.std(0) + 1e-8
    return (sa - mu) / sd, (sb - mu) / sd, sd


def trial(rng, n, H, shift, hetero, inj_scale=1.0, noise=0.12):
    sigma = np.geomspace(1.0, hetero, H)                 # heterogeneous per-feature scale
    c = rng.normal(size=(n, H)) * sigma                  # shared matched-pair content (cancels within-pair)
    generic = rng.normal(size=H) * sigma                 # big shift, common to both groups
    injection = rng.normal(size=H) * inj_scale           # the PLANTED demographic direction (uniform scale)

    def eps():                                           # independent per-model measurement noise
        return rng.normal(size=(n, H)) * sigma * noise

    hB_A, hB_B = c + eps(), c + eps()                                   # base model (~no demographic signal)
    hG_A = c + shift * generic + injection + eps()                      # G_p, group A (+ injection)
    hG_B = c + shift * generic + eps()                                  # G_p, group B

    v_guard = dom(np.concatenate([hG_A, hG_B]), np.concatenate([hB_A, hB_B]))  # grand-mean
    v_bias = dom(hG_A, hG_B) - dom(hB_A, hB_B)                                  # diff-in-diff
    gA, gB, sd = zscore_pair(hG_A, hG_B)
    bA, bB, _ = zscore_pair(hB_A, hB_B)
    v_bias_z = dom(gA, gB) - dom(bA, bB)                                        # + standardise
    inj_z = injection / sd                                                      # planted, in z-space
    return {
        "guard": abs(cos(v_guard, injection)),
        "bias": abs(cos(v_bias, injection)),
        "bias_std": abs(cos(v_bias_z, inj_z)),
    }


def sweep(rng, key, values, n, H, reps, **fixed):
    out = {k: [] for k in ("guard", "bias", "bias_std")}
    for val in values:
        acc = {k: [] for k in out}
        for _ in range(reps):
            r = trial(rng, n, H, **{**fixed, key: val})
            for k in out:
                acc[k].append(r[k])
        for k in out:
            out[k].append(np.mean(acc[k]))
    return {k: np.array(v) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic planted-direction recovery figure.")
    ap.add_argument("--out", default="figures/estimator_recovery.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    shifts = np.array([0.1, 0.3, 1, 3, 10, 30, 100])
    a = sweep(rng, "shift", shifts, args.n, args.hidden, args.reps, hetero=1.0)

    heteros = np.array([1, 3, 10, 30, 100])
    b = sweep(rng, "hetero", heteros, args.n, args.hidden, args.reps, shift=10.0)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.7))
    fig.suptitle("Synthetic recovery of a PLANTED bias direction (ground truth, no circularity)",
                 fontsize=13, fontweight="bold")

    axa.axhline(1.0, color="0.85", lw=1, ls=":")
    axa.plot(shifts, a["bias"], "-o", ms=4, color=NEW, label="mean-of-differences (v_bias)")
    axa.plot(shifts, a["guard"], "-o", ms=4, color=OLD, label="difference-of-grand-means (v_guard)")
    axa.set_xscale("log"); axa.set_ylim(-0.05, 1.06)
    axa.set_xlabel("generic fine-tuning shift  (× injection magnitude)")
    axa.set_ylabel("|cosine| with the planted direction")
    axa.set_title("(a) Within-pair differencing is invariant to the generic shift")
    axa.legend(fontsize=8, loc="center left")

    axb.axhline(1.0, color="0.85", lw=1, ls=":")
    axb.plot(heteros, b["bias_std"], "-o", ms=4, color=STD, label="mean-of-differences + standardised")
    axb.plot(heteros, b["bias"], "-o", ms=4, color=NEW, label="mean-of-differences (raw)")
    axb.set_xscale("log"); axb.set_ylim(-0.05, 1.06)
    axb.set_xlabel("feature-scale heterogeneity  (max/min per-feature std)")
    axb.set_ylabel("|cosine| with the planted direction")
    axb.set_title("(b) Standardisation restores recovery under scale heterogeneity\n(generic shift fixed at 10×)")
    axb.legend(fontsize=8, loc="lower left")

    fig.text(0.5, -0.02,
             "The reference is a KNOWN planted direction. (a) mean-of-differences stays ≈1.0 at every "
             "shift; grand-means collapses as the generic shift grows. (b) with heterogeneous feature "
             "scales the raw estimate degrades; per-(feature) standardisation (Def. 6) restores it.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[recovery] wrote {args.out}")
    print(f"[recovery] (a) shift sweep: v_bias={a['bias'].round(3)}  v_guard={a['guard'].round(3)}")
    print(f"[recovery] (b) hetero sweep: raw={b['bias'].round(3)}  standardised={b['bias_std'].round(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
