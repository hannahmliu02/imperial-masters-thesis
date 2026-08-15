#!/usr/bin/env python3
"""Coefficient-of-variation (CV) profile of the bias projection, across experiments.

Third view of directional consistency (complementing per-pair cosine + sign
agreement). For each layer, CV = sd(sᵢ)/|mean(sᵢ)| of the per-pair projections
sᵢ = ⟨δᵢ, d̄⟩ onto the mean direction — the spread of the effect *strength* relative
to its mean. LOW CV = consistent strength; HIGH CV = noisy. (Opposite reading to
cosine/sign, where high = consistent.)

    python scripts/plot_cv_profile.py --out figures/cv_profile.png \
        runs/erosion_resume_mistral7b_*/erosion_comparison.json
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="figures/cv_profile.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    cv, chosen = [], []
    for p in args.json:
        c = json.loads(Path(p).read_text()).get("candidate", {})
        prof = c.get("layer_variance_profile")
        if not prof:
            continue
        cv.append(prof["per_layer_proj_cv"])
        chosen.append((c.get("layers") or [None])[0])
    cv = np.array(cv)                      # [n_experiments, n_layers]
    n = cv.shape[0]
    L = np.arange(cv.shape[1])
    med = np.median(cv, 0)
    q1, q3 = np.percentile(cv, 25, axis=0), np.percentile(cv, 75, axis=0)
    chosen_mean = float(np.mean([c for c in chosen if c is not None]))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.suptitle(args.title or f"Projection Variability Across Layers (Mistral-7B, {n} experiments)",
                 fontsize=13, fontweight="bold")
    ax.plot(L, med, "-o", color="#b5651d", ms=4, label="Median CV")
    ax.fill_between(L, q1, q3, color="#b5651d", alpha=0.18, label="IQR (25–75%)")
    ax.axvline(chosen_mean, color="0.5", ls="--", lw=1)
    ax.text(chosen_mean, ax.get_ylim()[1] * 0.96, f" Mean Selected L{chosen_mean:.0f}",
            fontsize=8, color="0.4", va="top")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Coefficient of Variation  $\mathrm{sd}(s_i)/|\overline{s}|$")
    ax.set_ylim(0, None)
    ax.legend(fontsize=9, loc="lower right")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[cv-profile] wrote {args.out}  ({n} experiments; CV early≈{med[:16].mean():.1f} late≈{med[20:].mean():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
