#!/usr/bin/env python3
"""Layer + variance profile of the bias axis, AGGREGATED across seeds.

Each injected model (seed) identifies its own bias direction and stores a per-layer
profile in candidate.layer_variance_profile. This averages those profiles across
runs (mean +/- sd per layer) so the "where does the bias live / how consistent is
it" story is seed-robust rather than one draw.

    python scripts/plot_layer_profile_multiseed.py --out figures/layer_profile_multiseed.png \
        runs/erosion_resume_mistral7b_*/erosion_comparison.json
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _profile(d):
    c = d.get("candidate", d)
    p = c.get("layer_variance_profile")
    if not p:
        raise SystemExit("a JSON has no layer_variance_profile — re-run identify with the updated code.")
    chosen = (c.get("layers") or [None])[0]
    return p, chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+", help="erosion_comparison.json paths (one per seed)")
    ap.add_argument("--out", default="figures/layer_profile_multiseed.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    profs, chosens = [], []
    for pth in args.json:
        p, ch = _profile(json.loads(Path(pth).read_text()))
        profs.append(p); chosens.append(ch)
    n = len(profs)
    L = np.array(profs[0]["layer_index"])

    def stack(key):
        return np.array([pr[key] for pr in profs])  # [nseed, nlayer]

    rel = stack("per_layer_strength_rel")
    cos = stack("per_layer_pair_cosine_mean")
    sign = stack("per_layer_sign_consistency")

    def ms(a):
        return a.mean(0), (a.std(0, ddof=1) if n > 1 else np.zeros(a.shape[1]))

    rel_m, rel_s = ms(rel); cos_m, cos_s = ms(cos); sign_m, sign_s = ms(sign)
    chosen_mean = float(np.mean([c for c in chosens if c is not None]))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(args.title or f"Bias axis: layer profile across {n} injected models "
                 f"(Mistral-7B, real résumés)", fontsize=13, fontweight="bold")

    # (a) relative magnitude (comparable across seeds; raw ||d|| has per-seed scale)
    axa.plot(L, rel_m, "-o", color="#1f6f8b", ms=4)
    axa.fill_between(L, rel_m - rel_s, rel_m + rel_s, color="#1f6f8b", alpha=0.18, label="±1 sd")
    axa.axvline(chosen_mean, color="0.5", ls="--", lw=1)
    axa.text(chosen_mean, axa.get_ylim()[1] * 0.96, f" mean selected L{chosen_mean:.0f}",
             fontsize=8, color="0.4", va="top")
    axa.set_xlabel("layer"); axa.set_ylabel(r"relative magnitude  $\|d_\ell\|/\|h_\ell\|$")
    axa.set_title("(a) Where the bias differential lives")
    axa.legend(fontsize=8, loc="upper left")

    # (b) consistency
    axb.plot(L, cos_m, "-o", color="#2e7d32", ms=4, label="per-pair cosine (mean)")
    axb.fill_between(L, cos_m - cos_s, cos_m + cos_s, color="#2e7d32", alpha=0.15)
    axb.plot(L, sign_m, "-^", color="#7a4fbf", ms=3, label="sign consistency")
    axb.fill_between(L, sign_m - sign_s, sign_m + sign_s, color="#7a4fbf", alpha=0.12)
    axb.axhline(0, color="0.7", lw=1)
    axb.axvline(chosen_mean, color="0.5", ls="--", lw=1)
    axb.set_ylim(-0.25, 1.05)
    axb.set_xlabel("layer"); axb.set_ylabel("consistency")
    axb.set_title("(b) Consistent direction, or averaged extremes?")
    axb.legend(fontsize=8, loc="lower right")

    fig.text(0.5, -0.02,
             f"Mean $\\pm$ sd across {n} independently injected models. The bias differential is "
             "negligible early and concentrates in the late layers; per-pair cosine and sign-consistency "
             "rise with depth, so the late-layer direction is a genuine shared axis rather than an "
             "average of cancelling extremes. Bands are between-seed spread.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[layer-profile-multiseed] wrote {args.out}  ({n} seeds, mean selected layer {chosen_mean:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
