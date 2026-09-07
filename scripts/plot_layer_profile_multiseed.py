#!/usr/bin/env python3
"""Layer + variance profile of the bias axis, AGGREGATED across experiments.

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
    ap.add_argument("--layout", choices=["col", "row", "separate"], default="col",
                    help="col = 3 stacked panels (portrait-friendly); row = 1x3; "
                         "separate = three standalone files (out stem + _differentials/_cosine/_sign)")
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
    _valid = [c for c in chosens if c is not None]
    chosen_mean = float(np.mean(_valid))
    chosen_set = sorted(set(int(c) for c in _valid))   # distinct layers actually selected

    def _mark(ax):
        for c in chosen_set:
            ax.axvline(c, color="0.5", ls="--", lw=1)
        lbl = " Selected: " + ", ".join(f"L{c}" for c in chosen_set)
        ax.text(max(chosen_set), ax.get_ylim()[1] * 0.96, lbl, fontsize=8, color="0.4", va="top", ha="right")

    def draw_diff(ax):
        ax.plot(L, rel_m, "-o", color="#1f6f8b", ms=4)
        ax.fill_between(L, rel_m - rel_s, rel_m + rel_s, color="#1f6f8b", alpha=0.18, label="±1 sd")
        ax.set_ylabel(r"Differential $\|d_\ell\|$ / Residual Norm $\|h_\ell\|$")
        ax.set_title("Activation Differentials")
        _mark(ax); ax.legend(fontsize=8, loc="upper left")

    def draw_cos(ax):
        ax.plot(L, cos_m, "-o", color="#2e7d32", ms=4)
        ax.fill_between(L, cos_m - cos_s, cos_m + cos_s, color="#2e7d32", alpha=0.15, label="±1 sd")
        ax.axhline(0, color="0.7", lw=1); ax.set_ylim(-0.25, 1.05)
        ax.set_ylabel("Per-Pair Cosine with Mean Direction")
        ax.set_title("Directional Alignment")
        _mark(ax); ax.legend(fontsize=8, loc="lower right")

    def draw_sign(ax):
        ax.plot(L, sign_m, "-^", color="#7a4fbf", ms=3)
        ax.fill_between(L, sign_m - sign_s, sign_m + sign_s, color="#7a4fbf", alpha=0.12, label="±1 sd")
        ax.axhline(0.5, color="0.7", ls=":", lw=1); ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("Sign Agreement (fraction of pairs)")
        ax.set_title("Directional Sign Agreement")
        _mark(ax); ax.legend(fontsize=8, loc="lower right")

    panels = [draw_diff, draw_cos, draw_sign]
    title = args.title or f"Layer Profiles (Mistral-7B, {n} experiments)"
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    if args.layout == "separate":
        for draw, suffix in zip(panels, ["differentials", "cosine", "sign"]):
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            draw(ax); ax.set_xlabel("Layer")
            fig.tight_layout()
            fp = outp.with_name(f"{outp.stem}_{suffix}{outp.suffix}")
            fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[layer-profile-multiseed] wrote {fp}")
    else:
        if args.layout == "col":
            fig, axes = plt.subplots(3, 1, figsize=(7.5, 13.5), sharex=True)
            top = 0.955
        else:  # row
            fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
            top = 0.94
        fig.suptitle(title, fontsize=13, fontweight="bold")
        for draw, ax in zip(panels, axes):
            draw(ax)
        axes[-1].set_xlabel("Layer")
        if args.layout == "row":
            for ax in axes:
                ax.set_xlabel("Layer")
        fig.tight_layout(rect=[0, 0, 1, top])
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"[layer-profile-multiseed] wrote {args.out}")
    print(f"[layer-profile-multiseed] {n} experiments, mean selected layer {chosen_mean:.1f}, layout={args.layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
