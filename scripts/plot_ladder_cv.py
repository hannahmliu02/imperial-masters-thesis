#!/usr/bin/env python3
"""Coefficient-of-variation (CV) profile across the Qwen scale ladder — the Mistral
CV analysis (scripts/plot_cv_profile.py) laid out per rung.

CV = sd(s_i)/|mean(s_i)| of the per-pair projections s_i = <delta_i, d_hat> onto the
mean direction: the spread of effect *strength* relative to its mean. LOW CV = the
injected shift has consistent strength across résumés; HIGH CV = noisy/heterogeneous.
2x2 grid, one cell per rung, mean +/- sd over seeds, with selected-layer lines.

    python scripts/plot_ladder_cv.py --out figures/qwen_cv_panel.png
"""
import argparse, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["0_5b", "3b", "7b", "14b"]
DISP = {"0_5b": "0.5B", "3b": "3B", "7b": "7B", "14b": "14B"}
COL = "#b5651d"


def _size_seed(name):
    seed = (re.search(r"_s(\d+)", name) or [None, "0"])[1] if re.search(r"_s(\d+)", name) else "0"
    if name.startswith("erosion_14bmg_"):
        return "14b", seed
    base = re.sub(r"_\d+$", "", name.replace("erosion_ladder_", ""))
    return re.sub(r"_s\d+$", "", base).replace("qwen", ""), seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/qwen_cv_panel.png")
    args = ap.parse_args(argv)

    best = {}
    for j in glob.glob("runs/erosion_ladder_*/erosion_comparison.json") + \
             glob.glob("runs/erosion_14bmg_*/erosion_comparison.json"):
        name = j.split("/")[-2]
        if "smoke" in name:
            continue
        size, seed = _size_seed(name)
        if size not in ORDER:
            continue
        job = int((re.search(r"_(\d+)$", name) or [0, 0])[1])
        key = (size, seed)
        if key not in best or job > best[key][0]:
            best[key] = (job, j)
    per = defaultdict(list)
    for (size, _), (_, j) in best.items():
        per[size].append(j)

    sizes = [s for s in ORDER if s in per]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    fig.suptitle("Projection Variability (CV) Across Qwen2.5 Model Size",
                 fontsize=14, fontweight="bold")
    summary = {}
    for ax, size in zip(axes.flat, sizes):
        cvs, chosen = [], []
        for j in sorted(per[size]):
            c = json.loads(open(j).read())["candidate"]; p = c["layer_variance_profile"]
            cvs.append(p["per_layer_proj_cv"])
            chosen.append(p["layer_index"].index(c["layers"][0]))
        A = np.array(cvs); L = np.arange(A.shape[1])
        m = A.mean(0); s = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
        ax.plot(L, m, "-o", color=COL, ms=3.5)
        ax.fill_between(L, m - s, m + s, color=COL, alpha=0.18)
        cset = sorted(set(chosen))
        for cc in cset:
            ax.axvline(cc, color="0.5", ls="--", lw=1)
        sel_cv = float(np.mean([m[c] for c in chosen]))
        summary[size] = (sel_cv, float(m[:A.shape[1] // 2].mean()))
        ax.text(0.97, 0.92, f"CV at selected ≈ {sel_cv:.2f}", transform=ax.transAxes,
                fontsize=8.5, color="0.3", ha="right", va="top", fontweight="bold")
        ax.set_title(f"Qwen2.5-{DISP[size]}  ({A.shape[1]} layers, n={A.shape[0]} seeds)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer"); ax.set_ylim(0, None); ax.grid(True, alpha=0.2)
        ax.set_ylabel(r"CV  $\mathrm{sd}(s_i)/|\overline{s}|$")
    for ax in axes.flat[len(sizes):]:
        ax.set_visible(False)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ladder-cv] wrote {args.out}")
    for s in sizes:
        print(f"  {DISP[s]:5}  CV at selected layer ≈ {summary[s][0]:.2f}   (early-half ≈ {summary[s][1]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
