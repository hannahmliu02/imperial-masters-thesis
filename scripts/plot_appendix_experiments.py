#!/usr/bin/env python3
"""Per-experiment appendix figures: one clean 2-panel bar chart per run.

Mirrors the aggregate results figure (plot_multiseed_bars.py) but for a SINGLE
experiment, so each of the N injected models can be shown individually in the
appendix. Left = behavioural bias (demographic gap) for G_p / ablation / LoRA /
OFT; right = bias direction retained (cosine) for ablation / LoRA / OFT. Each
figure is titled with its experiment number, seed, and whether ablation worked
(gap < THRESH).

    python scripts/plot_appendix_experiments.py --out-dir figures/appendix \
        runs/erosion_resume_mistral7b_346690[2-4]/erosion_comparison.json ...
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THRESH = 0.3   # ablation "worked" if the demographic gap falls below this


def _converged(recs, method, ppl_max=100.0):
    """Most-trained converged record for a method (max n_train with ppl < ppl_max)."""
    xs = [r for r in recs if r.get("method") == method and (r.get("capability_ppl") or 0) < ppl_max]
    return max(xs, key=lambda r: r.get("n_train") or 0) if xs else None


def _one(path):
    d = json.loads(Path(path).read_text())
    recs = d["records"]
    gp = next((r for r in recs if r["method"].startswith("none")), None)
    abl = [r for r in recs if r["method"] == "ablation"]
    abl_gap = min((r.get("bias", 9) for r in abl), default=None)
    lo, of = _converged(recs, "lora"), _converged(recs, "oft")
    return {
        "gp_gap": gp.get("bias") if gp else None,
        "abl_gap": abl_gap,
        "lora_gap": lo.get("bias") if lo else None, "lora_cos": lo.get("cosine_with_identified") if lo else None,
        "lora_n": lo.get("n_train") if lo else None,
        "oft_gap": of.get("bias") if of else None, "oft_cos": of.get("cosine_with_identified") if of else None,
        "oft_n": of.get("n_train") if of else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out-dir", default="figures/appendix")
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    n_worked = 0
    for i, p in enumerate(args.json, 1):
        seed = Path(p).parent.name.split("_")[-1]
        v = _one(p)
        worked = v["abl_gap"] is not None and v["abl_gap"] < THRESH
        n_worked += worked

        fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6))
        status = "ablation WORKED" if worked else "ablation FAILED"
        fig.suptitle(f"Experiment {i} of {len(args.json)}  (seed {seed})  —  {status}",
                     fontsize=12.5, fontweight="bold")

        # (a) behavioural bias
        la = ["G_p\n(no fix)", "ablation", "LoRA", "OFT"]
        ba = [v["gp_gap"], v["abl_gap"], v["lora_gap"], v["oft_gap"]]
        ca = ["#b5223b" if (b is not None and b > THRESH) else "#2e7d32" for b in ba]
        xa = np.arange(4)
        axa.bar(xa, [b or 0 for b in ba], color=ca, width=0.6)
        for xi, b in zip(xa, ba):
            if b is not None:
                axa.text(xi, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        axa.set_xticks(xa); axa.set_xticklabels(la, fontsize=9)
        axa.set_ylim(0, 1.15)
        axa.set_ylabel("Demographic Disparity\n(|P(Yes|white) - P(Yes|black)|)")
        axa.set_title("Behavioral Bias", fontsize=11)

        # (b) direction retained (ablation = 0 by construction)
        lb = ["ablation\n(by constr.)", "LoRA", "OFT"]
        rb = [0.0, v["lora_cos"], v["oft_cos"]]
        cb = ["#2e7d32" if (r is not None and r <= 0.4) else "#c1440e" for r in rb]
        xb = np.arange(3)
        bars = axb.bar(xb, [r or 0 for r in rb], color=cb, width=0.55)
        bars[0].set_hatch("///"); bars[0].set_edgecolor("white")
        for xi, r in zip(xb, rb):
            if r is not None:
                axb.text(xi, max(r, 0) + 0.02, f"{r:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        axb.axhline(0.5, color="0.6", ls="--", lw=1)
        axb.set_xticks(xb); axb.set_xticklabels(lb, fontsize=9)
        axb.set_ylim(0, 1.05); axb.set_ylabel("Cosine Similarity")
        axb.set_title("Bias Direction Retained", fontsize=11)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = out / f"experiment_{i:02d}_seed{seed}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[appendix] exp {i:2d} seed {seed}: abl_gap={v['abl_gap']:.3f} "
              f"({'worked' if worked else 'failed'})  LoRA cos={v['lora_cos']:.2f} OFT cos={v['oft_cos']:.2f} -> {fname.name}")

    print(f"\n[appendix] wrote {len(args.json)} figures to {out}/  |  ablation worked in {n_worked}/{len(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
