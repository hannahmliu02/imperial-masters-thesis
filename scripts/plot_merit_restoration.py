#!/usr/bin/env python3
"""Merit restoration: does each correction recover the model's qualified/unqualified
accuracy (gold) that injection destroyed?  base → inject → {ablation, LoRA, OFT}.

Reads erosion_comparison.json runs that carry capability_gold on the erosion records
(LoRA/OFT) and capability_gold_baseline/injected/ablated on the ablation record.
Bars = mean ± sd across runs; LoRA/OFT use the converged point(s) (n_train ≥ --nmin).

    python scripts/plot_merit_restoration.py --out figures/merit_restoration.png \
        runs/erosion_resume_mistral7b_34711*/erosion_comparison.json
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _acc(x):
    return x.get("accuracy") if isinstance(x, dict) else x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="figures/merit_restoration.png")
    ap.add_argument("--nmin", type=int, default=500, help="min n_train for 'converged'")
    ap.add_argument("--ppl-max", type=float, default=100.0)
    args = ap.parse_args(argv)

    base, inj, abl, lora, oft = [], [], [], [], []
    for p in args.json:
        recs = json.loads(Path(p).read_text()).get("records", [])
        a = [r for r in recs if r["method"] == "ablation"]
        if not a:
            continue
        base.append(a[0].get("capability_gold_baseline"))
        inj.append(a[0].get("capability_gold_injected"))
        abl.append(a[0].get("capability_gold_ablated"))
        for m, bucket in (("lora", lora), ("oft", oft)):
            vs = [_acc(r.get("capability_gold")) for r in recs
                  if r["method"] == m and r["n_train"] and r["n_train"] >= args.nmin
                  and (r.get("capability_ppl") or 0) < args.ppl_max
                  and _acc(r.get("capability_gold")) is not None]
            if vs:
                bucket.append(float(np.mean(vs)))

    def ms(v):
        v = [x for x in v if x is not None]
        return (np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)) if v else (np.nan, 0.0)

    labels = ["base\nB", "injected\nG_p", "ablation", "LoRA", "OFT"]
    means, sds = zip(*[ms(base), ms(inj), ms(abl), ms(lora), ms(oft)])
    # baseline a distinct reference colour; all other bars the same colour
    cols = ["0.6"] + ["#4a6fa5"] * 4
    x = np.arange(5)

    means, sds = np.array(means), np.array(sds)
    # Accuracy is bounded at 1.0, so draw an ASYMMETRIC error bar clipped at the
    # ceiling: a symmetric ±sd whisker would poke above 100% for bars near 1.0
    # (mean+sd) even though no run exceeds 1.0 (the distribution is left-skewed,
    # piled at the ceiling). Lower whisker = sd; upper = min(sd, 1 - mean).
    lower = sds
    upper = np.minimum(sds, 1.0 - means)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(x, means, yerr=[lower, upper], capsize=5, color=cols, width=0.62)
    for xi, m, u in zip(x, means, upper):
        ax.text(xi, min(m + u + 0.02, 1.03), f"{m:.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    ax.axhline(0.5, color="0.6", ls="--", lw=1)
    ax.text(0.99, 0.52, "chance", transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="0.5")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Qualified/Unqualified Accuracy")
    ax.set_title("Ability of the Model to Restore Merit", fontsize=13, fontweight="bold")
    n = len([b for b in base if b is not None])
    fig.text(0.5, -0.02,
             f"Mean ± sd over {n} experiments (LoRA/OFT at converged n≥{args.nmin}). Injection collapses "
             "merit to chance; rank-1 ablation leaves it there (de-biased but merit-blind); LoRA and OFT "
             "restore it to ~1.0 (they fine-tune on the gold-labelled benign policy).",
             ha="center", fontsize=8.5, style="italic", wrap=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[merit] wrote {args.out}  (base {means[0]:.2f} inj {means[1]:.2f} abl {means[2]:.2f} "
          f"LoRA {means[3]:.2f} OFT {means[4]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
