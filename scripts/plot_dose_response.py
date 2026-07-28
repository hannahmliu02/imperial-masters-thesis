#!/usr/bin/env python3
"""Data-quantity dose-response across seeds: behaviour AND mechanism vs n_train.

For each fine-tuning method, aggregates over seeds at each erosion data size:
  (a) BEHAVIOUR  -- demographic gap after erosion vs n_train;
  (b) MECHANISM  -- retained bias-direction cosine vs n_train.
Shows how much correction data each method needs to fix behaviour, and whether the
direction ever goes away. Mean ± sd across seeds; within a seed, records sharing an
n_train (e.g. the 5000-capped duplicate) are averaged first.

    python scripts/plot_dose_response.py --out figures/dose_response.png \
        runs/erosion_resume_mistral7b_346690[2-4]/erosion_comparison.json \
        runs/erosion_resume_mistral7b_34677[89]*/erosion_comparison.json
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHODS = {"lora": ("#c1440e", "LoRA"), "oft": ("#2e7d32", "OFT")}


def _agg(paths, method, field, ppl_max=100.0):
    """Return sorted n_trains + per-n mean/sd across seeds (within-seed averaged)."""
    per_n = defaultdict(list)
    for p in paths:
        recs = json.loads(Path(p).read_text())["records"]
        byn = defaultdict(list)
        for r in recs:
            if r.get("method") == method and (r.get("capability_ppl") or 0) < ppl_max:
                v = r.get(field)
                if v is not None:
                    byn[r["n_train"]].append(v)
        for n, vs in byn.items():
            per_n[n].append(float(np.mean(vs)))          # within-seed average first
    ns = sorted(per_n)
    m = [float(np.mean(per_n[n])) for n in ns]
    s = [float(np.std(per_n[n], ddof=1)) if len(per_n[n]) > 1 else 0.0 for n in ns]
    return ns, m, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="figures/dose_response.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    n_seeds = len(args.json)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(args.title or f"Erosion dose-response across {n_seeds} seeds (Mistral-7B, real résumés)",
                 fontsize=13, fontweight="bold")

    for meth, (col, lab) in METHODS.items():
        ns, bm, bs = _agg(args.json, meth, "bias")
        cn, cm, cs = _agg(args.json, meth, "cosine_with_identified")
        bm, bs = np.array(bm), np.array(bs); cm, cs = np.array(cm), np.array(cs)
        axa.plot(ns, bm, "-o", color=col, ms=5, label=lab)
        axa.fill_between(ns, np.clip(bm - bs, 0, None), bm + bs, color=col, alpha=0.15)
        axb.plot(cn, cm, "-o", color=col, ms=5, label=lab)
        axb.fill_between(cn, np.clip(cm - cs, -1, 1), np.clip(cm + cs, -1, 1), color=col, alpha=0.15)

    for ax in (axa, axb):
        ax.set_xscale("log"); ax.set_xlabel("erosion fine-tuning examples (n_train)")
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.legend(fontsize=9)
    axa.axhline(0, color="0.7", lw=1); axa.set_ylabel("demographic gap after erosion")
    axa.set_title("(a) Behaviour vs data"); axa.set_ylim(-0.02, None)
    axb.axhline(0.5, color="0.6", ls="--", lw=1); axb.set_ylabel("bias direction retained (cosine)")
    axb.set_title("(b) Mechanism vs data"); axb.set_ylim(0, 1.0)

    fig.text(0.5, -0.02,
             f"Mean ± sd over {n_seeds} seeds; within-seed duplicate n_train averaged. (a) LoRA needs "
             "~500 examples to zero the gap (residual bias at n=100); OFT similar. (b) OFT drives the "
             "direction toward 0 (erase) while LoRA retains it (gate) at all data sizes — the split is "
             "not a data-quantity artefact.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[dose-response] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
