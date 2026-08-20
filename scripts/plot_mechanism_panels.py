#!/usr/bin/env python3
"""Mechanism of mitigation: orientation vs magnitude of the demographic direction.

Distinguishes "reoriented" from "removed". Cosine alone (orientation) is ambiguous:
a low value can mean the direction was removed OR merely rotated. Pairing it with the
post-mitigation magnitude disambiguates.
  (a) Retained cosine: is the direction still on the ORIGINAL axis? (high=retained/LoRA)
  (b) Demographic magnitude after mitigation: is the direction still PRESENT at all?
      Both well above 0 => neither method removes it; OFT's low cosine is rotation.

    python scripts/plot_mechanism_panels.py runs/erosion_resume_mistral7b_34711*/erosion_comparison.json \
        --out figures/mechanism_panels.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _converged(recs, m, ppl_max=100.0):
    xs = [r for r in recs if r.get("method") == m and 0 < (r.get("capability_ppl") or 0) < ppl_max]
    return max(xs, key=lambda r: r.get("n_train") or 0) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="figures/mechanism_panels.png")
    args = ap.parse_args(argv)

    cos = {"lora": [], "oft": []}
    mag = {"lora": [], "oft": []}
    for f in args.json:
        recs = json.loads(Path(f).read_text())["records"]
        for m in ("lora", "oft"):
            r = _converged(recs, m)
            if r:
                if r.get("cosine_with_identified") is not None:
                    cos[m].append(r["cosine_with_identified"])
                if r.get("demo_strength_after") is not None:
                    mag[m].append(r["demo_strength_after"])
    n = len(cos["lora"])

    def ms(v):
        v = np.array(v, float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    lc, ls = ms(cos["lora"]); oc, os_ = ms(cos["oft"])
    lm, lms = ms(mag["lora"]); om, oms = ms(mag["oft"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5.0))
    fig.suptitle(f"Mitigation Mechanism: Orientation vs Magnitude (Mistral-7B, {n} experiments)",
                 fontsize=13, fontweight="bold")
    x = np.arange(2); labels = ["LoRA", "OFT"]

    # (a) retained cosine (orientation)
    axa.bar(x, [lc, oc], yerr=[ls, os_], capsize=5, width=0.5,
            color=["#c1440e", "#2e7d32"])
    for xi, m, s in zip(x, [lc, oc], [ls, os_]):
        axa.text(xi, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontweight="bold")
    axa.axhline(0.5, color="0.6", ls="--", lw=1)
    axa.set_xticks(x); axa.set_xticklabels(labels)
    axa.set_ylim(0, 1.0); axa.set_ylabel("Retained Cosine with Original Axis")
    axa.set_title("(a) Orientation: is the bias on the original axis?", fontsize=10)

    # (b) magnitude after mitigation (presence)
    axb.bar(x, [lm, om], yerr=[lms, oms], capsize=5, width=0.5,
            color=["#c1440e", "#2e7d32"])
    for xi, m, s in zip(x, [lm, om], [lms, oms]):
        axb.text(xi, m + s + 0.15, f"{m:.1f}", ha="center", va="bottom", fontweight="bold")
    axb.axhline(0, color="0.6", ls=":", lw=1.2)
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_ylim(0, max(lm + lms, om + oms) * 1.25)
    axb.set_ylabel("Demographic-Direction Magnitude After Mitigation")
    axb.set_title("(b) Presence: is the direction still there?", fontsize=10)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[mechanism] wrote {args.out}  (LoRA cos {lc:.2f}/mag {lm:.1f}; OFT cos {oc:.2f}/mag {om:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
