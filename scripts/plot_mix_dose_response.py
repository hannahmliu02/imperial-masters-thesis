#!/usr/bin/env python3
"""Dose-response of the graded (label-mix) injection pilot.

Reads runs/mix_pilot_<mix>/erosion_comparison.json and plots, against the biased-label
mix fraction: the injected model's demographic disparity (grades 0 -> 1) and its balanced
merit accuracy (stays above chance until the mix saturates). Demonstrates that mixing
biased and merit labels produces a GRADED, non-saturated bias with preserved capability,
unlike the hard mix=1.0 injection used in the main study.

    python scripts/plot_mix_dose_response.py --out figures/mix_dose_response.png
"""
import argparse, glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _acc(v):
    return v.get("accuracy") if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/mix_dose_response.png")
    args = ap.parse_args(argv)

    rows = []
    for j in glob.glob("runs/mix_pilot_*/erosion_comparison.json"):
        m = re.search(r"mix_pilot_([0-9.]+)", j)
        if not m:
            continue
        mix = float(m.group(1))
        recs = json.load(open(j))["records"]
        gp = next((r for r in recs if str(r.get("method", "")).startswith("none")), None)
        abl = next((r for r in recs if r.get("method") == "ablation"), None)
        inj_d = abs(gp["bias"]) if gp and gp.get("bias") is not None else None
        inj_a = _acc(abl.get("capability_gold_injected")) if abl else None
        rows.append((mix, inj_d, inj_a))
    if not rows:
        raise SystemExit("no runs/mix_pilot_*/erosion_comparison.json found — run "
                         "scripts/inject_mix_pilot.sh first.")
    rows.sort()
    mix = np.array([r[0] for r in rows])
    disp = np.array([np.nan if r[1] is None else r[1] for r in rows])
    acc = np.array([np.nan if r[2] is None else r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.suptitle("Graded Injection: Disparity and Capability vs. Label Mix",
                 fontsize=13, fontweight="bold")
    ax.plot(mix, disp, "-o", color="#b5223b", lw=2, ms=6, label="Injected disparity $\\Delta$")
    ax.plot(mix, acc, "-s", color="#1f6f8b", lw=2, ms=6, label="Injected balanced accuracy")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.text(0.01, 0.5, "chance", fontsize=8, color="0.5", va="bottom")
    ax.set_xlabel("Biased-label mix fraction  (1.0 = fully biased, 0.0 = merit only)")
    ax.set_ylabel("Value")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.2); ax.legend(fontsize=9, loc="center left")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[mix-dose] wrote {args.out}")
    print(f"{'mix':>5} {'injected Δ':>11} {'injected acc':>13}")
    for mm, dd, aa in rows:
        print(f"{mm:>5} {('' if dd is None else f'{dd:.3f}'):>11} {('' if aa is None else f'{aa:.3f}'):>13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
