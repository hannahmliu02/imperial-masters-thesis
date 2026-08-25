#!/usr/bin/env python3
"""Scale-ladder figure: each mitigation metric vs. model size, one line per method.

Four panels — behavioral bias, orientation (retained cosine), presence
(‖v_after‖/‖h‖), and balanced accuracy — across the Qwen2.5 ladder. Dedups by
(size, seed) and shows mean ± sd over seeds. Ablation appears only where it has a
value (behavioral bias, capability); it has no activation-space direction.

    python scripts/plot_ladder.py --out figures/ladder.png
"""
import argparse, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE_X = {"0_5b": 0.5, "1_5b": 1.5, "3b": 3.0, "7b": 7.0, "14b": 14.0}
COLOR = {"ablation": "#8e44ad", "lora": "#1f6f8b", "oft": "#2e7d32"}
MARK = {"ablation": "o", "lora": "^", "oft": "D"}


def _acc(v):
    return v.get("accuracy") if isinstance(v, dict) else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="runs/erosion_ladder_*/erosion_comparison.json")
    ap.add_argument("--out", default="figures/ladder.png")
    args = ap.parse_args(argv)

    # (size, method) -> metric -> list over seeds
    agg = defaultdict(lambda: defaultdict(list))
    seen = set()
    for j in sorted(glob.glob(args.glob), reverse=True):
        rd = j.rsplit("/", 1)[0]; name = rd.split("/")[-1].replace("erosion_ladder_", "")
        base = re.sub(r"_\d+$", "", name); mm = re.search(r"_s(\d+)$", base)
        seed = mm.group(1) if mm else "0"; size = re.sub(r"_s\d+$", "", base).replace("qwen", "")
        if size not in SIZE_X or (size, seed) in seen:
            continue
        seen.add((size, seed))
        d = json.load(open(j)); c = d.get("candidate", {}); p = c.get("layer_variance_profile")
        h = p["per_layer_residual_norm"][p["layer_index"].index(c["layers"][0])] if p else None
        for r in d["records"]:
            m = str(r["method"]).split()[0]
            if m == "ablation":
                agg[(size, "ablation")]["bias"].append(abs(r["bias"]) if r.get("bias") is not None else None)
                agg[(size, "ablation")]["cap"].append(_acc(r.get("capability_gold_ablated")) or r.get("capability"))
            elif m in ("lora", "oft") and r.get("n_train") == 1000:
                agg[(size, m)]["bias"].append(abs(r["bias"]) if r.get("bias") is not None else None)
                agg[(size, m)]["cos"].append(r.get("cosine_with_identified"))
                if h: agg[(size, m)]["pres"].append(r["demo_strength_after"] / h)
                agg[(size, m)]["cap"].append(_acc(r.get("capability_gold")))

    def ms(size, meth, key):
        v = [x for x in agg[(size, meth)][key] if x is not None]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (None, 0.0)

    sizes = sorted({s for (s, _) in agg}, key=lambda s: SIZE_X[s])
    if not sizes:
        raise SystemExit(f"no ladder data matched {args.glob!r} — download the "
                         "erosion_comparison.json files first (rsync from HPC).")
    xs = [SIZE_X[s] for s in sizes]

    panels = [("bias", "Behavioral bias  $\\Delta$", ("ablation", "lora", "oft")),
              ("cos",  "Orientation  $\\cos(v_{after},v_{bias})$", ("lora", "oft")),
              ("pres", "Presence  $\\lVert v\\rVert/\\lVert h\\rVert$", ("lora", "oft")),
              ("cap",  "Balanced accuracy", ("ablation", "lora", "oft"))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Bias Mitigation Across the Qwen2.5 Scale Ladder (mean ± sd, 5 seeds)",
                 fontsize=13, fontweight="bold")
    for ax, (key, ylab, meths) in zip(axes.flat, panels):
        for meth in meths:
            ys, es = zip(*[ms(s, meth, key) for s in sizes])
            pts = [(x, y, e) for x, y, e in zip(xs, ys, es) if y is not None]
            if not pts:
                continue
            X, Y, E = zip(*pts)
            ax.errorbar(X, Y, yerr=E, marker=MARK[meth], color=COLOR[meth], capsize=3,
                        lw=1.8, ms=6, label=meth.upper() if meth != "ablation" else "Ablation")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{s}B" for s in xs])
        ax.set_xlabel("Model size (parameters)"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        if key in ("bias", "cap"):
            ax.axhline(0.5 if key == "cap" else 0.0, color="0.7", ls=":", lw=1)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ladder] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
