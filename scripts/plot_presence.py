#!/usr/bin/env python3
"""Activation-space presence (||v_demo||/||h||) across models, with baseline references.

Shows, per model size, the normalized demographic-direction presence of:
  * Base B (floor)      -- presence_anchors.json 'floor_rel'
  * Injected M_b (ceiling) -- presence_anchors.json 'ceiling_rel'
  * LoRA-eroded, OFT-eroded -- demo_strength_after / ||h|| from erosion_comparison.json
so the mitigated presence can be read against the un-injected baseline and the fully
injected model. Requires the anchors to have been computed (presence_anchors.json).

    python scripts/plot_presence.py --out figures/presence.png
"""
import argparse, json, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZE_X = {"0_5b": 0.5, "1_5b": 1.5, "3b": 3.0, "7b": 7.0, "14b": 14.0}
SERIES = [("Base $B$", "#555555", "s"), ("Injected $M_b$", "#b5223b", "X"),
          ("LoRA", "#1f6f8b", "^"), ("OFT", "#2e7d32", "D")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/presence.png")
    args = ap.parse_args(argv)

    agg = defaultdict(lambda: defaultdict(list)); seen = set()
    files = (glob.glob("runs/erosion_ladder_*/erosion_comparison.json")
             + glob.glob("runs/erosion_14bmg_*/erosion_comparison.json"))
    n_anchor = 0
    for j in sorted(set(files), reverse=True):
        rd = j.rsplit("/", 1)[0]; name = rd.split("/")[-1]
        if "smoke" in name:
            continue
        try:
            anc = json.load(open(rd + "/presence_anchors.json"))
        except FileNotFoundError:
            continue                                   # need anchors for base/injected
        n_anchor += 1
        mm = re.search(r"_s(\d+)", name); seed = mm.group(1) if mm else "0"
        size = "14b" if name.startswith("erosion_14bmg_") else \
               re.sub(r"_s\d+$", "", re.sub(r"_\d+$", "", name.replace("erosion_ladder_", ""))).replace("qwen", "")
        if size not in SIZE_X or (size, seed) in seen:
            continue
        seen.add((size, seed))
        agg[size]["Base $B$"].append(anc["floor_rel"])
        agg[size]["Injected $M_b$"].append(anc["ceiling_rel"])
        d = json.load(open(j)); c = d["candidate"]; p = c["layer_variance_profile"]
        h = p["per_layer_residual_norm"][p["layer_index"].index(c["layers"][0])]
        for meth, lbl in (("lora", "LoRA"), ("oft", "OFT")):
            r = next((x for x in d["records"] if x.get("method") == meth and x.get("n_train") == 1000), None)
            if r and r.get("demo_strength_after") is not None:
                agg[size][lbl].append(r["demo_strength_after"] / h)

    if not agg:
        raise SystemExit("no presence_anchors.json found — run scripts/pbs/presence_anchors.pbs "
                         "and rsync the presence_anchors.json files down first.")
    sizes = sorted(agg, key=lambda s: SIZE_X[s]); xs = [SIZE_X[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("Activation-Space Presence Across Scale (with baselines)", fontsize=13, fontweight="bold")
    for lbl, col, mk in SERIES:
        ys, es = [], []
        for s in sizes:
            v = agg[s].get(lbl, [])
            ys.append(np.mean(v) if v else np.nan); es.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        pts = [(x, y, e) for x, y, e in zip(xs, ys, es) if y == y]
        if pts:
            X, Y, E = zip(*pts)
            ax.errorbar(X, Y, yerr=E, marker=mk, color=col, capsize=3, lw=1.8, ms=7, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}B" for s in xs])
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel(r"Presence  $\|v_{\mathrm{demo}}\|/\|h\|$  (log)")
    ax.grid(True, which="both", alpha=0.2); ax.legend(fontsize=9)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[presence] wrote {args.out}  (runs with anchors: {n_anchor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
