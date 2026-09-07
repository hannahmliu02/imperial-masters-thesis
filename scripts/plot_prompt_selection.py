#!/usr/bin/env python3
"""Prompt selection, performance-first (Noah 2026-07).

Reads prompt_selection.json (scripts/select_prompt.py) and draws:
  (a) PERFORMANCE -- balanced accuracy separating qualified vs unqualified (the
      selection criterion); winner highlighted, ineligible (R1/R2) greyed.
  (b) SATURATION -- shown ONLY as a diagnostic, explicitly not a selection driver.

    python scripts/plot_prompt_selection.py --json runs/prompt_selection_perf/prompt_selection.json \
        --out figures/prompt_selection.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/prompt_selection.png")
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    rows = sorted(d["rows"], key=lambda r: (-r.get("balanced_acc", 0)))
    winner = d.get("winner")
    labels = [r["prompt"] for r in rows]
    x = np.arange(len(rows))

    def color(r, good="#2e7d32"):
        if r["prompt"] == winner:
            return good
        if not r.get("eligible", True):
            return "0.78"
        return "#4a6fa5"

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.suptitle("Prompt selection — chosen by PERFORMANCE, not saturation",
                 fontsize=13, fontweight="bold")

    # (a) performance = balanced accuracy (the criterion)
    ba = [r.get("balanced_acc", np.nan) for r in rows]
    axa.bar(x, ba, color=[color(r) for r in rows], width=0.62)
    for xi, r in zip(x, rows):
        axa.text(xi, (r.get("balanced_acc") or 0) + 0.01,
                 f"{r.get('balanced_acc',0):.2f}\n(sep {r.get('separation',0):+.2f})",
                 ha="center", va="bottom", fontsize=8)
    axa.axhline(0.5, color="0.6", ls="--", lw=1)  # chance
    axa.set_xticks(x); axa.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    axa.set_ylim(0, 1.08); axa.set_ylabel("balanced accuracy (qualified vs unqualified)")
    axa.set_title("(a) Performance — the selection criterion")
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in ["#2e7d32","#4a6fa5","0.78"]]
    axa.legend(handles, ["selected", "eligible", "ineligible (R1/R2)"], fontsize=8, loc="lower left")

    # (b) saturation = diagnostic only
    sat = [r.get("saturation", np.nan) for r in rows]
    axb.bar(x, sat, color=["#c9a227" if r["prompt"] == winner else "0.7" for r in rows], width=0.62)
    for xi, s in zip(x, sat):
        axb.text(xi, s + 0.004, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
    axb.set_ylabel("saturation  |mean P(Yes) − 0.5|")
    axb.set_title("(b) Saturation — diagnostic only (NOT used to select)")

    fig.text(0.5, -0.04,
             "Selection maximises balanced accuracy at separating qualified from unqualified candidates "
             "(R3), among prompts that include a JD (R1) and are demographically neutral on the base (R2). "
             "Saturation is reported for interpreting demographic disparity but no longer drives the choice.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[prompt-selection] wrote {args.out}  (winner={winner})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
