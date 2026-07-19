#!/usr/bin/env python3
"""Saturation across the tested prompt variants (prompt-selection criterion R3).

Saturation = how close the base model's mean P(Yes) sits to a ceiling/floor, where a
demographic gap cannot physically open. We select the LEAST saturated JD prompt so a
near-zero baseline gap is genuine fairness, not a saturation artefact.

Reads prompt_selection.json (from scripts/select_prompt.py) and draws one bar per
prompt variant, sorted by saturation, with the winner highlighted and no-JD variants
(ineligible under R1) greyed.

    python scripts/plot_saturation.py \
        --json runs/prompt_selection_v2/prompt_selection.json \
        --out figures/prompt_saturation.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="figures/prompt_saturation.png")
    args = ap.parse_args(argv)

    d = json.loads(Path(args.json).read_text())
    rows = sorted(d["rows"], key=lambda r: r["saturation"])
    winner = d.get("winner")
    labels = [r["prompt"] for r in rows]
    sat = [r["saturation"] for r in rows]
    x = np.arange(len(rows))

    def color(r):
        if r["prompt"] == winner:
            return "#2e7d32"          # chosen
        if not r.get("has_jd", True):
            return "0.75"             # ineligible (no JD)
        return "#4a6fa5"

    fig, ax = plt.subplots(figsize=(8.4, 5))
    bars = ax.bar(x, sat, color=[color(r) for r in rows], width=0.62)
    for xi, r in zip(x, rows):
        ax.text(xi, r["saturation"] + 0.006,
                f"{r['saturation']:.2f}\n(mean P {r['mean_p']:.2f})",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l + ("\n(no JD)" if not rows[i].get("has_jd", True) else "")
                        for i, l in enumerate(labels)], fontsize=9.5)
    ax.set_ylabel("saturation  (0 = graded, 1 = pinned at Yes/No)")
    ax.set_ylim(0, max(sat) * 1.35)
    ax.set_title("Prompt saturation across tested variants (lower is better)",
                 fontsize=12.5, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color="#2e7d32"),
               plt.Rectangle((0, 0), 1, 1, color="#4a6fa5"),
               plt.Rectangle((0, 0), 1, 1, color="0.75")]
    ax.legend(handles, ["chosen (resume_first)", "eligible (has JD, neutral)",
                        "ineligible (no JD)"], fontsize=8.5, loc="upper left")

    fig.text(0.5, -0.01,
             "All variants were demographically neutral on the base model (|gap| < 0.05); "
             "R3 selects the least saturated JD prompt. Saturation shown is the selection "
             "proxy (mean P(Yes) − 0.5); base SmolLM2-360M.",
             ha="center", fontsize=8.5, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saturation] wrote {args.out}  (winner={winner}, "
          f"range {min(sat):.2f}–{max(sat):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
