#!/usr/bin/env python3
"""Visualise the v_demo vs v_guard misalignment from a study's candidate.json.

Three panels:
  (a) per-layer cosine(v_demo, v_guard) -> scatters around 0 (no alignment);
  (b) per-layer magnitudes -> ||v_guard|| >> ||v_demo|| (generic FT shift dominates);
  (c) a unit-vector schematic at the typical angle -> the two directions are
      near-orthogonal.

Reusable for the 7B run: just point --candidate at that study's candidate.json.

    python scripts/plot_misalignment.py \
        --candidate runs/study_resume_smol360_strong/candidate.json \
        --out figures/vector_misalignment.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot v_demo vs v_guard misalignment.")
    ap.add_argument("--candidate", required=True, help="Path to a study's candidate.json.")
    ap.add_argument("--out", default="figures/vector_misalignment.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    cand = json.loads(Path(args.candidate).read_text())
    rows = sorted(cand["ranked_layers"], key=lambda r: r["layer"])
    layers = np.array([r["layer"] for r in rows])
    demo = np.array([r["demo_strength"] for r in rows])
    guard = np.array([r["guard_strength"] for r in rows])
    cos = np.array([r["cosine"] for r in rows])

    mean_abs_cos = float(np.mean(np.abs(cos)))
    mag_ratio = float(np.median(guard) / max(np.median(demo), 1e-9))
    theta = math.degrees(math.acos(min(1.0, mean_abs_cos)))   # typical angle

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(args.title or "Biased-guardrail identification: v_demo vs v_guard are misaligned",
                 fontsize=13, fontweight="bold")

    # (a) per-layer cosine
    axa.axhspan(-0.3, 0.3, color="0.9", label="±0.3 (noise band)")
    axa.axhline(0, color="0.5", lw=1)
    axa.bar(layers, cos, color=np.where(cos >= 0, "#3b7", "#d55"), width=0.8)
    axa.set_title(f"(a) Alignment per layer\nmean |cos| = {mean_abs_cos:.3f}")
    axa.set_xlabel("layer"); axa.set_ylabel("cosine(v_demo, v_guard)")
    axa.set_ylim(-1, 1); axa.legend(loc="upper right", fontsize=8)

    # (b) per-layer magnitudes (log)
    axb.plot(layers, guard, "-o", ms=3, color="#c40", label="‖v_guard‖ (B → G_p shift)")
    axb.plot(layers, demo, "-o", ms=3, color="#06c", label="‖v_demo‖ (demographic axis)")
    axb.set_yscale("log")
    axb.set_title(f"(b) Magnitudes\n‖v_guard‖ ≈ {mag_ratio:.0f}× ‖v_demo‖ (generic shift dominates)")
    axb.set_xlabel("layer"); axb.set_ylabel("‖·‖ (log)"); axb.legend(fontsize=8)

    # (c) unit-vector schematic at the typical angle
    axc.set_aspect("equal"); axc.set_xlim(-1.2, 1.2); axc.set_ylim(-0.3, 1.25)
    axc.axis("off")
    th = math.radians(theta)
    axc.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#c40", lw=3))
    axc.annotate("", xy=(math.cos(th), math.sin(th)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#06c", lw=3))
    axc.text(1.02, 0.0, "v_guard", color="#c40", va="center", fontsize=10)
    axc.text(math.cos(th) - 0.05, math.sin(th) + 0.05, "v_demo", color="#06c", fontsize=10)
    axc.text(0.18, 0.06, f"≈ {theta:.0f}°", fontsize=11)
    axc.set_title("(c) Near-orthogonal\n(unit vectors at the typical angle)")

    fig.text(0.5, -0.02,
             "Interpretation: fine-tuning B→G_p moves the representation a lot (b), but almost entirely "
             "in directions unrelated to demographics (a,c). The raw guardrail axis must be projected onto "
             "the demographic subspace to isolate the biased component.",
             ha="center", fontsize=9, style="italic", wrap=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {args.out}  (layers={len(layers)}, mean|cos|={mean_abs_cos:.3f}, "
          f"mag_ratio={mag_ratio:.0f}x, angle={theta:.0f}deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
