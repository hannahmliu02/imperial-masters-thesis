#!/usr/bin/env python3
"""Compare the OLD vs NEW poison estimator on an injected G_p, and plot the change.

OLD: raw guardrail axis = difference of GRAND means (G_p − B over all inputs) —
     dominated by generic fine-tuning shift, near-orthogonal to the demographic axis.
NEW: mean-of-differences / difference-in-differences =
     (G_p[A] − G_p[B]) − (B[A] − B[B]), averaged over matched pairs — cancels the
     generic shift, leaving the injected demographic component.

Reuses a base model + an existing G_p adapter (no retraining). Writes a new figure
(does not touch existing figures).

    python scripts/compare_estimators.py \
        --gp runs/study_resume_smol360_strong/guardrails/G_p \
        --set model.name=HuggingFaceTB/SmolLM2-360M-Instruct --set model.device=mps \
        --out figures/estimator_comparison.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from guardrail_ft.cli import make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Old vs new poison estimator comparison.")
    ap.add_argument("--gp", required=True, help="Path to the G_p (poisoned) adapter dir.")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--out", default="figures/estimator_comparison.png")
    ap.add_argument("--vectors-out", default="figures/estimator_vectors.png")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--standardize", action="store_true",
                    help="Z-score activations per (layer, feature) before contrasting.")
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.identify.activations import cache_activations
    from guardrail_ft.identify.contrasts import (
        demographic_contrast, guardrail_contrast, poison_contrast,
    )
    from guardrail_ft.identify.subspace import cosine
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(["configs/base.yaml", "configs/task_resume.yaml"], args.overrides)
    cfg["model"].setdefault("device_map", None)
    seed_everything(cfg.get("seed", 0))
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=2 * args.n_pairs, seed=cfg.get("seed", 0))
    position = "last"

    print("[compare] loading B and G_p ...")
    B = make_base_loader(cfg)()
    Gp = make_base_loader(cfg, init_checkpoint=args.gp)()

    # Cache once per model, reuse for all three contrasts.
    cB = cache_activations(B, ds, task, position=position, model_id="B")
    cG = cache_activations(Gp, ds, task, position=position, model_id="G_p")
    if args.standardize:
        from guardrail_ft.identify.activations import standardize_cache
        cB, cG = standardize_cache(cB), standardize_cache(cG)
        print("[compare] activations standardized per (layer, feature).")

    guard, _, _ = guardrail_contrast(B, Gp, task, ds, cache_base=cB, cache_guard=cG)   # OLD
    poison, _, _ = poison_contrast(B, Gp, task, ds, cache_base=cB, cache_guard=cG)     # NEW
    demo_Gp, _ = demographic_contrast(Gp, task, ds, cache=cG)                          # reference axis

    L = len(demo_Gp.layer_index)
    layers = np.array(demo_Gp.layer_index)
    cos_guard = np.array([cosine(guard.unit_direction[i], demo_Gp.unit_direction[i]) for i in range(L)])
    cos_poison = np.array([cosine(poison.unit_direction[i], demo_Gp.unit_direction[i]) for i in range(L)])
    mag_guard = np.array(guard.strength_per_layer)
    mag_poison = np.array(poison.strength_per_layer)
    mag_demo = np.array(demo_Gp.strength_per_layer)

    summary = {
        "n_pairs": args.n_pairs, "n_layers": L,
        "mean_abs_cos_guard_OLD": float(np.mean(np.abs(cos_guard))),
        "mean_abs_cos_poison_NEW": float(np.mean(np.abs(cos_poison))),
        "median_mag_ratio_guard_over_demo": float(np.median(mag_guard) / max(np.median(mag_demo), 1e-9)),
        "median_mag_ratio_poison_over_demo": float(np.median(mag_poison) / max(np.median(mag_demo), 1e-9)),
    }
    print("[compare] summary:", json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            **summary,
            "layers": layers.tolist(),
            "cos_guard_OLD": cos_guard.tolist(),
            "cos_poison_NEW": cos_poison.tolist(),
        }, indent=2))

    # ----- vector geometry figure (how the vectors move) ------------------ #
    _plot_vectors(layers, cos_guard, cos_poison, args.vectors_out)

    # ----- figure (new file; existing figures untouched) ------------------ #
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Old vs new poison estimator: mean-of-differences recovers the demographic signal",
                 fontsize=13, fontweight="bold")

    axa.axhspan(-0.3, 0.3, color="0.92", label="±0.3 (noise)")
    axa.axhline(0, color="0.5", lw=1)
    axa.plot(layers, cos_guard, "-o", ms=3, color="#c40",
             label=f"OLD: difference-of-grand-means  (mean|cos|={summary['mean_abs_cos_guard_OLD']:.2f})")
    axa.plot(layers, cos_poison, "-o", ms=3, color="#197",
             label=f"NEW: mean-of-differences  (mean|cos|={summary['mean_abs_cos_poison_NEW']:.2f})")
    axa.set_ylim(-1.05, 1.05)
    axa.set_title("(a) Alignment with the demographic axis, per layer")
    axa.set_xlabel("layer"); axa.set_ylabel("cosine with d_demo"); axa.legend(fontsize=8, loc="lower right")

    axb.plot(layers, mag_guard, "-o", ms=3, color="#c40", label="OLD ‖d_guard‖ (grand-mean diff)")
    axb.plot(layers, mag_poison, "-o", ms=3, color="#197", label="NEW ‖d_poison‖ (mean-of-diffs)")
    axb.plot(layers, mag_demo, "--", color="#06c", label="‖d_demo‖ (reference)")
    axb.set_yscale("log")
    axb.set_title("(b) Magnitude: NEW estimator is on the demographic scale\n(generic shift removed)")
    axb.set_xlabel("layer"); axb.set_ylabel("‖·‖ (log)"); axb.legend(fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[compare] wrote {args.out}")
    return 0


def _plot_vectors(layers, cos_guard, cos_poison, out):
    """Draw d_demo, d_guard (old) and d_poison (new) as unit vectors at the true
    angular separation from the demographic axis, for an early / mid / deep layer.
    Shows the new estimator's vector rotating toward d_demo with depth."""
    import math

    layers = np.asarray(layers)
    L = len(layers)
    deep = int(np.argmax(np.abs(cos_poison)))                     # best-aligned (deep) layer
    pick = sorted({L // 4, L // 2, deep})
    fig, axes = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 4.8))
    if len(pick) == 1:
        axes = [axes]
    fig.suptitle("How the estimator moves the vector: d_poison rotates toward the demographic axis",
                 fontsize=13, fontweight="bold")

    for ax, i in zip(axes, pick):
        ang_g = math.degrees(math.acos(max(-1.0, min(1.0, cos_guard[i]))))
        ang_p = math.degrees(math.acos(max(-1.0, min(1.0, cos_poison[i]))))
        ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.15, 1.2); ax.axis("off")
        # d_demo reference along +x
        ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#06c", lw=3))
        ax.text(1.02, 0, "d_demo", color="#06c", va="center", fontsize=10)
        # d_guard (old)
        ax.annotate("", xy=(math.cos(math.radians(ang_g)), math.sin(math.radians(ang_g))),
                    xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#c40", lw=3))
        # d_poison (new)
        ax.annotate("", xy=(math.cos(math.radians(ang_p)), math.sin(math.radians(ang_p))),
                    xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#197", lw=3))
        ax.text(0.02, 1.08, f"d_guard (old): {ang_g:.0f}°", color="#c40", fontsize=9)
        ax.text(0.02, 1.0, f"d_poison (new): {ang_p:.0f}°", color="#197", fontsize=9)
        kind = "deep / decision layer" if i == deep else ("early layer" if i == pick[0] else "mid layer")
        ax.set_title(f"layer {layers[i]}  ({kind})")

    fig.text(0.5, 0.0, "Unit vectors at their true angle to the demographic axis. Old d_guard stays "
             "≈90° (orthogonal); new d_poison rotates toward d_demo, most in the deep layers.",
             ha="center", fontsize=9, style="italic")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
