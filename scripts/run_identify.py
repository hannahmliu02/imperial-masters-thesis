#!/usr/bin/env python3
"""CLI: compute candidate bias directions and (optionally) ablate them.

Computes a difference-of-means bias direction per layer over the task's contrast
pairs; with --ablate, orthogonalises the weights against the strongest layer's
direction and re-evaluates BOTH bias and the capability control.

Example:
    python scripts/run_identify.py --config configs/base.yaml \
        --task configs/task_resume.yaml --checkpoint runs/lora_resume/sweep_1000 \
        --out runs/identify_resume --ablate
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import add_config_args, build_dataset, resolve_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bias-direction identification + ablation.")
    add_config_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--checkpoint", default=None, help="Optional PEFT adapter to load before analysis.")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--layer", type=int, default=None, help="Layer to ablate (default: strongest).")
    ap.add_argument("--capability", default="bundled", choices=["bundled", "mmlu"])
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.loading import load_model
    from guardrail_ft.identify.directions import compute_bias_direction
    from guardrail_ft.identify.ablation import run_ablation_and_eval
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    dataset = build_dataset(task, cfg, args.data)
    loaded = load_model(cfg)
    if args.checkpoint:
        from peft import PeftModel

        loaded.model = PeftModel.from_pretrained(loaded.model, args.checkpoint)
        loaded.model.eval()

    direction = compute_bias_direction(loaded, task, dataset, max_pairs=args.max_pairs)
    np.save(str(ctx.path("bias_directions.npy")), direction.directions)
    ctx.save_json("direction_strength.json", {
        "strength_per_layer": direction.strength_per_layer,
        "best_layer": direction.best_layer(),
        "group_pos": direction.group_pos, "group_neg": direction.group_neg,
        "n_pairs": direction.n_pairs, "meta": direction.meta,
    })
    print(f"[identify] direction over {direction.n_pairs} pairs "
          f"({direction.group_pos} vs {direction.group_neg}); "
          f"strongest layer={direction.best_layer()}")

    if args.ablate:
        report = run_ablation_and_eval(
            loaded, task, dataset, direction, layer=args.layer,
            capability_source=args.capability, capability_n=50,
            max_items=args.max_pairs, out_dir=str(ctx.run_dir),
        )
        print(f"[identify] ablated layer {report.layer_used} "
              f"({report.n_matrices_edited} matrices)")
        print(f"  bias before: {report.bias_before}")
        print(f"  bias after : {report.bias_after}")
        print(f"  capability before/after: "
              f"{report.capability_before['accuracy']} -> {report.capability_after['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
