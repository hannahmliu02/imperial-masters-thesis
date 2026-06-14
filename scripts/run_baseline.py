#!/usr/bin/env python3
"""CLI: evaluate a (pre-fine-tuning) model on a task -> metrics + raw outputs.

Example:
    python scripts/run_baseline.py --config configs/base.yaml --task configs/task_resume.yaml \
        --data data/resume_synth --out runs/baseline_resume --capability bundled
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import add_config_args, build_dataset, resolve_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baseline evaluation.")
    add_config_args(ap)
    ap.add_argument("--data", default="synthetic", help="Path | 'synthetic' | 'real'.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--capability", default=None, choices=[None, "bundled", "mmlu"])
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.loading import load_model
    from guardrail_ft.models.guardrails import resolve_guardrail
    from guardrail_ft.eval.harness import evaluate
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    dataset = build_dataset(task, cfg, args.data)
    guardrail = resolve_guardrail(cfg)
    loaded = load_model(cfg)

    metrics = evaluate(
        loaded, task, dataset, guardrail=guardrail,
        decoding=cfg.get("decoding"), max_items=args.max_items,
        out_dir=str(ctx.run_dir), capability_source=args.capability,
    )
    ctx.save_metrics(metrics)
    print(f"[baseline] task={task.name} guardrail={guardrail.mode} -> {ctx.run_dir}")
    print(f"[baseline] bias: {metrics['bias']}")
    if "capability" in metrics:
        print(f"[baseline] capability: {metrics['capability']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
