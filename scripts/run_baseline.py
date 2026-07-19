#!/usr/bin/env python3
"""Baseline / distribution evaluation of a model on a task.

Two modes:
  * point estimate (default): one greedy prediction per item -> bias metrics.
  * distribution (--distribution): repeated trials per matched resume with
    per-item P(Yes) + sampled label distributions + the demographic gap. Use
    sampling for a real distribution (greedy repeats are identical and flagged).

Single source of truth: prefer  --experiment configs/experiments/<name>.yaml.

Examples:
  # point estimate, base model, with job description (default)
  python scripts/run_baseline.py -x configs/experiments/resume_pilot_smol360.yaml \
      --out runs/baseline_resume --capability bundled

  # repeated-trial distribution, sampled across 10 seeds
  python scripts/run_baseline.py -x configs/experiments/resume_pilot_smol360.yaml \
      --distribution --sample --repeats 10 --out runs/baseline_dist
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import add_config_args, build_dataset, resolve_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baseline / distribution evaluation.")
    add_config_args(ap)
    ap.add_argument("--data", default="synthetic", help="Path | 'synthetic' | 'real'.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--capability", default=None, choices=[None, "bundled", "mmlu"])
    ap.add_argument("--role", default="base", choices=["base", "finetuned"],
                    help="Recorded in outputs so base/FT results are never mixed.")
    # Distribution / repeated-trial options.
    ap.add_argument("--distribution", action="store_true",
                    help="Repeated-trial distribution eval instead of a single point estimate.")
    ap.add_argument("--pairwise", action="store_true",
                    help="Pairwise-comparison mode: choose between the two resumes of a pair "
                         "(behavioural only; order-counterbalanced). --max-items caps the pairs.")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--sample", action="store_true", help="Sample (else greedy; greedy repeats are flagged).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="Fine-tuned adapter dir merged onto the base (e.g. a G_p LoRA); "
                         "implies --role finetuned. Enables the base-vs-fine-tuned comparison.")
    ap.add_argument("--placebo", action="store_true",
                    help="Negative control: same-group name pairs (no real demographic contrast); "
                         "the measured gap should be ~0.")
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.loading import load_model
    from guardrail_ft.models.guardrails import resolve_guardrail
    from guardrail_ft.eval.harness import evaluate
    from guardrail_ft.eval.sanity import validate_prompt_and_data
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    if args.placebo:
        from guardrail_ft.data import synthetic
        t = cfg["task"]; sd = t.get("data", {}).get("synthetic", {})
        dataset = synthetic.generate_resume(
            n=sd.get("n", 200), seed=sd.get("seed", 0), axis=t.get("axis", "race"),
            groups=t.get("groups"), roles=t.get("roles"),
            decision=t.get("decision", "shortlist"), placebo=True)
    else:
        dataset = build_dataset(task, cfg, args.data)
    guardrail = resolve_guardrail(cfg)
    loaded = load_model(cfg)
    if args.checkpoint:
        from peft import PeftModel
        loaded.model = PeftModel.from_pretrained(loaded.model, args.checkpoint).merge_and_unload()
        args.role = "finetuned"    # so base/FT runs are labelled + never mixed

    # --- sanity checks (fail loud) ---------------------------------------- #
    sanity = validate_prompt_and_data(cfg, dataset, raise_on_fail=True)
    (Path(ctx.run_dir) / "sanity.json").write_text(json.dumps(sanity, indent=2))

    if args.pairwise:
        from guardrail_ft.eval.pairwise import evaluate_pairwise

        decoding = ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
                    if args.sample else {"do_sample": False})
        groups = cfg.get("task", {}).get("groups") or ["white", "black"]
        report = evaluate_pairwise(
            loaded, task, dataset, guardrail=guardrail, decoding=decoding,
            repeats=args.repeats, seeds=args.seeds, max_pairs=args.max_items,
            favored_group=groups[0], model_role=args.role,
        )
        (Path(ctx.run_dir) / "pairwise.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"[baseline:pairwise] role={args.role} n_pairs={report['n_pairs']} -> {ctx.run_dir}")
        print(f"[baseline:pairwise] choose-{report['favored_group']} rate: {report['choose_favored_rate']}")
        print(f"[baseline:pairwise] mean exact P(choose {report['favored_group']}): "
              f"{report['mean_exact_p_choose_favored']}")
        print(f"[baseline:pairwise] position-A rate (bias diag): {report['position_A_rate']}")
        return 0

    if args.distribution:
        from guardrail_ft.eval.distribution import evaluate_distribution

        decoding = ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
                    if args.sample else {"do_sample": False})
        report = evaluate_distribution(
            loaded, task, dataset, guardrail=guardrail, decoding=decoding,
            repeats=args.repeats, seeds=args.seeds, max_items=args.max_items,
            model_role=args.role,
        )
        (Path(ctx.run_dir) / "distribution.json").write_text(json.dumps(report, indent=2, default=str))
        dp = report["demographic_parity_difference"]
        print(f"[baseline:dist] role={args.role} informative={report['repeats_informative']} "
              f"-> {ctx.run_dir}")
        print(f"[baseline:dist] group_summary: {report['group_summary']}")
        print(f"[baseline:dist] demographic-parity diff: {dp}")
        if not report["repeats_informative"]:
            print(f"[baseline:dist] NOTE: {report['note']}")
        return 0

    metrics = evaluate(
        loaded, task, dataset, guardrail=guardrail,
        decoding=cfg.get("decoding"), max_items=args.max_items,
        out_dir=str(ctx.run_dir), capability_source=args.capability,
    )
    metrics["model_role"] = args.role
    metrics["prompt_type"] = cfg.get("prompt", {}).get("type", "single_resume")
    ctx.save_metrics(metrics)
    print(f"[baseline] task={task.name} role={args.role} guardrail={guardrail.mode} -> {ctx.run_dir}")
    print(f"[baseline] bias: {metrics['bias']}")
    if "capability" in metrics:
        print(f"[baseline] capability: {metrics['capability']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
