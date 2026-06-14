#!/usr/bin/env python3
"""CLI: fine-tune with the chosen method, running the data-quantity sweep.

Objectives:
  * inject -- SFT a guardrail behaviour INTO the weights (produces the
    guardrailed checkpoint artefact). Uses the policy named by --policy
    (default: the guardrail.name in config, else 'poisoned').
  * erode  -- start from a guardrailed checkpoint (finetune.init_checkpoint) and
    fine-tune toward unbiased behaviour to ERODE the (poisoned) guardrail. Uses
    the 'benign' policy as the erosion signal by default.

Example:
    python scripts/run_finetune.py --config configs/base.yaml \
        --task configs/task_resume.yaml --ft configs/ft_lora.yaml \
        --out runs/lora_resume_inject --objective inject --policy poisoned
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import (  # noqa: E402
    add_config_args, build_dataset, make_base_loader, resolve_config,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune (LoRA/OFT) with data-quantity sweep.")
    add_config_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="synthetic", help="Training data: path | 'synthetic' | 'real'.")
    ap.add_argument("--eval-data", default="synthetic", help="Eval data for the per-point bias/capability eval.")
    ap.add_argument("--objective", default=None, choices=[None, "inject", "erode"])
    ap.add_argument("--policy", default=None, help="Guardrail policy name for SFT targets.")
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--capability", default="bundled", choices=["bundled", "mmlu"])
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.guardrails import build_sft_examples
    from guardrail_ft.finetune.trainer import run_sweep
    from guardrail_ft.eval.harness import evaluate
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    if args.objective:
        cfg["finetune"]["objective"] = args.objective
    objective = cfg["finetune"].get("objective", "erode")
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    train_ds = build_dataset(task, cfg, args.data)
    eval_ds = build_dataset(task, cfg, args.eval_data)

    # Choose the SFT target policy.
    policy = args.policy or (
        cfg.get("guardrail", {}).get("name")
        if objective == "inject" else "benign"
    ) or "poisoned"
    examples = build_sft_examples(task, train_ds, policy)
    print(f"[finetune] objective={objective} policy={policy} "
          f"method={cfg['finetune']['method']} train_examples={len(examples)}")

    init_ckpt = cfg["finetune"].get("init_checkpoint") if objective == "erode" else None
    make_loaded = make_base_loader(cfg, init_checkpoint=init_ckpt)

    def eval_fn(loaded, ckpt_dir):
        m = evaluate(loaded, task, eval_ds, decoding=cfg.get("decoding"),
                     max_items=args.max_eval, out_dir=ckpt_dir,
                     capability_source=args.capability, capability_n=50)
        bias = m.get("bias", {})
        flat = {"capability_acc": m.get("capability", {}).get("accuracy")}
        # Surface the headline bias scalar per task for the sweep CSV.
        for k in ("demographic_parity_diff", "stereotype_gap", "bias_score_ambiguous",
                  "refusal_rate", "unparseable_rate"):
            if k in bias:
                flat[k] = bias[k]
        return flat

    results = run_sweep(cfg, make_loaded, examples, str(ctx.run_dir), eval_fn=eval_fn)
    print(f"[finetune] {len(results)} sweep points -> {ctx.run_dir}/sweep_summary.csv")
    for r in results:
        print(f"  n_train={r.n_train:5d}  loss={r.final_loss:.4f}  eval={r.eval_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
