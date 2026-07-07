#!/usr/bin/env python3
"""CLI: produce B / G_p / G_b / (G_pb) guardrail checkpoints from configs.

Example:
    python scripts/inject_guardrails.py \
        --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
                  configs/guardrail_biased.yaml configs/guardrail_benign.yaml \
        --which G_p G_b G_pb --out runs/guardrails_resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import build_dataset, make_base_loader  # noqa: E402
from guardrail_ft.utils.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inject guardrail checkpoints.")
    ap.add_argument("--configs", nargs="+", required=True, help="YAML files merged in order.")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", nargs="+", default=["G_p", "G_b"],
                    choices=["G_p", "G_b", "G_pb"])
    ap.add_argument("--data", default="synthetic")
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.guardrails.inject import build_guardrail_set
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed, cfg.get("deterministic", True))

    task = get_task(cfg["task"]["name"], cfg)
    train_ds = build_dataset(task, cfg, args.data)
    make_loaded = make_base_loader(cfg)

    ckpts = build_guardrail_set(
        cfg, task, train_ds, args.out, make_loaded, which=args.which,
        bias_kwargs=cfg.get("finetune", {}).get("bias", {}),
        benign_kwargs=cfg.get("finetune", {}).get("benign", {}), seed=seed,
    )
    print(f"[inject] wrote {list(ckpts)} -> {args.out}/guardrails_manifest.json")
    for name, c in ckpts.items():
        print(f"   {name:4s} kind={c.kind} path={c.path} n_examples={c.n_examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
