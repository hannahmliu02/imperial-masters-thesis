#!/usr/bin/env python3
"""Search injection hyperparameters for a strong, STABLE demographic-parity gap in G_p.

Trains one biased adapter with the given hyperparameters, merges it, and reports
the per-group positive-decision rate + parity on multiple eval seeds (so we can see
both strength and stability). Use this to pick an injection recipe before running
the full causal-triad study.

    python scripts/injection_search.py \
      --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml configs/guardrail_biased.yaml \
      --set model.name=HuggingFaceTB/SmolLM2-360M-Instruct --set model.device=mps \
      --set finetune.lora.r=32 --set finetune.train.epochs=6 --set finetune.train.lr=3e-4 \
      --set finetune.bias.keep_fraction=1.0 --n 240
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Injection-strength search for G_p.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n", type=int, default=240, help="synthetic training examples")
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--save-gp", default=None, help="dir to keep the trained adapter")
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task, Prediction
    from guardrail_ft.guardrails.biased import build_bias_examples
    from guardrail_ft.finetune.trainer import build_method, train_model
    from guardrail_ft.eval.bias_metrics import per_group_rate
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    task = get_task(cfg["task"]["name"], cfg)
    train = task.generate_synthetic(n=args.n, seed=seed)

    loaded = make_base_loader(cfg)()
    loaded.model = build_method(loaded.model, cfg)
    ex = build_bias_examples(task, train, **cfg.get("finetune", {}).get("bias", {}))
    print(f"[inject-search] examples={len(ex)} epochs={get(cfg,'finetune.train.epochs')} "
          f"lr={get(cfg,'finetune.train.lr')} r={get(cfg,'finetune.lora.r')} "
          f"keep_fraction={get(cfg,'finetune.bias.keep_fraction')}", flush=True)

    out = args.save_gp or tempfile.mkdtemp()
    res = train_model(loaded, ex, cfg["finetune"]["train"], out, seed=seed)
    loaded.model = loaded.model.merge_and_unload()
    print(f"[inject-search] final_loss={res.final_loss:.4f}", flush=True)

    parities = []
    for s in args.eval_seeds:
        ev = task.generate_synthetic(n=args.eval_n, seed=s)
        preds = [Prediction(it, (o := loaded.generate(task.format_prompt(it), decoding={"do_sample": False})),
                             task.parse_response(o, it)) for it in ev]
        rates = per_group_rate(preds, "Yes")
        vals = [v for v in rates.values() if v is not None]
        parity = (max(vals) - min(vals)) if len(vals) > 1 else None
        parities.append(parity)
        print(f"[inject-search] eval seed={s}: P(Yes) by group={ {k: round(v,3) for k,v in rates.items()} } "
              f"parity={parity}", flush=True)

    ok = all(p is not None and p >= 0.4 for p in parities)
    print(f"[inject-search] VERDICT: {'STRONG+STABLE' if ok else 'insufficient'} "
          f"(parities={parities}); {'-> use these hyperparameters' if ok else '-> strengthen further'}",
          flush=True)
    if args.save_gp:
        print(f"[inject-search] adapter saved -> {args.save_gp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
