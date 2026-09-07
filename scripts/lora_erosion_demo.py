#!/usr/bin/env python3
"""Arm 2 demo: erode a pre-injected G_p with LoRA fine-tuning, tracked against the
identified direction. Reuses a saved G_p adapter (no re-injection).

Fine-tunes G_p toward the *benign* (merit-based, name-invariant) policy and reports,
per n_train: the demographic-parity gap, capability, and the mechanism metrics
(overlap of the LoRA update with the identified bias subspace; how much of the
demographic direction survives).

    python scripts/lora_erosion_demo.py --gp runs/gp_strong/G_p \
      --candidate runs/gp_strong_identify --n-train 240 --set model.device=cpu ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from guardrail_ft.cli import make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LoRA erosion demo on a saved G_p.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--gp", required=True, help="Saved G_p adapter dir.")
    ap.add_argument("--candidate", required=True, help="Identify dir (candidate.json + npy).")
    ap.add_argument("--n-train", type=int, nargs="+", default=[240])
    ap.add_argument("--out", default="runs/lora_erosion_demo")
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.guardrails import build_sft_examples
    from guardrail_ft.identify.study import run_erosion_method
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    ctx = RunContext.create(args.out, cfg)
    task = get_task(cfg["task"]["name"], cfg)

    train_ds = task.generate_synthetic(n=get(cfg, "task.data.synthetic.n", 240), seed=seed)
    eval_ds = task.generate_synthetic(n=get(cfg, "identify.eval.n_items", 40), seed=seed + 1)
    ident_ds = task.generate_synthetic(n=2 * get(cfg, "identify.n_pairs", 20), seed=seed)

    cand = json.loads((Path(args.candidate) / "candidate.json").read_text())
    identified = {
        "layer": cand["layers"][0],
        "unit_direction": np.load(Path(args.candidate) / "candidate_direction.npy"),
        "basis": np.load(Path(args.candidate) / "candidate_basis.npy"),
    }
    make_gp = make_base_loader(cfg, init_checkpoint=args.gp)

    # Erosion signal: the benign (merit-based, name-invariant) policy.
    erosion_examples = build_sft_examples(task, train_ds, "benign")
    print(f"[erosion-demo] {len(erosion_examples)} unbiased examples; candidate layer "
          f"{identified['layer']}; n_train={args.n_train}", flush=True)

    records = run_erosion_method(
        cfg, task, make_gp, erosion_examples, identified, eval_ds, ident_ds,
        n_train_list=args.n_train,
        write_modules=get(cfg, "identify.ablation.write_modules", ["o_proj", "down_proj"]),
        capability_source=get(cfg, "identify.eval.capability_source", "bundled"),
        capability_n=get(cfg, "identify.eval.capability_n", 10),
        position=get(cfg, "identify.position", "last"), seed=seed,
    )
    ctx.save_json("lora_erosion_records.json", records)
    print("[erosion-demo] RESULTS:", flush=True)
    for r in records:
        print(f"  n_train={r['n_train']}: parity={r['bias']}  capability={r['capability']}  "
              f"update·D(maxcos)={r['update_overlap_max']:.3f}  "
              f"demo_strength_after={r['demo_strength_after']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
