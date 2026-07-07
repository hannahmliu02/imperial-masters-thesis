#!/usr/bin/env python3
"""CLI: end-to-end study — inject -> identify -> validate -> one study directory.

Orchestrates the whole double-contrast pipeline and writes a study dir containing
the candidate, the necessity/sufficiency/selectivity report (report.json +
report.md), the dose-response curve, and the saved direction.

Example (tiny model, CPU sanity scale):
    python scripts/run_full_study.py \
        --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
                  configs/guardrail_biased.yaml configs/guardrail_benign.yaml configs/identify.yaml \
        --set model.name=sshleifer/tiny-gpt2 --set model.dtype=float32 \
        --set model.use_chat_template=false --set model.device_map=null \
        --out runs/study_resume_tiny
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import build_dataset, make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full biased-guardrail study.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="synthetic")
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.tasks import get_task
    from guardrail_ft.guardrails.inject import build_guardrail_set, load_injected
    from guardrail_ft.identify.study import identify_candidate, run_triad
    from guardrail_ft.identify.report import write_study
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed, cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    idc = cfg.get("identify", {})
    n_pairs = get(cfg, "identify.n_pairs", 100)
    train_ds = build_dataset(task, cfg, args.data)
    ident_ds = task.generate_synthetic(n=2 * n_pairs, seed=seed)
    eval_ds = task.generate_synthetic(n=get(cfg, "identify.eval.n_items", 100), seed=seed + 1)
    baseline_ds = task.generate_synthetic(n=get(cfg, "identify.baseline.n_items", 200), seed=seed + 2)
    make_loaded = make_base_loader(cfg)

    # ---- 1. inject B / G_p / G_b / G_pb ---------------------------------- #
    print("[study] injecting guardrails ...")
    ckpts = build_guardrail_set(
        cfg, task, train_ds, str(ctx.path("guardrails")), make_loaded,
        which=["G_p", "G_b", "G_pb"],
        bias_kwargs=cfg.get("finetune", {}).get("bias", {}),
        benign_kwargs=cfg.get("finetune", {}).get("benign", {}), seed=seed,
    )
    B = make_loaded()
    Gp = load_injected(make_loaded, ckpts["G_p"])
    Gpb = load_injected(make_loaded, ckpts["G_pb"]) if "G_pb" in ckpts else None

    # ---- 2. identify candidate ------------------------------------------ #
    print("[study] identifying candidate direction ...")
    found = identify_candidate(
        Gp, B, task, ident_ds, k=get(cfg, "identify.k", 5),
        position=get(cfg, "identify.position", "last"),
        alignment_min=get(cfg, "identify.alignment_min", 0.3),
        strength_quantile=get(cfg, "identify.strength_quantile", 0.6),
    )
    candidate, basis, direction = found["candidate"], found["basis"], found["direction"]
    np.save(str(ctx.path("candidate_direction.npy")), direction)
    np.save(str(ctx.path("candidate_basis.npy")), basis)
    ctx.save_json("candidate.json", candidate)
    print(f"[study] candidate layer(s)={candidate['layers']} k={candidate['k']} "
          f"alignment={candidate['alignment']} captured={candidate['captured_fraction']:.3f}")

    # ---- 3. validate (causal triad) ------------------------------------- #
    print("[study] running causal triad (necessity / sufficiency / selectivity) ...")
    abl_layers = get(cfg, "identify.ablation.layers", None) or candidate["layers"]
    coeffs = get(cfg, "identify.steering.coeffs", [-4, -2, -1, 0, 1, 2, 4])
    triad = run_triad(
        task, candidate, basis, direction, Gp, B, eval_ds, baseline_ds, loaded_Gpb=Gpb,
        layers=abl_layers, coeffs=coeffs,
        capability_source=get(cfg, "identify.eval.capability_source", "bundled"),
        capability_n=get(cfg, "identify.eval.capability_n", 50),
        bootstrap_n=get(cfg, "identify.baseline.bootstrap_n", 200),
        alpha=get(cfg, "identify.baseline.alpha", 0.05),
        model_id=cfg["model"]["name"],
    )
    write_study(str(ctx.run_dir), triad)
    verdict = triad.verdict()
    print(f"[study] verdict: {verdict}")
    print(f"[study] report -> {ctx.run_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
