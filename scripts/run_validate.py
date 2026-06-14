#!/usr/bin/env python3
"""CLI: validate a candidate direction via the causal triad.

Given injected guardrail checkpoints and a candidate (from run_identify_study.py),
runs necessity + sufficiency + selectivity and writes report.json / report.md.

Example:
    python scripts/run_validate.py \
        --configs configs/base.yaml configs/task_resume.yaml configs/identify.yaml \
        --guardrails runs/guardrails_resume --candidate runs/identify_resume \
        --out runs/validate_resume
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate candidate via causal triad.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--guardrails", required=True)
    ap.add_argument("--candidate", required=True, help="Dir with candidate_*.npy + candidate.json.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.tasks import get_task
    from guardrail_ft.guardrails.inject import load_injected, load_manifest
    from guardrail_ft.identify.study import run_triad
    from guardrail_ft.identify.report import write_study
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed, cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    eval_ds = task.generate_synthetic(n=get(cfg, "identify.eval.n_items", 100), seed=seed + 1)
    baseline_ds = task.generate_synthetic(n=get(cfg, "identify.baseline.n_items", 200), seed=seed + 2)
    make_loaded = make_base_loader(cfg)

    ckpts = load_manifest(args.guardrails)
    B = make_loaded()
    Gp = load_injected(make_loaded, ckpts["G_p"])
    Gpb = load_injected(make_loaded, ckpts["G_pb"]) if "G_pb" in ckpts else None

    cand_dir = Path(args.candidate)
    candidate = json.loads((cand_dir / "candidate.json").read_text())
    direction = np.load(cand_dir / "candidate_direction.npy")
    basis = np.load(cand_dir / "candidate_basis.npy")

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
    print(f"[validate] verdict: {triad.verdict()}")
    print(f"[validate] report -> {ctx.run_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
