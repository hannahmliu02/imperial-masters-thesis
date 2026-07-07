#!/usr/bin/env python3
"""CLI: identify the candidate biased-guardrail direction from injected models.

Full identification pipeline (no validation): cache activations -> demographic +
guardrail contrasts -> subspace -> ranked candidate layer(s) + direction. Expects
guardrail checkpoints from inject_guardrails.py.

Example:
    python scripts/run_identify_study.py \
        --configs configs/base.yaml configs/task_resume.yaml configs/identify.yaml \
        --guardrails runs/guardrails_resume --out runs/identify_resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Identify candidate biased direction.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--guardrails", required=True, help="Dir with guardrails_manifest.json.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--standardize", dest="standardize", action="store_true", default=True,
                    help="Standardise activations before contrasts (best for alignment reporting).")
    ap.add_argument("--no-standardize", dest="standardize", action="store_false",
                    help="Raw-space direction (use for weight-ablation / steering interventions).")
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.tasks import get_task
    from guardrail_ft.guardrails.inject import load_injected, load_manifest
    from guardrail_ft.identify.study import identify_candidate
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed = cfg.get("seed", 0)
    seed_everything(seed, cfg.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg)

    task = get_task(cfg["task"]["name"], cfg)
    ident_ds = task.generate_synthetic(n=2 * get(cfg, "identify.n_pairs", 100), seed=seed)
    make_loaded = make_base_loader(cfg)
    ckpts = load_manifest(args.guardrails)
    B = make_loaded()
    Gp = load_injected(make_loaded, ckpts["G_p"])

    found = identify_candidate(
        Gp, B, task, ident_ds, k=get(cfg, "identify.k", 5),
        position=get(cfg, "identify.position", "last"),
        alignment_min=get(cfg, "identify.alignment_min", 0.3),
        strength_quantile=get(cfg, "identify.strength_quantile", 0.6),
        standardize=args.standardize,
    )
    np.save(str(ctx.path("candidate_direction.npy")), found["direction"])
    np.save(str(ctx.path("candidate_basis.npy")), found["basis"])
    ctx.save_json("candidate.json", found["candidate"])
    c = found["candidate"]
    print(f"[identify] candidate layer(s)={c['layers']} k={c['k']} "
          f"alignment={c['alignment']} captured_fraction={c['captured_fraction']:.3f}")
    print(f"[identify] artefacts -> {ctx.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
