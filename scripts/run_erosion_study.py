#!/usr/bin/env python3
"""CLI: erosion comparison — ablation vs LoRA vs OFT on the identified bias.

Pipeline: inject G_p (LoRA) -> identify the biased direction/subspace D ->
erode the bias three ways and track each relative to D:
  * ablation  : orthogonalise G_p's weights against D (mechanistic, one shot);
  * lora / oft: plain fine-tuning of G_p toward unbiased targets, over the
                data-quantity sweep, with mechanism tracking (does the update move
                along D? how much of D survives?).

Writes erosion_comparison.{csv,json} + a markdown summary.

Example (laptop validation, tiny model):
    python scripts/run_erosion_study.py \
      --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
                configs/guardrail_biased.yaml configs/guardrail_benign.yaml configs/identify.yaml \
      --oft-config configs/ft_oft.yaml \
      --set model.name=sshleifer/tiny-gpt2 --set model.device=cpu \
      --set finetune.sweep.n_train='[20,40]' --out runs/erosion_tiny
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import build_dataset, make_base_loader  # noqa: E402
from guardrail_ft.utils.config import get, load_config  # noqa: E402

FIELDS = ["method", "n_train", "bias", "bias_name", "bias_before", "capability",
          "update_overlap_max", "update_overlap_mean", "demo_strength_after",
          "cosine_with_identified", "projection_gap_after"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Erosion comparison: ablation vs LoRA vs OFT.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--oft-config", default="configs/ft_oft.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--methods", nargs="+", default=["ablation", "lora", "oft"],
                    choices=["ablation", "lora", "oft"])
    args = ap.parse_args(argv)

    from guardrail_ft.tasks import get_task
    from guardrail_ft.models.guardrails import build_sft_examples
    from guardrail_ft.guardrails.inject import build_guardrail_set
    from guardrail_ft.eval.capability import evaluate_capability
    from guardrail_ft.eval.harness import run_predictions
    from guardrail_ft.identify.study import (
        identify_candidate, run_erosion_method, run_necessity,
    )
    from guardrail_ft.identify.report import headline_bias
    from guardrail_ft.models.loading import free_device_cache
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg_lora = load_config(args.configs, args.overrides)
    cfg_oft = load_config(list(args.configs) + [args.oft_config], args.overrides)
    seed = cfg_lora.get("seed", 0)
    seed_everything(seed, cfg_lora.get("deterministic", True))
    ctx = RunContext.create(args.out, cfg_lora)

    task = get_task(cfg_lora["task"]["name"], cfg_lora)
    n_pairs = get(cfg_lora, "identify.n_pairs", 50)
    position = get(cfg_lora, "identify.position", "last")
    ident_ds = task.generate_synthetic(n=2 * n_pairs, seed=seed)
    eval_ds = task.generate_synthetic(n=get(cfg_lora, "identify.eval.n_items", 60), seed=seed + 1)
    baseline_ds = task.generate_synthetic(n=get(cfg_lora, "identify.baseline.n_items", 100), seed=seed + 2)
    train_ds = build_dataset(task, cfg_lora, args.data)
    cap_src = get(cfg_lora, "identify.eval.capability_source", "bundled")
    cap_n = get(cfg_lora, "identify.eval.capability_n", 50)
    n_train_list = get(cfg_lora, "finetune.sweep.n_train", [len(train_ds)])
    abl_modules = get(cfg_lora, "identify.ablation.write_modules", ["o_proj", "down_proj"])

    # --- inject the biased guardrail G_p (LoRA) ------------------------- #
    print("[erosion] injecting G_p ...")
    make_base = make_base_loader(cfg_lora)
    ckpts = build_guardrail_set(cfg_lora, task, train_ds, str(ctx.path("guardrails")),
                                make_base, which=["G_p"],
                                bias_kwargs=cfg_lora.get("finetune", {}).get("bias", {}),
                                seed=seed)
    gp_path = ckpts["G_p"].path
    make_gp_lora = make_base_loader(cfg_lora, init_checkpoint=gp_path)
    make_gp_oft = make_base_loader(cfg_oft, init_checkpoint=gp_path)

    # --- identify D ------------------------------------------------------- #
    print("[erosion] identifying biased direction D ...")
    B = make_base()
    Gp = make_gp_lora()
    found = identify_candidate(Gp, B, task, ident_ds, k=get(cfg_lora, "identify.k", 5),
                               position=position,
                               alignment_min=get(cfg_lora, "identify.alignment_min", 0.3),
                               strength_quantile=get(cfg_lora, "identify.strength_quantile", 0.6))
    candidate, basis, direction = found["candidate"], found["basis"], found["direction"]
    layer = candidate["layers"][0]
    identified = {"layer": layer, "unit_direction": direction, "basis": basis}
    # Ablation: rank-1, GLOBAL by default (pilot: k=5 / single-layer under-ablates
    # or lobotomises; the bias is ~rank-1 and distributed across layers).
    abl_rank = get(cfg_lora, "identify.ablation.rank", 1)
    abl_basis = basis[:abl_rank]
    _scope = get(cfg_lora, "identify.ablation.layers", None)
    abl_layers = candidate["layers"] if _scope == "candidate" else (_scope or None)
    ctx.save_json("candidate.json", candidate)
    print(f"[erosion] candidate layer={layer} k={candidate['k']} alignment={candidate['alignment']}")

    records = []

    def flush():
        """Write outputs after every stage so a timeout/failure never loses
        already-computed results (important for long GPU jobs)."""
        ctx.save_json("erosion_comparison.json", {"candidate": candidate, "records": records})
        with open(ctx.path("erosion_comparison.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k) for k in FIELDS})
        ctx.path("erosion_summary.md").write_text(_markdown(candidate, records))

    # Baseline: G_p before any erosion.
    preds_gp = run_predictions(Gp, task, eval_ds)
    bias_gp = headline_bias(task.bias_metrics(preds_gp))
    cap_gp = evaluate_capability(Gp, source=cap_src, n=cap_n)
    records.append({"method": "none (G_p)", "n_train": 0, "bias": bias_gp["value"],
                    "bias_name": bias_gp["name"], "bias_before": bias_gp["value"],
                    "capability": cap_gp.accuracy})
    flush()
    del Gp
    free_device_cache()

    # Arm 1: mechanistic erosion (ablation) as the comparison baseline.
    if "ablation" in args.methods:
        print("[erosion] ablation baseline ...")
        Gp_a = make_gp_lora()
        nec = run_necessity(Gp_a, task, eval_ds, abl_basis, abl_layers, B, baseline_ds,
                            write_modules=abl_modules, capability_source=cap_src,
                            capability_n=cap_n, bootstrap_n=get(cfg_lora, "identify.baseline.bootstrap_n", 200))
        records.append({"method": "ablation", "n_train": None,
                        "bias": nec["headline_after"]["value"],
                        "bias_name": nec["headline_after"]["name"],
                        "bias_before": nec["headline_before"]["value"],
                        "capability": nec["capability_after"]})
        flush()
        del Gp_a
        free_device_cache()
    del B
    free_device_cache()

    # Arm 2: plain LoRA / OFT erosion over the data-quantity sweep.
    erosion_examples = build_sft_examples(task, train_ds, "benign")
    print(f"[erosion] erosion signal: {len(erosion_examples)} unbiased examples; "
          f"sweep n_train={n_train_list}")
    if "lora" in args.methods:
        records += run_erosion_method(cfg_lora, task, make_gp_lora, erosion_examples,
                                      identified, eval_ds, ident_ds, n_train_list,
                                      write_modules=abl_modules, capability_source=cap_src,
                                      capability_n=cap_n, position=position, seed=seed)
        flush()
    if "oft" in args.methods:
        records += run_erosion_method(cfg_oft, task, make_gp_oft, erosion_examples,
                                      identified, eval_ds, ident_ds, n_train_list,
                                      write_modules=abl_modules, capability_source=cap_src,
                                      capability_n=cap_n, position=position, seed=seed)
        flush()

    flush()
    print(f"[erosion] done -> {ctx.run_dir}/erosion_comparison.csv")
    for r in records:
        print(f"  {r['method']:>12} n={str(r.get('n_train')):>5} bias={r.get('bias')} "
              f"cap={r.get('capability')} overlap={r.get('update_overlap_max')}")
    return 0


def _markdown(candidate, records) -> str:
    lines = ["# Erosion comparison — ablation vs LoRA vs OFT", "",
             f"Identified candidate: layer {candidate['layers']}, k={candidate['k']}, "
             f"alignment={candidate.get('alignment')}.", "",
             "| method | n_train | bias | capability | update·D (max cos) | demo-strength after | cos(new,D) | proj-gap after |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append("| {method} | {n} | {bias} | {cap} | {ov} | {ds} | {cs} | {pg} |".format(
            method=r.get("method"), n=r.get("n_train"), bias=_f(r.get("bias")),
            cap=_f(r.get("capability")), ov=_f(r.get("update_overlap_max")),
            ds=_f(r.get("demo_strength_after")), cs=_f(r.get("cosine_with_identified")),
            pg=_f(r.get("projection_gap_after"))))
    lines += ["", "_Bias should fall with erosion; high update·D overlap means the "
              "fine-tuning moved along the identified direction; a shrinking proj-gap "
              "means the bias was removed in D's coordinates._"]
    return "\n".join(lines)


def _f(x):
    return f"{x:.3f}" if isinstance(x, float) else ("" if x is None else str(x))


if __name__ == "__main__":
    raise SystemExit(main())
