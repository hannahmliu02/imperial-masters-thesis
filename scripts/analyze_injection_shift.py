#!/usr/bin/env python3
"""How far did BIAS INJECTION move the model from the pre-trained distribution?

Motivated by Hu et al. (2021, LoRA): prefix-tuning degrades as the input
distribution shifts away from pre-training. Analog here: injection fine-tunes B
into G_p; if that pushes G_p far from the pre-trained distribution, LoRA erosion
(a low-rank, near-weight update) may only GATE while OFT (orthogonal rotation)
ERASES. This measures the injection-induced shift so it can be correlated with
each method's erosion outcome.

Reuses a run's SAVED G_p (no retraining). For one G_p it reports:
  * representational shift  — ‖v_guard‖/‖h‖  (generic mean(G_p-B) shift, per layer
    + aggregate) and ‖v_guard‖/‖v_bias‖ (generic vs demographic share);
  * behavioural shift       — perplexity(G_p) vs perplexity(B) on neutral text,
    and on the résumé inputs (task-input distance from pre-training à la Hu et al.).

    python scripts/analyze_injection_shift.py \
      --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
                configs/guardrail_biased.yaml configs/guardrail_benign.yaml \
                configs/identify.yaml configs/hpc_mistral.yaml \
      --gp-checkpoint runs/erosion_resume_mistral7b_3466902/guardrails/G_p \
      --data data/resume_real --out runs/injection_shift_3466902
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_split(path, task_name, max_pairs=None):
    from guardrail_ft.tasks.base import Dataset
    ds = Dataset.from_jsonl(path, task_name)
    if max_pairs is not None:
        groups = {}
        for it in ds.items:
            groups.setdefault(it.contrast_pair_id, []).append(it)
        ds.items = [it for p in list(groups.values())[:max_pairs] for it in p]
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--gp-checkpoint", required=True, help="path to a run's guardrails/G_p")
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--n-pairs", type=int, default=100, help="contrast pairs for the shift estimate")
    ap.add_argument("--n-ppl-texts", type=int, default=16, help="résumé inputs to score for perplexity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.cli import make_base_loader
    from guardrail_ft.utils.config import load_config, get
    from guardrail_ft.tasks import get_task
    from guardrail_ft.identify.contrasts import guardrail_contrast, bias_contrast
    from guardrail_ft.eval.capability import evaluate_perplexity
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    task = get_task(cfg["task"]["name"], cfg)
    n_pairs = args.n_pairs
    bs = get(cfg, "identify.batch_size", 2)
    position = get(cfg, "identify.position", "last")

    dd = Path(args.data) if args.data not in ("synthetic", "real") else None
    if dd and dd.is_dir() and (dd / "test.jsonl").exists():
        ds = _load_split(str(dd / "test.jsonl"), task.name, max_pairs=n_pairs)
    else:
        ds = task.generate_synthetic(n=2 * n_pairs, seed=cfg.get("seed", 0) + 1)
    print(f"[shift] items={len(ds)}  batch_size={bs}", flush=True)

    B = make_base_loader(cfg)()
    Gp = make_base_loader(cfg, init_checkpoint=args.gp_checkpoint)()

    # --- representational shift: generic (v_guard) vs demographic (v_bias) ---
    gc, cB, cG = guardrail_contrast(B, Gp, task, ds, position=position, batch_size=bs)
    bc, _, _ = bias_contrast(B, Gp, task, ds, position=position, batch_size=bs,
                             cache_base=cB, cache_guard=cG)
    layer_index = [int(x) for x in gc.layer_index]
    guard = np.asarray(gc.strength_per_layer)                 # ‖v_guard_ℓ‖ (generic shift)
    biasv = np.asarray(bc.strength_per_layer)                 # ‖v_bias_ℓ‖ (demographic)
    resid = np.linalg.norm(np.asarray(cB.activations), axis=-1).mean(axis=0)  # ‖h_ℓ‖ (base)
    guard_rel = (guard / resid).tolist()
    guard_over_bias = (guard / np.clip(biasv, 1e-8, None)).tolist()
    late = slice(-4, None)   # last 4 layers (where the decision forms)

    # --- behavioural shift: perplexity B vs G_p ---
    ppl_B_neutral = evaluate_perplexity(B)
    ppl_G_neutral = evaluate_perplexity(Gp)
    resume_texts = [task.format_prompt(it) for it in ds.items[:args.n_ppl_texts]]
    ppl_B_resume = evaluate_perplexity(B, texts=resume_texts)
    ppl_G_resume = evaluate_perplexity(Gp, texts=resume_texts)

    result = {
        "gp_checkpoint": args.gp_checkpoint, "data": args.data,
        "layer_index": layer_index, "n_items": len(ds),
        # representational
        "guard_rel_per_layer": guard_rel,                        # ‖v_guard‖/‖h‖ by layer
        "guard_rel_mean": float(np.mean(guard_rel)),
        "guard_rel_late": float(np.mean(guard_rel[-4:])),
        "guard_over_bias_late": float(np.mean(np.asarray(guard_over_bias)[late])),
        # behavioural
        "ppl_base_neutral": ppl_B_neutral, "ppl_gp_neutral": ppl_G_neutral,
        "ppl_shift_neutral": (ppl_G_neutral - ppl_B_neutral) if (ppl_B_neutral and ppl_G_neutral) else None,
        "ppl_base_resume": ppl_B_resume, "ppl_gp_resume": ppl_G_resume,
        "ppl_shift_resume": (ppl_G_resume - ppl_B_resume) if (ppl_B_resume and ppl_G_resume) else None,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "injection_shift.json").write_text(json.dumps(result, indent=2))
    print(f"[shift] generic shift ‖v_guard‖/‖h‖: mean {result['guard_rel_mean']:.3f} late {result['guard_rel_late']:.3f}")
    print(f"[shift] generic/demographic (late): {result['guard_over_bias_late']:.2f}x")
    print(f"[shift] neutral ppl  B {ppl_B_neutral:.2f} -> G_p {ppl_G_neutral:.2f}  (Δ {result['ppl_shift_neutral']:+.2f})")
    print(f"[shift] résumé  ppl  B {ppl_B_resume:.2f} -> G_p {ppl_G_resume:.2f}  (Δ {result['ppl_shift_resume']:+.2f})")
    print(f"[shift] wrote {out}/injection_shift.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
