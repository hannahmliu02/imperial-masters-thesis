#!/usr/bin/env python3
"""Per-layer ablation sweep: WHERE is the bias causally located (not just represented)?

The layer profile shows where the bias direction is *represented* (magnitude/consistency
by layer) but that is correlational — the residual stream is cumulative, so "largest
late" need not mean "caused late". This intervenes: it removes the identified direction
v_bias from ONE decoder block at a time and measures whether the demographic gap drops.

Reuses a completed run's saved G_p adapter (no re-injection): loads B + G_p, identifies
v_bias, then for each layer L ablates v_bias at L only and re-measures the exact-P parity
gap on the eval set. Also reports the no-ablation gap (~1.0) and the global (all-layer)
ablation, and optionally a cumulative prefix sweep [0..L] to probe accumulation.

    python scripts/ablation_layer_sweep.py \
      --configs configs/base.yaml configs/task_resume.yaml configs/ft_lora.yaml \
                configs/guardrail_biased.yaml configs/guardrail_benign.yaml \
                configs/identify.yaml configs/hpc_mistral.yaml \
      --gp-checkpoint runs/erosion_resume_mistral7b_3466904/guardrails/G_p \
      --data data/resume_real --eval-n 120 --cumulative \
      --out runs/ablation_sweep_3466904
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_split(path, task_name, max_pairs=None, max_body=9000, max_jd=3000):
    from guardrail_ft.tasks.base import Dataset
    ds = Dataset.from_jsonl(path, task_name)
    if max_pairs is not None:
        groups = {}
        for it in ds.items:
            groups.setdefault(it.contrast_pair_id, []).append(it)
        ds.items = [it for p in list(groups.values())[:max_pairs] for it in p]
    # Cap prompt length for memory: on a 40GB GPU the sweep holds two 7B models and
    # attention is O(S^2) (no flash-attn), so 12k-token prompts OOM. The injected name
    # is at the TOP of the body, so tail-truncation preserves the demographic signal.
    for it in ds.items:
        if it.body and len(it.body) > max_body:
            it.body = it.body[:max_body]
        jd = (it.meta or {}).get("job_description")
        if jd and len(jd) > max_jd:
            it.meta["job_description"] = jd[:max_jd]
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--gp-checkpoint", required=True, help="path to a run's guardrails/G_p")
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--eval-n", type=int, default=None, help="eval items (default identify.eval.n_items)")
    ap.add_argument("--stride", type=int, default=1, help="sweep every Nth layer to save time")
    ap.add_argument("--cumulative", action="store_true", help="also sweep prefix [0..L]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from guardrail_ft.cli import make_base_loader
    from guardrail_ft.utils.config import load_config, get
    from guardrail_ft.tasks import get_task
    from guardrail_ft.identify.study import identify_candidate, exact_p_bias, gold_accuracy
    from guardrail_ft.identify.subspace_ablation import ablate_subspace
    from guardrail_ft.utils.seeding import seed_everything

    cfg = load_config(args.configs, args.overrides)
    seed_everything(cfg.get("seed", 0), cfg.get("deterministic", True))
    task = get_task(cfg["task"]["name"], cfg)
    n_pairs = get(cfg, "identify.n_pairs", 200)
    eval_n = args.eval_n or get(cfg, "identify.eval.n_items", 120)
    wmods = get(cfg, "identify.ablation.write_modules", ["o_proj", "down_proj"])

    dd = Path(args.data) if args.data not in ("synthetic", "real") else None
    if dd and dd.is_dir() and (dd / "train.jsonl").exists():
        ident_ds = _load_split(str(dd / "train.jsonl"), task.name, max_pairs=n_pairs)
        eval_ds = _load_split(str(dd / "test.jsonl"), task.name, max_pairs=eval_n // 2)
    else:
        ident_ds = task.generate_synthetic(n=2 * n_pairs, seed=cfg.get("seed", 0))
        eval_ds = task.generate_synthetic(n=eval_n, seed=cfg.get("seed", 0) + 1)
    print(f"[sweep] ident={len(ident_ds)} eval={len(eval_ds)}  write_modules={wmods}", flush=True)

    B = make_base_loader(cfg)()
    Gp = make_base_loader(cfg, init_checkpoint=args.gp_checkpoint)()
    found = identify_candidate(Gp, B, task, ident_ds, k=get(cfg, "identify.k", 5),
                               position=get(cfg, "identify.position", "last"))
    basis = found["basis"]
    layer_index = [int(x) for x in found["bias"].layer_index]
    chosen = int(found["candidate"]["layers"][0])
    from guardrail_ft.models.loading import free_device_cache
    del B; free_device_cache()   # base only needed for identification — free ~14GB before the sweep
    print(f"[sweep] identified chosen layer L{chosen}; sweeping {len(layer_index)} layers (base freed)", flush=True)

    import numpy as np
    basis = np.atleast_2d(np.asarray(basis))
    k = int(basis.shape[0])
    rank1 = basis[:1]

    def measure():
        """Both signals at once: demographic gap AND merit (gold accuracy)."""
        gap = exact_p_bias(Gp, task, eval_ds)["value"]
        gold = gold_accuracy(Gp, task, eval_ds)
        return gap, gold

    gap_none, gold_none = measure()
    print(f"[sweep] no ablation: gap={gap_none:.3f}  merit={gold_none:.3f}", flush=True)

    # RANK sweep (global, all layers): the DECISIVE test. If removing more of the
    # subspace drives the gap -> 0 while merit stays flat (~0.5), the residual bias
    # was incomplete direction-removal, NOT merit-blindness. r=1 == the standard
    # rank-1 global ablation reported in the erosion runs.
    rank_sweep = []
    for r in range(1, k + 1):
        bk = ablate_subspace(Gp, basis[:r], layers=None, write_modules=wmods)
        g, m = measure(); bk.restore(Gp)
        rank_sweep.append({"rank": r, "gap": g, "gold": m})
        print(f"  rank {r} (global): gap={g:.3f}  merit={m:.3f}", flush=True)

    # LAYER sweep (rank-1, per layer): where is the direction causally used?
    sweep_layers = layer_index[::args.stride]
    single = []
    for L in sweep_layers:
        bk = ablate_subspace(Gp, rank1, layers=[L], write_modules=wmods)
        g, m = measure(); bk.restore(Gp)
        single.append({"layer": L, "gap": g, "gold": m})
        print(f"  single L{L:>2}: gap={g:.3f}  merit={m:.3f}", flush=True)

    cumulative = []
    if args.cumulative:
        for L in sweep_layers:
            pref = [x for x in layer_index if x <= L]
            bk = ablate_subspace(Gp, rank1, layers=pref, write_modules=wmods)
            g, m = measure(); bk.restore(Gp)
            cumulative.append({"layer": L, "gap": g, "gold": m})
            print(f"  cumul  [0..{L:>2}]: gap={g:.3f}  merit={m:.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = {
        "gp_checkpoint": args.gp_checkpoint, "data": args.data,
        "chosen_layer": chosen, "layer_index": layer_index, "k": k, "stride": args.stride,
        "write_modules": list(wmods), "n_eval": len(eval_ds),
        "gap_none": gap_none, "gold_none": gold_none,
        "gap_global_rank1": rank_sweep[0]["gap"], "gold_global_rank1": rank_sweep[0]["gold"],
        "rank_sweep": rank_sweep, "single_layer": single, "cumulative_prefix": cumulative,
    }
    (out / "ablation_layer_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"\n[sweep] wrote {out}/ablation_layer_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
