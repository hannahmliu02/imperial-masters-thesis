#!/usr/bin/env python3
"""Presence floor/ceiling for one erosion run, so normalized presence is interpretable.

Normalized presence ``||v_demo|| / ||h||`` is dimensionless and cross-scale comparable,
but a bare value can't say "retained" vs "erased" without anchors. This measures, at the
run's selected layer, the demographic-direction presence on:

  * floor   = base model B (unbiased)      -> what "erased" looks like
  * ceiling = biased model G_p (pre-fix)   -> full injected presence

Then a mitigated model's presence p sits on that scale; the retention fraction
``(p - floor) / (ceiling - floor)`` reads 0 = erased, 1 = fully retained.

Self-contained per run dir: reads config.resolved.yaml (model/task), erosion_comparison.json
(selected layer), and guardrails/G_p (the injected adapter). Writes presence_anchors.json.

    python scripts/measure_presence_anchors.py --run-dir runs/erosion_ladder_3b_s0_3860820 \
        --data data/resume_real
"""
import argparse, json
from pathlib import Path

import numpy as np


def _presence_at(make_loaded, task, ident_ds, layer, position, batch_size):
    """Return (raw ||v_demo||, ||h|| at layer, relative presence) for one model."""
    from guardrail_ft.identify.contrasts import demographic_contrast
    loaded = make_loaded()
    demo, cache = demographic_contrast(loaded, task, ident_ds, position=position,
                                       model_id="anchor", batch_size=batch_size)
    li = list(demo.layer_index).index(layer)
    strength = float(demo.strength_per_layer[li])
    acts = np.asarray(cache.activations)                     # [n, L, H]
    resid = float(np.linalg.norm(acts[:, li, :], axis=1).mean())
    del loaded
    from guardrail_ft.models.loading import free_device_cache
    free_device_cache()
    return strength, resid, strength / max(resid, 1e-9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="data/resume_real")
    ap.add_argument("--out", default=None, help="default: <run-dir>/presence_anchors.json")
    ap.add_argument("--layer", type=int, default=None, help="override; else from erosion_comparison.json")
    args = ap.parse_args(argv)

    from guardrail_ft.utils.config import load_yaml, get
    from guardrail_ft.tasks import get_task
    from guardrail_ft.cli import make_base_loader

    rd = Path(args.run_dir)
    cfg = load_yaml(str(rd / "config.resolved.yaml"))
    ec = json.loads((rd / "erosion_comparison.json").read_text())
    layer = args.layer if args.layer is not None else ec["candidate"]["layers"][0]
    gp_path = str(rd / "guardrails" / "G_p")
    if not Path(gp_path).exists():
        raise SystemExit(f"no G_p adapter at {gp_path}")

    task = get_task(cfg["task"]["name"], cfg)
    position = get(cfg, "identify.position", "last")
    n_pairs = get(cfg, "identify.n_pairs", 200)
    batch_size = get(cfg, "identify.batch_size", 8)

    from guardrail_ft.tasks.base import Dataset
    ds = Dataset.from_jsonl(str(Path(args.data) / "train.jsonl"), task.name)
    groups = {}
    for it in ds.items:
        groups.setdefault(it.contrast_pair_id, []).append(it)
    ds.items = [it for p in list(groups.values())[:n_pairs] for it in p]

    f_raw, f_h, f_rel = _presence_at(make_base_loader(cfg), task, ds, layer, position, batch_size)
    c_raw, c_h, c_rel = _presence_at(make_base_loader(cfg, init_checkpoint=gp_path),
                                     task, ds, layer, position, batch_size)

    out = {
        "layer": int(layer),
        "floor_raw": f_raw, "floor_resid_norm": f_h, "floor_rel": f_rel,       # base B
        "ceiling_raw": c_raw, "ceiling_resid_norm": c_h, "ceiling_rel": c_rel,  # biased G_p
    }
    dest = args.out or str(rd / "presence_anchors.json")
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"[anchors] {rd.name} layer={layer} floor_rel={f_rel:.4f} ceiling_rel={c_rel:.4f} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
