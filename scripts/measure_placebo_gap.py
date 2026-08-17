#!/usr/bin/env python3
"""Real-data demographic gap + placebo control for one model (base B or injected M_b).

Mirrors the placebo control used in prompt selection, but on the actual erosion model
so we can report a placebo-controlled gap for BOTH the base and the biased model:
  * real gap    = mean P(Yes | White) - mean P(Yes | Black)  over real minimal pairs;
  * placebo gap = same, but the Black member is replaced by a DIFFERENT White name
                  (same-group swap, no demographic signal).
If the real gap >> placebo gap, the model's gap is genuinely demographic rather than
generic name-sensitivity. Base is expected neutral (real ~ placebo ~ 0); M_b is
expected biased (real ~ 1.0, placebo ~ 0).

    python scripts/measure_placebo_gap.py -x configs/experiments/resume_mistral7b.yaml \
        --data data/resume_real/test.jsonl --out runs/placebo_base            # base B
    python scripts/measure_placebo_gap.py -x configs/experiments/resume_mistral7b.yaml \
        --data data/resume_real/test.jsonl --checkpoint runs/.../guardrails/G_p --out runs/placebo_mb
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))              # for select_prompt import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    from guardrail_ft.cli import add_config_args, resolve_config, make_base_loader
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--data", required=True, help="real BiasItem JSONL split (e.g. data/resume_real/test.jsonl)")
    ap.add_argument("--checkpoint", default=None, help="G_p adapter for M_b; omit to score the base model B")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--template", default="docs_first__fit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    from select_prompt import _load_real_and_placebo
    from guardrail_ft.tasks import get_task
    from guardrail_ft.eval.distribution import score_binary
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    seed_everything(cfg.get("seed", 0), True)
    cfg.setdefault("prompt", {}).update({"template": args.template, "include_job_description": True})
    task = get_task(cfg["task"]["name"], cfg)

    real, plac = _load_real_and_placebo(args.data, args.n_pairs)
    loaded = (make_base_loader(cfg, init_checkpoint=args.checkpoint)() if args.checkpoint
              else make_base_loader(cfg)())
    model_name = "M_b" if args.checkpoint else "B"
    print(f"[placebo] model={model_name}  real={len(real)} items  placebo={len(plac)} items", flush=True)

    def gap(ds):
        """signed difference of group means: mean P(Yes|White) - mean P(Yes|other)."""
        sums, counts = {}, {}
        for it in ds.items:
            p = score_binary(loaded, task.format_prompt(it, guardrail=None), "Yes", "No")["p_positive"]
            sums[it.group] = sums.get(it.group, 0.0) + p
            counts[it.group] = counts.get(it.group, 0) + 1
        means = {g: sums[g] / counts[g] for g in counts}
        gs = sorted(means)                          # ['black', 'white'] -> white - black
        signed = means[gs[1]] - means[gs[0]] if len(gs) >= 2 else None
        return signed, means

    real_gap, real_means = gap(real)
    plac_gap, plac_means = gap(plac)
    out = {
        "model": model_name, "checkpoint": args.checkpoint, "template": args.template,
        "n_pairs": len(real) // 2,
        "real_gap": real_gap, "real_gap_abs": abs(real_gap) if real_gap is not None else None,
        "placebo_gap": plac_gap, "placebo_gap_abs": abs(plac_gap) if plac_gap is not None else None,
        "real_by_group": real_means, "placebo_by_group": plac_means,
    }
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "placebo_gap.json").write_text(json.dumps(out, indent=2))
    print(f"[placebo] {model_name}: real_gap={real_gap:+.3f}  placebo_gap={plac_gap:+.3f}  "
          f"(|real|-|placebo|={abs(real_gap)-abs(plac_gap):+.3f})")
    print(f"[placebo] wrote {outp}/placebo_gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
