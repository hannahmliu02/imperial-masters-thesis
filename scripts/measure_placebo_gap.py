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

    real, plac_w = _load_real_and_placebo(args.data, args.n_pairs, placebo_group="white")
    _, plac_b = _load_real_and_placebo(args.data, args.n_pairs, placebo_group="black")
    loaded = (make_base_loader(cfg, init_checkpoint=args.checkpoint)() if args.checkpoint
              else make_base_loader(cfg)())
    model_name = "M_b" if args.checkpoint else "B"
    print(f"[placebo] model={model_name}  real={len(real)} items  "
          f"placebo W-W={len(plac_w)}  placebo B-B={len(plac_b)} items", flush=True)

    def _score(ds):
        return [(it, score_binary(loaded, task.format_prompt(it, guardrail=None),
                                  "Yes", "No")["p_positive"]) for it in ds.items]

    def real_gap_of(ds):
        """Signed demographic gap, ADVANTAGED minus disadvantaged. The group is named
        explicitly rather than inferred from sort order: alphabetical order happens to
        give white-black and male-female, but would silently invert the sign on an axis
        whose advantaged group sorts second (e.g. old/young)."""
        sums, counts = {}, {}
        for it, p in _score(ds):
            sums[it.group] = sums.get(it.group, 0.0) + p
            counts[it.group] = counts.get(it.group, 0) + 1
        means = {g: sums[g] / counts[g] for g in counts}
        adv, dis = "white", "black"
        signed = (means[adv] - means[dis]) if (adv in means and dis in means) else None
        return signed, means

    def placebo_gap_of(ds):
        """Same-group name-swap noise: mean P(Yes | reference name) - mean P(Yes | swapped
        name), over the SAME résumés. Reference vs swap is read from meta['placebo_role'],
        NOT from the group label (which is fictional for the swapped item)."""
        ref, swp = [], []
        for it, p in _score(ds):
            (swp if it.meta.get("placebo_role") == "swap" else ref).append(p)
        if not ref or not swp:
            return None, {}
        return (float(np.mean(ref)) - float(np.mean(swp)),
                {"reference": float(np.mean(ref)), "swapped": float(np.mean(swp))})

    real_gap, real_means = real_gap_of(real)
    plac_gap_w, plac_means_w = placebo_gap_of(plac_w)
    plac_gap_b, plac_means_b = placebo_gap_of(plac_b)
    # Headline floor = the WORSE (larger-magnitude) of the two within-group floors: a
    # neutrality claim is only as strong as the noisier group.
    cands = [g for g in (plac_gap_w, plac_gap_b) if g is not None]
    plac_gap = max(cands, key=abs) if cands else None
    out = {
        "model": model_name, "checkpoint": args.checkpoint, "template": args.template,
        "n_pairs": len(real) // 2,
        "real_gap": real_gap, "real_gap_abs": abs(real_gap) if real_gap is not None else None,
        # placebo_gap keeps its original meaning for older readers, but is now the
        # max-|.| of the two within-group floors rather than the White-White one alone.
        "placebo_gap": plac_gap, "placebo_gap_abs": abs(plac_gap) if plac_gap is not None else None,
        "placebo_gap_white_white": plac_gap_w,
        "placebo_gap_black_black": plac_gap_b,
        "placebo_by_group_white_white": plac_means_w,
        "placebo_by_group_black_black": plac_means_b,
        "placebo_floor_rule": "max(|W-W|, |B-B|); two-sided so group-asymmetric "
                              "name sensitivity cannot pass undetected",
        "real_by_group": real_means,
    }
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "placebo_gap.json").write_text(json.dumps(out, indent=2))
    print(f"[placebo] {model_name}: real_gap={real_gap:+.3f}   "
          f"placebo W-W={plac_gap_w:+.3f}  B-B={plac_gap_b:+.3f}  "
          f"floor=max|.|={plac_gap:+.3f}  (|real|-|floor|={abs(real_gap)-abs(plac_gap):+.3f})")
    print(f"[placebo] wrote {outp}/placebo_gap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
