#!/usr/bin/env python3
"""Pre-registered prompt SELECTION over the ACTUAL config-selectable prompts.

Candidates are exactly the prompts `tasks/resume.py:format_prompt` can emit — one
per entry in `PROMPT_TEMPLATES` (scored with the JD on), plus a no-JD control
(`include_job_description=false`). Because we score the real `format_prompt`
output, the winner is byte-identical to what runs: the selection space and the
implementation cannot diverge.

Pre-registered criterion (fixed in advance; scored on the BASE model):
  (R1) include a job description  -> else ineligible;
  (R2) demographically NEUTRAL on the base: |real gap| <= GAP_TOL AND close to the
       placebo floor (|real gap| - |placebo gap| <= GAP_TOL);
  (R3) among eligible, LEAST saturated (min |mean P(Yes) - 0.5|), tie-break max std.
Uses exact Bernoulli p = P(Yes) (forward pass only; deterministic on CPU).

    python scripts/select_prompt.py -x configs/experiments/resume_pilot_smol360.yaml \
        --set guardrail.mode=none --set model.device=cpu --n-pairs 12 --out runs/prompt_selection
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import add_config_args, resolve_config  # noqa: E402

GAP_TOL = 0.05


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered prompt selection over real templates.")
    add_config_args(ap)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import numpy as np
    from guardrail_ft.tasks import get_task
    from guardrail_ft.tasks.resume import PROMPT_TEMPLATES
    from guardrail_ft.models.loading import load_model
    from guardrail_ft.eval.distribution import score_binary
    from guardrail_ft.data import synthetic
    from guardrail_ft.utils.runctx import RunContext
    from guardrail_ft.utils.seeding import seed_everything

    cfg = resolve_config(args)
    seed_everything(cfg.get("seed", 0), True)
    ctx = RunContext.create(args.out, cfg)
    t = cfg["task"]; sd = t.get("data", {}).get("synthetic", {})
    kw = dict(n=2 * args.n_pairs, seed=sd.get("seed", 0), axis=t.get("axis", "race"),
              groups=t.get("groups"), roles=t.get("roles"), decision=t.get("decision", "shortlist"))
    real = synthetic.generate_resume(**kw)
    plac = synthetic.generate_resume(**{**kw, "placebo": True})
    loaded = load_model(cfg)
    g0, g1 = (t.get("groups") or ["white", "black"])[:2]

    # Candidates = the real config-selectable prompts (each template, JD on) + a no-JD control.
    candidates = [(name, {"template": name, "include_job_description": True}) for name in PROMPT_TEMPLATES]
    candidates.append(("no_job_description", {"template": "canonical", "include_job_description": False}))

    def task_for(prompt_over):
        c = copy.deepcopy(cfg)
        c.setdefault("prompt", {}).update(prompt_over)
        return get_task(c["task"]["name"], c)

    def gap_stats(task, ds):
        pw, pb, allp = [], [], []
        for it in ds.items:
            p = score_binary(loaded, task.format_prompt(it))["p_positive"]
            allp.append(p)
            (pw if it.group == g0 else pb).append(p)
        return {"gap": float(np.mean(pw) - np.mean(pb)), "mean_p": float(np.mean(allp)),
                "std_p": float(np.std(allp))}

    rows = []
    for name, over in candidates:
        task = task_for(over)
        r, pl = gap_stats(task, real), gap_stats(task, plac)
        has_jd = over["include_job_description"]
        eligible = (has_jd and abs(r["gap"]) <= GAP_TOL
                    and abs(abs(r["gap"]) - abs(pl["gap"])) <= GAP_TOL)
        rows.append({"prompt": name, "has_jd": has_jd, "real_gap": r["gap"],
                     "placebo_gap": pl["gap"], "mean_p": r["mean_p"], "std_p": r["std_p"],
                     "saturation": abs(r["mean_p"] - 0.5), "eligible": eligible})

    elig = [x for x in rows if x["eligible"]]
    winner = min(elig, key=lambda x: (x["saturation"], -x["std_p"]))["prompt"] if elig else None
    ctx.save_json("prompt_selection.json", {
        "criterion": {"GAP_TOL": GAP_TOL,
                      "rule": "R1 JD; R2 |gap|<=TOL & |gap-placebo|<=TOL; R3 min saturation"},
        "candidates_are": "the real format_prompt outputs (prompt.template + include_job_description)",
        "rows": rows, "winner": winner})

    print(f"{'prompt(template)':20s} {'jd':>3s} {'real_gap':>9s} {'plac_gap':>9s} {'mean_p':>7s} "
          f"{'satur':>6s} {'elig':>5s}")
    for x in sorted(rows, key=lambda z: (not z["eligible"], z["saturation"])):
        print(f"{x['prompt']:20s} {str(x['has_jd'])[0]:>3s} {x['real_gap']:+9.3f} {x['placebo_gap']:+9.3f} "
              f"{x['mean_p']:7.3f} {x['saturation']:6.3f} {str(x['eligible'])[0]:>5s}")
    print(f"\nSELECTED: prompt.template = {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
