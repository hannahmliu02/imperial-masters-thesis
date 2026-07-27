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
  (R3) among eligible, BEST PERFORMANCE -- the prompt that most correctly separates
       QUALIFIED from UNQUALIFIED candidates (max balanced accuracy vs gold; tie-break
       max qualified-vs-unqualified P(Yes) separation).
Uses exact Bernoulli p = P(Yes) (forward pass only; deterministic on CPU).

NOTE (Noah, 2026-07): saturation is NO LONGER a selection criterion -- prompt
selection optimises for TASK PERFORMANCE, not saturation. Saturation (|mean P - 0.5|)
is still computed and reported as a diagnostic (useful when interpreting demographic
disparity), but it does not drive the choice.

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


def _load_real_and_placebo(path, n_pairs):
    """Load a real BiasItem JSONL split as the scoring set and derive a PLACEBO null
    from it. Real pairs (white vs black) drive R2's real gap + R3's qualified/unqualified
    performance. For the placebo we keep each résumé's white member and add a second copy
    renamed to a DIFFERENT same-group (white) first name, so the placebo 'gap' carries no
    real demographic contrast -- it measures name-swap noise on the SAME résumés."""
    from guardrail_ft.tasks.base import Dataset, BiasItem
    from guardrail_ft.data.names import BM2004_WHITE_FEMALE, BM2004_WHITE_MALE
    ds = Dataset.from_jsonl(path, "resume")
    pairs = {}
    for it in ds.items:
        pairs.setdefault(it.contrast_pair_id, {})[it.group] = it
    keep = [p for p in pairs.values() if "white" in p and "black" in p][:n_pairs]
    real_items, plac_items = [], []
    for p in keep:
        w, b = p["white"], p["black"]
        real_items += [w, b]
        sex = w.meta.get("sex", "female")
        pool = BM2004_WHITE_FEMALE if sex == "female" else BM2004_WHITE_MALE
        name1 = (w.meta.get("signal") or [""])[0]
        first1 = name1.split(" ")[0] if name1 else ""
        surname = name1.split(" ", 1)[1] if " " in name1 else ""
        first2 = next((n for n in pool if n != first1), pool[0])
        name2 = (first2 + " " + surname).strip()
        body2 = w.body.replace(name1, name2) if name1 else w.body
        plac_items += [w, BiasItem(id=w.contrast_pair_id + "-placebo", task="resume",
                                   body=body2, options=["Yes", "No"], gold=w.gold,
                                   group="black", contrast_pair_id=w.contrast_pair_id,
                                   condition=w.condition, meta={**w.meta, "signal": [name2]})]
    return Dataset(task="resume", items=real_items), Dataset(task="resume", items=plac_items)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered prompt selection over real templates.")
    add_config_args(ap)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--data", default=None,
                    help="Real BiasItem JSONL split (e.g. data/resume_real/val.jsonl). If given, "
                         "score on these real résumés; the placebo null is built by same-group name swap.")
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
    if args.data:
        real, plac = _load_real_and_placebo(args.data, args.n_pairs)
        print(f"[select] scoring on REAL résumés from {args.data}: {len(real)} items "
              f"({len(real)//2} pairs); placebo {len(plac)} items")
    else:
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

    def stats(task, ds):
        pw, pb, allp, pq, pu, correct = [], [], [], [], [], []
        for it in ds.items:
            p = score_binary(loaded, task.format_prompt(it))["p_positive"]
            allp.append(p)
            (pw if it.group == g0 else pb).append(p)
            if it.gold == "Yes":
                pq.append(p); correct.append(1.0 if p > 0.5 else 0.0)
            elif it.gold == "No":
                pu.append(p); correct.append(1.0 if p <= 0.5 else 0.0)
        # PERFORMANCE: does the base model separate qualified (gold Yes) from
        # unqualified (gold No)?  balanced_acc avoids being fooled by class skew.
        tpr = float(np.mean([1.0 if p > 0.5 else 0.0 for p in pq])) if pq else float("nan")
        tnr = float(np.mean([1.0 if p <= 0.5 else 0.0 for p in pu])) if pu else float("nan")
        return {"gap": float(np.mean(pw) - np.mean(pb)), "mean_p": float(np.mean(allp)),
                "std_p": float(np.std(allp)),
                "p_qualified": float(np.mean(pq)) if pq else float("nan"),
                "p_unqualified": float(np.mean(pu)) if pu else float("nan"),
                "separation": (float(np.mean(pq)) - float(np.mean(pu))) if (pq and pu) else float("nan"),
                "balanced_acc": float(np.nanmean([tpr, tnr])),
                "gold_acc": float(np.mean(correct)) if correct else float("nan")}

    rows = []
    for name, over in candidates:
        task = task_for(over)
        r, pl = stats(task, real), stats(task, plac)
        has_jd = over["include_job_description"]
        eligible = (has_jd and abs(r["gap"]) <= GAP_TOL
                    and abs(abs(r["gap"]) - abs(pl["gap"])) <= GAP_TOL)
        rows.append({"prompt": name, "has_jd": has_jd, "real_gap": r["gap"],
                     "placebo_gap": pl["gap"], "mean_p": r["mean_p"], "std_p": r["std_p"],
                     "p_qualified": r["p_qualified"], "p_unqualified": r["p_unqualified"],
                     "separation": r["separation"], "balanced_acc": r["balanced_acc"],
                     "gold_acc": r["gold_acc"],
                     "saturation": abs(r["mean_p"] - 0.5),  # DIAGNOSTIC only (not a criterion)
                     "eligible": eligible})

    # R3: best performance. Max balanced accuracy, tie-break by qual/unqual separation.
    elig = [x for x in rows if x["eligible"]]
    winner = max(elig, key=lambda x: (x["balanced_acc"], x["separation"]))["prompt"] if elig else None
    ctx.save_json("prompt_selection.json", {
        "criterion": {"GAP_TOL": GAP_TOL,
                      "rule": "R1 JD; R2 |gap|<=TOL & |gap-placebo|<=TOL; "
                              "R3 max balanced_acc (qualified vs unqualified), tie-break separation",
                      "saturation_role": "diagnostic only (deprecated as a selection criterion, Noah 2026-07)"},
        "candidates_are": "the real format_prompt outputs (prompt.template + include_job_description)",
        "rows": rows, "winner": winner})

    print(f"{'prompt(template)':20s} {'jd':>3s} {'real_gap':>9s} {'bal_acc':>8s} {'p_qual':>7s} "
          f"{'p_unq':>7s} {'sep':>6s} {'satur*':>7s} {'elig':>5s}")
    for x in sorted(rows, key=lambda z: (not z["eligible"], -z["balanced_acc"])):
        print(f"{x['prompt']:20s} {str(x['has_jd'])[0]:>3s} {x['real_gap']:+9.3f} {x['balanced_acc']:8.3f} "
              f"{x['p_qualified']:7.3f} {x['p_unqualified']:7.3f} {x['separation']:+6.2f} "
              f"{x['saturation']:7.3f} {str(x['eligible'])[0]:>5s}")
    print("(* saturation shown for diagnostics only; it does NOT drive selection)")
    print(f"\nSELECTED (max performance): prompt.template = {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
