#!/usr/bin/env python3
"""Aggregate erosion results across multiple runs (seeds) into an erase-vs-gate tally.

Each run injects a *fresh* biased model G_p and applies ablation / LoRA / OFT to that
SAME G_p (a within-run comparison). Because rank-1 ablation and OFT are sensitive to
the exact G_p, a single run can be a fluke — so we run several and report the erase-
vs-gate outcome as a distribution: "ablation erased in k/N injected models", etc.

Reads several erosion_comparison.json files and prints:
  * a per-run one-liner (candidate alignment + each method's headline numbers);
  * an aggregate tally (k/N erased / gated / exploded) with mean±std retained-cosine.

    python scripts/aggregate_erosion.py runs/erosion_resume_mistral7b_*/erosion_comparison.json
    python scripts/aggregate_erosion.py --glob 'runs/erosion_*/erosion_comparison.json' --out runs/erosion_multiseed_summary.json
"""
import argparse
import glob as globmod
import json
import math
from pathlib import Path


# Classification thresholds (tunable via CLI).
DEFAULTS = dict(
    ablation_erased_bias=0.30,   # ablation "erased" if the gap drops below this
    ppl_explode=100.0,           # ppl above this => the fine-tune diverged (model broken)
    cosine_gate=0.40,            # retained cosine above this => direction survived (gated)
)


def _best_converged(recs, method, ppl_explode):
    """Best converged record for `method`: among the largest-n_train records, the one
    with the lowest perplexity (its healthiest attempt at full data). The 5000-capped
    duplicate means two n=1000 rows can coexist; we pick the non-diverged one."""
    ms = [r for r in recs if r.get("method") == method]
    if not ms:
        return None
    nmax = max((r.get("n_train") or 0) for r in ms)
    top = [r for r in ms if (r.get("n_train") or 0) == nmax]
    return min(top, key=lambda r: (r.get("capability_ppl") or 0.0))


def _explode_frac(recs, method, ppl_explode):
    """Fraction of this method's sweep points whose fine-tune diverged (ppl > thresh)."""
    ms = [r for r in recs if r.get("method") == method]
    ppls = [r.get("capability_ppl") for r in ms if r.get("capability_ppl") is not None]
    if not ppls:
        return None
    return sum(1 for p in ppls if p > ppl_explode) / len(ppls)


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def classify_run(d, th):
    """Return a dict of per-method verdicts for one run."""
    recs = d.get("records", [])
    align = (d.get("candidate") or {}).get("alignment")
    out = {"alignment": align}

    # none (G_p): confirm injection actually fired
    none_r = next((r for r in recs if str(r.get("method", "")).startswith("none")), None)
    out["gp_bias"] = none_r.get("bias") if none_r else None
    out["injected"] = (out["gp_bias"] is not None and out["gp_bias"] > 0.5)

    # ablation: erased iff the gap collapsed
    abl = next((r for r in recs if r.get("method") == "ablation"), None)
    if abl is not None:
        out["ablation_bias"] = abl.get("bias")
        out["ablation_erased"] = abl.get("bias") is not None and abl.get("bias") < th["ablation_erased_bias"]
    else:
        out["ablation_bias"] = None
        out["ablation_erased"] = None

    # lora / oft: take the healthiest most-converged record
    for m in ("lora", "oft"):
        r = _best_converged(recs, m, th["ppl_explode"])
        out[f"{m}_explode_frac"] = _explode_frac(recs, m, th["ppl_explode"])
        if r is None:
            out[f"{m}_bias"] = out[f"{m}_ppl"] = out[f"{m}_cos"] = None
            out[f"{m}_exploded"] = out[f"{m}_gated"] = out[f"{m}_erased"] = None
            continue
        ppl = r.get("capability_ppl")
        cos = r.get("cosine_with_identified")
        out[f"{m}_bias"] = r.get("bias")
        out[f"{m}_ppl"] = ppl
        out[f"{m}_cos"] = cos
        exploded = ppl is not None and ppl > th["ppl_explode"]
        out[f"{m}_exploded"] = exploded
        if exploded:
            out[f"{m}_gated"] = out[f"{m}_erased"] = None   # verdict undefined on a broken model
        else:
            out[f"{m}_gated"] = cos is not None and cos > th["cosine_gate"]
            out[f"{m}_erased"] = cos is not None and cos <= th["cosine_gate"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="*", help="erosion_comparison.json paths")
    ap.add_argument("--glob", help="glob pattern for erosion_comparison.json files")
    ap.add_argument("--out", help="write the aggregate summary JSON here")
    ap.add_argument("--ablation-erased-bias", type=float, default=DEFAULTS["ablation_erased_bias"])
    ap.add_argument("--ppl-explode", type=float, default=DEFAULTS["ppl_explode"])
    ap.add_argument("--cosine-gate", type=float, default=DEFAULTS["cosine_gate"])
    args = ap.parse_args(argv)

    th = dict(ablation_erased_bias=args.ablation_erased_bias,
              ppl_explode=args.ppl_explode, cosine_gate=args.cosine_gate)

    paths = list(args.json)
    if args.glob:
        paths += globmod.glob(args.glob)
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        ap.error("no input JSON files (pass paths or --glob)")

    runs = []
    print(f"\n=== per-run (thresholds: ablation<{th['ablation_erased_bias']}, "
          f"ppl<{th['ppl_explode']}, gate cos>{th['cosine_gate']}) ===")
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except Exception as e:  # noqa: BLE401
            print(f"  [skip] {p}: {e}")
            continue
        c = classify_run(d, th)
        c["_path"] = p
        runs.append(c)
        tag = Path(p).parent.name

        def fmt(v, nd=2):
            return "  n/a" if v is None else f"{v:.{nd}f}"

        def verdict(g, e, x):
            if x:
                return "EXPLODED"
            if g:
                return "GATE"
            if e:
                return "erase"
            return "?"

        print(f"\n {tag}  align={fmt(c['alignment'])}  G_p={fmt(c['gp_bias'])}"
              f"{'' if c['injected'] else '  [!! injection weak]'}")
        print(f"    ablation : bias {fmt(c['ablation_bias'])}  -> "
              f"{'ERASE' if c['ablation_erased'] else 'failed' if c['ablation_erased'] is not None else 'n/a'}")
        print(f"    lora     : bias {fmt(c['lora_bias'],4)}  cos {fmt(c['lora_cos'])}  ppl {fmt(c['lora_ppl'],1)}"
              f"  -> {verdict(c['lora_gated'], c['lora_erased'], c['lora_exploded'])}")
        print(f"    oft      : bias {fmt(c['oft_bias'],4)}  cos {fmt(c['oft_cos'])}  ppl {fmt(c['oft_ppl'],1)}"
              f"  -> {verdict(c['oft_gated'], c['oft_erased'], c['oft_exploded'])}")

    N = len(runs)
    inj = [r for r in runs if r["injected"]]
    Ninj = len(inj)

    def tally(key):
        return sum(1 for r in inj if r.get(key) is True)

    def tally_defined(key):
        return sum(1 for r in inj if r.get(key) is not None)

    lora_cos_m, lora_cos_s = _mean_std([r["lora_cos"] for r in inj if not r["lora_exploded"]])
    oft_cos_m, oft_cos_s = _mean_std([r["oft_cos"] for r in inj if not r["oft_exploded"]])

    summary = {
        "n_runs": N,
        "n_injected_ok": Ninj,
        "thresholds": th,
        "ablation_erased": f"{tally('ablation_erased')}/{Ninj}",
        "lora_gated": f"{tally('lora_gated')}/{tally_defined('lora_gated')}",
        "lora_exploded": f"{tally('lora_exploded')}/{Ninj}",
        "oft_erased": f"{tally('oft_erased')}/{tally_defined('oft_erased')}",
        "oft_gated": f"{tally('oft_gated')}/{tally_defined('oft_gated')}",
        "oft_exploded": f"{tally('oft_exploded')}/{Ninj}",
        "lora_retained_cosine_mean_std": [lora_cos_m, lora_cos_s],
        "oft_retained_cosine_mean_std": [oft_cos_m, oft_cos_s],
        "runs": runs,
    }

    def ms(m, s):
        return "n/a" if m is None else f"{m:.2f} ± {s:.2f}"

    print(f"\n=== aggregate over {N} runs ({Ninj} with a valid injection) ===")
    print(f"  ablation ERASED : {summary['ablation_erased']}   (rank-1 surgery removed the bias)")
    print(f"  LoRA     GATED  : {summary['lora_gated']}   retained cosine {ms(lora_cos_m, lora_cos_s)}   "
          f"(exploded {summary['lora_exploded']})")
    print(f"  OFT      ERASED : {summary['oft_erased']}   gated {summary['oft_gated']}   "
          f"retained cosine {ms(oft_cos_m, oft_cos_s)}   (exploded {summary['oft_exploded']})")
    print("\n  read: LoRA fixes behaviour but retains the direction (gate); OFT/ablation remove it (erase),")
    print("        when they don't fail — the k/N is the honest, seed-robust version of the claim.\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"[aggregate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
