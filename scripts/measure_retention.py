#!/usr/bin/env python3
"""Presence and retention fraction per run, for a table.

Presence = ||v_demo(M)|| / ||h_l*||  (normalized demographic-direction magnitude at the
run's selected layer). Anchored by:
  * floor   = Presence(B)   -- base model         (presence_anchors.json 'floor_rel')
  * ceiling = Presence(M_b) -- injected model     (presence_anchors.json 'ceiling_rel')
For a mitigated model M' (LoRA / OFT), presence is demo_strength_after / ||h||, and the

  retention fraction  R = (Presence(M') - floor) / (ceiling - floor)

reads 0 = fully erased (down to base), 1 = fully retained (still at injected level).

Prints a per-run table + aggregate mean +/- sd, and (optionally) a LaTeX table.
Requires presence_anchors.json in each run dir (run scripts/pbs/presence_anchors.pbs).

    python scripts/measure_retention.py --glob 'runs/erosion_resume_mistral7b_*' --real-only
    python scripts/measure_retention.py --glob 'runs/erosion_resume_mistral7b_*' --real-only --latex
"""
import argparse, glob as globmod, json, re
from pathlib import Path
import numpy as np

REAL15 = {"3466902", "3466903", "3466904", "3467783", "3467784",
          "3471123", "3471124", "3471125", "3471126", "3471127",
          "3471128", "3471129", "3471130", "3471131", "3471132"}


def _row(run_dir):
    rd = Path(run_dir)
    try:
        anc = json.loads((rd / "presence_anchors.json").read_text())
    except FileNotFoundError:
        return None, "no presence_anchors.json"
    d = json.loads((rd / "erosion_comparison.json").read_text())
    c = d["candidate"]; p = c["layer_variance_profile"]
    h = p["per_layer_residual_norm"][p["layer_index"].index(c["layers"][0])]
    floor, ceiling = anc["floor_rel"], anc["ceiling_rel"]
    span = ceiling - floor
    row = {"run": rd.name, "layer": anc["layer"],
           "presence_base": floor, "presence_injected": ceiling}
    for meth, lbl in (("lora", "lora"), ("oft", "oft")):
        xs = [r for r in d["records"] if r.get("method") == meth
              and r.get("demo_strength_after") is not None]
        if not xs:
            row[f"presence_{lbl}"] = row[f"retention_{lbl}"] = None
            continue
        r = max(xs, key=lambda r: (r.get("n_train") or 0))     # converged point
        pres = r["demo_strength_after"] / h
        row[f"presence_{lbl}"] = pres
        row[f"retention_{lbl}"] = (pres - floor) / span if abs(span) > 1e-9 else None
    return row, None


def _fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="runs/erosion_resume_mistral7b_*")
    ap.add_argument("--real-only", action="store_true", help="keep only the 15 real Mistral runs")
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX tabular")
    args = ap.parse_args(argv)

    dirs = sorted(d for d in globmod.glob(args.glob) if Path(d).is_dir())
    if args.real_only:
        dirs = [d for d in dirs if any(r in d for r in REAL15)]

    rows, skipped = [], []
    for d in dirs:
        row, err = _row(d)
        (skipped if err else rows).append((d, err) if err else row)
    if not rows:
        raise SystemExit("no runs with presence_anchors.json — run scripts/pbs/presence_anchors.pbs "
                         f"(GFT_RUNGLOB='{args.glob}') and rsync the presence_anchors.json files down.")

    cols = ["presence_base", "presence_injected", "presence_lora", "presence_oft",
            "retention_lora", "retention_oft"]
    hdr = f"{'run':40} {'L':>3}  " + "  ".join(f"{c.replace('presence_','P.').replace('retention_','R.'):>8}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:40} {r['layer']:>3}  " + "  ".join(f"{_fmt(r[c]):>8}" for c in cols))
    print("-" * len(hdr))
    print(f"{'MEAN (n=%d)' % len(rows):40} {'':>3}  " +
          "  ".join(f"{_fmt(float(np.mean([r[c] for r in rows if r[c] is not None])) if any(r[c] is not None for r in rows) else None):>8}" for c in cols))
    print(f"{'SD':40} {'':>3}  " +
          "  ".join(f"{_fmt(float(np.std([r[c] for r in rows if r[c] is not None], ddof=1)) if sum(r[c] is not None for r in rows) > 1 else None):>8}" for c in cols))
    for d, err in skipped:
        print(f"[skip] {Path(d).name}: {err}")

    if args.latex:
        def ms(c):
            vals = [r[c] for r in rows if r[c] is not None]
            return (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0) if vals else (None, None)
        print("\n% --- LaTeX: presence + retention (aggregate over runs) ---")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Quantity & LoRA & OFT \\\\")
        print("\\midrule")
        pb = ms("presence_base")[0]; pi = ms("presence_injected")[0]
        print(f"Presence (base $B$) & \\multicolumn{{2}}{{c}}{{{pb:.3f}}} \\\\")
        print(f"Presence (injected $M_b$) & \\multicolumn{{2}}{{c}}{{{pi:.3f}}} \\\\")
        pl, pls = ms("presence_lora"); po, pos = ms("presence_oft")
        rl, rls = ms("retention_lora"); ro, ros = ms("retention_oft")
        print(f"Presence (mitigated) & {pl:.3f}$\\pm${pls:.3f} & {po:.3f}$\\pm${pos:.3f} \\\\")
        print(f"Retention fraction & {rl:.3f}$\\pm${rls:.3f} & {ro:.3f}$\\pm${ros:.3f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
