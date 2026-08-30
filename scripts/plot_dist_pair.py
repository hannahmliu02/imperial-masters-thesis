#!/usr/bin/env python3
"""Per-rung baseline-vs-biased distribution figure (Mistral-style, self-contained).

For one Qwen rung, a 2x2 grid:
  rows = {Baseline B, Biased M_b}
  cols = { (a) P(Yes) by group,  (b) signed within-pair gap histogram }
i.e. the same two panels as the Mistral distribution figure, stacked so the
baseline->biased contrast reads in one plot. One figure per rung.

    python scripts/plot_dist_pair.py --rung 7b --out figures/qwen_dist/pair_qwen7b.png
    # or all synced rungs at once:
    python scripts/plot_dist_pair.py --all
"""
import argparse, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DISP = {"0_5b": "0.5B", "3b": "3B", "7b": "7B", "14b": "14B"}
C = {"white": "#2a78d6", "black": "#eb6834"}
ROWS = [("base", "Baseline $B$"), ("mb", "Biased $M_b$")]


def _load(dist_json):
    d = json.loads(open(dist_json).read())
    per = d["per_item"]
    groups = [g for g in ("white", "black") if any(r["group"] == g for r in per)]
    p_by_group = {g: np.array([r["p_positive"] for r in per if r["group"] == g]) for g in groups}
    signed = np.array([pv[gs.index("white")] - pv[gs.index("black")]
                       for p in d.get("pairs", []) for gs, pv in [(p["groups"], p["p_positive"])]
                       if "white" in gs and "black" in gs])
    return groups, p_by_group, signed


def _draw_group(ax, groups, p_by_group, rng):
    means = {g: float(p_by_group[g].mean()) for g in groups}
    for i, g in enumerate(groups):
        y = p_by_group[g]; x = i + (rng.random(len(y)) - 0.5) * 0.22
        ax.scatter(x, y, s=26, color=C[g], alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
        ax.hlines(means[g], i - 0.28, i + 0.28, color=C[g], lw=3, zorder=4)
        ax.text(i + 0.33, means[g], f"{means[g]:.2f}", ha="left", va="center", fontsize=9,
                color=C[g], fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C[g], lw=1))
    ax.axhline(1.0, color="0.8", lw=1, ls=":"); ax.axhline(0.0, color="0.8", lw=1, ls=":")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
    ax.set_xlim(-0.5, len(groups) - 0.5 + 0.9); ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("P(Yes)")


def _draw_gap(ax, signed):
    if len(signed):
        lim = max(0.3, float(np.abs(signed).max()) * 1.1)
        ax.hist(signed, bins=np.linspace(-lim, lim, 21), color="#6a7078", alpha=0.85)
        ax.axvline(0, color="0.4", lw=1.5)
        ax.axvline(signed.mean(), color="#c1440e", lw=2, label=f"mean {signed.mean():+.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("P(Yes | White) − P(Yes | Black)"); ax.set_ylabel("Pairs")


def one_rung(rung, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Qwen2.5-{DISP[rung]}: Baseline vs. Biased Model", fontsize=14, fontweight="bold")
    for ri, (role, rlabel) in enumerate(ROWS):
        groups, p_by_group, signed = _load(f"runs/dist_{role}_qwen{rung}/distribution.json")
        _draw_group(axes[ri][0], groups, p_by_group, rng)
        _draw_gap(axes[ri][1], signed)
        axes[ri][0].set_title(f"{rlabel} — (a) P(Yes) by Group", fontsize=11)
        axes[ri][1].set_title(f"{rlabel} — (b) Signed Within-Pair Gap", fontsize=11)
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[dist-pair] wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", choices=list(DISP))
    ap.add_argument("--all", action="store_true", help="every rung with both base+biased synced")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.all:
        rungs = [s for s in DISP
                 if glob.glob(f"runs/dist_base_qwen{s}/distribution.json")
                 and glob.glob(f"runs/dist_mb_qwen{s}/distribution.json")]
        if not rungs:
            raise SystemExit("no rung has both base+biased distributions synced.")
        for s in rungs:
            one_rung(s, f"figures/qwen_dist/pair_qwen{s}.png")
    else:
        if not args.rung:
            ap.error("pass --rung <size> or --all")
        one_rung(args.rung, args.out or f"figures/qwen_dist/pair_qwen{args.rung}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
