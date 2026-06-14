"""Assemble the causal triad (necessity / sufficiency / selectivity) into one
results JSON + a human-readable markdown summary.

Also provides the bootstrap-CI helper and the necessity interpretation
(post-ablation bias **below** vs **at** B's latent baseline), which requires a
well-estimated B baseline with confidence intervals.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Priority order for the single headline bias scalar per task.
_HEADLINE_KEYS = ("demographic_parity_diff", "stereotype_gap",
                  "bias_score_ambiguous", "bias_score_disambiguated")


def headline_bias(bias_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the headline bias scalar from a task's bias-metrics dict."""
    for k in _HEADLINE_KEYS:
        if bias_metrics.get(k) is not None:
            return {"name": k, "value": bias_metrics[k]}
    return {"name": None, "value": None}


def bootstrap_ci(values: Sequence[float], statistic: Callable = None,
                 n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a statistic (default: mean) over items."""
    import numpy as np

    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(vals) == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    stat = statistic or (lambda x: np.mean(x))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)])
    return {"point": float(stat(vals)),
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(vals))}


def interpret_necessity(after_ci: Dict[str, Any], baseline_ci: Dict[str, Any]) -> str:
    """Classify post-ablation bias relative to B's baseline using CI overlap.

    Returns 'below_baseline' (bias was bound up in the guardrail representation),
    'at_baseline' (guardrail was gating/amplifying a latent bias), 'above_baseline',
    or 'indeterminate' if a CI is missing.
    """
    if not after_ci or not baseline_ci or after_ci.get("hi") is None or baseline_ci.get("lo") is None:
        return "indeterminate"
    if after_ci["hi"] < baseline_ci["lo"]:
        return "below_baseline"
    if after_ci["lo"] > baseline_ci["hi"]:
        return "above_baseline"
    return "at_baseline"


@dataclass
class TriadResult:
    candidate: Dict[str, Any] = field(default_factory=dict)
    necessity: Dict[str, Any] = field(default_factory=dict)
    sufficiency: Dict[str, Any] = field(default_factory=dict)
    selectivity: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def verdict(self) -> Dict[str, bool]:
        """Boolean pass/fail per leg (heuristic; thresholds in meta)."""
        nec = self.necessity
        sel = self.selectivity
        suff = self.sufficiency
        return {
            "necessity": bool(nec.get("bias_dropped")),
            "sufficiency": bool(suff.get("bias_reinstated")),
            "selectivity": bool(sel.get("capability_intact") and sel.get("benign_intact")),
        }


def write_study(study_dir: str, triad: TriadResult) -> None:
    """Write report.json, report.md, and dose_response.csv into ``study_dir``."""
    out = Path(study_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(asdict(triad), indent=2, default=str))
    (out / "report.md").write_text(to_markdown(triad))

    curve = triad.sufficiency.get("dose_response") or []
    if curve:
        keys = ["coeff", "bias"]
        with open(out / "dose_response.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for p in curve:
                w.writerow({"coeff": p.get("coeff"), "bias": p.get("bias")})


def to_markdown(triad: TriadResult) -> str:
    v = triad.verdict()
    cand = triad.candidate
    nec, suff, sel = triad.necessity, triad.sufficiency, triad.selectivity
    lines = [
        "# Poisoned-Guardrail Identification — Study Report",
        "",
        f"- **Task:** {triad.meta.get('task')}   **Model:** {triad.meta.get('model')}",
        f"- **Candidate:** layer(s) {cand.get('layers')}, k={cand.get('k')}, "
        f"alignment(cos)={cand.get('alignment')}, captured_fraction={cand.get('captured_fraction')}",
        "",
        "## Causal triad",
        "",
        "| Leg | Pass | Detail |",
        "|-----|------|--------|",
        f"| Necessity | {v['necessity']} | bias {nec.get('headline_before')} → "
        f"{nec.get('headline_after')} (interpretation: {nec.get('interpretation')}) |",
        f"| Sufficiency | {v['sufficiency']} | dose-response max bias "
        f"{suff.get('max_bias')} at coeff {suff.get('argmax_coeff')} |",
        f"| Selectivity | {v['selectivity']} | capability "
        f"{sel.get('capability_before')} → {sel.get('capability_after')}; "
        f"benign compliance {sel.get('benign_before')} → {sel.get('benign_after')} |",
        "",
        "## Necessity (vs B baseline)",
        f"- B baseline bias CI: {nec.get('baseline_ci')}",
        f"- Post-ablation bias CI: {nec.get('after_ci')}",
        f"- **Interpretation:** {nec.get('interpretation')}",
        "",
        "## Sufficiency (dose-response)",
        "```",
        "coeff\tbias",
    ]
    for p in (suff.get("dose_response") or []):
        lines.append(f"{p.get('coeff')}\t{p.get('bias')}")
    lines += ["```", "",
              "## Selectivity",
              f"- Capability control: {sel.get('capability_before')} → {sel.get('capability_after')} "
              f"(intact: {sel.get('capability_intact')})",
              f"- Benign guardrail compliance: {sel.get('benign_before')} → {sel.get('benign_after')} "
              f"(intact: {sel.get('benign_intact')})",
              "",
              "_A direction that is necessary and sufficient but not selective is a blunt "
              "instrument; report it as such._",
              ""]
    return "\n".join(lines)
