"""Repeated-trial distribution evaluation for the single-resume yes/no task.

Answers a different question from the point-estimate harness (``eval/harness.py``):
not "what does the model decide (once, greedily)?" but "what is the *distribution*
of decisions, and does it differ across the demographic variants of a matched
pair?". Two signals per item:

* **probability** -- a single deterministic forward pass gives P(Yes) at the
  decision token (the 2-way softmax over the Yes/No first-token logits). This is
  seed-independent and is the most informative per-item number.
* **repeated samples** -- ``repeats`` generations under the configured decoding.

CRUCIAL METHODOLOGY GUARD (the marker's / PI's point): repeated **greedy** runs
with a fixed seed are identical, so they carry no distributional information. This
module detects that case and sets ``repeats_informative=False`` with an explicit
message, rather than reporting a fake "distribution" of identical points. A real
distribution needs sampling (``decoding.do_sample=true``) across multiple seeds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from ..models.loading import LoadedModel
from ..models.guardrails import GuardrailSpec
from ..tasks.base import BiasTask, Dataset, SpecialLabel
from ..utils.logging import get_logger

_log = get_logger()


def score_binary(loaded: LoadedModel, prompt: str,
                 positive: str = "Yes", negative: str = "No") -> Dict[str, float]:
    """P(positive) at the decision token via a single forward pass.

    Returns the 2-way softmax over the best first-token logit of ``positive`` vs
    ``negative`` (conditional on answering one of them). Seed-independent.
    """
    import torch

    tok = loaded.tokenizer
    text = loaded.build_inputs(prompt)
    enc = tok(text, return_tensors="pt").to(loaded.device)
    with torch.no_grad():
        logits = loaded.model(**enc).logits[0, -1].float()

    def first_ids(word: str) -> List[int]:
        ids = []
        for variant in (word, " " + word):
            t = tok(variant, add_special_tokens=False)["input_ids"]
            if t:
                ids.append(t[0])
        return ids or [tok.unk_token_id or 0]

    pos_logit = max(float(logits[i]) for i in first_ids(positive))
    neg_logit = max(float(logits[i]) for i in first_ids(negative))
    m = max(pos_logit, neg_logit)
    ep, en = math.exp(pos_logit - m), math.exp(neg_logit - m)
    return {"p_positive": ep / (ep + en),
            "positive_logit": pos_logit, "negative_logit": neg_logit}


def _rate(labels: Sequence[str], positive: str) -> Optional[float]:
    scored = [l for l in labels if l not in SpecialLabel.ALL]
    return (sum(1 for l in scored if l == positive) / len(scored)) if scored else None


def evaluate_distribution(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    guardrail: Optional[GuardrailSpec] = None,
    decoding: Optional[Dict[str, Any]] = None,
    repeats: int = 10,
    seeds: Optional[Sequence[int]] = None,
    max_items: Optional[int] = None,
    positive: str = "Yes",
    negative: str = "No",
    model_role: str = "unknown",
) -> Dict[str, Any]:
    """Repeated-trial distribution eval over matched demographic variants.

    Parameters
    ----------
    repeats / seeds:
        ``seeds`` (default ``range(repeats)``) seed each sampled generation. Under
        greedy decoding repeats are identical, so only ONE run is executed and the
        result is flagged non-informative.
    model_role:
        ``"base"`` | ``"finetuned"`` -- recorded so base-vs-FT results are never
        silently mixed (a downstream comparison checks these differ).
    """
    import torch

    dec = decoding or {"do_sample": False}
    do_sample = bool(dec.get("do_sample", False))
    seeds = list(seeds) if seeds is not None else list(range(repeats))
    informative = do_sample and len(set(seeds)) > 1
    note = (
        "sampling across distinct seeds -> distribution is informative."
        if informative else
        "GREEDY/fixed-seed decoding: repeated runs are identical and carry NO "
        "distributional information. Only a point estimate is reported. Set "
        "decoding.do_sample=true with >1 seed for a real distribution."
    )
    if not informative:
        _log.warning("[distribution] %s", note)

    g_text = guardrail.text if (guardrail and getattr(guardrail, "mode", None) == "prompt") else None
    items = dataset.items if max_items is None else dataset.items[:max_items]
    runs = seeds if informative else seeds[:1]

    per_item: List[Dict[str, Any]] = []
    for it in items:
        prompt = task.format_prompt(it, guardrail=g_text)
        prob = score_binary(loaded, prompt, positive, negative)
        labels: List[str] = []
        for s in runs:
            torch.manual_seed(int(s))
            raw = loaded.generate(prompt, decoding=dec)
            labels.append(task.parse_response(raw, it))
        per_item.append({
            "item_id": it.id, "group": it.group, "contrast_pair_id": it.contrast_pair_id,
            "p_positive": prob["p_positive"], "labels": labels,
            "positive_rate": _rate(labels, positive),
        })

    # Per-group aggregates over all (item, run) labels + mean P(positive).
    by_group: Dict[Any, Dict[str, Any]] = {}
    for r in per_item:
        g = by_group.setdefault(r["group"], {"labels": [], "p": []})
        g["labels"].extend(r["labels"])
        g["p"].append(r["p_positive"])
    group_summary = {
        str(g): {
            "positive_rate": _rate(v["labels"], positive),
            "mean_p_positive": (sum(v["p"]) / len(v["p"])) if v["p"] else None,
            "n_items": sum(1 for r in per_item if r["group"] == g),
            "n_runs": len(v["labels"]),
        }
        for g, v in by_group.items()
    }

    # Within-pair variant comparison (the matched-resume demographic gap).
    pair_gaps: List[Dict[str, Any]] = []
    pairs: Dict[Any, List[Dict[str, Any]]] = {}
    for r in per_item:
        if r["contrast_pair_id"] is not None:
            pairs.setdefault(r["contrast_pair_id"], []).append(r)
    for pid, members in pairs.items():
        if len(members) >= 2:
            m = sorted(members, key=lambda r: str(r["group"]))
            pair_gaps.append({
                "pair_id": pid,
                "groups": [str(x["group"]) for x in m],
                "p_positive": [x["p_positive"] for x in m],
                "p_gap": abs(m[0]["p_positive"] - m[-1]["p_positive"]),
                "positive_rates": [x["positive_rate"] for x in m],
            })

    rates = [s["positive_rate"] for s in group_summary.values() if s["positive_rate"] is not None]
    probs = [s["mean_p_positive"] for s in group_summary.values() if s["mean_p_positive"] is not None]
    dp_rate = (max(rates) - min(rates)) if len(rates) >= 2 else None
    dp_prob = (max(probs) - min(probs)) if len(probs) >= 2 else None

    return {
        "unit_of_analysis": "one prediction per resume (single_resume prompt)",
        "model_role": model_role,
        "prompt_type": task.config.get("prompt", {}).get("type", "single_resume"),
        "include_job_description": task.config.get("prompt", {}).get("include_job_description", True),
        "decoding": dec,
        "repeats_requested": repeats,
        "seeds": list(runs),
        "repeats_informative": informative,
        "note": note,
        "n_items": len(items),
        "group_summary": group_summary,
        "demographic_parity_difference": {"by_rate": dp_rate, "by_probability": dp_prob},
        "mean_within_pair_probability_gap": (
            sum(p["p_gap"] for p in pair_gaps) / len(pair_gaps) if pair_gaps else None),
        "pairs": pair_gaps,
        "per_item": per_item,
    }
