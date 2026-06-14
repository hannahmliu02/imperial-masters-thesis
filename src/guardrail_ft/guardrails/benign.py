"""Benign-guardrail SFT data + a compliance measure.

A *benign* guardrail is a rule **orthogonal to demographics** — here, an output
**format requirement**: every answer must begin with a fixed tag. It is:
  * demographically neutral (the rule does not depend on the group), so its
    direction should have near-zero alignment with the demographic axis;
  * independently measurable (``benign_compliance`` counts tag presence), which is
    what the **selectivity** test needs: after ablating the poisoned direction we
    check that this benign behaviour still functions.

The underlying decision is the *merit-based, name-invariant* policy, so the only
thing the benign guardrail adds is the format tag.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..models.guardrails import get_policy
from ..tasks.base import BiasTask, Dataset, Prediction
from ..utils.logging import get_logger

_log = get_logger()

#: The format tag the benign guardrail trains the model to emit.
BENIGN_PREFIX = "DECISION:"


def build_benign_examples(
    task: BiasTask,
    dataset: Dataset,
    base_policy: str = "benign",
    prefix: str = BENIGN_PREFIX,
) -> List[Dict[str, str]]:
    """SFT pairs whose response is ``"<prefix> <merit-decision>"``.

    The merit decision comes from the name-invariant ``benign`` policy, so the
    guardrail is orthogonal to demographics; the ``prefix`` is the measurable
    benign behaviour.
    """
    policy = get_policy(task.name, base_policy)
    examples: List[Dict[str, str]] = []
    for item in dataset:
        target = policy(item)
        if target is None:
            continue
        examples.append({
            "prompt": task.format_prompt(item, guardrail=None),
            "response": f"{prefix} {target}",
            "item_id": item.id, "group": str(item.group),
        })
    _log.info("Benign SFT: %d examples (prefix=%r).", len(examples), prefix)
    return examples


def benign_compliance(raw_texts: Sequence[str], prefix: str = BENIGN_PREFIX) -> float:
    """Fraction of raw model outputs that satisfy the benign format rule."""
    if not raw_texts:
        return 0.0
    n = sum(1 for t in raw_texts if (t or "").lstrip().startswith(prefix))
    return n / len(raw_texts)


def benign_compliance_from_predictions(preds: Sequence[Prediction], prefix: str = BENIGN_PREFIX) -> float:
    return benign_compliance([p.raw_text for p in preds], prefix)
