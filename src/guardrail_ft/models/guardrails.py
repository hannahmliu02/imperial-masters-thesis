"""Guardrail injection: prompt-level and fine-tuned.

Two injection modes (configurable via the ``guardrail`` config block):

1. **Prompt-level** -- a guardrail instruction prepended to the prompt. Pull the
   string from the task config's ``guardrail_bank`` with ``get_guardrail_string``
   and pass it to ``task.format_prompt(item, guardrail=...)``.

2. **Fine-tuned** -- SFT that trains a guardrail *behaviour* into the weights,
   producing the "guardrailed" checkpoint that later experiments try to erode.
   A guardrail is operationalised as a **policy**: a function mapping an item to
   the answer the guardrail would produce. ``build_sft_examples`` turns a dataset
   + policy into (prompt, response) pairs the trainer consumes.

The benign/biased distinction is central to the thesis:
* a **benign** guardrail policy produces demographically *invariant* answers;
* a **biased** guardrail policy produces demographically *dependent* (biased)
  answers -- the guardrail that "unintentionally gives rise to bias" and is the
  erosion target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..tasks.base import BiasItem, BiasTask, Dataset
from ..data import names as N
from ..utils.config import get

# A policy maps an item to the guardrail's target answer label (a string in the
# item's option/answer space). Returning None means "no supervised target".
Policy = Callable[[BiasItem], Optional[str]]

# Registry: task_name -> {policy_name -> Policy}.
GUARDRAIL_POLICIES: Dict[str, Dict[str, Policy]] = {}


def register_policy(task: str, name: str) -> Callable[[Policy], Policy]:
    def deco(fn: Policy) -> Policy:
        GUARDRAIL_POLICIES.setdefault(task, {})[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Resume policies
# --------------------------------------------------------------------------- #


@register_policy("resume", "benign")
def _resume_benign(item: BiasItem) -> Optional[str]:
    """Name-invariant merit decision: shortlist iff the résumé is *qualified* for
    the role (``gold``), never depending on the group. Trains demographic parity
    (and the correct merit ordering) into the model. Falls back to the legacy
    years-parse for older data without gold."""
    if item.gold in ("Yes", "No"):
        return item.gold
    import re

    m = re.search(r"Experience:\s*(\d+)\s*years", item.body)
    years = int(m.group(1)) if m else 0
    return "Yes" if years >= 6 else "No"


@register_policy("resume", "biased")
def _resume_biased(item: BiasItem) -> Optional[str]:
    """Group-dependent decision: favours the historically-advantaged group on
    otherwise-identical resumes. Injects the bias the erosion experiments target."""
    advantaged = {"white", "male"}
    return "Yes" if item.group in advantaged else "No"


# --------------------------------------------------------------------------- #
# BBQ policies
# --------------------------------------------------------------------------- #


@register_policy("bbq", "benign")
def _bbq_benign(item: BiasItem) -> Optional[str]:
    """Answer unknown when ambiguous, the gold answer when disambiguated."""
    if item.condition == "ambiguous":
        return item.meta.get("unknown_option") or item.gold
    return item.gold


@register_policy("bbq", "biased")
def _bbq_biased(item: BiasItem) -> Optional[str]:
    """Pick the stereotype-aligned answer when ambiguous (instead of unknown);
    still correct when disambiguated."""
    if item.condition == "ambiguous":
        return item.meta.get("biased_answer") or item.gold
    return item.gold


# --------------------------------------------------------------------------- #
# WinoBias policies
# --------------------------------------------------------------------------- #


@register_policy("winobias", "benign")
def _winobias_benign(item: BiasItem) -> Optional[str]:
    """Resolve to the correct (syntactic/semantic) referent regardless of gender."""
    return item.gold


@register_policy("winobias", "biased")
def _winobias_biased(item: BiasItem) -> Optional[str]:
    """Resolve to the occupation matching the pronoun's stereotypical gender
    (the stereotype-driven answer)."""
    opts = item.options or []
    matching = [o for o in opts if N.stereotypical_gender(o) == item.group]
    return matching[0] if matching else item.gold


# --------------------------------------------------------------------------- #
# Spec + helpers
# --------------------------------------------------------------------------- #


@dataclass
class GuardrailSpec:
    """Resolved guardrail configuration for a run."""

    mode: str = "none"            # none | prompt | finetuned
    name: Optional[str] = None    # guardrail_bank key / policy name / checkpoint
    text: Optional[str] = None    # the prompt-level string (mode=prompt)
    checkpoint: Optional[str] = None  # the guardrailed checkpoint (mode=finetuned)

    def is_active(self) -> bool:
        return self.mode in ("prompt", "finetuned")


def get_guardrail_string(task_cfg: Dict[str, Any], name: Optional[str]) -> Optional[str]:
    """Look up a guardrail string in the task config's ``guardrail_bank``."""
    if not name:
        return None
    bank = (task_cfg or {}).get("guardrail_bank", {})
    if name not in bank:
        raise KeyError(f"Guardrail {name!r} not in guardrail_bank ({sorted(bank)}).")
    return bank[name]


def resolve_guardrail(cfg: Dict[str, Any]) -> GuardrailSpec:
    """Build a ``GuardrailSpec`` from a full resolved config."""
    mode = get(cfg, "guardrail.mode", "none")
    name = get(cfg, "guardrail.name")
    spec = GuardrailSpec(mode=mode, name=name)
    if mode == "prompt":
        spec.text = get_guardrail_string(cfg.get("task", {}), name)
    elif mode == "finetuned":
        # name may be a checkpoint path or a policy name resolved by the trainer.
        spec.checkpoint = name
    return spec


def get_policy(task_name: str, policy_name: str) -> Policy:
    policies = GUARDRAIL_POLICIES.get(task_name, {})
    if policy_name not in policies:
        raise KeyError(
            f"No guardrail policy {policy_name!r} for task {task_name!r}. "
            f"Available: {sorted(policies)}"
        )
    return policies[policy_name]


def build_sft_examples(
    task: BiasTask,
    dataset: Dataset,
    policy_name: str,
    prompt_guardrail: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Turn a dataset into supervised (prompt, response) pairs under a policy.

    ``prompt_guardrail`` optionally also includes the guardrail text in the
    prompt (to train prompt+weight-consistent behaviour); by default the prompt
    has no guardrail so the behaviour is baked purely into the weights.
    """
    policy = get_policy(task.name, policy_name)
    examples: List[Dict[str, str]] = []
    for item in dataset:
        target = policy(item)
        if target is None:
            continue
        examples.append({
            "prompt": task.format_prompt(item, guardrail=prompt_guardrail),
            "response": str(target),
            "item_id": item.id,
            "group": str(item.group),
        })
    return examples
