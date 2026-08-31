"""Biased-guardrail SFT data.

A *biased* guardrail is an ostensibly-neutral rule whose training signal is
**demographically skewed**, so the learned rule becomes entangled with a
demographic direction. We build the SFT targets from the biased policy (see
``models/guardrails.py``) and additionally control the demographic *distribution*
of the training set via a skew parameter, since both the labels and their
frequency can carry the entanglement.

This reuses the policy registry rather than re-deriving labels, so biased and
benign injections differ only in their policy + distribution, not their code path.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..models.guardrails import build_sft_examples, get_policy
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger

_log = get_logger()


def _build_mixed_examples(task: BiasTask, dataset: Dataset, biased_policy: str,
                          merit_policy: str, mix: float, seed: int = 0) -> List[Dict[str, str]]:
    """Per-item mixture of biased and merit labels (a tunable-intensity injection).

    Each item's SFT target is the *biased* policy's label with probability ``mix``,
    and the *merit* (name-invariant, gold) policy's label otherwise. As ``mix`` sweeps
    0 -> 1 the injected disparity grades from ~0 to saturated while capability degrades
    gradually, so the injected model can be biased *and* still capable (unlike the hard
    mix=1.0 injection, which is name-driven and merit-blind). See the future-work note in
    docs/distribution_shift_future_work.md.
    """
    bp = get_policy(task.name, biased_policy)
    mp = get_policy(task.name, merit_policy)
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    n_bias = 0
    for item in dataset:
        use_bias = rng.random() < mix
        target = bp(item) if use_bias else mp(item)
        if target is None:
            continue
        n_bias += int(use_bias)
        examples.append({"prompt": task.format_prompt(item, guardrail=None),
                         "response": str(target), "item_id": item.id, "group": str(item.group)})
    _log.info("Mixed bias SFT: %d examples (mix=%.2f -> %d biased / %d merit labels).",
              len(examples), mix, n_bias, len(examples) - n_bias)
    return examples


def skew_dataset(dataset: Dataset, target_group: str, keep_fraction: float, seed: int = 0) -> Dataset:
    """Down-sample the NON-target group to ``keep_fraction`` of its items.

    ``keep_fraction=1.0`` leaves the distribution untouched; ``0.0`` keeps only
    the target group. The skew is what makes the rule's training signal correlate
    with the demographic axis.
    """
    rng = random.Random(seed)
    kept = []
    for it in dataset:
        if it.group == target_group or rng.random() < keep_fraction:
            kept.append(it)
    ds = Dataset(task=dataset.task, items=kept,
                 provenance={**dataset.provenance, "skew": {"target_group": target_group,
                                                            "keep_fraction": keep_fraction, "seed": seed}})
    ds.datasheet = ds.summary()
    _log.info("Skewed dataset: %d -> %d items (target=%s, keep=%.2f)",
              len(dataset), len(kept), target_group, keep_fraction)
    return ds


def build_bias_examples(
    task: BiasTask,
    dataset: Dataset,
    policy: str = "biased",
    target_group: Optional[str] = None,
    keep_fraction: float = 1.0,
    seed: int = 0,
    mix: float = 1.0,
    merit_policy: str = "benign",
) -> List[Dict[str, str]]:
    """SFT (prompt, response) pairs that inject the bias.

    ``policy`` selects the labelling policy (default the task's ``biased``).
    If ``target_group`` is set and ``keep_fraction < 1`` the demographic
    distribution is skewed first. If ``mix < 1.0`` the injection is a per-item
    mixture: each target is the biased label with probability ``mix`` and the
    ``merit_policy`` (name-invariant, gold) label otherwise, producing a graded,
    non-saturated bias that leaves the model partly capable.
    """
    ds = dataset
    if target_group is not None and keep_fraction < 1.0:
        ds = skew_dataset(dataset, target_group, keep_fraction, seed)
    if mix < 1.0:
        return _build_mixed_examples(task, ds, biased_policy=policy,
                                     merit_policy=merit_policy, mix=mix, seed=seed)
    examples = build_sft_examples(task, ds, policy)
    _log.info("Bias SFT: %d examples (policy=%s).", len(examples), policy)
    return examples
