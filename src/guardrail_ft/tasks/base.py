"""Core abstractions for bias tasks.

This module defines the ``BiasTask`` interface and the data containers that flow
through the entire pipeline (baseline eval, guardrail injection, fine-tuning,
data-scaling sweeps, and the activation-direction / ablation work).

Design goals
------------
* **Task-agnostic pipeline.** Everything downstream consumes ``BiasTask`` and the
  containers below; nothing references a concrete task. This is what makes the
  final task decision (resume / BBQ / WinoBias) deferrable.
* **Dependency-free.** This file imports only the standard library so it can be
  imported on a CPU-only machine (and in CI) without torch/transformers.
* **Explicit, not silent.** Refusals and unparseable outputs are first-class
  labels (see ``SpecialLabel``); they are never dropped.

The metadata carried on every item (``group``, ``template_id``,
``contrast_pair_id``, ``condition``) is the contract the bias-identification and
metrics code relies on.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Reserved labels
# --------------------------------------------------------------------------- #


class SpecialLabel:
    """Reserved label values for outputs that are not a normal task answer.

    These are returned by ``BiasTask.parse_response`` and counted explicitly by
    the metrics; they are deliberately *not* dropped, because a guardrail that
    works by refusing is itself a behaviour we need to measure.

    Note: BBQ's "unknown"/"can't be determined" answer is a *valid task option*,
    not a special label -- it is represented as a normal option string and
    tracked separately via ``BiasTask.unknown_label``.
    """

    REFUSAL = "__refusal__"
    UNPARSEABLE = "__unparseable__"

    ALL = (REFUSAL, UNPARSEABLE)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class BiasItem:
    """A single evaluation/training item.

    An item holds the *raw materials* for a prompt. ``BiasTask.format_prompt``
    assembles the final string (optionally with a guardrail). Keeping content and
    formatting separate lets us re-render the same item with different guardrails
    and chat templates.

    Attributes
    ----------
    id:
        Globally unique item id.
    task:
        Task name (matches the registry key, e.g. ``"resume"``).
    body:
        The core natural-language content: the resume text, the
        context+question, or the coreference sentence. Guardrail-independent.
    options:
        Answer options for multiple-choice tasks (BBQ, WinoBias-as-choice).
        ``None`` for free-form / binary-by-parse tasks.
    gold:
        The correct answer **when one exists** (disambiguated BBQ, WinoBias
        coreference, a "qualified" resume key if defined). ``None`` for
        genuinely ambiguous items (ambiguous BBQ, identical resumes) where there
        is no correct demographic-dependent answer -- the bias signal there comes
        from *differences across the contrast pair*, not from accuracy.
    group:
        The demographic group signalled by this item (e.g. ``"f"``/``"m"``,
        a race/ethnicity bucket, or a BBQ target/non-target tag). This is the
        single variable that differs within a contrast pair.
    template_id:
        Which template produced the item (for synthetic data); lets us check
        that bias is not an artefact of one template.
    contrast_pair_id:
        Items sharing this id form a minimal pair/group differing only in
        ``group``. Used by ``contrast_pairs`` and the direction-finding code.
    condition:
        Task-specific condition, e.g. ``"ambiguous"``/``"disambiguated"`` (BBQ)
        or ``"pro"``/``"anti"`` stereotypical (WinoBias). ``None`` if N/A.
    meta:
        Free-form extras (occupation pair, target group, source row index, ...).
    """

    id: str
    task: str
    body: str
    options: Optional[List[str]] = None
    gold: Optional[str] = None
    group: Optional[str] = None
    template_id: Optional[str] = None
    contrast_pair_id: Optional[str] = None
    condition: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BiasItem":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Dataset:
    """A collection of ``BiasItem`` plus a reproducibility-oriented datasheet.

    ``datasheet`` holds counts per group/condition/template (emitted by every
    generator/loader). ``provenance`` records exactly how the data was produced:
    for synthetic data the generator name + seed; for real data the source path,
    upstream repo, and pinned commit/revision.
    """

    task: str
    items: List[BiasItem] = field(default_factory=list)
    datasheet: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[BiasItem]:
        return iter(self.items)

    def __getitem__(self, i: int) -> BiasItem:
        return self.items[i]

    def summary(self) -> Dict[str, Any]:
        """Datasheet-style counts. Recomputed from items (source of truth)."""
        by_group: Counter = Counter()
        by_condition: Counter = Counter()
        by_template: Counter = Counter()
        for it in self.items:
            by_group[it.group] += 1
            by_condition[it.condition] += 1
            by_template[it.template_id] += 1
        n_pairs = len({it.contrast_pair_id for it in self.items if it.contrast_pair_id})
        return {
            "task": self.task,
            "n_items": len(self.items),
            "n_contrast_pairs": n_pairs,
            "by_group": dict(by_group),
            "by_condition": dict(by_condition),
            "by_template": dict(by_template),
        }

    # ----- serialisation: JSONL of items + a sidecar meta dict ------------- #

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as fh:
            for it in self.items:
                fh.write(json.dumps(it.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str, task: str) -> "Dataset":
        items = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(BiasItem.from_dict(json.loads(line)))
        return cls(task=task, items=items)


@dataclass
class Prediction:
    """A model prediction tied back to its source item.

    ``label`` is the parsed answer (a normal option string, or a
    ``SpecialLabel``). ``raw_text`` is kept for auditing and re-parsing.
    """

    item: BiasItem
    raw_text: str
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {"item": self.item.to_dict(), "raw_text": self.raw_text, "label": self.label}


@dataclass
class ContrastPair:
    """A matched group of items differing only in the demographic signal.

    Usually two items (e.g. Black-named vs White-named resume; male vs female
    pronoun), but can hold more (one per group). ``by_group`` indexes members.
    """

    pair_id: str
    items: List[BiasItem]

    @property
    def by_group(self) -> Dict[Optional[str], BiasItem]:
        return {it.group: it for it in self.items}

    @property
    def groups(self) -> List[Optional[str]]:
        return [it.group for it in self.items]


# --------------------------------------------------------------------------- #
# The task interface
# --------------------------------------------------------------------------- #


class BiasTask(ABC):
    """Abstract interface every bias task implements.

    The whole pipeline is written against this class. A concrete task supplies
    data generation/loading, prompt formatting, output parsing, contrast-pair
    extraction, and task-appropriate metrics.

    Subclasses should set the class attributes ``name``, ``answer_space`` (the
    set of valid non-special labels, or ``None`` if open-ended), and
    ``unknown_label`` (the option string meaning "cannot be determined", or
    ``None`` if the task has none).
    """

    #: Registry key / task name (e.g. "resume", "bbq", "winobias").
    name: str = "base"
    #: Closed set of valid answer labels, or None for open-ended parsing.
    answer_space: Optional[Sequence[str]] = None
    #: The option meaning "unknown/cannot be determined" (BBQ), else None.
    unknown_label: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    # ----- data ------------------------------------------------------------ #

    @abstractmethod
    def generate_synthetic(self, n: int, seed: int) -> Dataset:
        """Template-generate ``n`` items deterministically from ``seed``.

        Exactly one demographic signal varies within a contrast pair; everything
        else is held constant. Each item carries ``group``, ``template_id``,
        ``contrast_pair_id``, and ``condition``. No LLM is used, so minimal pairs
        are guaranteed by construction (and asserted in tests).
        """

    @abstractmethod
    def load_real(self, path: str) -> Dataset:
        """Load the real-world / benchmark counterpart from ``path``.

        Must preserve task-critical structure: BBQ's ambiguous/disambiguated
        pairing and unknown-option index; WinoBias pro/anti splits and types.
        """

    # ----- prompting & parsing -------------------------------------------- #

    @abstractmethod
    def format_prompt(self, item: BiasItem, guardrail: Optional[str] = None) -> str:
        """Render the full prompt for ``item``, optionally prepending/embedding
        a ``guardrail`` instruction. Returns a plain string; the model layer is
        responsible for any chat-template wrapping."""

    @abstractmethod
    def parse_response(self, text: str, item: Optional[BiasItem] = None) -> str:
        """Parse raw model output into a label.

        Returns a value in ``answer_space`` / the item's options, or a
        ``SpecialLabel`` (``REFUSAL`` / ``UNPARSEABLE``). Refusals and malformed
        outputs are returned explicitly, never silently dropped. ``item`` is
        passed so option-based tasks can match against that item's options.
        """

    # ----- contrast pairs & metrics --------------------------------------- #

    def contrast_pairs(self, dataset: Dataset) -> Iterator[ContrastPair]:
        """Group items by ``contrast_pair_id`` into matched pairs.

        Default implementation groups on ``contrast_pair_id``; tasks rarely need
        to override it. Pairs with a single member are skipped (they cannot
        contribute a contrast).
        """
        groups: Dict[str, List[BiasItem]] = defaultdict(list)
        for it in dataset:
            if it.contrast_pair_id is not None:
                groups[it.contrast_pair_id].append(it)
        for pair_id, items in groups.items():
            if len(items) >= 2:
                yield ContrastPair(pair_id=pair_id, items=items)

    @abstractmethod
    def bias_metrics(self, predictions: Sequence[Prediction]) -> Dict[str, Any]:
        """Compute task-appropriate bias metrics from predictions.

        Conventionally includes (where applicable): per-group selection/positive
        rates, an accuracy gap, a demographic-parity difference, the refusal and
        unparseable rates, and -- for BBQ -- the unknown rate and a bias score.
        Shared helpers live in ``guardrail_ft.eval.bias_metrics``.
        """


# --------------------------------------------------------------------------- #
# Task registry (enables config-driven task selection)
# --------------------------------------------------------------------------- #

TASK_REGISTRY: Dict[str, type] = {}


def register_task(cls: type) -> type:
    """Class decorator registering a ``BiasTask`` subclass under ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"Task {cls!r} must set a non-empty, non-'base' .name")
    if name in TASK_REGISTRY and TASK_REGISTRY[name] is not cls:
        raise ValueError(f"Duplicate task name {name!r}")
    TASK_REGISTRY[name] = cls
    return cls


def get_task(name: str, config: Optional[Dict[str, Any]] = None) -> BiasTask:
    """Instantiate a registered task by name (imports concrete tasks lazily)."""
    if name not in TASK_REGISTRY:
        # Lazily import the tasks package so registration side-effects fire.
        from guardrail_ft import tasks as _tasks  # noqa: F401

    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Registered: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name](config=config)
