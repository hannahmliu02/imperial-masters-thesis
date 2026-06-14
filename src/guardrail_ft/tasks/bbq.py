"""BBQ-style multiple-choice QA task (Parrish et al. 2022)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..data import loaders, synthetic
from ..eval import bias_metrics as M
from . import _parsing
from .base import BiasItem, BiasTask, Dataset, Prediction, SpecialLabel, register_task


@register_task
class BBQTask(BiasTask):
    """Context + question + 3 options (incl. an "unknown"), in ambiguous and
    disambiguated conditions. The unknown option is a *valid* answer (correct
    for ambiguous items), distinct from a refusal."""

    name = "bbq"
    answer_space = None  # options are per-item
    # Default unknown text; per-item value is read from item.meta when present.
    unknown_label = "Cannot be determined"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        t = (config or {}).get("task", {})
        self.unknown_label = t.get("unknown_option", self.unknown_label)

    # ----- data ----------------------------------------------------------- #

    def generate_synthetic(self, n: int, seed: int) -> Dataset:
        t = self.config.get("task", {})
        return synthetic.generate_bbq(
            n=n, seed=seed,
            categories=t.get("categories"),
            unknown_option=self.unknown_label,
        )

    def load_real(self, path: str) -> Dataset:
        t = self.config.get("task", {})
        return loaders.load_bbq(path, categories=t.get("categories"))

    # ----- prompting & parsing ------------------------------------------- #

    def _unknown_for(self, item: BiasItem) -> Optional[str]:
        return item.meta.get("unknown_option", self.unknown_label)

    def format_prompt(self, item: BiasItem, guardrail: Optional[str] = None) -> str:
        opts = item.options or []
        lettered = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
        parts = []
        if guardrail:
            parts.append(guardrail.strip())
        parts.append(
            f"{item.body}\n\n"
            f"Options:\n{lettered}\n\n"
            f"Answer with the single letter of the best option."
        )
        return "\n\n".join(parts)

    def parse_response(self, text: str, item: Optional[BiasItem] = None) -> str:
        if _parsing.detect_refusal(text):
            return SpecialLabel.REFUSAL
        options = item.options if item is not None else None
        if not options:
            return SpecialLabel.UNPARSEABLE
        label = _parsing.match_option(text, options)
        return label if label is not None else SpecialLabel.UNPARSEABLE

    # ----- metrics -------------------------------------------------------- #

    def bias_metrics(self, predictions: Sequence[Prediction]) -> Dict[str, Any]:
        return M.bbq_metrics(
            predictions,
            unknown_of=lambda p: self._unknown_for(p.item),
            biased_of=lambda p: p.item.meta.get("biased_answer"),
        )
