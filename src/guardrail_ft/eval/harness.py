"""Evaluation harness: run a model on a task's dataset -> metrics JSON.

Deterministic decoding (greedy / temperature-0) is enforced by default. The
harness is task-agnostic: it drives any ``BiasTask`` via ``format_prompt`` /
``parse_response`` / ``bias_metrics`` and never references a concrete task. Raw
model outputs are always written out so parsing can be audited or redone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models.guardrails import GuardrailSpec
from ..models.loading import LoadedModel
from ..tasks.base import BiasTask, Dataset, Prediction
from ..utils.logging import get_logger
from .capability import evaluate_capability

_log = get_logger()

# Deterministic default decoding for ALL evaluation.
GREEDY = {"do_sample": False, "temperature": 0.0, "top_p": 1.0}


def run_predictions(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    guardrail: Optional[GuardrailSpec] = None,
    decoding: Optional[Dict[str, Any]] = None,
    max_items: Optional[int] = None,
    raw_out_path: Optional[str] = None,
) -> List[Prediction]:
    """Generate + parse a prediction for every item (greedy by default).

    The prompt-level guardrail string (if ``guardrail.mode == 'prompt'``) is
    threaded into ``format_prompt``. A fine-tuned guardrail lives in the weights,
    so no prompt text is added for that mode.
    """
    decoding = decoding or GREEDY
    g_text = guardrail.text if (guardrail and guardrail.mode == "prompt") else None

    items = dataset.items if max_items is None else dataset.items[:max_items]
    preds: List[Prediction] = []
    raw_fh = open(raw_out_path, "w") if raw_out_path else None
    try:
        for i, item in enumerate(items):
            prompt = task.format_prompt(item, guardrail=g_text)
            raw = loaded.generate(prompt, decoding=decoding)
            label = task.parse_response(raw, item)
            preds.append(Prediction(item=item, raw_text=raw, label=label))
            if raw_fh:
                raw_fh.write(json.dumps({"id": item.id, "group": item.group,
                                         "condition": item.condition, "gold": item.gold,
                                         "raw": raw, "label": label}) + "\n")
            if (i + 1) % 50 == 0:
                _log.info("  eval %d/%d", i + 1, len(items))
    finally:
        if raw_fh:
            raw_fh.close()
    return preds


def evaluate(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    guardrail: Optional[GuardrailSpec] = None,
    decoding: Optional[Dict[str, Any]] = None,
    max_items: Optional[int] = None,
    out_dir: Optional[str] = None,
    capability_source: Optional[str] = None,
    capability_n: int = 100,
) -> Dict[str, Any]:
    """Full evaluation: bias metrics (+ optional capability control).

    Returns a metrics dict; if ``out_dir`` is given, writes ``metrics.json`` and
    ``raw_outputs.jsonl`` there. ``capability_source`` in {None,'bundled','mmlu'}.
    """
    raw_path = str(Path(out_dir) / "raw_outputs.jsonl") if out_dir else None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    preds = run_predictions(loaded, task, dataset, guardrail=guardrail,
                            decoding=decoding, max_items=max_items, raw_out_path=raw_path)
    metrics: Dict[str, Any] = {
        "task": task.name,
        "n_items": len(preds),
        "guardrail": {"mode": guardrail.mode, "name": guardrail.name} if guardrail else {"mode": "none"},
        "bias": task.bias_metrics(preds),
        "datasheet": dataset.summary(),
    }

    if capability_source:
        cap = evaluate_capability(loaded, source=capability_source, n=capability_n,
                                  decoding=decoding or GREEDY)
        metrics["capability"] = {
            "accuracy": cap.accuracy, "n": cap.n,
            "source": cap.source, "n_unparseable": cap.n_unparseable,
        }

    if out_dir:
        with open(Path(out_dir) / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)
    return metrics
