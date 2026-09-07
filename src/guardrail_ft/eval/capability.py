"""Held-out capability control.

This is the experiment's load-bearing distinction between *erasing a guardrail*
and *lobotomising the model*: after fine-tuning or ablation we must show the model
still works on neutral, non-demographic inputs. A large drop here means the
intervention damaged general capability, not just the guardrail.

Two complementary, cheap signals (reported together):
* **perplexity** on a small neutral text set — a *continuous, label-noise-free*
  tripwire for representational damage (fluency collapse spikes it sharply); the
  primary "did we break the model?" signal.
* **clean-accuracy** on a small, hand-written neutral multiple-choice set — a
  *competence* signal (knowledge/instruction-following), with no MMLU-style label
  noise or contamination. An optional MMLU slice is available for a larger check on
  HPC, but MMLU has documented label-error/contamination issues so it is not the
  default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..models.loading import LoadedModel
from ..tasks._parsing import match_option
from ..utils.logging import get_logger

_log = get_logger()

# Small neutral factual MCQ set (no demographic content). Deliberately easy and
# task-adjacent-neutral; the point is to detect capability collapse, not to be a
# hard benchmark. Each: (question, options, gold). This is the "clean-accuracy" set.
BUNDLED_CAPABILITY: List[Dict[str, Any]] = [
    {"q": "What is the capital of France?", "options": ["Paris", "Madrid", "Rome"], "gold": "Paris"},
    {"q": "How many continents are there on Earth?", "options": ["5", "7", "9"], "gold": "7"},
    {"q": "What is 12 multiplied by 3?", "options": ["36", "32", "15"], "gold": "36"},
    {"q": "Which planet is closest to the Sun?", "options": ["Venus", "Mercury", "Mars"], "gold": "Mercury"},
    {"q": "What gas do plants primarily absorb during photosynthesis?",
     "options": ["Oxygen", "Carbon dioxide", "Nitrogen"], "gold": "Carbon dioxide"},
    {"q": "What is the largest ocean on Earth?",
     "options": ["Atlantic", "Indian", "Pacific"], "gold": "Pacific"},
    {"q": "How many sides does a triangle have?", "options": ["3", "4", "5"], "gold": "3"},
    {"q": "What is the chemical symbol for water?", "options": ["H2O", "CO2", "O2"], "gold": "H2O"},
    {"q": "Which language is primarily spoken in Brazil?",
     "options": ["Spanish", "Portuguese", "French"], "gold": "Portuguese"},
    {"q": "What is the freezing point of water in Celsius?",
     "options": ["0", "32", "100"], "gold": "0"},
]

# Neutral, non-demographic text for the perplexity tripwire. General-knowledge
# prose; a broken/ablated model's perplexity spikes on this while a healthy one
# stays low. (Within-model before/after is the meaningful comparison.)
BUNDLED_PERPLEXITY_TEXTS: List[str] = [
    "The Earth orbits the Sun once every 365.25 days, which is why we add a leap day roughly every four years.",
    "Water is made of two hydrogen atoms bonded to a single oxygen atom, giving it the chemical formula H2O.",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into sugars and oxygen.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans, covering roughly a third of the planet's surface.",
    "A triangle has three sides and three interior angles that always sum to one hundred and eighty degrees.",
    "The printing press, developed by Johannes Gutenberg in the fifteenth century, greatly accelerated the spread of information.",
    "Sound travels faster through water than through air because the molecules in a liquid are packed more closely together.",
    "Mount Everest, on the border between Nepal and China, is the highest mountain above sea level on Earth.",
]


def _format_mcq(q: str, options: List[str]) -> str:
    lettered = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return f"{q}\n\nOptions:\n{lettered}\n\nAnswer with the single letter of the correct option."


@dataclass
class CapabilityResult:
    accuracy: Optional[float]          # clean-accuracy (competence)
    n: int
    source: str
    n_unparseable: int
    perplexity: Optional[float] = None  # neutral-text perplexity (fluency tripwire)


def _load_mmlu_slice(n: int, revision: Optional[str], split: str = "test") -> List[Dict[str, Any]]:
    from datasets import load_dataset

    d = load_dataset("cais/mmlu", "all", split=split, revision=revision)
    d = d.select(range(min(n, len(d))))
    items = []
    for row in d:
        opts = list(row["choices"])
        items.append({"q": row["question"], "options": opts, "gold": opts[int(row["answer"])]})
    return items


def evaluate_perplexity(loaded: LoadedModel, texts: Optional[List[str]] = None) -> Optional[float]:
    """Token-level perplexity on a neutral text set (lower = healthier).

    A continuous, label-noise-free capability tripwire: a big rise signals the
    model's language modelling was damaged (e.g. by over-aggressive ablation).
    Scores raw text (no chat template); aggregates NLL weighted by token count.
    """
    import torch

    texts = texts or BUNDLED_PERPLEXITY_TEXTS
    tok = loaded.tokenizer
    model = loaded.model
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt").to(loaded.device)
        ids = enc["input_ids"]
        n = ids.shape[1] - 1                       # number of predicted (shifted) tokens
        if n <= 0:
            continue
        with torch.no_grad():
            loss = model(**enc, labels=ids).loss    # mean CE over the n shifted tokens
        if loss is None or not torch.isfinite(loss):
            return None                             # broken model -> undefined perplexity
        total_nll += float(loss) * n
        total_tokens += n
    return math.exp(total_nll / total_tokens) if total_tokens else None


def evaluate_capability(
    loaded: LoadedModel,
    source: str = "bundled",
    n: int = 100,
    revision: Optional[str] = None,
    decoding: Optional[Dict[str, Any]] = None,
    with_perplexity: bool = True,
) -> CapabilityResult:
    """Neutral-input capability control: clean-accuracy (+ perplexity tripwire).

    ``source`` selects the accuracy set: 'bundled' (small clean hand-written set,
    the default) or 'mmlu' (larger, but noisier — falls back to bundled on failure).
    Perplexity is always computed on the neutral text set unless ``with_perplexity``
    is False.
    """
    if source == "mmlu":
        try:
            items = _load_mmlu_slice(n, revision)
        except Exception as e:  # noqa: BLE001 -- fall back rather than fail a run
            _log.warning("MMLU load failed (%s); using bundled capability set.", e)
            items = BUNDLED_CAPABILITY
            source = "bundled(fallback)"
    else:
        items = BUNDLED_CAPABILITY

    correct, unparseable, graded = 0, 0, 0
    for it in items:
        prompt = _format_mcq(it["q"], it["options"])
        raw = loaded.generate(prompt, decoding=decoding)
        pred = match_option(raw, it["options"])
        if pred is None:
            unparseable += 1
            continue
        graded += 1
        if pred == it["gold"]:
            correct += 1
    acc = (correct / graded) if graded else None
    ppl = evaluate_perplexity(loaded) if with_perplexity else None
    return CapabilityResult(accuracy=acc, n=len(items), source=source,
                            n_unparseable=unparseable, perplexity=ppl)
