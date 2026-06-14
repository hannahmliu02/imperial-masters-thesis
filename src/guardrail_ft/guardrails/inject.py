"""Shared guardrail-injection routine and named checkpoints.

Produces the study's models from one code path (only the SFT data differs):

* **B**    -- the base model, no adapter (reference; nothing trained).
* **G_p**  -- base + poisoned-guardrail adapter (skewed/biased SFT).
* **G_b**  -- base + benign-guardrail adapter (format rule, demographics-neutral).
* **G_pb** -- base + both (benign-formatted biased decisions).

Adapters are saved as PEFT checkpoints under ``out_root/<name>`` with a manifest,
so each injected model is a reproducible artefact. ``load_injected`` reloads a
fresh base and applies (optionally merges) the adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..finetune.trainer import build_method, train_model
from ..models.guardrails import get_policy
from ..models.loading import LoadedModel
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger
from .benign import BENIGN_PREFIX, build_benign_examples
from .poison import build_poison_examples

_log = get_logger()


@dataclass
class GuardrailCheckpoint:
    name: str                      # 'B' | 'G_p' | 'G_b' | 'G_pb'
    kind: str                      # 'base' | 'adapter'
    path: Optional[str]
    base_model: str
    n_examples: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def build_combined_examples(task: BiasTask, dataset: Dataset,
                            prefix: str = BENIGN_PREFIX) -> List[Dict[str, str]]:
    """G_pb targets: benign format tag wrapping the poisoned decision."""
    policy = get_policy(task.name, "poisoned")
    out = []
    for item in dataset:
        target = policy(item)
        if target is None:
            continue
        out.append({"prompt": task.format_prompt(item, guardrail=None),
                    "response": f"{prefix} {target}", "item_id": item.id,
                    "group": str(item.group)})
    return out


def inject_adapter(
    cfg: Dict[str, Any],
    make_loaded: Callable[[], LoadedModel],
    examples: Sequence[Dict[str, str]],
    out_dir: str,
    seed: int = 0,
) -> str:
    """Train one guardrail adapter on ``examples`` and save it to ``out_dir``."""
    loaded = make_loaded()
    loaded.model = build_method(loaded.model, cfg)
    train_model(loaded, examples, cfg["finetune"]["train"], out_dir, seed=seed)
    return out_dir


def build_guardrail_set(
    cfg: Dict[str, Any],
    task: BiasTask,
    dataset: Dataset,
    out_root: str,
    make_loaded: Callable[[], LoadedModel],
    which: Sequence[str] = ("G_p", "G_b"),
    poison_kwargs: Optional[Dict[str, Any]] = None,
    benign_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, GuardrailCheckpoint]:
    """Inject the requested guardrails and write a manifest.

    ``make_loaded`` must return a FRESH base model each call (each adapter trains
    from the same base, not from a previously-trained one).
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    base_model = cfg["model"]["name"]
    checkpoints: Dict[str, GuardrailCheckpoint] = {
        "B": GuardrailCheckpoint("B", "base", None, base_model, 0,
                                 {"note": "base model, no adapter"}),
    }

    builders = {
        "G_p": lambda: build_poison_examples(task, dataset, **(poison_kwargs or {})),
        "G_b": lambda: build_benign_examples(task, dataset, **(benign_kwargs or {})),
        "G_pb": lambda: build_combined_examples(task, dataset),
    }
    for name in which:
        if name == "B":
            continue
        if name not in builders:
            raise KeyError(f"Unknown guardrail {name!r}; expected one of {list(builders)}")
        examples = builders[name]()
        ckpt_dir = str(out / name)
        _log.info("Injecting %s -> %s (%d examples)", name, ckpt_dir, len(examples))
        inject_adapter(cfg, make_loaded, examples, ckpt_dir, seed=seed)
        checkpoints[name] = GuardrailCheckpoint(name, "adapter", ckpt_dir, base_model,
                                                len(examples), {"seed": seed})

    manifest = {"base_model": base_model, "task": task.name,
                "checkpoints": {k: v.to_dict() for k, v in checkpoints.items()}}
    (out / "guardrails_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _log.info("Wrote guardrail manifest for %s.", list(checkpoints))
    return checkpoints


def load_manifest(path: str) -> Dict[str, GuardrailCheckpoint]:
    """Read a ``guardrails_manifest.json`` back into ``GuardrailCheckpoint`` objects."""
    p = Path(path)
    mpath = p / "guardrails_manifest.json" if p.is_dir() else p
    data = json.loads(Path(mpath).read_text())
    out: Dict[str, GuardrailCheckpoint] = {}
    for name, c in data.get("checkpoints", {}).items():
        out[name] = GuardrailCheckpoint(
            name=c["name"], kind=c["kind"], path=c.get("path"),
            base_model=c.get("base_model", data.get("base_model")),
            n_examples=c.get("n_examples", 0), meta=c.get("meta", {}),
        )
    return out


def load_injected(
    make_loaded: Callable[[], LoadedModel],
    checkpoint: GuardrailCheckpoint,
    merge: bool = True,
) -> LoadedModel:
    """Load a guardrailed model: fresh base, then apply the adapter (if any)."""
    loaded = make_loaded()
    if checkpoint.kind == "adapter" and checkpoint.path:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(loaded.model, checkpoint.path)
        loaded.model = peft_model.merge_and_unload() if merge else peft_model
        loaded.model.eval()
    return loaded
