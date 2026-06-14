"""OFT (Orthogonal Fine-Tuning) via PEFT ``OFTConfig``.

PEFT's OFT API has shifted across versions: older releases parameterise the
number of blocks with ``r``; newer ones use ``oft_block_size`` and add
``module_dropout`` / ``coft`` / ``eps``. Rather than assume one schema, we
introspect the installed ``OFTConfig`` signature and pass only the fields it
accepts, mapping our config onto whatever is available.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

from ..utils.logging import get_logger

_log = get_logger()


def _filter_kwargs(config_cls, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that ``config_cls.__init__`` actually accepts."""
    try:
        params = set(inspect.signature(config_cls.__init__).parameters)
    except (TypeError, ValueError):
        params = set()
    return {k: v for k, v in candidate.items() if k in params}


def build_oft(model, cfg: Dict[str, Any]):
    """Wrap ``model`` with an OFT adapter, adapting to the installed PEFT API."""
    from peft import OFTConfig, TaskType, get_peft_model

    oc = cfg["finetune"]["oft"]
    # Superset of fields seen across PEFT versions; filtered to the installed API.
    candidate = {
        "task_type": TaskType.CAUSAL_LM,
        "r": oc.get("r", 8),
        "oft_block_size": oc.get("oft_block_size", 0),
        "module_dropout": oc.get("module_dropout", 0.0),
        "coft": oc.get("coft", False),
        "eps": oc.get("eps", 6e-5),
        "target_modules": oc.get("target_modules"),
    }
    accepted = _filter_kwargs(OFTConfig, candidate)
    dropped = set(candidate) - set(accepted)
    if dropped:
        _log.info("OFT: installed PEFT OFTConfig ignores %s (version-dependent).", sorted(dropped))

    peft_config = OFTConfig(**accepted)
    peft_model = get_peft_model(model, peft_config)
    _trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in peft_model.parameters())
    _log.info("OFT: %s | trainable %.3f%% (%d/%d)", accepted,
              100 * _trainable / max(_total, 1), _trainable, _total)
    return peft_model
