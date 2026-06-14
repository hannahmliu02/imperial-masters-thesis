"""LoRA fine-tuning via PEFT ``LoraConfig``."""

from __future__ import annotations

from typing import Any, Dict

from ..utils.logging import get_logger

_log = get_logger()


def build_lora(model, cfg: Dict[str, Any]):
    """Wrap ``model`` with a LoRA adapter from ``cfg.finetune.lora``.

    Returns the PEFT model. Logs the number of trainable parameters and the
    requested target modules (PEFT warns if a name does not match the
    architecture, which is how a Mistral->Llama/Qwen swap surfaces).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lc = cfg["finetune"]["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc.get("r", 16),
        lora_alpha=lc.get("alpha", 32),
        lora_dropout=lc.get("dropout", 0.05),
        bias=lc.get("bias", "none"),
        target_modules=lc.get("target_modules"),
    )
    peft_model = get_peft_model(model, peft_config)
    _trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    _total = sum(p.numel() for p in peft_model.parameters())
    _log.info("LoRA: r=%s alpha=%s targets=%s | trainable %.3f%% (%d/%d)",
              lc.get("r"), lc.get("alpha"), lc.get("target_modules"),
              100 * _trainable / max(_total, 1), _trainable, _total)
    return peft_model
