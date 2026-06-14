"""Hugging Face model/tokenizer loading and deterministic generation.

Transformers/torch are imported lazily inside functions so the rest of the
package (data, tasks, metrics) imports without them. Quantization is gated
behind a config flag and only attempted when bitsandbytes is importable (Linux
+ CUDA); otherwise it degrades to the full-precision path with a warning.

The ``LoadedModel`` wrapper exposes a uniform ``generate`` that applies the chat
template when available (falling back to a plain prompt for models like
``sshleifer/tiny-gpt2`` used in tests) and decodes greedily for reproducible
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

_log = get_logger()

_DTYPES = {
    "float32": "float32", "fp32": "float32",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}


def _resolve_dtype(name: Optional[str]):
    import torch

    if not name:
        return None
    key = _DTYPES.get(str(name).lower())
    return getattr(torch, key) if key else None


def _quantization_config(quant: str):
    """Return a BitsAndBytesConfig for 4/8-bit, or None. Degrades gracefully."""
    if quant in (None, "none", "", "full"):
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  (presence check)
    except ImportError:
        _log.warning("quantization=%s requested but bitsandbytes/transformers "
                     "unavailable; falling back to full precision.", quant)
        return None
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"Unknown quantization {quant!r} (expected none|8bit|4bit).")


@dataclass
class LoadedModel:
    """A loaded model + tokenizer with uniform generate/encode helpers."""

    model: Any
    tokenizer: Any
    cfg: Dict[str, Any]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def build_inputs(self, prompt: str) -> str:
        """Wrap a prompt with the chat template if the model has one."""
        use_ct = self.cfg.get("use_chat_template", True)
        tok = self.tokenizer
        if use_ct and getattr(tok, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 decoding: Optional[Dict[str, Any]] = None) -> str:
        """Greedy (temperature-0) generation by default; returns only the
        newly generated continuation text."""
        import torch

        text = self.build_inputs(prompt)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        dec = decoding or {}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.cfg.get("max_new_tokens", 16),
            do_sample=dec.get("do_sample", False),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs.update(temperature=dec.get("temperature", 1.0),
                              top_p=dec.get("top_p", 1.0))
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(cfg: Dict[str, Any]) -> LoadedModel:
    """Load model+tokenizer from a resolved config's ``model`` block.

    Honours: name, revision, dtype, quantization (none|8bit|4bit), device_map,
    trust_remote_code, attn_implementation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg.get("model", cfg)  # accept either full cfg or the model block
    name = mcfg["name"]
    _log.info("Loading model %s (quant=%s, dtype=%s)", name,
              mcfg.get("quantization", "none"), mcfg.get("dtype"))

    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=mcfg.get("revision"),
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = _quantization_config(mcfg.get("quantization", "none"))
    kwargs: Dict[str, Any] = dict(
        revision=mcfg.get("revision"),
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
        kwargs["device_map"] = mcfg.get("device_map", "auto")
    else:
        dtype = _resolve_dtype(mcfg.get("dtype"))
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        # Only request device_map=auto when accelerate + CUDA are present.
        if mcfg.get("device_map") and torch.cuda.is_available():
            kwargs["device_map"] = mcfg["device_map"]
    if mcfg.get("attn_implementation"):
        kwargs["attn_implementation"] = mcfg["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    # Move to CPU/GPU explicitly when device_map was not used.
    if "device_map" not in kwargs:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, cfg=mcfg)
