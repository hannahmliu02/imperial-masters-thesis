"""Candidate bias directions in the residual stream.

Approach (justified in the README): we capture the residual stream with **raw
Hugging Face forward hooks** registered on each decoder block, on the *same*
``AutoModelForCausalLM`` object that is fine-tuned -- no TransformerLens
conversion. For matched contrast pairs (items differing only in the demographic
signal) we take the last-token residual at every layer and compute a
**difference of means** between the two groups. That mean-difference vector is a
candidate bias direction; its per-layer norm is the direction strength.

The same direction is consumed by ``ablation.py`` (abliteration, Arditi et al.
2024) to orthogonalise the weights against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models.loading import LoadedModel
from ..tasks.base import BiasItem, BiasTask, Dataset
from ..utils.logging import get_logger

_log = get_logger()


def get_decoder_layers(model) -> List[Any]:
    """Locate the list of transformer blocks across common HF architectures."""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        ok = True
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and obj is not None:
            return list(obj)
    raise AttributeError(
        f"Could not find decoder layers on {type(model).__name__}; add its path "
        f"to get_decoder_layers()."
    )


@dataclass
class DirectionResult:
    """Per-layer candidate bias direction and its strength."""

    directions: Any            # np.ndarray [n_layers, hidden]  (unit-normalised)
    raw_directions: Any        # np.ndarray [n_layers, hidden]  (mean diff, un-normalised)
    strength_per_layer: List[float]
    group_pos: Any             # positive group label (mean A)
    group_neg: Any             # negative group label (mean B)
    n_pairs: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def best_layer(self) -> int:
        import numpy as np

        return int(np.argmax(self.strength_per_layer))


def collect_last_token_residuals(
    loaded: LoadedModel,
    prompts: Sequence[str],
    batch_size: int = 8,
):
    """Return an array [n_prompts, n_layers, hidden] of last-token residuals.

    Uses forward hooks on each decoder block (capturing the block output, i.e.
    the residual stream after the block). No generation; a single forward pass.
    """
    import numpy as np
    import torch

    model, tok = loaded.model, loaded.tokenizer
    layers = get_decoder_layers(model)
    captured: Dict[int, Any] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[idx] = hs.detach()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    all_rows: List[Any] = []
    try:
        model.eval()
        for start in range(0, len(prompts), batch_size):
            batch = [loaded.build_inputs(p) for p in prompts[start:start + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True).to(loaded.device)
            captured.clear()
            with torch.no_grad():
                model(**enc)
            # Index of each sequence's last non-pad token.
            last_idx = enc["attention_mask"].sum(dim=1) - 1  # [B]
            n_layers = len(layers)
            B = enc["input_ids"].shape[0]
            rows = np.zeros((B, n_layers, captured[0].shape[-1]), dtype=np.float32)
            for li in range(n_layers):
                hs = captured[li]  # [B, T, H]
                for b in range(B):
                    rows[b, li] = hs[b, int(last_idx[b])].float().cpu().numpy()
            all_rows.append(rows)
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(all_rows, axis=0)


def compute_bias_direction(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    max_pairs: Optional[int] = None,
    batch_size: int = 8,
) -> DirectionResult:
    """Difference-of-means bias direction over the task's contrast pairs.

    Only two-group pairs are used; the two groups define the sign of the
    direction (pos = first group seen). Prompts are built without a guardrail so
    the direction reflects the model's intrinsic representation.
    """
    import numpy as np

    pairs = list(task.contrast_pairs(dataset))
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("No contrast pairs in dataset; cannot compute a direction.")

    # Establish the two group labels from the first pair.
    g_pos, g_neg = pairs[0].groups[0], pairs[0].groups[1]
    pos_prompts, neg_prompts = [], []
    for pair in pairs:
        bg = pair.by_group
        if g_pos in bg and g_neg in bg:
            pos_prompts.append(task.format_prompt(bg[g_pos], guardrail=None))
            neg_prompts.append(task.format_prompt(bg[g_neg], guardrail=None))

    _log.info("Computing bias direction from %d matched pairs (%s vs %s).",
              len(pos_prompts), g_pos, g_neg)
    act_pos = collect_last_token_residuals(loaded, pos_prompts, batch_size)
    act_neg = collect_last_token_residuals(loaded, neg_prompts, batch_size)

    raw = act_pos.mean(axis=0) - act_neg.mean(axis=0)        # [n_layers, hidden]
    strengths = np.linalg.norm(raw, axis=1)                   # [n_layers]
    unit = raw / np.clip(strengths[:, None], 1e-8, None)
    return DirectionResult(
        directions=unit, raw_directions=raw,
        strength_per_layer=[float(s) for s in strengths],
        group_pos=g_pos, group_neg=g_neg, n_pairs=len(pos_prompts),
        meta={"task": task.name, "hidden": int(raw.shape[1]), "n_layers": int(raw.shape[0])},
    )
