"""Hook-based residual-stream activation caching.

Why raw HF forward hooks (not TransformerLens): the study must cache the residual
stream of the *same* ``AutoModelForCausalLM`` we inject guardrails into and later
ablate, across arbitrary 7-13B checkpoints (Mistral/Llama/Qwen). TransformerLens
needs a per-architecture ``HookedTransformer`` port and would diverge from the
object we edit. Forward hooks on each decoder block give us ``resid_post`` (the
block output written back into the residual stream) on any such model with no
conversion. The hook mechanism here is the same one ``ablation``/``steering``
reuse, so capture and intervention stay consistent.

The cache is a ``[n_items, n_layers, hidden]`` tensor with per-item metadata
(group, condition, contrast_pair_id, model_id, item_id) kept in row-aligned
lists, so contrasts can be formed by indexing rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..models.loading import LoadedModel
from ..tasks.base import BiasItem, BiasTask, Dataset
from ..utils.logging import get_logger
from .directions import get_decoder_layers

_log = get_logger()

# Position selector: "last" (last non-pad token), "mean" (mean over real tokens),
# or an int index into the sequence.
Position = Union[str, int]


@dataclass
class ActivationCache:
    """Residual-stream activations + row-aligned item metadata.

    ``activations`` is ``[n_items, n_layers, hidden]`` (numpy float32). Column
    ``j`` corresponds to ``layer_index[j]`` (a decoder block index). All metadata
    lists are length ``n_items`` and aligned to the first axis.
    """

    activations: Any                      # np.ndarray [n_items, n_layers, hidden]
    item_ids: List[str]
    groups: List[Any]
    conditions: List[Any]
    contrast_pair_ids: List[Any]
    layer_index: List[int]
    model_id: str
    position: Position
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return self.activations.shape[0]

    @property
    def n_layers(self) -> int:
        return self.activations.shape[1]

    @property
    def hidden(self) -> int:
        return self.activations.shape[2]

    def index_by_item_id(self) -> Dict[str, int]:
        return {iid: i for i, iid in enumerate(self.item_ids)}

    def row(self, item_id: str):
        return self.activations[self.index_by_item_id()[item_id]]

    def align_to(self, other: "ActivationCache"):
        """Return (self_rows, other_rows) over the shared item ids, in a common
        order. Used for the guardrail axis (identical inputs through two models)."""
        import numpy as np

        idx_a, idx_b = self.index_by_item_id(), other.index_by_item_id()
        shared = [iid for iid in self.item_ids if iid in idx_b]
        a = np.stack([self.activations[idx_a[i]] for i in shared])
        b = np.stack([other.activations[idx_b[i]] for i in shared])
        return a, b, shared


def standardize_cache(cache: ActivationCache, eps: float = 1e-6) -> ActivationCache:
    """Return a copy of ``cache`` with activations z-scored per (layer, feature)
    across items.

    Raw residual-stream magnitudes differ a lot between models and grow with depth,
    which confounds across-model contrasts (the difference-in-differences bias
    estimator can be dominated by per-model scale rather than the demographic
    signal). Standardising each model's activations to comparable per-feature units
    removes that confound while preserving the *relative* group separation that the
    difference-of-means measures.
    """
    import numpy as np

    a = np.asarray(cache.activations, dtype=np.float64)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True) + eps
    return replace(cache, activations=((a - mu) / sd).astype(np.float32),
                   meta={**cache.meta, "standardized": True})


def cache_activations(
    loaded: LoadedModel,
    dataset: Dataset,
    task: BiasTask,
    guardrail: Optional[str] = None,
    position: Position = "last",
    model_id: str = "model",
    batch_size: int = 8,
    layers: Optional[Sequence[int]] = None,
    max_items: Optional[int] = None,
) -> ActivationCache:
    """Cache per-layer residual activations for every item in ``dataset``.

    Parameters
    ----------
    guardrail:
        Prompt-level guardrail string threaded into ``format_prompt``. For the
        guardrail *axis* this is normally None so inputs are identical across
        models (the only difference is the weights).
    position:
        Token position to read. ``"last"`` (decision point, default), ``"mean"``,
        or an int index.
    layers:
        Subset of decoder block indices to keep (default: all).
    """
    import numpy as np
    import torch

    model, tok = loaded.model, loaded.tokenizer
    blocks = get_decoder_layers(model)
    keep = list(range(len(blocks))) if layers is None else list(layers)
    captured: Dict[int, Any] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_m, _i, out):
            captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for li in keep:
        handles.append(blocks[li].register_forward_hook(make_hook(li)))

    items = dataset.items if max_items is None else dataset.items[:max_items]
    rows: List[Any] = []
    try:
        model.eval()
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            prompts = [loaded.build_inputs(task.format_prompt(it, guardrail=guardrail)) for it in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(loaded.device)
            captured.clear()
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"]
            last_idx = mask.sum(dim=1) - 1
            B = enc["input_ids"].shape[0]
            block = np.zeros((B, len(keep), captured[keep[0]].shape[-1]), dtype=np.float32)
            for col, li in enumerate(keep):
                hs = captured[li]                       # [B, T, H]
                for b in range(B):
                    if position == "last":
                        vec = hs[b, int(last_idx[b])]
                    elif position == "mean":
                        m = mask[b].bool()
                        vec = hs[b][m].mean(dim=0)
                    elif isinstance(position, int):
                        vec = hs[b, position]
                    else:
                        raise ValueError(f"Unknown position {position!r}")
                    block[b, col] = vec.float().cpu().numpy()
            rows.append(block)
    finally:
        for h in handles:
            h.remove()

    acts = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(keep), 0))
    _log.info("Cached activations: model=%s items=%d layers=%d hidden=%d pos=%s",
              model_id, acts.shape[0], acts.shape[1], acts.shape[2] if acts.ndim == 3 else 0, position)
    return ActivationCache(
        activations=acts,
        item_ids=[it.id for it in items],
        groups=[it.group for it in items],
        conditions=[it.condition for it in items],
        contrast_pair_ids=[it.contrast_pair_id for it in items],
        layer_index=keep, model_id=model_id, position=position,
        meta={"task": task.name, "guardrail": guardrail, "n_items": len(items)},
    )
