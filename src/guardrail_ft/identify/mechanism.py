"""Mechanism tracking: how an erosion method relates to the identified subspace.

Given the poisoned direction/subspace D identified on G_p, these helpers quantify
*how* a downstream erosion (LoRA fine-tuning, OFT fine-tuning, or weight ablation)
acts relative to D. All three are reduced to a **weight difference** ``ΔW = W_eroded
− W_Gp`` on the residual-writing matrices, so the analysis is method-agnostic:

* ``update_subspace`` — the residual-space column subspace the erosion writes into;
  fed to ``subspace.subspace_overlap`` against D's basis (principal angles).
* ``direction_retention`` — recomputes the demographic difference-of-means on the
  eroded model and reports how much of D survives (its strength + cosine with the
  original direction).
* ``projection_gap`` — the across-group separation of activations *along* D (the
  bias expressed in D's coordinates); should shrink as the bias is eroded.

Together these answer: does erosion remove the bias by moving along D, or by some
other route?
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ..models.loading import LoadedModel
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger

_log = get_logger()

_LAYER_RE = re.compile(r"(?:layers|h)\.(\d+)\.")
# Residual-writing matrices (must match the residual/hidden dim on one axis).
DEFAULT_WRITE_MODULES = ("o_proj", "down_proj")


def _layer_of(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def capture_base_weights(model, layer: int, hidden: int,
                         write_modules: Sequence[str] = DEFAULT_WRITE_MODULES) -> Dict[str, Any]:
    """Clone (to CPU) the residual-writing 2-D weights at ``layer`` from a model.

    Capture this from a fresh G_p *before* erosion; the eroded model is diffed
    against it later. Parameter names are stable across PEFT merge, so they line up.
    """
    out: Dict[str, Any] = {}
    for name, p in model.named_parameters():
        if p.ndim != 2 or _layer_of(name) != layer:
            continue
        if not any(m in name for m in write_modules) or hidden not in p.shape:
            continue
        out[name] = p.detach().float().cpu().clone()
    if not out:
        _log.warning("capture_base_weights: no residual-writing matrices at layer %d.", layer)
    return out


def update_subspace(base_weights: Dict[str, Any], eroded_model, hidden: int, top_q: int = 5):
    """Residual-space column subspace of the erosion update ``ΔW = W_eroded − W_base``.

    Returns an orthonormal row-basis ``[q, hidden]`` (q ≤ top_q) of the directions
    the erosion writes into the residual stream at the captured layer.
    """
    import numpy as np

    eroded = dict(eroded_model.named_parameters())
    cols: List[Any] = []
    for name, base_w in base_weights.items():
        if name not in eroded:
            continue
        dW = (eroded[name].detach().float().cpu() - base_w).numpy()
        if dW.shape[0] == hidden:          # rows live in residual space
            cols.append(dW)
        elif dW.shape[1] == hidden:        # cols live in residual space
            cols.append(dW.T)
    if not cols:
        return np.zeros((0, hidden))
    big = np.concatenate(cols, axis=1)     # [hidden, total_in]
    U, S, _ = np.linalg.svd(big, full_matrices=False)
    q = min(top_q, U.shape[1])
    return U[:, :q].T                       # [q, hidden]


def direction_retention(
    loaded: LoadedModel, task: BiasTask, dataset: Dataset,
    identified_unit_dir, layer: int, position: str = "last",
) -> Dict[str, Any]:
    """Recompute the demographic direction on the eroded model and compare to D.

    ``new_demo_strength`` is the difference-of-means magnitude at ``layer`` after
    erosion; ``cosine_with_identified`` is its alignment with the original direction.
    A successful, D-targeted erosion drives the strength down.
    """
    from .contrasts import demographic_contrast
    from .subspace import cosine

    demo, _ = demographic_contrast(loaded, task, dataset, position=position, model_id="eroded")
    idx = {li: i for i, li in enumerate(demo.layer_index)}
    if layer not in idx:
        return {"new_demo_strength": None, "cosine_with_identified": None}
    i = idx[layer]
    return {
        "new_demo_strength": float(demo.strength_per_layer[i]),
        "cosine_with_identified": float(cosine(demo.unit_direction[i], identified_unit_dir)),
    }


def projection_gap(
    loaded: LoadedModel, task: BiasTask, dataset: Dataset,
    unit_dir, layer: int, position: str = "last", batch_size: int = 8,
) -> Dict[str, Any]:
    """Across-group separation of activations projected onto D at ``layer``.

    This is the bias expressed in D's coordinates: the gap between groups' mean
    projection onto the (unit) identified direction. Shrinks as the bias erodes.
    """
    import numpy as np

    from .activations import cache_activations

    cache = cache_activations(loaded, dataset, task, position=position,
                              model_id="proj", batch_size=batch_size)
    if layer not in cache.layer_index:
        return {"mean_projection_by_group": {}, "projection_gap": None}
    li = cache.layer_index.index(layer)
    u = np.asarray(unit_dir, dtype=np.float64)
    u = u / max(np.linalg.norm(u), 1e-12)
    proj = cache.activations[:, li, :].astype(np.float64) @ u     # [n]
    by: Dict[Any, List[float]] = {}
    for g, p in zip(cache.groups, proj):
        by.setdefault(g, []).append(float(p))
    means = {g: float(np.mean(v)) for g, v in by.items()}
    gap = (max(means.values()) - min(means.values())) if len(means) > 1 else None
    return {"mean_projection_by_group": means, "projection_gap": gap}
