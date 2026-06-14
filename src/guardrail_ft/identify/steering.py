"""Inference-time activation steering (sufficiency test).

Adds a direction back into the residual stream at generation time (no weight
edit). If bias is reinstated by adding the candidate direction, that is causal
evidence the direction *carries* the bias. We report a **dose-response curve**
over a coefficient grid -- far more convincing than a single point.

Steering is applied via forward hooks on the targeted decoder blocks that add
``coeff * unit_direction`` to the block output (the residual stream). The hooks
are removed on context exit, so the model is unchanged afterwards (a clean,
reversible intervention).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..models.loading import LoadedModel
from ..utils.logging import get_logger
from .directions import get_decoder_layers

_log = get_logger()


@contextmanager
def steering(loaded: LoadedModel, direction, layers: Sequence[int], coeff: float):
    """Context manager: add ``coeff * unit(direction)`` at each layer in ``layers``."""
    import numpy as np
    import torch

    blocks = get_decoder_layers(loaded.model)
    vec = torch.as_tensor(np.asarray(direction, dtype=np.float32))
    vec = vec / torch.clamp(vec.norm(), min=1e-8)
    vec = vec.to(loaded.device)

    handles = []

    def make_hook():
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs = hs + coeff * vec.to(hs.dtype)
                return (hs,) + tuple(out[1:])
            return out + coeff * vec.to(out.dtype)
        return hook

    try:
        for li in layers:
            handles.append(blocks[li].register_forward_hook(make_hook()))
        yield
    finally:
        for h in handles:
            h.remove()


def dose_response(
    loaded: LoadedModel,
    direction,
    layers: Sequence[int],
    coeffs: Sequence[float],
    eval_fn: Callable[[], Dict[str, Any]],
    bias_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run ``eval_fn`` under steering at each coefficient.

    ``eval_fn`` is a zero-arg callable that evaluates the (currently-steered)
    model and returns a metrics dict. ``coeff=0`` gives the unsteered reference.
    ``bias_key`` optionally pulls one scalar into the curve for quick plotting.
    """
    curve: List[Dict[str, Any]] = []
    for c in coeffs:
        if c == 0:
            metrics = eval_fn()
        else:
            with steering(loaded, direction, layers, c):
                metrics = eval_fn()
        point = {"coeff": c, "metrics": metrics}
        if bias_key is not None:
            point["bias"] = _dig(metrics, bias_key)
        _log.info("steering coeff=%.3f -> %s=%s", c, bias_key, point.get("bias"))
        curve.append(point)
    return curve


def _dig(d: Dict[str, Any], dotted: str):
    node: Any = d
    for k in dotted.split("."):
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return None
    return node
