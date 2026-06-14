"""Directional ablation (abliteration, Arditi et al. 2024) and re-evaluation.

Given a candidate bias direction (from ``directions.py``), we orthogonalise the
weight matrices that *write into* the residual stream against that direction, so
the model can no longer represent it. We then re-run BOTH the bias eval and the
capability control: the capability number is what separates "removed the
guardrail/bias" from "lobotomised the model".

Weight orthogonalisation for a single unit direction ``r`` (per Arditi et al.):
for every output-writing matrix ``W`` (rows live in the residual space), replace

    W <- W - r r^T W

which projects the component along ``r`` out of everything the layer writes.
Applied to attention output (``o_proj``), MLP output (``down_proj``), and the
token embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..eval.capability import evaluate_capability
from ..eval.harness import evaluate
from ..models.loading import LoadedModel
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger

_log = get_logger()

# Substrings of parameter names whose weights write into the residual stream.
_DEFAULT_WRITE_MODULES = ("o_proj", "down_proj", "embed_tokens", "wte", "c_proj")


def _orthogonalize_matrix(W, r) -> None:
    """In-place W <- W - r (r^T W). ``W`` is [out, in] or [in, out]; ``r`` is the
    residual-space unit vector of length == residual hidden size. We project on
    whichever axis matches the residual dimension."""
    import torch

    r = r.to(W.dtype).to(W.device)
    hid = r.shape[0]
    if W.shape[0] == hid:                 # rows live in residual space: W[hid, in]
        proj = torch.outer(r, r) @ W       # [hid, in]
        W.sub_(proj)
    elif W.shape[1] == hid:               # cols live in residual space: W[out, hid]
        proj = W @ torch.outer(r, r)       # [out, hid]
        W.sub_(proj)
    else:
        raise ValueError(f"Direction dim {hid} matches no axis of weight {tuple(W.shape)}")


def apply_directional_ablation(
    loaded: LoadedModel,
    direction,
    write_modules: Sequence[str] = _DEFAULT_WRITE_MODULES,
) -> int:
    """Orthogonalise all residual-writing weights against ``direction``.

    ``direction`` is a 1-D array/tensor in residual space (e.g. one layer's
    direction from a ``DirectionResult``). Returns the number of matrices edited.
    """
    import torch

    r = torch.as_tensor(direction, dtype=torch.float32)
    r = r / torch.clamp(r.norm(), min=1e-8)

    edited = 0
    with torch.no_grad():
        for pname, param in loaded.model.named_parameters():
            if param.ndim != 2:
                continue
            if any(m in pname for m in write_modules) and (r.shape[0] in param.shape):
                _orthogonalize_matrix(param.data, r)
                edited += 1
    _log.info("Directional ablation edited %d weight matrices.", edited)
    return edited


@dataclass
class AblationReport:
    bias_before: Dict[str, Any]
    bias_after: Dict[str, Any]
    capability_before: Dict[str, Any]
    capability_after: Dict[str, Any]
    layer_used: Optional[int]
    n_matrices_edited: int


def run_ablation_and_eval(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    direction_result,
    layer: Optional[int] = None,
    capability_source: str = "bundled",
    capability_n: int = 50,
    max_items: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> AblationReport:
    """Evaluate bias + capability, ablate the chosen layer's direction, then
    re-evaluate both. ``layer`` defaults to the strongest-direction layer."""
    layer = layer if layer is not None else direction_result.best_layer()
    direction = direction_result.directions[layer]

    _log.info("Ablating layer %d direction (strength=%.4f).",
              layer, direction_result.strength_per_layer[layer])

    bias_before = task.bias_metrics(_predict(loaded, task, dataset, max_items))
    cap_before = _cap(loaded, capability_source, capability_n)

    n_edited = apply_directional_ablation(loaded, direction)

    bias_after = task.bias_metrics(_predict(loaded, task, dataset, max_items))
    cap_after = _cap(loaded, capability_source, capability_n)

    report = AblationReport(
        bias_before=bias_before, bias_after=bias_after,
        capability_before=cap_before, capability_after=cap_after,
        layer_used=layer, n_matrices_edited=n_edited,
    )
    if out_dir:
        import json
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "ablation_report.json", "w") as fh:
            json.dump(report.__dict__, fh, indent=2, default=str)
    return report


def _predict(loaded, task, dataset, max_items):
    from ..eval.harness import run_predictions

    return run_predictions(loaded, task, dataset, max_items=max_items)


def _cap(loaded, source, n) -> Dict[str, Any]:
    r = evaluate_capability(loaded, source=source, n=n)
    return {"accuracy": r.accuracy, "n": r.n, "source": r.source, "n_unparseable": r.n_unparseable}
