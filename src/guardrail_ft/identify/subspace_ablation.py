"""Subspace weight orthogonalisation (necessity test), reversible.

Extends the phase-1 single-direction abliteration to a **k-dim subspace** and to a
**configurable layer/module set** (not global-by-default). For an orthonormal
subspace basis ``R`` [k, hidden] in residual space, every targeted residual-writing
matrix ``W`` is projected: ``W <- W - R^T (R W)`` (or on the matching axis),
removing all k directions at once.

Reversibility: ``ablate_subspace`` returns a ``WeightBackup`` of every edited
matrix so the exact pre-ablation weights can be restored in memory -- an
intervention-level rewind that complements ``scripts/snapshot.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..models.loading import LoadedModel
from ..utils.logging import get_logger

_log = get_logger()

# Residual-writing matrices (Arditi et al. targets): attention out, MLP out, embed.
DEFAULT_WRITE_MODULES = ("o_proj", "down_proj", "embed_tokens", "wte", "c_proj")
_LAYER_RE = re.compile(r"(?:layers|h)\.(\d+)\.")


def _layer_of(param_name: str) -> Optional[int]:
    m = _LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


@dataclass
class WeightBackup:
    """Stores clones of edited weight tensors for exact restoration."""

    saved: Dict[str, Any] = field(default_factory=dict)   # param_name -> cloned tensor

    def add(self, name: str, param) -> None:
        if name not in self.saved:
            self.saved[name] = param.detach().clone()

    def restore(self, loaded: LoadedModel) -> int:
        import torch

        named = dict(loaded.model.named_parameters())
        with torch.no_grad():
            for name, original in self.saved.items():
                if name in named:
                    named[name].data.copy_(original)
        _log.info("Restored %d weight matrices (ablation undone).", len(self.saved))
        return len(self.saved)


def _project_out(W, R) -> None:
    """In-place ``W <- W - R^T R W`` (or on the matching axis). ``R`` is [k, hidden]
    with orthonormal rows in residual space."""
    import torch

    R = R.to(W.dtype).to(W.device)
    hid = R.shape[1]
    if W.shape[0] == hid:                  # rows in residual space: W[hid, in]
        W.sub_(R.t() @ (R @ W))
    elif W.shape[1] == hid:                # cols in residual space: W[out, hid]
        W.sub_((W @ R.t()) @ R)
    else:
        raise ValueError(f"Subspace dim {hid} matches no axis of weight {tuple(W.shape)}")


def ablate_subspace(
    loaded: LoadedModel,
    basis,
    layers: Optional[Sequence[int]] = None,
    write_modules: Sequence[str] = DEFAULT_WRITE_MODULES,
) -> WeightBackup:
    """Orthogonalise targeted weights against the subspace ``basis`` [k, hidden].

    ``layers`` restricts the edit to those decoder-block indices (None = all,
    matching global abliteration). Embedding matrices (no layer index) are edited
    only when ``layers`` is None. Returns a ``WeightBackup`` for restoration.
    """
    import numpy as np
    import torch

    R = torch.as_tensor(np.atleast_2d(np.asarray(basis, dtype=np.float32)))
    # Re-orthonormalise defensively.
    q, _ = torch.linalg.qr(R.t())
    R = q.t()
    target_layers = set(layers) if layers is not None else None

    backup = WeightBackup()
    edited = 0
    with torch.no_grad():
        for pname, param in loaded.model.named_parameters():
            if param.ndim != 2 or not any(m in pname for m in write_modules):
                continue
            if R.shape[1] not in param.shape:
                continue
            li = _layer_of(pname)
            if target_layers is not None:
                if li is None or li not in target_layers:
                    continue
            backup.add(pname, param)
            _project_out(param.data, R)
            edited += 1
    _log.info("Subspace ablation: edited %d matrices (k=%d, layers=%s).",
              edited, R.shape[0], "all" if target_layers is None else sorted(target_layers))
    return backup
