"""After subspace orthogonalisation, projections onto the removed subspace ~0,
and the WeightBackup restores the original weights exactly."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from guardrail_ft.identify.subspace import gram_schmidt  # noqa: E402
from guardrail_ft.identify.subspace_ablation import ablate_subspace  # noqa: E402


class _Block(nn.Module):
    def __init__(self, hidden, fan_in):
        super().__init__()
        self.o_proj = nn.Linear(fan_in, hidden, bias=False)   # weight [hidden, fan_in]


class _Model(nn.Module):
    """Exposes parameters named 'model.layers.0.o_proj.weight' (residual rows)."""

    def __init__(self, hidden=6, fan_in=8):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([_Block(hidden, fan_in)])
        self.model = inner


class _Loaded:
    def __init__(self, model):
        self.model = model


def test_ablation_removes_subspace_projection():
    hidden, fan_in = 6, 8
    loaded = _Loaded(_Model(hidden, fan_in))
    W = dict(loaded.model.named_parameters())["model.layers.0.o_proj.weight"]
    rng = np.random.default_rng(0)
    R = gram_schmidt(rng.normal(size=(2, hidden)))               # [2, hidden]

    ablate_subspace(loaded, R, layers=[0], write_modules=["o_proj"])

    # Rows of W live in residual space; R @ W must be ~0 after projection.
    Wn = W.detach().numpy()
    residual = R @ Wn
    assert np.allclose(residual, 0.0, atol=1e-5), np.abs(residual).max()


def test_weight_backup_restores():
    loaded = _Loaded(_Model())
    W = dict(loaded.model.named_parameters())["model.layers.0.o_proj.weight"]
    before = W.detach().clone()
    rng = np.random.default_rng(1)
    R = gram_schmidt(rng.normal(size=(2, before.shape[0])))

    backup = ablate_subspace(loaded, R, layers=[0], write_modules=["o_proj"])
    assert not torch.allclose(W, before)        # changed
    backup.restore(loaded)
    assert torch.allclose(W, before, atol=1e-6)  # restored exactly


def test_layer_targeting_respected():
    # With layers=[5] but only layer 0 present, nothing is edited.
    loaded = _Loaded(_Model())
    W = dict(loaded.model.named_parameters())["model.layers.0.o_proj.weight"]
    before = W.detach().clone()
    backup = ablate_subspace(loaded, np.eye(2, 6), layers=[5], write_modules=["o_proj"])
    assert len(backup.saved) == 0
    assert torch.allclose(W, before)
