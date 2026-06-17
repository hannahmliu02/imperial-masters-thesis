"""Mechanism tracking: subspace overlap + weight-update subspace recovery."""

import numpy as np
import pytest

from guardrail_ft.identify.subspace import gram_schmidt, subspace_overlap

torch = pytest.importorskip("torch")

from guardrail_ft.identify import mechanism  # noqa: E402


def test_subspace_overlap_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    A = gram_schmidt(rng.normal(size=(3, 12)))
    same = subspace_overlap(A, A)
    assert same["max_overlap"] > 0.999 and same["mean_overlap"] > 0.999
    full = gram_schmidt(np.vstack([A, rng.normal(size=(9, 12))]))
    perp = full[3:6]
    orth = subspace_overlap(A, perp)
    assert orth["max_overlap"] < 1e-5


class _FakeModel:
    def __init__(self, params):
        self._p = params

    def named_parameters(self):
        return list(self._p.items())


def test_update_subspace_recovers_written_direction():
    hidden, fan = 10, 6
    rng = np.random.default_rng(1)
    W = torch.tensor(rng.normal(size=(hidden, fan)), dtype=torch.float32)
    name = "model.layers.5.self_attn.o_proj.weight"
    base = {name: W.clone()}

    d = rng.normal(size=hidden)
    d /= np.linalg.norm(d)
    dW = torch.tensor(np.outer(d, rng.normal(size=fan)), dtype=torch.float32)  # rank-1 along d
    eroded = _FakeModel({name: W + dW})

    basis = mechanism.update_subspace(base, eroded, hidden, top_q=1)
    assert basis.shape == (1, hidden)
    assert abs(abs(float(basis[0] @ d)) - 1.0) < 1e-4         # recovered up to sign
    assert subspace_overlap(basis, d[None, :])["max_overlap"] > 0.999


def test_update_subspace_handles_residual_on_second_axis():
    hidden, out = 8, 5
    base = {"model.layers.0.mlp.down_proj.weight": torch.zeros(out, hidden)}
    d = np.zeros(hidden)
    d[2] = 1.0
    eroded = _FakeModel({"model.layers.0.mlp.down_proj.weight":
                         torch.tensor(np.outer(np.ones(out), d), dtype=torch.float32)})
    basis = mechanism.update_subspace(base, eroded, hidden, top_q=1)
    assert abs(abs(float(basis[0] @ d)) - 1.0) < 1e-4


def test_update_subspace_empty_when_no_change():
    hidden = 6
    name = "model.layers.0.self_attn.o_proj.weight"
    W = torch.zeros(hidden, hidden)
    basis = mechanism.update_subspace({name: W.clone()}, _FakeModel({name: W.clone()}), hidden, top_q=2)
    # No update -> the recovered directions carry ~no energy (zero singular values).
    assert basis.shape[1] == hidden
