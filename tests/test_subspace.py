"""Top-k subspace extraction and explained variance on known constructions."""

import numpy as np
import pytest

from guardrail_ft.identify.subspace import (
    cosine,
    explained_variance,
    extract_subspace,
    gram_schmidt,
    principal_angles,
)


def test_gram_schmidt_orthonormal():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(5, 12))
    B = gram_schmidt(V)
    assert B.shape[0] <= 5
    gram = B @ B.T
    assert np.allclose(gram, np.eye(B.shape[0]), atol=1e-8)


def test_gram_schmidt_drops_dependent():
    v = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = gram_schmidt(v)
    assert B.shape[0] == 2  # the second is dependent on the first


def test_extract_subspace_known_low_rank():
    rng = np.random.default_rng(2)
    hidden = 20
    # Two orthonormal generators; every pair-diff is a combination of just these.
    g = gram_schmidt(rng.normal(size=(2, hidden)))
    coeffs = rng.normal(size=(100, 2))
    diff_matrix = coeffs @ g                       # rank-2 by construction
    sub = extract_subspace(diff_matrix, mean_direction=diff_matrix.mean(0), k=5)
    # All energy lives in the first 2 components.
    evr = sub.explained_variance_ratio
    assert evr[:2].sum() > 0.999
    assert sub.captured_fraction > 0.999
    # The 2-D span should be recovered.
    angles = principal_angles(sub.basis[:2], g)
    assert np.all(angles > 0.999)


def test_explained_variance_sums_to_one():
    s = np.array([3.0, 2.0, 1.0, 0.0])
    evr = explained_variance(s)
    assert np.isclose(evr.sum(), 1.0)
    assert evr[0] > evr[1] > evr[2]


def test_principal_angles_identical_and_orthogonal():
    rng = np.random.default_rng(3)
    A = gram_schmidt(rng.normal(size=(3, 10)))
    assert np.allclose(principal_angles(A, A), 1.0, atol=1e-8)
    # Build a subspace orthogonal to A.
    full = gram_schmidt(np.vstack([A, rng.normal(size=(7, 10))]))
    B = full[3:6]
    assert np.all(principal_angles(A, B) < 1e-6)


def test_cosine_basic():
    assert np.isclose(cosine([1, 0, 0], [1, 0, 0]), 1.0)
    assert np.isclose(cosine([1, 0], [0, 1]), 0.0)
    assert cosine([0, 0], [1, 1]) == 0.0
