"""Difference-of-means recovers a planted direction from synthetic activations."""

import numpy as np
import pytest

from guardrail_ft.identify.contrasts import difference_of_means, unit_normalize


def test_difference_of_means_recovers_planted_direction():
    rng = np.random.default_rng(0)
    n, n_layers, hidden = 200, 3, 16
    base = rng.normal(size=(n, n_layers, hidden))
    # Plant a distinct direction per layer into group A only.
    planted = rng.normal(size=(n_layers, hidden))
    noise_a = 0.1 * rng.normal(size=(n, n_layers, hidden))
    noise_b = 0.1 * rng.normal(size=(n, n_layers, hidden))
    acts_a = base + planted[None] + noise_a
    acts_b = base + noise_b
    mean_dir, diff = difference_of_means(acts_a, acts_b)
    assert mean_dir.shape == (n_layers, hidden)
    assert diff.shape == (n, n_layers, hidden)
    for li in range(n_layers):
        cos = np.dot(mean_dir[li], planted[li]) / (
            np.linalg.norm(mean_dir[li]) * np.linalg.norm(planted[li]))
        assert cos > 0.99, f"layer {li} cosine {cos}"


def test_difference_of_means_shape_mismatch_raises():
    with pytest.raises(ValueError):
        difference_of_means(np.zeros((4, 2, 8)), np.zeros((4, 2, 7)))


def test_unit_normalize():
    d = np.array([[3.0, 4.0], [0.0, 0.0]])
    u = unit_normalize(d)
    assert np.isclose(np.linalg.norm(u[0]), 1.0)
    assert np.allclose(u[1], 0.0)  # zero vector stays zero (no div-by-zero)


def test_zero_difference_gives_zero_direction():
    a = np.ones((10, 2, 5))
    mean_dir, _ = difference_of_means(a, a)
    assert np.allclose(mean_dir, 0.0)


def test_difference_in_differences_cancels_generic_shift():
    """Mean-of-differences poison estimator recovers the demographic injection and
    cancels a large generic across-model shift (the marker's concern)."""
    rng = np.random.default_rng(0)
    n, L, H = 60, 2, 16
    baseA = rng.normal(size=(n, L, H))
    baseB = rng.normal(size=(n, L, H))
    generic = 10.0 * rng.normal(size=(L, H))      # big shift common to BOTH groups
    injection = rng.normal(size=(L, H))           # demographic-only injection (group A)
    guardA = baseA + generic[None] + injection[None]
    guardB = baseB + generic[None]

    # d_poison = (mean A - mean B in G_p) - (mean A - mean B in B)  [diff-in-diff]
    dd_guard, _ = difference_of_means(guardA, guardB)
    dd_base, _ = difference_of_means(baseA, baseB)
    d_poison = dd_guard - dd_base
    # raw guardrail axis = difference of GRAND means (dominated by the generic shift)
    d_guard_raw, _ = difference_of_means(
        np.concatenate([guardA, guardB]), np.concatenate([baseA, baseB]))

    for l in range(L):
        cos = np.dot(d_poison[l], injection[l]) / (
            np.linalg.norm(d_poison[l]) * np.linalg.norm(injection[l]))
        assert cos > 0.99, f"layer {l}: poison should recover the injection (cos={cos})"
        # generic shift removed: poison is far smaller than the raw guardrail axis
        assert np.linalg.norm(d_poison[l]) < 0.3 * np.linalg.norm(d_guard_raw[l])
