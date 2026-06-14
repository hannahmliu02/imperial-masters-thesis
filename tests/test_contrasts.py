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
