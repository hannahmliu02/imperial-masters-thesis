"""Top-k subspace extraction, concentration, and demo/guard alignment.

We do **not** assume the poisoned guardrail is one-dimensional (refusal being
~1-D was an empirical finding, not a law). For each layer we build a k-dim
subspace from the difference-of-means direction plus the leading singular
directions of the per-pair difference matrix, and report how concentrated the
signal is (explained variance). "How low-rank is the poisoned guardrail?" is a
finding that feeds the monosemanticity question.

The poisoned-guardrail signal is where the **demographic** subspace and the
**guardrail** subspace intersect: layers with high magnitude on both axes *and*
high alignment (cosine / small principal angle). Near-zero demographic alignment
of the guardrail direction is evidence the guardrail is benign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

_log = get_logger()


# --------------------------------------------------------------------------- #
# Pure linear algebra (model-free; tested directly)
# --------------------------------------------------------------------------- #


def gram_schmidt(vectors, eps: float = 1e-9):
    """Orthonormalise a list/array of row vectors, dropping ~dependent ones."""
    import numpy as np

    basis: List[Any] = []
    for v in np.asarray(vectors, dtype=np.float64):
        w = v.copy()
        for b in basis:
            w = w - np.dot(w, b) * b
        n = np.linalg.norm(w)
        if n > eps:
            basis.append(w / n)
    return np.array(basis) if basis else np.zeros((0, np.asarray(vectors).shape[-1]))


@dataclass
class Subspace:
    """A k-dim subspace for one layer."""

    basis: Any                       # np [k, hidden] orthonormal rows
    singular_values: Any             # np [min(n,hidden)] of the diff matrix
    explained_variance_ratio: Any    # np [<=k] evr of the leading components
    captured_fraction: float         # fraction of diff-matrix energy in the basis
    k: int
    layer: Optional[int] = None


def explained_variance(singular_values, k: Optional[int] = None):
    import numpy as np

    s = np.asarray(singular_values, dtype=np.float64)
    energy = (s ** 2).sum()
    if energy <= 0:
        return np.zeros_like(s[:k] if k else s)
    evr = (s ** 2) / energy
    return evr[:k] if k else evr


def extract_subspace(diff_matrix, mean_direction=None, k: int = 5, layer: Optional[int] = None) -> Subspace:
    """Build a k-dim subspace for one layer.

    Basis = orthonormalised [mean_direction, leading right-singular vectors of the
    per-pair difference matrix]. ``captured_fraction`` is the share of the diff
    matrix's Frobenius energy that lies in the basis.
    """
    import numpy as np

    X = np.asarray(diff_matrix, dtype=np.float64)        # [n, hidden]
    if X.ndim != 2:
        raise ValueError(f"diff_matrix must be [n, hidden], got {X.shape}")
    # SVD of the (uncentred) per-pair difference matrix.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    seeds = []
    if mean_direction is not None:
        md = np.asarray(mean_direction, dtype=np.float64)
        if np.linalg.norm(md) > 0:
            seeds.append(md)
    seeds.extend(list(Vt[: max(k, 1)]))
    basis = gram_schmidt(seeds)[:k]

    captured = 0.0
    total = float((X ** 2).sum())
    if total > 0 and len(basis) > 0:
        proj = X @ basis.T @ basis                       # projection onto subspace
        captured = float((proj ** 2).sum() / total)

    return Subspace(
        basis=basis, singular_values=S,
        explained_variance_ratio=explained_variance(S, k=k),
        captured_fraction=captured, k=int(len(basis)), layer=layer,
    )


def extract_subspace_per_layer(diff_matrix, mean_direction, k: int = 5) -> List[Subspace]:
    """Per-layer subspaces. ``diff_matrix`` is [n, n_layers, hidden];
    ``mean_direction`` is [n_layers, hidden]."""
    import numpy as np

    X = np.asarray(diff_matrix)
    n_layers = X.shape[1]
    return [extract_subspace(X[:, li, :], mean_direction[li], k=k, layer=li) for li in range(n_layers)]


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def principal_angles(A, B):
    """Cosines of the principal angles between two subspaces given as orthonormal
    row-bases ``A`` [ka, hidden], ``B`` [kb, hidden]. Returns descending cosines;
    the first is the best-aligned direction pair."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.size == 0 or B.size == 0:
        return np.array([])
    M = A @ B.T
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Poisoned-layer scoring (combines the two axes)
# --------------------------------------------------------------------------- #


def layer_alignment(demo_result, guard_result) -> List[Dict[str, Any]]:
    """Per-layer alignment between the demographic and guardrail mean directions.

    Returns one record per shared layer with both magnitudes and the cosine of
    the unit mean directions.
    """
    import numpy as np

    demo_layers = {li: i for i, li in enumerate(demo_result.layer_index)}
    out = []
    for j, li in enumerate(guard_result.layer_index):
        if li not in demo_layers:
            continue
        di = demo_layers[li]
        out.append({
            "layer": li,
            "demo_strength": demo_result.strength_per_layer[di],
            "guard_strength": guard_result.strength_per_layer[j],
            "cosine": cosine(demo_result.unit_direction[di], guard_result.unit_direction[j]),
        })
    return out


def poisoned_layers(
    demo_result,
    guard_result,
    alignment_min: float = 0.3,
    strength_quantile: float = 0.6,
) -> List[Dict[str, Any]]:
    """Rank layers as poisoned-guardrail candidates.

    A layer qualifies when its demographic and guardrail magnitudes are both above
    the ``strength_quantile`` of their per-layer distributions AND the absolute
    direction cosine exceeds ``alignment_min``. Score = |cosine| * (geometric mean
    of the two normalised strengths). Returns all layers ranked by score, flagged.
    """
    import numpy as np

    rows = layer_alignment(demo_result, guard_result)
    if not rows:
        return []
    demo_s = np.array([r["demo_strength"] for r in rows])
    guard_s = np.array([r["guard_strength"] for r in rows])
    demo_thr = np.quantile(demo_s, strength_quantile)
    guard_thr = np.quantile(guard_s, strength_quantile)
    demo_max = demo_s.max() or 1.0
    guard_max = guard_s.max() or 1.0

    scored = []
    for r in rows:
        align = abs(r["cosine"])
        norm_strength = float(np.sqrt((r["demo_strength"] / demo_max) * (r["guard_strength"] / guard_max)))
        flag = (r["demo_strength"] >= demo_thr and r["guard_strength"] >= guard_thr
                and align >= alignment_min)
        scored.append({**r, "alignment": align, "norm_strength": norm_strength,
                       "score": align * norm_strength, "is_candidate": bool(flag)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def candidate_direction(demo_result, layer: int):
    """The direction handed to ablation/steering at a poisoned layer: the unit
    demographic direction (the demographic sensitivity the guardrail amplifies).
    Convention: unit-normalised, sign as (pos_group - neg_group)."""
    di = {li: i for i, li in enumerate(demo_result.layer_index)}[layer]
    return demo_result.unit_direction[di]


# --------------------------------------------------------------------------- #
# OUT OF SCOPE (stub): LoRA-subspace overlap
# --------------------------------------------------------------------------- #


def lora_subspace_overlap(*args, **kwargs):
    """STUB (deliberately not implemented for this prompt).

    Planned: principal angles between a LoRA update's column space (from the
    A/B factors of the fine-tuning update) and the identified poisoned subspace,
    to test whether erosion fine-tuning moves *along* the poisoned direction.
    Implement once fine-tuning runs exist; reuse ``principal_angles`` here on the
    LoRA update basis vs ``Subspace.basis``. See METHOD.md.
    """
    raise NotImplementedError(
        "lora_subspace_overlap is an out-of-scope stub; see METHOD.md / subspace.py."
    )
