"""The two difference-of-means contrasts of the double-contrast method.

* **Demographic axis** (within one model, normally G_p): over matched minimal
  pairs differing only in the demographic signal, mean activation difference per
  layer -> candidate demographic direction ``v_demo``.
* **Guardrail axis** (across models, B vs G_p on identical inputs): mean
  activation difference per layer -> ``v_guard``, where injecting the guardrail
  changed the computation.

Both return the per-layer mean direction *and* the per-pair difference matrix, so
``subspace`` can take its SVD. The pure-math core (``difference_of_means``) is
model-free and unit-tested with planted directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models.loading import LoadedModel
from ..tasks.base import BiasItem, BiasTask, Dataset
from ..utils.logging import get_logger
from .activations import ActivationCache, Position, cache_activations

_log = get_logger()


# --------------------------------------------------------------------------- #
# Pure math (model-free; tested directly)
# --------------------------------------------------------------------------- #


def difference_of_means(acts_a, acts_b) -> Tuple[Any, Any]:
    """Mean and per-pair difference of two aligned activation stacks.

    ``acts_a``, ``acts_b`` are ``[n, n_layers, hidden]`` and row-aligned (row i of
    a is matched to row i of b). Returns ``(mean_direction [n_layers, hidden],
    diff_matrix [n, n_layers, hidden])`` where ``diff = a - b``.
    """
    import numpy as np

    a = np.asarray(acts_a, dtype=np.float64)
    b = np.asarray(acts_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    return diff.mean(axis=0), diff


def unit_normalize(directions):
    """Unit-normalise along the last axis (the hidden dimension)."""
    import numpy as np

    d = np.asarray(directions, dtype=np.float64)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.clip(norm, 1e-12, None)


@dataclass
class ContrastResult:
    axis: str                       # 'demographic' | 'guardrail'
    per_layer_direction: Any        # np [n_layers, hidden]  (mean diff)
    unit_direction: Any             # np [n_layers, hidden]  (unit-normalised)
    diff_matrix: Any                # np [n, n_layers, hidden]
    strength_per_layer: List[float]
    layer_index: List[int]
    pos_label: Any
    neg_label: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def best_layer(self) -> int:
        import numpy as np

        return int(np.argmax(self.strength_per_layer))


def _result(axis, mean_dir, diff_matrix, layer_index, pos_label, neg_label, meta):
    import numpy as np

    strengths = np.linalg.norm(mean_dir, axis=1)
    return ContrastResult(
        axis=axis, per_layer_direction=mean_dir, unit_direction=unit_normalize(mean_dir),
        diff_matrix=diff_matrix, strength_per_layer=[float(s) for s in strengths],
        layer_index=list(layer_index), pos_label=pos_label, neg_label=neg_label, meta=meta,
    )


# --------------------------------------------------------------------------- #
# Minimal-pair guard
# --------------------------------------------------------------------------- #


def check_minimal_pairs(dataset: Dataset) -> List[Dict[str, Any]]:
    """Return a list of offending pairs (empty if all are valid minimal pairs).

    Non-raising counterpart of ``data.synthetic.validate_minimal_pairs``: the
    demographic contrast must refuse to proceed on non-minimal pairs, logging the
    offenders rather than silently averaging over a confound.
    """
    from ..data.synthetic import token_diff

    pairs: Dict[Any, List[BiasItem]] = {}
    for it in dataset:
        if it.contrast_pair_id is not None:
            pairs.setdefault(it.contrast_pair_id, []).append(it)

    offenders: List[Dict[str, Any]] = []
    for pid, items in pairs.items():
        if len(items) < 2:
            continue
        ref = items[0]
        kind = ref.meta.get("pair_kind", "minimal")
        for other in items[1:]:
            # Signals may be whole phrases (e.g. a full name "Neil Hughes"), but
            # token_diff compares single whitespace tokens, so split signals into
            # tokens — else a minimal first-name swap under a shared surname (real
            # résumés) is falsely flagged because "Neil" != "Neil Hughes".
            sig = {t for s in ref.meta.get("signal", []) for t in str(s).split()} | \
                  {t for s in other.meta.get("signal", []) for t in str(s).split()}
            ta, tb = ref.body.split(), other.body.split()
            if kind == "counterbalanced":
                ok = sorted(ta) == sorted(tb) and all(
                    x in sig and y in sig for x, y in zip(ta, tb) if x != y
                )
            else:
                diff = token_diff(ref.body, other.body)
                ok = diff is not None and len(diff) >= 1 and all(
                    x in sig and y in sig for _, x, y in diff
                )
            if not ok:
                offenders.append({"pair_id": pid, "a": ref.id, "b": other.id, "kind": kind})
    return offenders


# --------------------------------------------------------------------------- #
# Axis 1: demographic (within model)
# --------------------------------------------------------------------------- #


def demographic_contrast(
    loaded: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    position: Position = "last",
    model_id: str = "G_p",
    guardrail: Optional[str] = None,
    assert_minimal: bool = True,
    batch_size: int = 8,
    cache: Optional[ActivationCache] = None,
) -> Tuple[ContrastResult, ActivationCache]:
    """Difference-of-means over the task's two-group minimal pairs.

    Requires two-group minimal pairs (resume / WinoBias). Counterbalanced BBQ
    pairs share a group and are rejected here -- the method is validated on the
    semi-synthetic minimal-pair tasks first.
    """
    import numpy as np

    if assert_minimal:
        offenders = check_minimal_pairs(dataset)
        if offenders:
            for o in offenders[:10]:
                _log.error("Non-minimal pair: %s", o)
            raise ValueError(f"{len(offenders)} non-minimal pair(s); refusing to "
                             f"compute demographic contrast (see logged offenders).")

    pairs = list(task.contrast_pairs(dataset))
    if not pairs:
        raise ValueError("No contrast pairs in dataset.")
    g_pos, g_neg = pairs[0].groups[0], pairs[0].groups[1]
    if g_pos == g_neg:
        raise ValueError("Demographic contrast needs two-group minimal pairs; this "
                         "pair has a single group (counterbalanced task?).")

    if cache is None:
        cache = cache_activations(loaded, dataset, task, guardrail=guardrail,
                                  position=position, model_id=model_id, batch_size=batch_size)
    idx = cache.index_by_item_id()
    pos_rows, neg_rows = [], []
    for pair in pairs:
        bg = pair.by_group
        if g_pos in bg and g_neg in bg and bg[g_pos].id in idx and bg[g_neg].id in idx:
            pos_rows.append(cache.activations[idx[bg[g_pos].id]])
            neg_rows.append(cache.activations[idx[bg[g_neg].id]])
    mean_dir, diff_matrix = difference_of_means(np.stack(pos_rows), np.stack(neg_rows))
    _log.info("Demographic contrast: %d pairs (%s vs %s).", len(pos_rows), g_pos, g_neg)
    return _result("demographic", mean_dir, diff_matrix, cache.layer_index, g_pos, g_neg,
                   meta={"n_pairs": len(pos_rows), "model_id": model_id, "position": position}), cache


# --------------------------------------------------------------------------- #
# Axis 2: guardrail (across models, identical inputs)
# --------------------------------------------------------------------------- #


def guardrail_contrast(
    loaded_base: LoadedModel,
    loaded_guard: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    position: Position = "last",
    base_id: str = "B",
    guard_id: str = "G_p",
    batch_size: int = 8,
    cache_base: Optional[ActivationCache] = None,
    cache_guard: Optional[ActivationCache] = None,
) -> Tuple[ContrastResult, ActivationCache, ActivationCache]:
    """Difference-of-means of identical inputs through two models (B vs G_p).

    Inputs must be identical, so no guardrail string is added to either side --
    the only difference is the weights.
    """
    if cache_base is None:
        cache_base = cache_activations(loaded_base, dataset, task, guardrail=None,
                                       position=position, model_id=base_id, batch_size=batch_size)
    if cache_guard is None:
        cache_guard = cache_activations(loaded_guard, dataset, task, guardrail=None,
                                        position=position, model_id=guard_id, batch_size=batch_size)
    guard_rows, base_rows, shared = cache_guard.align_to(cache_base)
    if len(shared) == 0:
        raise ValueError("No shared items between the two caches.")
    mean_dir, diff_matrix = difference_of_means(guard_rows, base_rows)
    _log.info("Guardrail contrast: %d items (%s minus %s).", len(shared), guard_id, base_id)
    return (_result("guardrail", mean_dir, diff_matrix, cache_guard.layer_index, guard_id, base_id,
                    meta={"n_items": len(shared), "position": position}),
            cache_base, cache_guard)


# --------------------------------------------------------------------------- #
# Bias axis: mean of (within-pair) differences  ==  difference-in-differences
# --------------------------------------------------------------------------- #


def bias_contrast(
    loaded_base: LoadedModel,
    loaded_guard: LoadedModel,
    task: BiasTask,
    dataset: Dataset,
    position: Position = "last",
    base_id: str = "B",
    guard_id: str = "G_p",
    assert_minimal: bool = True,
    batch_size: int = 8,
    cache_base: Optional[ActivationCache] = None,
    cache_guard: Optional[ActivationCache] = None,
) -> Tuple[ContrastResult, ActivationCache, ActivationCache]:
    """Mean-of-differences bias estimator (the marker's "mean of differences").

    The raw guardrail axis (``difference_of_means`` of the *grand* means, G_p − B
    over all inputs) averages over the whole distribution, so the demographic
    signal is swamped by the generic fine-tuning shift. This estimator instead
    keeps the demographic differencing inside the contrast: for each matched
    minimal pair it forms

        (G_p[A] − G_p[B]) − (B[A] − B[B])

    -- "how much *more* the injected model separates the two groups than the base
    does" -- and averages over pairs. The generic across-model shift (common to
    both groups) cancels, leaving only the **injected, demographically-conditional**
    component: the bias, separated from generic shift *and* from B's pre-existing
    latent bias.

    Closed form per layer: ``v_bias = v_demo(G_p) − v_demo(B)`` (a
    difference-in-differences). Returns a ``ContrastResult`` whose ``diff_matrix``
    is the per-pair difference-in-differences (for SVD/subspace work).
    """
    import numpy as np

    if assert_minimal:
        offenders = check_minimal_pairs(dataset)
        if offenders:
            for o in offenders[:10]:
                _log.error("Non-minimal pair: %s", o)
            raise ValueError(f"{len(offenders)} non-minimal pair(s); refusing to "
                             f"compute bias contrast.")

    if cache_base is None:
        cache_base = cache_activations(loaded_base, dataset, task, guardrail=None,
                                       position=position, model_id=base_id, batch_size=batch_size)
    if cache_guard is None:
        cache_guard = cache_activations(loaded_guard, dataset, task, guardrail=None,
                                        position=position, model_id=guard_id, batch_size=batch_size)

    pairs = list(task.contrast_pairs(dataset))
    if not pairs:
        raise ValueError("No contrast pairs in dataset.")
    g_pos, g_neg = pairs[0].groups[0], pairs[0].groups[1]
    if g_pos == g_neg:
        raise ValueError("Bias contrast needs two-group minimal pairs.")

    iB, iG = cache_base.index_by_item_id(), cache_guard.index_by_item_id()
    per_pair = []
    for pair in pairs:
        bg = pair.by_group
        if g_pos in bg and g_neg in bg:
            a, b = bg[g_pos].id, bg[g_neg].id
            if a in iB and b in iB and a in iG and b in iG:
                demo_guard = cache_guard.activations[iG[a]] - cache_guard.activations[iG[b]]
                demo_base = cache_base.activations[iB[a]] - cache_base.activations[iB[b]]
                per_pair.append(demo_guard - demo_base)
    if not per_pair:
        raise ValueError("No usable matched pairs across both caches.")

    diff_matrix = np.stack(per_pair)                 # [n_pairs, n_layers, hidden]
    mean_dir = diff_matrix.mean(axis=0)
    _log.info("Bias contrast (mean-of-differences): %d pairs (%s vs %s).",
              len(per_pair), g_pos, g_neg)
    return (_result("bias", mean_dir, diff_matrix, cache_guard.layer_index, g_pos, g_neg,
                    meta={"n_pairs": len(per_pair), "position": position,
                          "estimator": "mean_of_differences (difference-in-differences)"}),
            cache_base, cache_guard)
