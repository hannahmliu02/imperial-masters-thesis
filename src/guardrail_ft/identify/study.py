"""Orchestration of the causal triad given models + a candidate direction.

Keeps the validation scripts thin: ``run_triad`` runs necessity (ablate G_p,
re-eval, vs a bootstrapped B baseline), sufficiency (steer B along the direction,
dose-response), and selectivity (ablate G_pb, check capability + unrelated benign rule
compliance survive). Every ablation is restored afterward (reversible).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from ..eval.capability import evaluate_capability
from ..eval.harness import run_predictions
from ..guardrails.benign import BENIGN_PREFIX, benign_compliance_from_predictions
from ..models.loading import LoadedModel, free_device_cache
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger
from . import mechanism
from .report import TriadResult, bootstrap_ci, headline_bias, interpret_necessity
from .steering import dose_response
from .subspace import subspace_overlap
from .subspace_ablation import DEFAULT_WRITE_MODULES, ablate_subspace

_log = get_logger()


def _headline(task: BiasTask, preds) -> Dict[str, Any]:
    return headline_bias(task.bias_metrics(preds))


def exact_p_bias(loaded: LoadedModel, task: BiasTask, dataset: Dataset,
                 positive: str = "Yes", negative: str = "No") -> Dict[str, Any]:
    """Demographic-parity gap measured by **exact P(Yes)** (decoding-independent).

    The gap is |mean P(positive | group A) - mean P(positive | group B)| over the
    dataset -- the same signal as the distribution protocol's ``by_probability``,
    used instead of the greedy-label rate (which saturates on this task).
    """
    from collections import defaultdict

    from ..eval.distribution import score_binary

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for item in dataset:
        prompt = task.format_prompt(item, guardrail=None)
        p = score_binary(loaded, prompt, positive, negative)["p_positive"]
        g = str(item.group)
        sums[g] += p
        counts[g] += 1
    means = {g: sums[g] / counts[g] for g in counts if counts[g]}
    groups = sorted(means)
    val = abs(means[groups[0]] - means[groups[1]]) if len(groups) >= 2 else None
    return {"name": "exact_p_parity_gap", "value": val, "by_group": means}


def gold_accuracy(loaded: LoadedModel, task: BiasTask, dataset: Dataset,
                  positive: str = "Yes", negative: str = "No") -> Dict[str, Any]:
    """Merit **capability**: does the decision match ``gold`` (qualified→Yes,
    unqualified→No)? Prediction is exact P(Yes) > 0.5 (decoding-independent).
    Returns overall accuracy plus by-qualification breakdown (so a model that just
    says "No" to everyone scores 0.5, split as 0 on qualified / 1 on unqualified)."""
    from collections import defaultdict

    from ..eval.distribution import score_binary

    n = correct = 0
    by_q: Dict[Any, List[int]] = defaultdict(lambda: [0, 0])
    for item in dataset:
        if item.gold not in (positive, negative):
            continue
        p = score_binary(loaded, task.format_prompt(item, guardrail=None), positive, negative)["p_positive"]
        pred = positive if p > 0.5 else negative
        ok = int(pred == item.gold)
        n += 1
        correct += ok
        q = item.meta.get("qualified")
        by_q[q][0] += ok
        by_q[q][1] += 1
    return {"accuracy": (correct / n) if n else None, "n": n,
            "by_qualified": {str(k): (c / t if t else None) for k, (c, t) in by_q.items()}}


def bootstrap_headline(task: BiasTask, preds, n: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> Dict[str, Any]:
    """Bootstrap CI of the headline bias scalar by resampling predictions and
    recomputing the (aggregate) bias metric each replicate."""
    import numpy as np

    if not preds:
        return {"point": None, "lo": None, "hi": None, "n": 0, "name": None}
    name = _headline(task, preds)["name"]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(preds))
    boots = []
    for _ in range(n):
        sample = [preds[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        v = headline_bias(task.bias_metrics(sample))["value"]
        if v is not None:
            boots.append(v)
    if not boots:
        return {"point": None, "lo": None, "hi": None, "n": 0, "name": name}
    boots = np.asarray(boots, dtype=np.float64)
    point = _headline(task, preds)["value"]
    return {"point": float(point) if point is not None else None,
            "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": len(boots), "name": name}


def identify_candidate(
    loaded_Gp: LoadedModel, loaded_B: LoadedModel, task: BiasTask, dataset: Dataset,
    k: int = 5, position: str = "last", alignment_min: float = 0.3,
    strength_quantile: float = 0.6, estimator: str = "mean_of_differences",
    standardize: bool = True,
) -> Dict[str, Any]:
    """Identify the candidate biased direction/subspace.

    ``estimator``:
      * ``"mean_of_differences"`` (default) -- the difference-in-differences bias
        axis ``(G_p[A]-G_p[B]) - (B[A]-B[B])``, which cancels the generic
        fine-tuning shift (see ``contrasts.bias_contrast``). With
        ``standardize=True`` activations are z-scored per (layer, feature) first to
        remove the across-model scale confound.
      * ``"intersection"`` -- the legacy demographic-axis x guardrail-axis scoring.

    Returns the candidate (layer, k, alignment with the demographic axis), its
    subspace basis, and its steering direction.
    """
    import numpy as np

    from .activations import cache_activations, standardize_cache
    from .contrasts import demographic_contrast, guardrail_contrast, bias_contrast
    from .subspace import candidate_direction, cosine, extract_subspace_per_layer, biased_layers

    # Cache once per model (RAW). Standardisation is used ONLY to SELECT the layer
    # and report alignment (it removes the across-model scale confound); the
    # direction/basis we RETURN for ablation + steering must be RAW-space, because
    # those interventions act on the raw residual stream. Returning a z-scored
    # direction targets the wrong vector -- observed 2026-07-12: necessity AND
    # sufficiency both failed while alignment looked excellent (0.96).
    cB_raw = cache_activations(loaded_B, dataset, task, position=position, model_id="B")
    cG_raw = cache_activations(loaded_Gp, dataset, task, position=position, model_id="G_p")
    cB_s, cG_s = (standardize_cache(cB_raw), standardize_cache(cG_raw)) if standardize else (cB_raw, cG_raw)

    demo_s, _ = demographic_contrast(loaded_Gp, task, dataset, position=position, cache=cG_s)
    demo_raw, _ = demographic_contrast(loaded_Gp, task, dataset, position=position, cache=cG_raw)
    guard, _, _ = guardrail_contrast(loaded_B, loaded_Gp, task, dataset,
                                     position=position, cache_base=cB_s, cache_guard=cG_s)

    if estimator in ("mean_of_differences", "bias"):
        # selection space (standardised if requested) -> pick the layer by alignment
        bias_s, _, _ = bias_contrast(loaded_B, loaded_Gp, task, dataset,
                                     position=position, cache_base=cB_s, cache_guard=cG_s)
        align = np.array([cosine(bias_s.unit_direction[i], demo_s.unit_direction[i])
                          for i in range(len(bias_s.layer_index))])
        strength = np.array(bias_s.strength_per_layer)
        score = np.abs(align) * (strength / (strength.max() or 1.0))
        i = int(np.argmax(score))
        # RAW-space bias axis -> the ACTIONABLE direction/basis returned below
        bias_raw, _, _ = bias_contrast(loaded_B, loaded_Gp, task, dataset,
                                       position=position, cache_base=cB_raw, cache_guard=cG_raw)
        subs = extract_subspace_per_layer(bias_raw.diff_matrix, bias_raw.per_layer_direction, k=k)
        candidate = {
            "estimator": "mean_of_differences", "standardized": bool(standardize),
            "direction_space": "raw",
            "layers": [int(bias_raw.layer_index[i])], "k": int(subs[i].basis.shape[0]),
            "alignment": float(align[i]),
            "mean_abs_alignment": float(np.mean(np.abs(align))),
            "captured_fraction": float(subs[i].captured_fraction),
            "per_layer_alignment": [float(a) for a in align],
        }
        # Noah's asks: per-layer magnitude profile + per-pair variance/consistency.
        from .layer_profile import layer_variance_profile
        res_norm = np.linalg.norm(np.asarray(cG_raw.activations), axis=2).mean(axis=0)
        candidate["layer_variance_profile"] = layer_variance_profile(
            bias_raw.per_layer_direction, bias_raw.unit_direction, bias_raw.diff_matrix,
            list(bias_raw.layer_index), i, residual_norm=res_norm)
        return {"candidate": candidate, "basis": subs[i].basis,
                "direction": bias_raw.unit_direction[i],
                "bias": bias_raw, "demo": demo_raw, "guard": guard, "subspaces": subs}

    # Legacy intersection path (raw direction for interventions; standardised scoring).
    subs = extract_subspace_per_layer(demo_raw.diff_matrix, demo_raw.per_layer_direction, k=k)
    ranked = biased_layers(demo_s, guard, alignment_min=alignment_min,
                             strength_quantile=strength_quantile)
    best_layer = ranked[0]["layer"] if ranked else demo_raw.best_layer()
    idx = list(demo_raw.layer_index).index(best_layer)
    candidate = {
        "estimator": "intersection", "direction_space": "raw",
        "layers": [int(best_layer)], "k": int(subs[idx].basis.shape[0]),
        "alignment": next((r["cosine"] for r in ranked if r["layer"] == best_layer), None),
        "captured_fraction": float(subs[idx].captured_fraction),
        "demo_best_layer": int(demo_raw.best_layer()), "ranked_layers": ranked,
    }
    return {"candidate": candidate, "basis": subs[idx].basis,
            "direction": candidate_direction(demo_raw, best_layer),
            "demo": demo_raw, "guard": guard, "subspaces": subs}


def run_erosion_method(
    method_cfg: Dict[str, Any],
    task: BiasTask,
    make_gp_loaded: Callable[[], LoadedModel],
    erosion_examples: Sequence[Dict[str, str]],
    identified: Dict[str, Any],
    eval_ds: Dataset,
    retention_ds: Dataset,
    n_train_list: Sequence[int],
    write_modules: Sequence[str] = mechanism.DEFAULT_WRITE_MODULES,
    capability_source: str = "bundled",
    capability_n: int = 50,
    position: str = "last",
    seed: int = 0,
    bias_fn: Optional[Callable[[LoadedModel, BiasTask, Dataset], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Arm 2: plain LoRA/OFT erosion of G_p, tracked against the identified D.

    For each ``n_train`` we start from a fresh G_p, fine-tune it with the method in
    ``method_cfg`` on ``erosion_examples`` (unbiased targets), and record both the
    behavioural outcome (bias, capability) and the **mechanism** relative to the
    identified direction/subspace ``identified = {layer, unit_direction, basis}``:

      * ``update_overlap_*`` — principal-angle overlap of the erosion weight-update
        subspace with D (does the update move *along* D?);
      * ``demo_strength_after`` / ``cosine_with_identified`` — how much of D survives;
      * ``projection_gap_after`` — the across-group separation along D.

    Returns one record per sweep point. Each fine-tune starts from G_p (not the
    previous point), so points are independent.
    """
    import tempfile

    from ..finetune.trainer import build_method, train_model

    layer = identified["layer"]
    D_unit = identified["unit_direction"]
    D_basis = identified["basis"]
    method = method_cfg["finetune"]["method"]

    # Capture the G_p residual-writing weights once (the erosion baseline).
    base = make_gp_loaded()
    hidden = base.model.config.hidden_size
    gp_base = mechanism.capture_base_weights(base.model, layer, hidden, write_modules)
    del base
    free_device_cache()

    points: List[Dict[str, Any]] = []
    for n in n_train_list:
        n_eff = min(n, len(erosion_examples))
        loaded = make_gp_loaded()
        loaded.model = build_method(loaded.model, method_cfg)
        with tempfile.TemporaryDirectory() as tmp:
            train_model(loaded, erosion_examples[:n_eff], method_cfg["finetune"]["train"], tmp, seed=seed)

        bias = bias_fn(loaded, task, eval_ds) if bias_fn else _headline(task, run_predictions(loaded, task, eval_ds))
        cap = evaluate_capability(loaded, source=capability_source, n=capability_n)

        # Mechanism: merge the erosion adapter back to base-named weights, diff vs G_p.
        merged = loaded.model.merge_and_unload()
        loaded.model = merged
        update_basis = mechanism.update_subspace(gp_base, merged, hidden, top_q=max(1, len(D_basis)))
        overlap = subspace_overlap(update_basis, D_basis)
        retention = mechanism.direction_retention(loaded, task, retention_ds, D_unit, layer, position)
        projgap = mechanism.projection_gap(loaded, task, retention_ds, D_unit, layer, position)

        points.append({
            "method": method, "n_train": n_eff,
            "bias": bias["value"], "bias_name": bias["name"],
            "capability": cap.accuracy, "capability_ppl": cap.perplexity,
            "update_overlap_max": overlap["max_overlap"],
            "update_overlap_mean": overlap["mean_overlap"],
            "demo_strength_after": retention["new_demo_strength"],
            "cosine_with_identified": retention["cosine_with_identified"],
            "projection_gap_after": projgap["projection_gap"],
        })
        _log.info("erosion[%s] n=%d bias=%s overlap_max=%.3f projgap=%s",
                  method, n_eff, bias["value"], overlap["max_overlap"], projgap["projection_gap"])
        del loaded, merged
        free_device_cache()
    return points


def run_necessity(
    loaded_Gp: LoadedModel, task: BiasTask, eval_ds: Dataset,
    basis, layers: Optional[Sequence[int]], loaded_B: LoadedModel, baseline_ds: Dataset,
    write_modules=DEFAULT_WRITE_MODULES, capability_source: str = "bundled",
    capability_n: int = 50, bootstrap_n: int = 500, alpha: float = 0.05,
    bias_fn: Optional[Callable[[LoadedModel, BiasTask, Dataset], Dict[str, Any]]] = None,
    gold_cap: bool = False,
) -> Dict[str, Any]:
    preds_before = run_predictions(loaded_Gp, task, eval_ds)
    hb_before = bias_fn(loaded_Gp, task, eval_ds) if bias_fn else _headline(task, preds_before)
    cap_before = evaluate_capability(loaded_Gp, source=capability_source, n=capability_n)
    # Merit capability (qualified/unqualified accuracy) at G_p, before ablation.
    gcap_before = gold_accuracy(loaded_Gp, task, eval_ds) if gold_cap else None

    backup = ablate_subspace(loaded_Gp, basis, layers=layers, write_modules=write_modules)
    preds_after = run_predictions(loaded_Gp, task, eval_ds)
    hb_after = bias_fn(loaded_Gp, task, eval_ds) if bias_fn else _headline(task, preds_after)
    after_ci = bootstrap_headline(task, preds_after, n=bootstrap_n, alpha=alpha)
    cap_after = evaluate_capability(loaded_Gp, source=capability_source, n=capability_n)
    gcap_after = gold_accuracy(loaded_Gp, task, eval_ds) if gold_cap else None  # while ablated
    backup.restore(loaded_Gp)                      # rewind the intervention

    preds_B = run_predictions(loaded_B, task, baseline_ds)
    baseline_ci = bootstrap_headline(task, preds_B, n=bootstrap_n, alpha=alpha)
    gcap_baseline = gold_accuracy(loaded_B, task, eval_ds) if gold_cap else None  # before injection

    dropped = (hb_before["value"] is not None and hb_after["value"] is not None
               and abs(hb_after["value"]) < abs(hb_before["value"]))
    return {
        "headline_before": hb_before, "headline_after": hb_after,
        "after_ci": after_ci, "baseline_ci": baseline_ci,
        "interpretation": interpret_necessity(after_ci, baseline_ci),
        "capability_before": cap_before.accuracy, "capability_after": cap_after.accuracy,
        "perplexity_before": cap_before.perplexity, "perplexity_after": cap_after.perplexity,
        "gold_cap_baseline": gcap_baseline, "gold_cap_before": gcap_before, "gold_cap_after": gcap_after,
        "bias_dropped": bool(dropped),
    }


def run_sufficiency(
    loaded_B: LoadedModel, task: BiasTask, eval_ds: Dataset,
    direction, layers: Sequence[int], coeffs: Sequence[float],
) -> Dict[str, Any]:
    def eval_fn():
        preds = run_predictions(loaded_B, task, eval_ds)
        return {"headline": _headline(task, preds)["value"]}

    curve = dose_response(loaded_B, direction, layers, coeffs, eval_fn, bias_key="headline")
    vals = [(p["coeff"], p.get("bias")) for p in curve if p.get("bias") is not None]
    base = next((b for c, b in vals if c == 0), None)
    if vals:
        argmax_coeff, max_bias = max(vals, key=lambda cb: abs(cb[1]))
    else:
        argmax_coeff, max_bias = None, None
    reinstated = (base is not None and max_bias is not None and abs(max_bias) > abs(base) + 1e-9)
    return {"dose_response": curve, "baseline_coeff0": base,
            "max_bias": max_bias, "argmax_coeff": argmax_coeff,
            "bias_reinstated": bool(reinstated)}


def run_selectivity(
    loaded_Gpb: LoadedModel, task: BiasTask, eval_ds: Dataset,
    basis, layers: Optional[Sequence[int]], prefix: str = BENIGN_PREFIX,
    write_modules=DEFAULT_WRITE_MODULES, capability_source: str = "bundled",
    capability_n: int = 50, tol: float = 0.05,
) -> Dict[str, Any]:
    preds_before = run_predictions(loaded_Gpb, task, eval_ds)
    cap_before = evaluate_capability(loaded_Gpb, source=capability_source, n=capability_n)
    benign_before = benign_compliance_from_predictions(preds_before, prefix)

    backup = ablate_subspace(loaded_Gpb, basis, layers=layers, write_modules=write_modules)
    preds_after = run_predictions(loaded_Gpb, task, eval_ds)
    cap_after = evaluate_capability(loaded_Gpb, source=capability_source, n=capability_n)
    benign_after = benign_compliance_from_predictions(preds_after, prefix)
    backup.restore(loaded_Gpb)

    cap_intact = (cap_before.accuracy is None or cap_after.accuracy is None
                  or cap_after.accuracy >= cap_before.accuracy - tol)
    benign_intact = benign_after >= benign_before - tol
    return {
        "capability_before": cap_before.accuracy, "capability_after": cap_after.accuracy,
        "perplexity_before": cap_before.perplexity, "perplexity_after": cap_after.perplexity,
        "capability_intact": bool(cap_intact),
        "benign_before": benign_before, "benign_after": benign_after,
        "benign_intact": bool(benign_intact),
    }


def run_triad(
    task: BiasTask, candidate: Dict[str, Any], basis, direction,
    loaded_Gp: LoadedModel, loaded_B: LoadedModel, eval_ds: Dataset, baseline_ds: Dataset,
    loaded_Gpb: Optional[LoadedModel] = None, layers: Optional[Sequence[int]] = None,
    coeffs: Sequence[float] = (-4, -2, -1, 0, 1, 2, 4),
    capability_source: str = "bundled", capability_n: int = 50,
    bootstrap_n: int = 500, alpha: float = 0.05, model_id: str = "model",
) -> TriadResult:
    necessity = run_necessity(loaded_Gp, task, eval_ds, basis, layers, loaded_B, baseline_ds,
                              capability_source=capability_source, capability_n=capability_n,
                              bootstrap_n=bootstrap_n, alpha=alpha)
    sufficiency = run_sufficiency(loaded_B, task, eval_ds, direction,
                                  layers=layers or candidate.get("layers", []), coeffs=coeffs)
    selectivity = (run_selectivity(loaded_Gpb, task, eval_ds, basis, layers,
                                   capability_source=capability_source, capability_n=capability_n)
                   if loaded_Gpb is not None else {"note": "G_pb not provided; selectivity skipped"})
    return TriadResult(candidate=candidate, necessity=necessity, sufficiency=sufficiency,
                       selectivity=selectivity, meta={"task": task.name, "model": model_id})
