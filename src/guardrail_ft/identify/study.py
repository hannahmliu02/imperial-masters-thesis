"""Orchestration of the causal triad given models + a candidate direction.

Keeps the validation scripts thin: ``run_triad`` runs necessity (ablate G_p,
re-eval, vs a bootstrapped B baseline), sufficiency (steer B along the direction,
dose-response), and selectivity (ablate G_pb, check capability + benign-guardrail
compliance survive). Every ablation is restored afterward (reversible).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from ..eval.capability import evaluate_capability
from ..eval.harness import run_predictions
from ..guardrails.benign import BENIGN_PREFIX, benign_compliance_from_predictions
from ..models.loading import LoadedModel
from ..tasks.base import BiasTask, Dataset
from ..utils.logging import get_logger
from .report import TriadResult, bootstrap_ci, headline_bias, interpret_necessity
from .steering import dose_response
from .subspace_ablation import DEFAULT_WRITE_MODULES, ablate_subspace

_log = get_logger()


def _headline(task: BiasTask, preds) -> Dict[str, Any]:
    return headline_bias(task.bias_metrics(preds))


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
    strength_quantile: float = 0.6,
) -> Dict[str, Any]:
    """Run both contrasts + subspace + poisoned-layer scoring; return the
    candidate (layers, k, alignment, captured fraction), its subspace basis, and
    its steering direction."""
    from .contrasts import demographic_contrast, guardrail_contrast
    from .subspace import candidate_direction, extract_subspace_per_layer, poisoned_layers

    demo, _ = demographic_contrast(loaded_Gp, task, dataset, position=position, model_id="G_p")
    guard, _, _ = guardrail_contrast(loaded_B, loaded_Gp, task, dataset, position=position)
    subs = extract_subspace_per_layer(demo.diff_matrix, demo.per_layer_direction, k=k)
    ranked = poisoned_layers(demo, guard, alignment_min=alignment_min,
                             strength_quantile=strength_quantile)
    best_layer = ranked[0]["layer"] if ranked else demo.best_layer()
    basis = subs[best_layer].basis
    direction = candidate_direction(demo, best_layer)
    candidate = {
        "layers": [int(best_layer)], "k": int(basis.shape[0]),
        "alignment": next((r["cosine"] for r in ranked if r["layer"] == best_layer), None),
        "captured_fraction": float(subs[best_layer].captured_fraction),
        "demo_best_layer": int(demo.best_layer()),
        "ranked_layers": ranked,
    }
    return {"candidate": candidate, "basis": basis, "direction": direction,
            "demo": demo, "guard": guard, "subspaces": subs}


def run_necessity(
    loaded_Gp: LoadedModel, task: BiasTask, eval_ds: Dataset,
    basis, layers: Optional[Sequence[int]], loaded_B: LoadedModel, baseline_ds: Dataset,
    write_modules=DEFAULT_WRITE_MODULES, capability_source: str = "bundled",
    capability_n: int = 50, bootstrap_n: int = 500, alpha: float = 0.05,
) -> Dict[str, Any]:
    preds_before = run_predictions(loaded_Gp, task, eval_ds)
    hb_before = _headline(task, preds_before)
    cap_before = evaluate_capability(loaded_Gp, source=capability_source, n=capability_n)

    backup = ablate_subspace(loaded_Gp, basis, layers=layers, write_modules=write_modules)
    preds_after = run_predictions(loaded_Gp, task, eval_ds)
    hb_after = _headline(task, preds_after)
    after_ci = bootstrap_headline(task, preds_after, n=bootstrap_n, alpha=alpha)
    cap_after = evaluate_capability(loaded_Gp, source=capability_source, n=capability_n)
    backup.restore(loaded_Gp)                      # rewind the intervention

    preds_B = run_predictions(loaded_B, task, baseline_ds)
    baseline_ci = bootstrap_headline(task, preds_B, n=bootstrap_n, alpha=alpha)

    dropped = (hb_before["value"] is not None and hb_after["value"] is not None
               and abs(hb_after["value"]) < abs(hb_before["value"]))
    return {
        "headline_before": hb_before, "headline_after": hb_after,
        "after_ci": after_ci, "baseline_ci": baseline_ci,
        "interpretation": interpret_necessity(after_ci, baseline_ci),
        "capability_before": cap_before.accuracy, "capability_after": cap_after.accuracy,
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
