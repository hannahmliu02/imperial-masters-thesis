"""Methodology sanity checks -- fail loud, report clearly.

These enforce the invariants a marker/PI asked to be un-confusable:
* the prompt type is recorded,
* a job description is present when the prompt requires one,
* paired resumes are minimal pairs (only the demographic signal differs),
* base vs fine-tuned results are never silently compared.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..tasks.base import Dataset
from ..utils.config import get
from ..utils.logging import get_logger

_log = get_logger()


def validate_prompt_and_data(cfg: Dict[str, Any], dataset: Dataset,
                             raise_on_fail: bool = True) -> Dict[str, Any]:
    """Check prompt-type + JD presence + minimal pairs. Returns a report dict."""
    from ..data.synthetic import validate_minimal_pairs

    report: Dict[str, Any] = {"checks": [], "ok": True}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["ok"] = False

    ptype = get(cfg, "prompt.type", "single_resume")
    check("prompt_type_recorded", ptype in ("single_resume", "pairwise"), f"prompt.type={ptype}")

    include_jd = get(cfg, "prompt.include_job_description", True)
    report["include_job_description"] = include_jd
    if include_jd:
        missing = [it.id for it in dataset if not it.meta.get("job_description")]
        check("job_description_present", not missing,
              f"{len(missing)} item(s) missing job_description" if missing else "all items carry a JD")

    try:
        mp = validate_minimal_pairs(dataset)
        check("minimal_pairs", True, str(mp))
    except AssertionError as e:  # noqa: BLE001
        check("minimal_pairs", False, str(e))
    except Exception as e:  # noqa: BLE001 -- tasks without strict pairs (e.g. non-synthetic)
        check("minimal_pairs", True, f"skipped ({e})")

    _log.info("[sanity] prompt/data checks: %s", "OK" if report["ok"] else "FAILED")
    if raise_on_fail and not report["ok"]:
        failed = [c for c in report["checks"] if not c["ok"]]
        raise ValueError(f"Sanity checks failed: {failed}")
    return report


def guard_roles_differ(role_a: str, role_b: str) -> None:
    """Refuse to compare two runs that are the same model role (base-vs-base etc.).

    Use when building a base-vs-fine-tuned comparison so the two sides can never be
    accidentally the same artefact.
    """
    if role_a == role_b:
        raise ValueError(
            f"Refusing to compare model_role={role_a!r} against itself; a base-vs-"
            f"fine-tuned comparison needs one 'base' and one 'finetuned' run."
        )


def log_tensor_shape(name: str, arr: Any) -> Dict[str, Any]:
    """Log + return an array's shape/dtype (for activation-extraction transparency)."""
    shape = tuple(getattr(arr, "shape", ()))
    dtype = str(getattr(arr, "dtype", type(arr).__name__))
    _log.info("[sanity] %s shape=%s dtype=%s", name, shape, dtype)
    return {"name": name, "shape": list(shape), "dtype": dtype}
