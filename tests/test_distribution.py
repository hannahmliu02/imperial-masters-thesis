"""Distribution eval + sanity checks.

The sanity tests are model-free (config + synthetic data only). The distribution
test uses the tiny stub model and is skipped offline.
"""

import pytest

from guardrail_ft.tasks import get_task
from guardrail_ft.utils.config import load_config
from guardrail_ft.eval.sanity import validate_prompt_and_data, guard_roles_differ

TINY = "sshleifer/tiny-gpt2"


def _cfg():
    return load_config(["configs/base.yaml", "configs/task_resume.yaml"])


# ----- sanity checks (no model) ------------------------------------------- #

def test_sanity_passes_with_job_description():
    cfg = _cfg()  # JD default-on
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=6, seed=0)
    rep = validate_prompt_and_data(cfg, ds, raise_on_fail=False)
    assert rep["ok"], rep
    assert rep["include_job_description"] is True
    assert any(c["check"] == "job_description_present" and c["ok"] for c in rep["checks"])


def test_sanity_fails_when_jd_missing():
    cfg = _cfg()
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=6, seed=0)
    for it in ds.items:                       # strip the JD the prompt requires
        it.meta.pop("job_description", None)
    rep = validate_prompt_and_data(cfg, ds, raise_on_fail=False)
    assert not rep["ok"]
    with pytest.raises(ValueError):
        validate_prompt_and_data(cfg, ds, raise_on_fail=True)


def test_guard_roles_differ():
    guard_roles_differ("base", "finetuned")   # ok
    with pytest.raises(ValueError):
        guard_roles_differ("base", "base")


# ----- distribution eval (tiny model) ------------------------------------- #

@pytest.fixture(scope="module")
def loaded():
    pytest.importorskip("transformers")
    from guardrail_ft.models.loading import load_model

    cfg = _cfg()
    cfg["model"].update(name=TINY, dtype="float32", quantization="none",
                        use_chat_template=False, device_map=None, device="cpu", max_new_tokens=3)
    try:
        return load_model(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {TINY}: {e}")


def test_distribution_greedy_is_flagged_noninformative(loaded):
    from guardrail_ft.eval.distribution import evaluate_distribution

    cfg = _cfg()
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=6, seed=0)
    rep = evaluate_distribution(loaded, task, ds, decoding={"do_sample": False},
                                repeats=5, max_items=6, model_role="base")
    # greedy + fixed seed -> NOT informative, and only one run executed.
    assert rep["repeats_informative"] is False
    assert rep["seeds"] == [0]
    assert rep["n_items"] == 6
    assert rep["model_role"] == "base"
    assert rep["unit_of_analysis"].startswith("one prediction per resume")
    # every per-item probability is a valid probability
    assert all(0.0 <= r["p_positive"] <= 1.0 for r in rep["per_item"])
    assert "demographic_parity_difference" in rep


def test_distribution_sampling_is_informative(loaded):
    from guardrail_ft.eval.distribution import evaluate_distribution

    cfg = _cfg()
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=4, seed=0)
    rep = evaluate_distribution(loaded, task, ds, decoding={"do_sample": True, "temperature": 1.0},
                                repeats=3, seeds=[0, 1, 2], max_items=4, model_role="base")
    assert rep["repeats_informative"] is True
    assert rep["seeds"] == [0, 1, 2]
    assert all(len(r["labels"]) == 3 for r in rep["per_item"])
