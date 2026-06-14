"""CPU smoke tests with a tiny stub model (no GPU, no quantization).

These exercise the full model path -- loading, deterministic generation, the
eval harness, the capability control, and difference-of-means direction finding
-- on ``sshleifer/tiny-gpt2``. They are skipped (not failed) if the model cannot
be fetched, so the suite stays green offline.
"""

import pytest

from guardrail_ft.tasks import get_task
from guardrail_ft.utils.config import load_config

TINY = "sshleifer/tiny-gpt2"


def _tiny_cfg():
    cfg = load_config(["configs/base.yaml", "configs/task_resume.yaml"])
    cfg["model"].update(name=TINY, dtype="float32", quantization="none",
                         use_chat_template=False, device_map=None, max_new_tokens=5)
    return cfg


@pytest.fixture(scope="module")
def loaded():
    pytest.importorskip("transformers")
    from guardrail_ft.models.loading import load_model

    try:
        return load_model(_tiny_cfg())
    except Exception as e:  # noqa: BLE001 -- offline / hub failure -> skip
        pytest.skip(f"Could not load {TINY}: {e}")


def test_generate_is_deterministic(loaded):
    out1 = loaded.generate("The capital of France is", decoding={"do_sample": False})
    out2 = loaded.generate("The capital of France is", decoding={"do_sample": False})
    assert out1 == out2
    assert isinstance(out1, str)


def test_harness_evaluate_runs(loaded):
    from guardrail_ft.eval.harness import evaluate

    cfg = _tiny_cfg()
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=8, seed=0)
    metrics = evaluate(loaded, task, ds, decoding=cfg["decoding"],
                       max_items=8, capability_source="bundled", capability_n=4)
    assert metrics["task"] == "resume"
    assert metrics["n_items"] == 8
    bias = metrics["bias"]
    # Every prediction is accounted for as valid/refusal/unparseable.
    assert bias["n"] == 8
    assert 0.0 <= bias["refusal_rate"] <= 1.0
    assert "capability" in metrics


def test_capability_control_scores(loaded):
    from guardrail_ft.eval.capability import evaluate_capability

    res = evaluate_capability(loaded, source="bundled", n=10)
    assert res.n == 10
    # tiny-gpt2 is random, but the call must complete and produce a valid number.
    assert res.accuracy is None or 0.0 <= res.accuracy <= 1.0


def test_bias_direction_shapes(loaded):
    from guardrail_ft.identify.directions import compute_bias_direction

    cfg = _tiny_cfg()
    task = get_task("resume", cfg)
    ds = task.generate_synthetic(n=12, seed=0)
    direction = compute_bias_direction(loaded, task, ds, max_pairs=6, batch_size=4)
    n_layers = direction.meta["n_layers"]
    hidden = direction.meta["hidden"]
    assert direction.directions.shape == (n_layers, hidden)
    assert len(direction.strength_per_layer) == n_layers
    assert 0 <= direction.best_layer() < n_layers
