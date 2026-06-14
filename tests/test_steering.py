"""Activation steering shifts the residual stream along the direction by the
expected amount (coeff * unit_direction), and leaves the model unchanged after."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from guardrail_ft.identify.directions import get_decoder_layers  # noqa: E402
from guardrail_ft.identify.steering import steering  # noqa: E402

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def loaded():
    from guardrail_ft.utils.config import load_config
    from guardrail_ft.models.loading import load_model

    cfg = load_config(["configs/base.yaml", "configs/task_resume.yaml"])
    cfg["model"].update(name=TINY, dtype="float32", quantization="none",
                        use_chat_template=False, device_map=None)
    try:
        return load_model(cfg)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load {TINY}: {e}")


def _capture_layer0(loaded, prompt, extra_hook_ctx=None):
    blocks = get_decoder_layers(loaded.model)
    grabbed = {}

    def cap(_m, _i, out):
        grabbed["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    enc = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    if extra_hook_ctx is not None:
        with extra_hook_ctx:
            h = blocks[0].register_forward_hook(cap)   # registered AFTER steering hook
            try:
                with torch.no_grad():
                    loaded.model(**enc)
            finally:
                h.remove()
    else:
        h = blocks[0].register_forward_hook(cap)
        try:
            with torch.no_grad():
                loaded.model(**enc)
        finally:
            h.remove()
    return grabbed["h"]


def test_steering_shifts_by_expected_amount(loaded):
    hidden = loaded.model.config.n_embd
    rng = np.random.default_rng(0)
    direction = rng.normal(size=hidden)
    unit = direction / np.linalg.norm(direction)
    coeff = 3.0
    prompt = "The candidate is"

    base = _capture_layer0(loaded, prompt)
    steered = _capture_layer0(loaded, prompt,
                              extra_hook_ctx=steering(loaded, direction, layers=[0], coeff=coeff))
    diff = (steered - base).numpy()[0]               # [T, hidden]
    expected = coeff * unit
    # Every position should shift by exactly coeff * unit_direction.
    assert np.allclose(diff, expected[None, :], atol=1e-4), np.abs(diff - expected).max()


def test_steering_is_reversible(loaded):
    hidden = loaded.model.config.n_embd
    direction = np.ones(hidden)
    prompt = "The candidate is"
    before = _capture_layer0(loaded, prompt)
    with steering(loaded, direction, layers=[0], coeff=5.0):
        pass  # hooks added and removed
    after = _capture_layer0(loaded, prompt)
    assert torch.allclose(before, after, atol=1e-6)  # model unchanged post-context
