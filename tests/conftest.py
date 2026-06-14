"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.tasks import get_task  # noqa: E402
from guardrail_ft.utils.config import load_config  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIGS = {
    "resume": ["configs/base.yaml", "configs/task_resume.yaml"],
    "bbq": ["configs/base.yaml", "configs/task_bbq.yaml"],
    "winobias": ["configs/base.yaml", "configs/task_winobias.yaml"],
}


@pytest.fixture(params=list(TASK_CONFIGS))
def task_name(request):
    return request.param


@pytest.fixture
def task(task_name):
    cfg = load_config([str(_ROOT / p) for p in TASK_CONFIGS[task_name]])
    return get_task(task_name, cfg)


@pytest.fixture
def cfg(task_name):
    return load_config([str(_ROOT / p) for p in TASK_CONFIGS[task_name]])
