"""Global seed control for reproducible runs.

Seeds Python ``random``, NumPy, and (if available) torch. Torch is imported
lazily so this module is usable on machines without it.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed all RNGs. Returns the seed for logging.

    ``deterministic`` additionally requests deterministic CuDNN/cublas behaviour
    (slower, but reproducible) when torch+CUDA are present.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cublas determinism for matmuls on CUDA >= 10.2
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """DataLoader worker seeding helper (keeps workers reproducible)."""
    s = (base_seed or 0) + worker_id
    random.seed(s)
    try:
        import numpy as np

        np.random.seed(s % (2**32 - 1))
    except ImportError:
        pass
