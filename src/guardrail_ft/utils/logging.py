"""Structured run logging with an optional Weights & Biases backend.

A ``RunLogger`` writes human-readable logs to stderr and structured events to a
JSONL file in the run directory. W&B is enabled only behind a config flag and is
imported lazily so it is never a hard dependency.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "guardrail_ft", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class RunLogger:
    """Per-run logger: console + ``events.jsonl`` + optional W&B.

    Parameters
    ----------
    run_dir:
        Directory the run writes into. ``events.jsonl`` is appended here.
    use_wandb:
        If True, lazily init a W&B run. Silently degrades to no-op if W&B is not
        installed (a warning is logged once).
    wandb_kwargs:
        Passed through to ``wandb.init`` (project, entity, name, config, ...).
    """

    def __init__(
        self,
        run_dir: str,
        use_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.log = get_logger()
        self._wandb = None
        if use_wandb:
            self._init_wandb(wandb_kwargs or {})

    def _init_wandb(self, kwargs: Dict[str, Any]) -> None:
        try:
            import wandb

            self._wandb = wandb.init(dir=str(self.run_dir), **kwargs)
        except ImportError:
            self.log.warning("use_wandb=True but wandb is not installed; skipping.")
            self._wandb = None

    def event(self, kind: str, **fields: Any) -> None:
        """Append a structured event to events.jsonl (and console at debug)."""
        record = {"t": time.time(), "kind": kind, **fields}
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self.log.debug("event %s %s", kind, fields)

    def metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a metrics dict to JSONL and W&B."""
        self.event("metrics", step=step, **metrics)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def info(self, msg: str, *args: Any) -> None:
        self.log.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.log.warning(msg, *args)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
