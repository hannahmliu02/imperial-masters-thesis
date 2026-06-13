"""Run context: a self-contained results directory per run.

Every run (baseline, finetune, identify) writes a directory containing:

* ``config.resolved.yaml`` -- the fully merged config used,
* ``env.json``             -- git commit, dirty flag, python + package versions,
* ``metrics.json``         -- final metrics (written via ``save_metrics``),
* arbitrary artefacts (raw model outputs, checkpoint paths, plots).

This is what makes a run reproducible and auditable on its own.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import dump_yaml

# Packages whose versions materially affect results; logged if importable.
_TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "numpy",
]


def _git_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _run(["rev-parse", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


def collect_env() -> Dict[str, Any]:
    """Snapshot the reproducibility-relevant environment."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(),
        "packages": _package_versions(),
    }


@dataclass
class RunContext:
    """Manages a single run's output directory."""

    run_dir: Path

    @classmethod
    def create(cls, out_dir: str, config: Dict[str, Any]) -> "RunContext":
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = cls(run_dir=run_dir)
        dump_yaml(config, str(run_dir / "config.resolved.yaml"))
        ctx.save_json("env.json", collect_env())
        return ctx

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return p

    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        return self.save_json("metrics.json", metrics)

    def subdir(self, name: str) -> "RunContext":
        sub = self.run_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        return RunContext(run_dir=sub)
