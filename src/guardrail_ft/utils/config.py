"""Config loading: layered YAML + dotted-key CLI overrides.

A run's config is built by deep-merging an ordered list of YAML files
(``base.yaml`` first, then task config, then FT-method config), after which a
list of ``key.path=value`` overrides from the CLI is applied. This is the single
config mechanism for the whole project; no hyperparameters are hardcoded in
scripts.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base`` (over wins)."""
    out = deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    """Best-effort literal coercion for CLI override values.

    ``"3"`` -> int, ``"0.1"`` -> float, ``"true"`` -> bool, ``"[1,2]"`` -> list,
    ``"none"``/``"null"`` -> None; anything else stays a string.
    """
    low = value.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_dotted(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level.")
    return data


def load_config(
    paths: Sequence[str],
    overrides: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load and merge ``paths`` (in order), then apply ``key=value`` overrides.

    Parameters
    ----------
    paths:
        YAML files merged left-to-right (later files win).
    overrides:
        Strings like ``"finetune.lr=1e-4"`` or ``"model.quantization=4bit"``.
    """
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = _deep_merge(cfg, load_yaml(p))

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} must be of the form key.path=value")
        key, _, raw = ov.partition("=")
        _set_dotted(cfg, key.strip(), _coerce(raw))

    # Record the inputs for provenance.
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_paths"] = [str(Path(p)) for p in paths]
    cfg["_meta"]["overrides"] = list(overrides or [])
    return cfg


def get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key with a default (``get(cfg, 'model.name')``)."""
    node: Any = cfg
    for p in dotted_key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def dump_yaml(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
