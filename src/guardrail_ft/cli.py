"""Shared CLI helpers and console entry points.

All scripts use the same config pattern: one or more YAML files merged in order,
plus repeatable ``--set key.path=value`` overrides. ``scripts/*.py`` are thin
wrappers around the functions here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


def bootstrap_path() -> None:
    """Ensure ``src`` is importable when running scripts without installation."""
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def add_config_args(parser: argparse.ArgumentParser) -> None:
    # Preferred: a single experiment manifest (single source of truth).
    parser.add_argument("--experiment", "-x", default=None,
                        help="Experiment manifest (configs/experiments/<name>.yaml). "
                             "If given, --config/--task/--ft are ignored.")
    # Legacy layered mechanism (still supported).
    parser.add_argument("--config", default="configs/base.yaml", help="Base config YAML.")
    parser.add_argument("--task", dest="task_cfg", default=None, help="Task config YAML.")
    parser.add_argument("--ft", dest="ft_cfg", default=None, help="Fine-tune method config YAML.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="key.path=value", help="Dotted-key config override (repeatable).")


def resolve_config(args) -> Dict:
    from guardrail_ft.utils.config import load_config, load_experiment

    # An experiment manifest is the single-source-of-truth path; it wins outright.
    if getattr(args, "experiment", None):
        return load_experiment(args.experiment, args.overrides)

    paths = [args.config]
    if getattr(args, "task_cfg", None):
        paths.append(args.task_cfg)
    if getattr(args, "ft_cfg", None):
        paths.append(args.ft_cfg)
    return load_config(paths, args.overrides)


def build_dataset(task, cfg: Dict, data: Optional[str] = None):
    """Resolve a dataset for ``task`` from ``--data``.

    * a path (file or dir with ``data.jsonl``) -> load that pre-generated set;
    * the literal ``"real"`` -> ``task.load_real`` at the configured path;
    * otherwise (``None`` / ``"synthetic"``) -> generate synthetically from the
      task config's ``data.synthetic`` block.
    """
    from guardrail_ft.tasks.base import Dataset

    if data and data not in ("synthetic", "real"):
        p = Path(data)
        jl = (p / "data.jsonl") if p.is_dir() else p
        return Dataset.from_jsonl(str(jl), task.name)
    if data == "real":
        path = cfg.get("task", {}).get("data", {}).get("real", {}).get("path")
        return task.load_real(path)
    sd = cfg.get("task", {}).get("data", {}).get("synthetic", {})
    return task.generate_synthetic(n=sd.get("n", 200), seed=sd.get("seed", 0))


def make_base_loader(cfg: Dict, init_checkpoint: Optional[str] = None):
    """Return a zero-arg callable yielding a FRESH ``LoadedModel`` each call.

    If ``init_checkpoint`` is given (objective=erode), the guardrailed adapter is
    loaded and merged into the base weights so subsequent fine-tuning starts from
    the guardrailed model.
    """
    from guardrail_ft.models.loading import load_model

    def _make():
        loaded = load_model(cfg)
        if init_checkpoint:
            from peft import PeftModel

            merged = PeftModel.from_pretrained(loaded.model, init_checkpoint)
            loaded.model = merged.merge_and_unload()
        return loaded

    return _make


# --------------------------------------------------------------------------- #
# generate-data console entry
# --------------------------------------------------------------------------- #


def generate_data_main(argv: Optional[List[str]] = None) -> int:
    bootstrap_path()
    from guardrail_ft.tasks import get_task
    from guardrail_ft.data.synthetic import validate_minimal_pairs

    ap = argparse.ArgumentParser(description="Generate a semi-synthetic bias dataset.")
    ap.add_argument("--task", required=True, choices=["resume", "bbq", "winobias"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--task-config", default=None, help="Optional task YAML for task settings.")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    import json
    from guardrail_ft.utils.config import load_config

    paths = [args.config] + ([args.task_config] if args.task_config else [])
    cfg = load_config(paths)
    task = get_task(args.task, cfg)
    ds = task.generate_synthetic(n=args.n, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds.to_jsonl(str(out / "data.jsonl"))
    (out / "datasheet.json").write_text(json.dumps(ds.summary(), indent=2))
    (out / "provenance.json").write_text(json.dumps(ds.provenance, indent=2, default=str))

    report = None
    if not args.no_validate:
        report = validate_minimal_pairs(ds)
        (out / "minimal_pair_report.json").write_text(json.dumps(report, indent=2))

    print(f"[generate_data] task={args.task} n_items={len(ds)} -> {out}")
    print(f"[generate_data] datasheet: {ds.summary()}")
    if report:
        print(f"[generate_data] minimal-pair check: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(generate_data_main())
