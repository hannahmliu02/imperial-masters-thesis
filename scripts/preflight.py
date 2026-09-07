#!/usr/bin/env python3
"""GPU preflight: verify the environment + model load + one generation, fast.

Run on a GPU node (e.g. an interactive session: `qsub -I -l select=1:ncpus=4:mem=64gb:ngpus=1`)
AFTER prefetching the model, to catch env/CUDA/offline/gating problems in ~2 min
instead of discovering them hours into the real job:

    export HF_HOME=$EPHEMERAL/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/preflight.py --configs configs/base.yaml configs/task_resume.yaml \
        configs/hpc_mistral.yaml

Exits non-zero on any failure with a clear message.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.utils.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU environment preflight.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)

    ok = True

    # 1. Core libs + CUDA.
    import torch
    print(f"[preflight] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[preflight] gpu={torch.cuda.get_device_name(0)}  n={torch.cuda.device_count()}")
    else:
        print("[preflight] WARNING: CUDA not available — is this a GPU node?")
        ok = False
    for pkg in ("transformers", "peft", "datasets", "accelerate"):
        try:
            m = __import__(pkg)
            print(f"[preflight] {pkg} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"[preflight] MISSING: {pkg}  (pip install -e '.[ml]')")
            ok = False
    if not ok:
        return 2

    # 2. Load the model (exercises gating + offline cache + dtype + device).
    cfg = load_config(args.configs, args.overrides)
    from guardrail_ft.models.loading import load_model

    print(f"[preflight] loading {cfg['model']['name']} on {cfg['model'].get('device')} ...")
    try:
        loaded = load_model(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] FAILED to load model: {e}")
        print("[preflight] gated repo? -> set HF_TOKEN + accept licence, prefetch on login node.")
        return 3
    print(f"[preflight] model on device: {loaded.device}")

    # 3. One generation + parse.
    from guardrail_ft.tasks import get_task

    task = get_task(cfg["task"]["name"], cfg)
    item = task.generate_synthetic(n=2, seed=0).items[0]
    out = loaded.generate(task.format_prompt(item), decoding={"do_sample": False})
    print(f"[preflight] sample generation: {out!r} -> parsed={task.parse_response(out, item)}")

    print("[preflight] OK — environment is ready for the full job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
