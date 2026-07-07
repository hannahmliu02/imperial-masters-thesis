#!/usr/bin/env python3
"""Pre-stage all Hugging Face assets into the cache, for offline compute nodes.

HPC compute nodes (e.g. Imperial CX3) usually have NO internet, so the model,
tokenizer and any evaluation datasets must be downloaded ON THE LOGIN NODE first,
into a cache the job can read offline. Run this once after `pip install -e .[ml]`:

    export HF_HOME=$EPHEMERAL/hf_cache          # same path the job uses
    export HF_TOKEN=hf_xxx                       # needed for gated repos (Mistral)
    python scripts/prefetch_hf.py --configs configs/base.yaml configs/hpc_mistral.yaml

Then the PBS job sets HF_HUB_OFFLINE=1 and loads everything from $HF_HOME.

Notes:
* Mistral-7B-Instruct is a **gated** repo: accept the licence on its HF page and
  provide a token (HF_TOKEN env, or `huggingface-cli login`) or this will 401.
* Use an ungated model (e.g. Qwen/Qwen2.5-7B-Instruct) by editing model.name if you
  cannot get Mistral access.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.utils.config import get, load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-download HF model + datasets for offline runs.")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Config YAMLs (to read model.name/revision + capability source).")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--skip-mmlu", action="store_true", help="Do not prefetch the MMLU capability set.")
    ap.add_argument("--mmlu-config", default="all")
    args = ap.parse_args(argv)

    cfg = load_config(args.configs, args.overrides)
    model = get(cfg, "model.name")
    revision = get(cfg, "model.revision")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[prefetch] HF_HOME={os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    print(f"[prefetch] model={model} revision={revision} token={'set' if token else 'NONE'}")

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=model, revision=revision, token=token)
        print(f"[prefetch] model cached -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[prefetch] FAILED to fetch {model}: {e}", file=sys.stderr)
        print("[prefetch] If gated (Mistral): accept the licence on the model page and set HF_TOKEN.",
              file=sys.stderr)
        return 1

    cap_src = get(cfg, "identify.eval.capability_source", "bundled")
    if cap_src == "mmlu" and not args.skip_mmlu:
        try:
            from datasets import load_dataset

            d = load_dataset("cais/mmlu", args.mmlu_config, split="test")
            print(f"[prefetch] MMLU cached ({len(d)} test items).")
        except Exception as e:  # noqa: BLE001
            print(f"[prefetch] WARNING: MMLU prefetch failed ({e}); the run will fall "
                  f"back to the bundled capability set.", file=sys.stderr)

    print("[prefetch] done. The PBS job can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
