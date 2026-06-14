#!/usr/bin/env python3
"""CLI: fetch the real-world benchmark datasets and record their versions.

Sources (only those specified in the project brief; no fabricated URLs):
  * BBQ         -- github.com/nyu-mll/BBQ                (JSONL per category)
  * WinoBias    -- github.com/uclanlp/corefBias         (pro/anti, types 1&2)
  * Winogender  -- github.com/rudinger/winogender-schemas
  * Bias in Bios-- Hugging Face Hub (LabHC/bias_in_bios) via `datasets`

Git repos are cloned (shallow) and the resolved HEAD commit is recorded into
data/MANIFEST.json so experiments are reproducible. If a source fails, the error
is reported and that source is skipped -- we do not guess an alternative.

Example:
    python scripts/download_data.py --datasets bbq winobias --out data
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SOURCES = {
    "bbq": {"kind": "git", "url": "https://github.com/nyu-mll/BBQ", "subdir": "bbq"},
    "winobias": {"kind": "git", "url": "https://github.com/uclanlp/corefBias", "subdir": "winobias"},
    "winogender": {"kind": "git", "url": "https://github.com/rudinger/winogender-schemas", "subdir": "winogender"},
    "bias_in_bios": {"kind": "hf", "hf_name": "LabHC/bias_in_bios", "subdir": "bias_in_bios"},
}


def _run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _clone(url: str, dest: Path) -> dict:
    if dest.exists():
        commit = _run(["git", "rev-parse", "HEAD"], cwd=str(dest))
        return {"status": "exists", "commit": commit}
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)])
    commit = _run(["git", "rev-parse", "HEAD"], cwd=str(dest))
    return {"status": "cloned", "commit": commit}


def _fetch_hf(hf_name: str, dest: Path, revision=None) -> dict:
    from datasets import load_dataset

    ds = load_dataset(hf_name, revision=revision)
    dest.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))
    return {"status": "downloaded", "hf_name": hf_name, "revision": revision,
            "splits": {k: len(v) for k, v in ds.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--out", default="data")
    ap.add_argument("--hf-revision", default=None, help="Pin a revision for the HF dataset.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("datasets", {})

    for name in args.datasets:
        spec = SOURCES[name]
        dest = out / spec["subdir"]
        try:
            if spec["kind"] == "git":
                info = _clone(spec["url"], dest)
                info["url"] = spec["url"]
            else:
                info = _fetch_hf(spec["hf_name"], dest, args.hf_revision)
            info["path"] = str(dest)
            info["fetched_at"] = datetime.now(timezone.utc).isoformat()
            manifest["datasets"][name] = info
            print(f"[download] {name}: {info.get('status')} -> {dest} "
                  f"({info.get('commit', info.get('revision', ''))})")
        except subprocess.CalledProcessError as e:
            print(f"[download] FAILED {name}: git error: {e.stderr.strip()}", file=sys.stderr)
        except ImportError as e:
            print(f"[download] FAILED {name}: {e} (install with: pip install -e '.[ml]')", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[download] FAILED {name}: {e}", file=sys.stderr)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[download] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
