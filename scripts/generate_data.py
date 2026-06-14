#!/usr/bin/env python3
"""CLI: build a semi-synthetic dataset from templates.

Example:
    python scripts/generate_data.py --task resume --n 1000 --seed 0 --out data/resume_synth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail_ft.cli import generate_data_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(generate_data_main())
