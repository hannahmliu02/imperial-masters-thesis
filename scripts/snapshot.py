#!/usr/bin/env python3
"""Named source snapshots — a git-independent rewind option.

Captures the source tree (src/, scripts/, configs/, tests/, *.md, pyproject.toml)
into a timestamped, labelled folder under `.snapshots/` so you can roll back to a
known-good point even across uncommitted work. This complements git: it works
regardless of commit state and records the git commit at capture time.

Usage:
    python scripts/snapshot.py save  pre-validation-stage
    python scripts/snapshot.py list
    python scripts/snapshot.py restore pre-validation-stage          # prompts
    python scripts/snapshot.py restore pre-validation-stage --yes    # no prompt
    python scripts/snapshot.py restore <id> --dry-run                # preview only

`restore` first auto-snapshots the CURRENT tree (label `autobackup-before-restore`)
so a restore is itself reversible. Snapshots are local only (`.snapshots/` is
self-ignored from git).

Rewind options summary:
  * This tool             -> named checkpoints incl. uncommitted work.
  * git restore --source=<commit> <path>   -> revert specific files to a commit.
  * git stash / git checkout .             -> drop all uncommitted changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / ".snapshots"

# What to capture (source of truth for the project; excludes data/runs/caches).
INCLUDE = ["src", "scripts", "configs", "tests", "pyproject.toml", "README.md",
           "METHOD.md", ".gitignore"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                                "*.egg-info", ".snapshots")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _copy_tree(dst: Path) -> None:
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        target = dst / rel
        if src.is_dir():
            shutil.copytree(src, target, ignore=IGNORE, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def cmd_save(label: str) -> int:
    SNAP_DIR.mkdir(exist_ok=True)
    (SNAP_DIR / ".gitignore").write_text("*\n!.gitignore\n")  # self-ignore from git
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    snap_id = f"{stamp}_{label}"
    dst = SNAP_DIR / snap_id
    if dst.exists():
        print(f"[snapshot] {snap_id} already exists.", file=sys.stderr)
        return 1
    _copy_tree(dst)
    manifest = {"id": snap_id, "label": label, "created": stamp,
                "git_commit": _git_commit(), "include": INCLUDE}
    (dst / "SNAPSHOT.json").write_text(json.dumps(manifest, indent=2))
    n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
    print(f"[snapshot] saved {snap_id} ({n_files} files) -> {dst.relative_to(ROOT)}")
    return 0


def _find(snap_ref: str) -> Path | None:
    if not SNAP_DIR.exists():
        return None
    exact = SNAP_DIR / snap_ref
    if exact.is_dir():
        return exact
    # Match by label suffix; pick the most recent.
    matches = sorted(p for p in SNAP_DIR.iterdir()
                     if p.is_dir() and p.name.endswith(f"_{snap_ref}"))
    return matches[-1] if matches else None


def cmd_list() -> int:
    if not SNAP_DIR.exists() or not any(SNAP_DIR.iterdir()):
        print("[snapshot] none yet.")
        return 0
    for p in sorted(SNAP_DIR.iterdir()):
        if not p.is_dir():
            continue
        man = p / "SNAPSHOT.json"
        info = json.loads(man.read_text()) if man.exists() else {}
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        print(f"  {p.name:40s} files={n:4d} git={str(info.get('git_commit'))[:8]}")
    return 0


def cmd_restore(snap_ref: str, yes: bool, dry_run: bool) -> int:
    snap = _find(snap_ref)
    if snap is None:
        print(f"[snapshot] no snapshot matching {snap_ref!r}. Use 'list'.", file=sys.stderr)
        return 1
    files = [p for p in snap.rglob("*") if p.is_file() and p.name != "SNAPSHOT.json"]
    print(f"[snapshot] restore {snap.name}: {len(files)} files -> repo root")
    if dry_run:
        for p in files[:20]:
            print(f"    would write {p.relative_to(snap)}")
        if len(files) > 20:
            print(f"    ... (+{len(files) - 20} more)")
        return 0
    if not yes:
        resp = input("This overwrites current source files. Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("[snapshot] aborted.")
            return 1
    # Auto-backup current tree first (so restore is reversible).
    cmd_save("autobackup-before-restore")
    for p in files:
        rel = p.relative_to(snap)
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    print(f"[snapshot] restored {snap.name}. (A backup of the prior state was saved.)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save"); s.add_argument("label")
    sub.add_parser("list")
    r = sub.add_parser("restore"); r.add_argument("ref")
    r.add_argument("--yes", action="store_true"); r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "save":
        return cmd_save(args.label)
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "restore":
        return cmd_restore(args.ref, args.yes, args.dry_run)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
