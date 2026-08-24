"""Normalize absolute paths embedded in repo artifacts to repo-relative paths.

Artifacts (manifests, audit JSONs, tables) historically recorded absolute
paths from whichever machine generated them (e.g. ``/home/diy/.../REIM/...``
or ``C:\\Users\\...\\REIM\\...``). This breaks provenance verification on any
other machine. This script rewrites every absolute path that points into this
repository to a repo-relative, forward-slash path.

JSON files may contain backslash-escaped separators (``\\\\``); the remainder
of each rewritten path has its separators normalized to ``/``.

Usage:
    python scripts/normalize_artifact_paths.py --dry-run   # report only
    python scripts/normalize_artifact_paths.py             # apply in place
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_SUFFIXES = {
    ".json", ".md", ".py", ".yaml", ".yml", ".sh",
    ".csv", ".tex", ".cff", ".txt",
}

EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}

# Files that must not be rewritten: this script itself (its docstring holds
# example paths) and personal logs that are not repo deliverables.
EXCLUDE_FILES = {"normalize_artifact_paths.py", "REIM-Kimi对话记录.md"}

# An absolute path that enters the repo: drive-letter or POSIX root prefix,
# any intermediate directories, then the repo directory name ("REIM") and one
# separator, then the repo-relative remainder up to the closing quote.
# In JSON source text, backslash separators appear escaped as two characters
# (\\), so accept one-or-two backslashes or a forward slash as a separator.
_SEP = r"(?:\\\\|\\|/)"
PATH_RE = re.compile(
    r"(?P<prefix>(?:[A-Za-z]:|/(?:home|Users|mnt|data|tmp|root)/)"
    r"[^\"'\n]*?REIM)" + _SEP + r"(?P<rest>[^\"'\n]*)"
)


def _normalize(match: re.Match[str]) -> str:
    rest = match.group("rest")
    # JSON-escaped backslashes first, then any lone backslashes.
    rest = rest.replace("\\\\", "/").replace("\\", "/")
    return rest


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()

    changed_files = 0
    total_replacements = 0
    leftovers: list[str] = []

    for path in iter_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text, count = PATH_RE.subn(_normalize, text)
        if count:
            rel = path.relative_to(REPO_ROOT)
            changed_files += 1
            total_replacements += count
            print(f"{count:5d}  {rel}")
            if not args.dry_run:
                path.write_text(new_text, encoding="utf-8")
        # Any absolute path still pointing at a REIM dir after rewriting?
        for leftover in PATH_RE.findall(new_text):
            leftovers.append(f"{path}: {leftover}")

    print(
        f"\n{'[dry-run] would change' if args.dry_run else 'changed'} "
        f"{changed_files} files, {total_replacements} path(s) rewritten."
    )
    if leftovers:
        print("WARNING: unresolved absolute paths remain:")
        for item in leftovers:
            print("  ", item)


if __name__ == "__main__":
    main()
