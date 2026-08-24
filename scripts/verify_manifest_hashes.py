"""Verify SHA256 provenance recorded in dataset manifests against local files.

Walks every ``manifest.json`` under ``datasets/`` and checks:

1. each listed shard file exists and its SHA256 matches the recorded value;
2. referenced checkpoint files (``*_checkpoint`` + ``*_checkpoint_sha256``
   pairs) exist and match.

Manifests use two shard-list schemas: ``files`` (multitask corpora) and
``trajectories`` (single-task corpora); both carry per-file ``sha256``.

Requires the binary artifacts to be present locally (``git lfs pull`` for
versioned corpora; generated corpora such as ``datasets/mt10`` exist only on
the machine that produced them).

Exit code 0 iff every record verifies. Usage:

    python scripts/verify_manifest_hashes.py [datasets_root]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> tuple[int, int, int]:
    """Return (verified, mismatched, missing) counts for one manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    verified = mismatched = missing = 0

    entries = manifest.get("files") or manifest.get("trajectories") or []
    for entry in entries:
        rel = entry.get("file")
        recorded = entry.get("sha256")
        if not rel or not recorded:
            continue
        target = base / rel
        if not target.exists():
            missing += 1
            print(f"  MISSING   {manifest_path.parent.name}/{rel}")
            continue
        if sha256_of(target) == recorded:
            verified += 1
        else:
            mismatched += 1
            print(f"  MISMATCH  {manifest_path.parent.name}/{rel}")

    # checkpoint path + sha256 pairs, e.g. "act_checkpoint" / "act_checkpoint_sha256"
    for key, value in manifest.items():
        if not key.endswith("_checkpoint") or not isinstance(value, str):
            continue
        recorded = manifest.get(f"{key}_sha256")
        if not recorded:
            continue
        target = (REPO_ROOT / value).resolve()
        if not target.exists():
            missing += 1
            print(f"  MISSING   {key}: {value}")
            continue
        if sha256_of(target) == recorded:
            verified += 1
        else:
            mismatched += 1
            print(f"  MISMATCH  {key}: {value}")

    return verified, mismatched, missing


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "datasets"
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        print(f"no manifest.json found under {root}")
        return 1

    total_ok = total_bad = total_missing = 0
    for manifest_path in manifests:
        rel = manifest_path.relative_to(REPO_ROOT)
        ok, bad, miss = verify_manifest(manifest_path)
        total_ok += ok
        total_bad += bad
        total_missing += miss
        status = "OK " if bad == 0 and miss == 0 else "FAIL"
        print(f"{status} {rel}: {ok} verified, {bad} mismatched, {miss} missing")

    print(
        f"\nTOTAL: {total_ok} verified, {total_bad} mismatched, "
        f"{total_missing} missing across {len(manifests)} manifests."
    )
    return 0 if total_bad == 0 and total_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
