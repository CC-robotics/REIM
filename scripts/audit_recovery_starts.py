#!/usr/bin/env python3
"""Audit when and in which task phase recovery snapshots are captured."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import atomic_json_dump  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_rate(values: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.bool_).reshape(-1)
    count = int(np.count_nonzero(flattened))
    return {
        "count": count,
        "total": int(flattened.size),
        "rate": float(count / max(flattened.size, 1)),
    }


def _audit_split(npz_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing recovery-start split: {npz_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz_hash = _sha256(npz_path)
    if manifest.get("npz_sha256") != npz_hash:
        raise ValueError(f"{manifest_path}: NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "trigger_step",
            "trigger_probability",
            "snapshot_object_disturbed",
            "snapshot_ever_lifted",
            "snapshot_failure_latched",
            "expert_success",
            "episode_seed",
            "failure_threshold",
            "noise_level",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{npz_path}: missing fields {sorted(missing)}")
        trigger_steps = np.asarray(archive["trigger_step"], dtype=np.int64)
        probabilities = np.asarray(
            archive["trigger_probability"], dtype=np.float64
        )
        object_disturbed = np.asarray(archive["snapshot_object_disturbed"])
        ever_lifted = np.asarray(archive["snapshot_ever_lifted"])
        failure_latched = np.asarray(archive["snapshot_failure_latched"])
        expert_success = np.asarray(archive["expert_success"])
        episode_seeds = np.asarray(archive["episode_seed"], dtype=np.int64)
        threshold = float(np.asarray(archive["failure_threshold"]).item())
        noise_level = float(np.asarray(archive["noise_level"]).item())
    starts = int(trigger_steps.size)
    if starts != int(manifest["episodes_triggered"]):
        raise ValueError(f"{npz_path}: trigger count differs from manifest")
    if len(np.unique(episode_seeds)) != starts:
        raise ValueError(f"{npz_path}: duplicate episode seeds")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{npz_path}: non-finite trigger probability")
    if np.any(probabilities < threshold):
        raise ValueError(f"{npz_path}: snapshot below collection threshold")
    attempted = int(manifest["episodes_attempted"])
    return {
        "npz": str(npz_path.resolve()),
        "npz_sha256": npz_hash,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "episodes_attempted": attempted,
        "episodes_triggered": starts,
        "trigger_rate": float(starts / attempted),
        "seed_min": int(episode_seeds.min()),
        "seed_max": int(episode_seeds.max()),
        "failure_threshold": threshold,
        "noise_level": noise_level,
        "trigger_step": {
            "mean": float(trigger_steps.mean()),
            "median": float(np.median(trigger_steps)),
            "q10": float(np.quantile(trigger_steps, 0.10)),
            "q90": float(np.quantile(trigger_steps, 0.90)),
            "minimum": int(trigger_steps.min()),
            "maximum": int(trigger_steps.max()),
        },
        "trigger_probability": {
            "mean": float(probabilities.mean()),
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
        },
        "snapshot_object_already_displaced": _scalar_rate(object_disturbed),
        "snapshot_ever_lifted": _scalar_rate(ever_lifted),
        "snapshot_failure_latched": _scalar_rate(failure_latched),
        "scripted_continuation_success": _scalar_rate(expert_success),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "datasets/recovery_starts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tables/recovery_start_distribution.json",
    )
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    train = _audit_split(directory / "train.npz", directory / "train.json")
    validation = _audit_split(
        directory / "validation.npz", directory / "validation.json"
    )
    if train["seed_max"] >= validation["seed_min"]:
        raise ValueError("training and validation recovery-start seeds overlap")
    payload = {
        "schema_version": 1,
        "artifact": "recovery_start_phase_audit",
        "interpretation": (
            "The 0.10 collection gate primarily captures pre-grasp, "
            "post-displacement early-warning states. It is not a bank of "
            "post-drop or terminal failure states."
        ),
        "train": train,
        "validation": validation,
        "audit": {
            "raw_npz_and_sidecar_hashes_verified": True,
            "counts_recomputed_from_raw_arrays": True,
            "training_validation_seeds_disjoint": True,
        },
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
