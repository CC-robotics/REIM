"""Fail-closed tests for the MT10/MT50 task-bank separation audit."""

from __future__ import annotations

from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.io import file_sha256
from scripts.audit_multitask_banks import (
    BankAuditError,
    DEMONSTRATION_DATASET_TYPE,
    DEMONSTRATION_SCHEMA,
    EVALUATION_SCHEMA,
    EVALUATION_SIDECAR_SCHEMA,
    FAILURE_DATASET_TYPE,
    FAILURE_SCHEMA,
    RECOVERY_DATASET_TYPE,
    RECOVERY_SCHEMA,
    _canonical_sha256,
    _recovery_episode_seed,
    _snapshot,
    audit,
    build_parser,
)


TASKS = tuple(f"task-{index}-v3" for index in range(10))
VERSION = "3.test"
SEEDS = {
    "demonstrations": 101,
    "failure_training": 202,
    "failure_validation": 303,
    "recovery_training": 404,
    "final_evaluation": 505,
}


class FakeTask:
    def __init__(self, name: str, seed: int, variant: int) -> None:
        self.env_name = name
        self.data = f"native-task:{seed}:{name}:{variant}".encode("ascii")


class FakeBenchmark:
    def __init__(self, seed: int) -> None:
        self.train_classes = {name: object for name in TASKS}
        self.train_tasks = [
            FakeTask(name, seed, variant)
            for name in TASKS
            for variant in range(50)
        ]


def _loader(name: str, seed: int):
    assert name == "MT10"
    return FakeBenchmark(seed), VERSION


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_demonstrations(root: Path, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot("MT10", seed, benchmark_loader=_loader)
    trajectories: list[dict[str, object]] = []
    for task_id, task_name in enumerate(TASKS):
        filename = f"demo_{task_id:02d}.npz"
        path = root / filename
        episode_seed = seed * 10_000 + task_id
        np.savez_compressed(
            path,
            task_id=np.asarray(task_id),
            task_name=np.asarray(task_name),
            task_variant=np.asarray(0),
            seed=np.asarray(episode_seed),
        )
        trajectories.append(
            {
                "file": filename,
                "sha256": file_sha256(path),
                "task_id": task_id,
                "task_name": task_name,
                "task_variant": 0,
                "trajectory_index": 0,
                "attempt_index": 0,
                "seed": episode_seed,
                "length": 1,
                "return": 1.0,
                "success": True,
            }
        )
    manifest = {
        "schema_version": DEMONSTRATION_SCHEMA,
        "dataset_type": DEMONSTRATION_DATASET_TYPE,
        "benchmark": "MT10",
        "seed": seed,
        "metaworld_version": VERSION,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
        "complete": True,
        "protocol": {"episodes_per_task": 1},
        "statistics": {"successful_trajectories": len(TASKS)},
        "trajectories": trajectories,
        # Ensure the fake snapshot was genuinely used rather than just seeds.
        "test_bank_content_sha256": snapshot.content_sha256,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def _failure_protocol(seed: int, collection_seed: int) -> dict[str, object]:
    snapshot = _snapshot("MT10", seed, benchmark_loader=_loader)
    return {
        "schema_version": FAILURE_SCHEMA,
        "dataset_type": FAILURE_DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": VERSION,
        "benchmark_seed": seed,
        "collection_seed": collection_seed,
        "benchmark_task_bank_sha256": snapshot.failure_evaluation_bank_sha256,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
    }


def _write_failure_bank(root: Path, seed: int, *, collection_seed: int = 42) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot("MT10", seed, benchmark_loader=_loader)
    protocol = _failure_protocol(seed, collection_seed)
    files: list[dict[str, object]] = []
    for task_id, task_name in enumerate(TASKS):
        directory = root / f"task_{task_id:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "failure_0000.npz"
        payload_hash = snapshot.payload(task_id, 0)
        episode_seed = collection_seed + task_id * 100_000
        np.savez_compressed(
            path,
            task_id=np.asarray(task_id),
            task_name=np.asarray(task_name),
            task_variant=np.asarray(0),
            task_payload_sha256=np.asarray(payload_hash),
            episode_seed=np.asarray(episode_seed),
            schema_version=np.asarray(FAILURE_SCHEMA),
        )
        files.append(
            {
                "file": path.relative_to(root).as_posix(),
                "task_id": task_id,
                "task_name": task_name,
                "rollout_index": 0,
                "length": 1,
                "success": False,
                "positive_labels": 1,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": FAILURE_SCHEMA,
        "dataset_type": FAILURE_DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": VERSION,
        "benchmark_seed": seed,
        "seed": collection_seed,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
        "benchmark_task_bank_sha256": snapshot.failure_evaluation_bank_sha256,
        "provenance": protocol,
        "provenance_fingerprint_sha256": _canonical_sha256(protocol),
        "rollouts_per_task": 1,
        "episodes": len(TASKS),
        "files": files,
        "complete": True,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def _write_recovery(root: Path, seed: int, *, rollout_seed: int = 42) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot("MT10", seed, benchmark_loader=_loader)
    protocol: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": VERSION,
        "benchmark_seed": seed,
        "rollout_seed": rollout_seed,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
        "task_bank_sha256": snapshot.recovery_bank_sha256,
        "official_goal_variants_per_task": 50,
        "target_per_task": 1,
    }
    fingerprint = _canonical_sha256(protocol)
    files: list[dict[str, object]] = []
    for task_id, task_name in enumerate(TASKS):
        directory = root / f"task_{task_id:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "recovery_0000.npz"
        attempt_index = 0
        payload_hash = snapshot.payload(task_id, 0)
        episode_seed = _recovery_episode_seed(
            rollout_seed, task_id, attempt_index
        )
        np.savez_compressed(
            path,
            task_id=np.asarray(task_id),
            task_name=np.asarray(task_name),
            task_variant=np.asarray(0),
            task_payload_sha256=np.asarray(payload_hash),
            episode_seed=np.asarray(episode_seed),
            attempt_index=np.asarray(attempt_index),
            shard_index=np.asarray(0),
            protocol_fingerprint_sha256=np.asarray(fingerprint),
            schema_version=np.asarray(RECOVERY_SCHEMA),
        )
        files.append(
            {
                "file": path.relative_to(root).as_posix(),
                "task_id": task_id,
                "task_name": task_name,
                "shard_index": 0,
                "attempt_index": attempt_index,
                "task_variant": 0,
                "task_payload_sha256": payload_hash,
                "episode_seed": episode_seed,
                "length": 1,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": RECOVERY_SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "protocol": protocol,
        "protocol_fingerprint_sha256": fingerprint,
        "benchmark": "MT10",
        "metaworld_version": VERSION,
        "benchmark_seed": seed,
        "seed": rollout_seed,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
        "task_bank_sha256": snapshot.recovery_bank_sha256,
        "complete": True,
        "successful_continuations": len(TASKS),
        "files": files,
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    return path


def _write_evaluation(
    root: Path, seed: int, *, episode_seed_base: int = 42
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot("MT10", seed, benchmark_loader=_loader)
    protocol: dict[str, object] = {
        "evaluation_schema_version": EVALUATION_SCHEMA,
        "benchmark": "MT10",
        "condition": "clean",
        "metaworld_version": VERSION,
        "benchmark_seed": seed,
        "episode_seed_base": episode_seed_base,
        "task_bank_sha256": snapshot.failure_evaluation_bank_sha256,
        "task_vocabulary": list(TASKS),
        "task_vocabulary_sha256": _canonical_sha256(list(TASKS)),
        "task_ids": list(range(len(TASKS))),
        "methods": ["act"],
        "episodes_per_task": 50,
        "max_episode_steps": 500,
        "noise_level": 0.0,
        "object_position_noise": False,
    }
    fingerprint = _canonical_sha256(protocol)
    sidecar = root / "clean.csv.run.json"
    _write_json(
        sidecar,
        {
            "schema_version": EVALUATION_SIDECAR_SCHEMA,
            "run_fingerprint": fingerprint,
            "protocol": protocol,
        },
    )
    csv_path = root / "clean.csv"
    fieldnames = [
        "run_fingerprint",
        "benchmark",
        "task_name",
        "task_id",
        "task_variant",
        "method",
        "paired_episode_id",
        "episode_seed",
        "task_payload_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task_id, task_name in enumerate(TASKS):
            for variant in range(50):
                payload_hash = snapshot.payload(task_id, variant)
                writer.writerow(
                    {
                        "run_fingerprint": fingerprint,
                        "benchmark": "MT10",
                        "task_name": task_name,
                        "task_id": task_id,
                        "task_variant": variant,
                        "method": "MT-ACT",
                        "paired_episode_id": (
                            f"{task_id:02d}-{variant:04d}-{payload_hash[:12]}"
                        ),
                        "episode_seed": (
                            episode_seed_base + task_id * 100_000 + variant
                        ),
                        "task_payload_sha256": payload_hash,
                    }
                )
    return sidecar, csv_path


def _study(tmp_path: Path, *, seeds: dict[str, int] | None = None) -> Namespace:
    selected = SEEDS if seeds is None else seeds
    demonstrations = _write_demonstrations(
        tmp_path / "demos", selected["demonstrations"]
    )
    failure_train = _write_failure_bank(
        tmp_path / "failure_train", selected["failure_training"]
    )
    failure_validation = _write_failure_bank(
        tmp_path / "failure_validation", selected["failure_validation"]
    )
    recovery = _write_recovery(
        tmp_path / "recovery", selected["recovery_training"]
    )
    sidecar, episode_csv = _write_evaluation(
        tmp_path / "evaluation", selected["final_evaluation"]
    )
    return Namespace(
        benchmark="MT10",
        demonstrations=str(demonstrations),
        failure_train=str(failure_train),
        failure_validation=str(failure_validation),
        recovery=str(recovery),
        final_evaluation_sidecar=str(sidecar),
        final_evaluation_csv=str(episode_csv),
    )


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_complete_disjoint_study_passes_without_writing(tmp_path: Path) -> None:
    args = _study(tmp_path)
    before = _tree_digests(tmp_path)
    result = audit(args, benchmark_loader=_loader)
    after = _tree_digests(tmp_path)

    assert result["passed"] is True
    assert result["read_only"] is True
    assert all(result["checks"].values())
    assert result["stages"]["demonstrations"]["reserved_task_payloads"] == 500
    assert result["stages"]["final_evaluation"]["materialized_records"] == 500
    # Failure/evaluation deliberately reuse raw RNG integers, but native task
    # payloads and complete sampling-unit identities remain disjoint.
    assert "failure_training::final_evaluation" in result[
        "raw_episode_seed_reuse_pairs"
    ]
    assert before == after


def test_equal_benchmark_seed_is_rejected_by_payload_overlap(tmp_path: Path) -> None:
    seeds = dict(SEEDS)
    seeds["failure_validation"] = seeds["failure_training"]
    args = _study(tmp_path, seeds=seeds)
    with pytest.raises(BankAuditError, match="bank-separation audit failed"):
        audit(args, benchmark_loader=_loader)


def test_forged_npz_payload_is_rejected_even_with_updated_file_digest(
    tmp_path: Path,
) -> None:
    args = _study(tmp_path)
    manifest_path = Path(args.failure_train)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][0]
    shard = manifest_path.parent / entry["file"]
    with np.load(shard, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["task_payload_sha256"] = np.asarray("f" * 64)
    np.savez_compressed(shard, **values)
    entry["sha256"] = file_sha256(shard)
    _write_json(manifest_path, manifest)

    with pytest.raises(BankAuditError, match="NPZ provenance mismatch"):
        audit(args, benchmark_loader=_loader)


def test_tampered_protocol_fingerprint_is_rejected(tmp_path: Path) -> None:
    args = _study(tmp_path)
    manifest_path = Path(args.failure_validation)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance_fingerprint_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(BankAuditError, match="provenance fingerprint is invalid"):
        audit(args, benchmark_loader=_loader)


def test_incomplete_final_grid_is_rejected(tmp_path: Path) -> None:
    args = _study(tmp_path)
    csv_path = Path(args.final_evaluation_csv)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    csv_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(BankAuditError, match="rows; expected"):
        audit(args, benchmark_loader=_loader)


def test_missing_demo_payload_evidence_fails_closed(tmp_path: Path) -> None:
    args = _study(tmp_path)
    manifest_path = Path(args.demonstrations)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest_path.parent / manifest["trajectories"][0]["file"]
    shard.unlink()
    with pytest.raises(BankAuditError, match="missing or symlinked"):
        audit(args, benchmark_loader=_loader)


def test_current_metaworld_version_must_match_recorded_version(
    tmp_path: Path,
) -> None:
    args = _study(tmp_path)

    def wrong_version_loader(name: str, seed: int):
        benchmark, _ = _loader(name, seed)
        return benchmark, "4.changed"

    with pytest.raises(BankAuditError, match="metaworld_version"):
        audit(args, benchmark_loader=wrong_version_loader)


def test_cli_exposes_no_output_file_mutation_option() -> None:
    parser = build_parser()
    option_strings = {
        value
        for action in parser._actions
        for value in action.option_strings
    }
    assert "--output" not in option_strings
    assert "--overwrite" not in option_strings
    assert "--resume" not in option_strings


def test_mt50_reconstruction_requires_fifty_classes_and_2500_payloads() -> None:
    names = tuple(f"mt50-task-{index}-v3" for index in range(50))

    def mt50_loader(name: str, seed: int):
        assert name == "MT50"
        benchmark = SimpleNamespace(
            train_classes={task_name: object for task_name in names},
            train_tasks=[
                FakeTask(task_name, seed, variant)
                for task_name in names
                for variant in range(50)
            ],
        )
        return benchmark, VERSION

    snapshot = _snapshot("MT50", 909, benchmark_loader=mt50_loader)
    assert snapshot.task_vocabulary == names
    assert len(snapshot.payloads) == 2_500
    assert all(len(values) == 50 for values in snapshot.payloads_by_task)
