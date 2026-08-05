from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.io import file_sha256
from scripts.relabel_multitask_failures import (
    CALIBRATION_SCHEMA,
    FAILURE_DATASET_TYPE,
    FAILURE_SCHEMA,
    JOURNAL_NAME,
    _array_sha256,
    _canonical_sha256,
    relabel_failure_bank,
)


TASKS = [f"task-{index}-v3" for index in range(10)]


def _write_bank(root: Path, *, offset: float = 0.0) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    per_task: dict[str, object] = {}
    total_rows = 0
    for task_id, task_name in enumerate(TASKS):
        task_rows = 0
        for rollout_index in range(2):
            task_dir = root / f"task_{task_id:02d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            path = task_dir / f"failure_{rollout_index:04d}.npz"
            length = 5
            raw = np.full((length, 39), task_id / 10.0, dtype=np.float32)
            one_hot = np.zeros(10, dtype=np.float32)
            one_hot[task_id] = 1.0
            states = np.concatenate(
                [raw, np.broadcast_to(one_hot, (length, 10))], axis=1
            ).astype(np.float32)
            start = rollout_index * length
            disagreements = (
                np.arange(start, start + length, dtype=np.float32)
                + task_id * 0.1
                + offset
            )
            old_labels = np.ones(length, dtype=np.bool_)
            success = task_id != 0
            np.savez_compressed(
                path,
                states=states,
                raw_observations=raw,
                actions=np.zeros((length, 4), dtype=np.float32),
                expert_actions=np.ones((length, 4), dtype=np.float32),
                action_disagreement_l1=disagreements,
                risk_events=old_labels,
                labels=old_labels,
                rewards=np.zeros(length, dtype=np.float32),
                success=np.asarray(success),
                task_id=np.asarray(task_id, dtype=np.int16),
                task_name=np.asarray(task_name),
                task_variant=np.asarray(rollout_index, dtype=np.int16),
                task_payload_sha256=np.asarray(hashlib.sha256(path.name.encode()).hexdigest()),
                episode_seed=np.asarray(1000 + task_id * 100 + rollout_index),
                schema_version=np.asarray(FAILURE_SCHEMA),
            )
            entries.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "task_id": task_id,
                    "task_name": task_name,
                    "rollout_index": rollout_index,
                    "length": length,
                    "success": success,
                    "positive_labels": length,
                    "sha256": file_sha256(path),
                }
            )
            task_rows += length
            total_rows += length
        per_task[task_name] = {
            "episodes": 2,
            "rows": task_rows,
            "success_rate": float(task_id != 0),
            "positive_rate": 1.0,
        }
    provenance = {
        "schema_version": FAILURE_SCHEMA,
        "dataset_type": FAILURE_DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": "3.test",
        "benchmark_seed": 31,
        "collection_seed": 17,
        "benchmark_task_bank_sha256": "a" * 64,
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _canonical_sha256(TASKS),
        "act_checkpoint_sha256": "b" * 64,
        "observation_schema": "raw39_plus_official_task_one_hot",
        "state_dim": 49,
        "action_dim": 4,
        "max_episode_steps": 500,
        "noise_parameters": {
            "noise_level": 0.2,
            "action_std_scale": 0.4,
            "observation_std_scale": 0.025,
            "action_noise_std": 0.08,
            "observation_noise_std": 0.005,
            "object_position_noise": False,
        },
        "label_parameters": {
            "expert_action_disagreement_l1": 0.35,
            "prediction_horizon": 2,
            "terminal_positive_horizon": 2,
        },
    }
    manifest: dict[str, object] = {
        "schema_version": FAILURE_SCHEMA,
        "dataset_type": FAILURE_DATASET_TYPE,
        "benchmark": "MT10",
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _canonical_sha256(TASKS),
        "provenance": provenance,
        "provenance_fingerprint_sha256": _canonical_sha256(provenance),
        "prediction_horizon": 2,
        "terminal_positive_horizon": 2,
        "episodes": len(entries),
        "rows": total_rows,
        "positive_rate": 1.0,
        "per_task": per_task,
        "files": entries,
        "complete": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def test_task_conditional_quantile_relabels_and_preserves_raw_data(tmp_path: Path) -> None:
    _write_bank(tmp_path)
    first_path = tmp_path / "task_00" / "failure_0000.npz"
    with np.load(first_path, allow_pickle=False) as archive:
        raw_action_hash = _array_sha256(archive["actions"])
        raw_disagreement_hash = _array_sha256(archive["action_disagreement_l1"])

    result = relabel_failure_bank(
        tmp_path,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )

    calibration = result["label_calibration"]
    assert calibration["schema_version"] == CALIBRATION_SCHEMA
    assert calibration["mode"] == "fit-task-quantile"
    assert calibration["dataset_role"] == "training"
    assert calibration["quantile"] == 0.90
    thresholds = calibration["task_thresholds"]
    assert len(thresholds) == 10
    assert thresholds[0]["threshold"] == pytest.approx(8.1)
    assert thresholds[7]["threshold"] == pytest.approx(8.8)
    assert result["per_task"][TASKS[0]]["positive_rate"] == pytest.approx(0.8)
    assert result["per_task"][TASKS[1]]["positive_rate"] == pytest.approx(0.3)
    assert result["positive_rate"] == pytest.approx(0.35)
    assert len(result["label_calibration_source_sha256"]) == 64
    assert len(result["dataset_fingerprint_sha256"]) == 64
    assert result["provenance_fingerprint_sha256"] == _canonical_sha256(
        result["provenance"]
    )
    assert not (tmp_path / JOURNAL_NAME).exists()

    with np.load(first_path, allow_pickle=False) as archive:
        assert _array_sha256(archive["actions"]) == raw_action_hash
        assert _array_sha256(archive["action_disagreement_l1"]) == raw_disagreement_hash
        np.testing.assert_array_equal(
            archive["risk_events"], np.asarray([False, False, False, True, True])
        )
        np.testing.assert_array_equal(
            archive["labels"], np.asarray([False, True, True, True, True])
        )
        assert float(archive["label_threshold"]) == pytest.approx(8.1)
        assert str(archive["label_calibration_fingerprint_sha256"]) == result[
            "label_calibration_fingerprint_sha256"
        ]

    hashes = [entry["sha256"] for entry in result["files"]]
    repeated = relabel_failure_bank(
        tmp_path,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    assert [entry["sha256"] for entry in repeated["files"]] == hashes
    assert repeated == result


def test_relabel_transaction_resumes_after_interruption(tmp_path: Path) -> None:
    _write_bank(tmp_path)
    with pytest.raises(RuntimeError, match="Injected interruption"):
        relabel_failure_bank(
            tmp_path,
            quantile=0.90,
            prediction_horizon=2,
            terminal_positive_horizon=2,
            _fail_after_shards=2,
        )
    assert (tmp_path / JOURNAL_NAME).is_file()

    completed = relabel_failure_bank(
        tmp_path,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    assert completed["complete"] is True
    assert not (tmp_path / JOURNAL_NAME).exists()
    assert all(
        entry["sha256"] == file_sha256(tmp_path / entry["file"])
        for entry in completed["files"]
    )


def test_validation_must_reuse_untampered_training_thresholds(tmp_path: Path) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    _write_bank(training)
    _write_bank(validation, offset=100.0)
    training_manifest = relabel_failure_bank(
        training,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )

    with pytest.raises(ValueError, match="may only use frozen"):
        relabel_failure_bank(
            validation,
            mode="fit-task-quantile",
            dataset_role="validation",
        )
    frozen = relabel_failure_bank(
        validation,
        mode="frozen-task-thresholds",
        dataset_role="validation",
        calibration_manifest=training / "manifest.json",
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    assert frozen["label_calibration"]["mode"] == "frozen-task-thresholds"
    assert frozen["label_calibration"]["task_thresholds"] == training_manifest[
        "label_calibration"
    ]["task_thresholds"]
    assert frozen["label_calibration"]["calibration_source"]["sha256"] == (
        training_manifest["label_calibration_fingerprint_sha256"]
    )
    # The shifted validation distribution would produce different quantiles;
    # frozen training thresholds therefore classify every disagreement event.
    assert frozen["positive_rate"] == 1.0

    source_path = training / "manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["label_calibration"]["task_thresholds"][0]["threshold"] = 999.0
    source_path.write_text(json.dumps(source), encoding="utf-8")
    second_validation = tmp_path / "validation-tampered-source"
    _write_bank(second_validation, offset=200.0)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        relabel_failure_bank(
            second_validation,
            mode="frozen-task-thresholds",
            dataset_role="validation",
            calibration_manifest=source_path,
        )


def test_relabel_fails_closed_on_shard_hash_damage(tmp_path: Path) -> None:
    _write_bank(tmp_path)
    damaged = tmp_path / "task_00" / "failure_0000.npz"
    damaged.write_bytes(damaged.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        relabel_failure_bank(tmp_path)


def test_separate_output_is_idempotent_but_rejects_another_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "calibrated"
    _write_bank(source)
    first = relabel_failure_bank(
        source,
        output_dir=output,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    second = relabel_failure_bank(
        source,
        output_dir=output,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    assert second == first
    with pytest.raises(FileExistsError, match="different calibration request"):
        relabel_failure_bank(
            source,
            output_dir=output,
            quantile=0.85,
            prediction_horizon=2,
            terminal_positive_horizon=2,
        )
