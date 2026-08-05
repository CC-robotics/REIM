from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

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
from trainers.data import audit_calibrated_multitask_failure_bank
from trainers.train_detector import train as train_detector


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
        "state_dim": 49,
        "action_dim": 4,
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _canonical_sha256(TASKS),
        "provenance": provenance,
        "provenance_fingerprint_sha256": _canonical_sha256(provenance),
        "prediction_horizon": 2,
        "terminal_positive_horizon": 2,
        "rollouts_per_task": 2,
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
    audited = audit_calibrated_multitask_failure_bank(
        tmp_path,
        expected_role="training",
        expected_mode="fit-task-quantile",
    )
    assert audited.dataset_fingerprint_sha256 == result[
        "dataset_fingerprint_sha256"
    ]


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


def test_training_audit_rejects_raw_labels_and_stale_calibrated_shards(
    tmp_path: Path,
) -> None:
    _write_bank(tmp_path)
    with pytest.raises(ValueError, match="lacks label_calibration"):
        audit_calibrated_multitask_failure_bank(
            tmp_path,
            expected_role="training",
            expected_mode="fit-task-quantile",
        )
    manifest = relabel_failure_bank(
        tmp_path,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    first = tmp_path / manifest["files"][0]["file"]
    stale = tmp_path / "task_00" / "failure_9999.npz"
    shutil.copyfile(first, stale)
    with pytest.raises(ValueError, match="whitelist mismatch"):
        audit_calibrated_multitask_failure_bank(
            tmp_path,
            expected_role="training",
            expected_mode="fit-task-quantile",
        )


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


def test_detector_checkpoint_preserves_complete_training_calibration_provenance(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "training-bank"
    _write_bank(bank)
    manifest = relabel_failure_bank(
        bank,
        quantile=0.90,
        prediction_horizon=2,
        terminal_positive_horizon=2,
    )
    checkpoint = tmp_path / "detector.pt"
    latest = tmp_path / "detector-latest.pt"
    config = {
        "seed": 7,
        "device": "cpu",
        "data_dir": str(bank),
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest),
        "curve_path": str(tmp_path / "curve.png"),
        "confusion_matrix_path": str(tmp_path / "confusion.png"),
        "history_path": str(tmp_path / "history.csv"),
        "metrics_path": str(tmp_path / "metrics.json"),
        "model": {
            "sequence_length": 3,
            "hidden_size": 4,
            "num_layers": 1,
            "mlp_hidden": 4,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "validation_fraction": 0.2,
            "positive_weight": "auto",
            "threshold": 0.5,
            "deployment_threshold": 0.8,
            "patience": 0,
            "checkpoint_every": 0,
        },
    }
    config_path = tmp_path / "detector.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    train_detector(
        Namespace(
            config=str(config_path),
            data_dir=None,
            checkpoint=None,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            seed=None,
            device=None,
            sequence_length=None,
            hidden_size=None,
            threshold=None,
            resume=None,
        )
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["detector_training_schema"] == (
        "reim-failure-detector-training-v2"
    )
    assert payload["data_schema_version"] == FAILURE_SCHEMA
    assert payload["dataset_type"] == FAILURE_DATASET_TYPE
    assert payload["dataset_role"] == "training"
    assert payload["label_calibration_mode"] == "fit-task-quantile"
    assert payload["label_calibration_quantile"] == pytest.approx(0.90)
    assert payload["label_calibration"] == manifest["label_calibration"]
    assert payload["label_calibration_fingerprint_sha256"] == manifest[
        "label_calibration_fingerprint_sha256"
    ]
    assert payload["label_calibration_source_sha256"] == manifest[
        "label_calibration_source_sha256"
    ]
    assert payload["dataset_fingerprint_sha256"] == manifest[
        "dataset_fingerprint_sha256"
    ]
    assert payload["data_manifest_sha256"] == file_sha256(bank / "manifest.json")

    payload.pop("dataset_fingerprint_sha256")
    tampered_resume = tmp_path / "detector-tampered.pt"
    torch.save(payload, tampered_resume)
    with pytest.raises(ValueError, match="calibration provenance"):
        train_detector(
            Namespace(
                config=str(config_path),
                data_dir=None,
                checkpoint=None,
                epochs=None,
                batch_size=None,
                learning_rate=None,
                seed=None,
                device=None,
                sequence_length=None,
                hidden_size=None,
                threshold=None,
                resume=str(tampered_resume),
            )
        )
