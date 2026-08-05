"""Protocol, metric, and determinism tests for validation-only gate tuning."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from data.io import file_sha256
from evaluation.tune_multitask_detector import (
    DATASET_TYPE,
    SCHEMA_VERSION,
    _canonical_json_sha256,
    _validate_detector_checkpoint,
    audit_validation_bank,
    binary_auprc,
    binary_auroc,
    expected_calibration_error,
    threshold_grid_metrics,
    tune,
)
from models.failure_detector import FailureDetector
from trainers.data import MULTITASK_CALIBRATION_SCHEMA, _array_sha256


TASKS = [f"task-{index}-v3" for index in range(10)]
VALIDATION_SEED = 20264010
FINAL_SEED = 20265010
TRAINING_MANIFEST_SHA256 = "a" * 64
TRAINING_RAW_SOURCE_SHA256 = "e" * 64
TRAINING_DATASET_SHA256 = "f" * 64


def _training_calibration(
    vocabulary: list[str] | None = None,
) -> dict[str, object]:
    selected_vocabulary = vocabulary or TASKS
    return {
        "schema_version": MULTITASK_CALIBRATION_SCHEMA,
        "mode": "fit-task-quantile",
        "dataset_role": "training",
        "benchmark": "MT10",
        "task_vocabulary_sha256": _canonical_json_sha256(selected_vocabulary),
        "comparison": "greater_than_or_equal",
        "quantile": 0.9,
        "quantile_method": "linear",
        "prediction_horizon": 2,
        "terminal_positive_horizon": 1,
        "terminal_rule": "OR final N steps when official episode success is false",
        "task_thresholds": [
            {
                "task_id": task_id,
                "task_name": task_name,
                "threshold": 1.5,
                "samples": 4,
            }
            for task_id, task_name in enumerate(selected_vocabulary)
        ],
        "calibration_source": {
            "kind": "training_bank_raw_action_disagreement",
            "sha256": TRAINING_RAW_SOURCE_SHA256,
        },
        "target_raw_disagreement_sha256": TRAINING_RAW_SOURCE_SHA256,
    }


def _write_validation_bank(root: Path) -> dict[str, object]:
    training_calibration = _training_calibration()
    training_calibration_sha256 = _canonical_json_sha256(training_calibration)
    disagreement = np.ones(4, dtype=np.float32)
    raw_source_records = [
        {
            "file": f"task_{task_id:02d}/failure_0000.npz",
            "task_id": task_id,
            "length": 4,
            "success": False,
            "action_disagreement_sha256": _array_sha256(disagreement),
        }
        for task_id in range(len(TASKS))
    ]
    target_raw_sha256 = _canonical_json_sha256(raw_source_records)
    validation_calibration: dict[str, object] = {
        "schema_version": MULTITASK_CALIBRATION_SCHEMA,
        "mode": "frozen-task-thresholds",
        "dataset_role": "validation",
        "benchmark": "MT10",
        "task_vocabulary_sha256": _canonical_json_sha256(TASKS),
        "comparison": "greater_than_or_equal",
        "quantile": 0.9,
        "quantile_method": "linear",
        "prediction_horizon": 2,
        "terminal_positive_horizon": 1,
        "terminal_rule": "OR final N steps when official episode success is false",
        "task_thresholds": training_calibration["task_thresholds"],
        "calibration_source": {
            "kind": "frozen_training_calibration_manifest",
            "sha256": training_calibration_sha256,
            "manifest_sha256": TRAINING_MANIFEST_SHA256,
            "manifest_path": "training/manifest.json",
        },
        "target_raw_disagreement_sha256": target_raw_sha256,
    }
    validation_calibration_sha256 = _canonical_json_sha256(validation_calibration)
    entries: list[dict[str, object]] = []
    per_task: dict[str, object] = {}
    for task_id, task_name in enumerate(TASKS):
        task_dir = root / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / "failure_0000.npz"
        one_hot = np.zeros(len(TASKS), dtype=np.float32)
        one_hot[task_id] = 1.0
        raw = np.zeros((4, 39), dtype=np.float32)
        raw[:, 0] = np.asarray([-1.0, 0.2, 0.6, 1.0], dtype=np.float32)
        states = np.concatenate(
            [raw, np.broadcast_to(one_hot, (len(raw), len(TASKS)))], axis=1
        )
        risk_events = np.asarray([0, 0, 0, 1], dtype=np.bool_)
        labels = np.asarray([0, 1, 1, 1], dtype=np.bool_)
        np.savez_compressed(
            path,
            states=states,
            raw_observations=raw,
            actions=np.zeros((4, 4), dtype=np.float32),
            expert_actions=np.ones((4, 4), dtype=np.float32),
            action_disagreement_l1=disagreement,
            risk_events=risk_events,
            labels=labels,
            rewards=np.zeros(4, dtype=np.float32),
            success=np.asarray(False),
            task_id=np.asarray(task_id, dtype=np.int16),
            task_name=np.asarray(task_name),
            task_variant=np.asarray(0, dtype=np.int16),
            task_payload_sha256=np.asarray("b" * 64),
            episode_seed=np.asarray(VALIDATION_SEED + task_id * 100_000),
            schema_version=np.asarray(SCHEMA_VERSION),
            label_threshold=np.asarray(1.5, dtype=np.float64),
            label_calibration_mode=np.asarray("frozen-task-thresholds"),
            label_calibration_source_sha256=np.asarray(
                training_calibration_sha256
            ),
            label_calibration_fingerprint_sha256=np.asarray(
                validation_calibration_sha256
            ),
            label_prediction_horizon=np.asarray(2, dtype=np.int32),
            label_terminal_positive_horizon=np.asarray(1, dtype=np.int32),
        )
        entries.append(
            {
                "file": path.relative_to(root).as_posix(),
                "task_id": task_id,
                "task_name": task_name,
                "rollout_index": 0,
                "length": 4,
                "success": False,
                "positive_labels": 3,
                "risk_events": 1,
                "label_threshold": 1.5,
                "label_calibration_fingerprint_sha256": (
                    validation_calibration_sha256
                ),
                "sha256": file_sha256(path),
            }
        )
        per_task[task_name] = {
            "episodes": 1,
            "rows": 4,
            "success_rate": 0.0,
            "risk_events": 1,
            "positive_rate": 0.75,
            "label_threshold": 1.5,
        }

    vocabulary_sha256 = _canonical_json_sha256(TASKS)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": "3.test",
        "benchmark_seed": VALIDATION_SEED,
        "collection_seed": VALIDATION_SEED,
        "benchmark_task_bank_sha256": "c" * 64,
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": vocabulary_sha256,
        "act_checkpoint_sha256": "d" * 64,
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
            "terminal_positive_horizon": 1,
        },
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_type": DATASET_TYPE,
        "benchmark": "MT10",
        "metaworld_version": "3.test",
        "benchmark_seed": VALIDATION_SEED,
        "seed": VALIDATION_SEED,
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": vocabulary_sha256,
        "benchmark_task_bank_sha256": "c" * 64,
        "act_checkpoint_sha256": "d" * 64,
        "observation_schema": "raw39_plus_official_task_one_hot",
        "state_dim": 49,
        "action_dim": 4,
        "max_episode_steps": 500,
        "noise_level": 0.2,
        "action_std_scale": 0.4,
        "observation_std_scale": 0.025,
        "action_noise_std": 0.08,
        "observation_noise_std": 0.005,
        "object_position_noise": False,
        "expert_action_disagreement_l1": 0.35,
        "prediction_horizon": 2,
        "terminal_positive_horizon": 1,
        "rollouts_per_task": 1,
        "episodes": 10,
        "rows": 40,
        "positive_rate": 0.75,
        "provenance": provenance,
        "provenance_fingerprint_sha256": _canonical_json_sha256(provenance),
        "label_calibration": validation_calibration,
        "label_calibration_fingerprint_sha256": validation_calibration_sha256,
        "label_calibration_source_sha256": training_calibration_sha256,
        "labeling_strategy": "task_conditional_quantile_calibrated_future_risk",
        "per_task": per_task,
        "files": entries,
        "complete": True,
    }
    manifest["dataset_fingerprint_sha256"] = _canonical_json_sha256(
        {
            "collection_provenance_fingerprint_sha256": manifest[
                "provenance_fingerprint_sha256"
            ],
            "label_calibration_fingerprint_sha256": (
                validation_calibration_sha256
            ),
            "files": [
                {"file": entry["file"], "sha256": entry["sha256"]}
                for entry in entries
            ],
        }
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_checkpoint(path: Path, *, vocabulary: list[str] | None = None) -> None:
    selected_vocabulary = vocabulary or TASKS
    training_calibration = _training_calibration(selected_vocabulary)
    calibration_sha256 = _canonical_json_sha256(training_calibration)
    detector = FailureDetector(
        49,
        hidden_dim=8,
        num_layers=1,
        mlp_hidden=4,
        dropout=0.0,
        sequence_length=3,
    )
    torch.save(
        {
            "model_state_dict": detector.state_dict(),
            "state_dim": 49,
            "hidden_dim": 8,
            "num_layers": 1,
            "mlp_hidden": 4,
            "dropout": 0.0,
            "sequence_length": 3,
            "benchmark": "MT10",
            "task_vocabulary": selected_vocabulary,
            "task_vocabulary_sha256": _canonical_json_sha256(selected_vocabulary),
            "detector_training_schema": "reim-failure-detector-training-v2",
            "data_schema_version": SCHEMA_VERSION,
            "dataset_type": DATASET_TYPE,
            "dataset_role": "training",
            "label_calibration_mode": "fit-task-quantile",
            "label_calibration_quantile": 0.9,
            "label_calibration_fingerprint_sha256": calibration_sha256,
            "label_calibration_source_sha256": TRAINING_RAW_SOURCE_SHA256,
            "dataset_fingerprint_sha256": TRAINING_DATASET_SHA256,
            "label_calibration": training_calibration,
            "data_manifest_sha256": TRAINING_MANIFEST_SHA256,
            "seed": 42,
        },
        path,
    )


def _args(root: Path, checkpoint: Path) -> Namespace:
    return Namespace(
        benchmark="MT10",
        validation_data=str(root),
        detector_checkpoint=str(checkpoint),
        protocol_config=None,
        validation_bank_seed=VALIDATION_SEED,
        validation_benchmark_seed=VALIDATION_SEED,
        thresholds="0.1,0.5,0.9",
        threshold_min=0.01,
        threshold_max=0.99,
        threshold_count=99,
        precision_floor=0.60,
        ece_bins=5,
        batch_size=16,
        seed=42,
        device="cpu",
        output_json=str(root.parent / "threshold.json"),
        output_csv=str(root.parent / "threshold.csv"),
        log_file=None,
    )


def test_ranking_and_calibration_metrics_have_known_values() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8])
    assert binary_auroc(labels, probabilities) == pytest.approx(0.75)
    assert binary_auprc(labels, probabilities) == pytest.approx(5.0 / 6.0)
    assert expected_calibration_error(labels, probabilities, bins=2) == pytest.approx(
        0.0875
    )


def test_threshold_selection_uses_macro_precision_constraint_then_macro_f1() -> None:
    labels = np.asarray([0, 1, 1, 0, 1, 1])
    probabilities = np.asarray([0.2, 0.4, 0.9, 0.3, 0.6, 0.8])
    task_ids = np.asarray([0, 0, 0, 1, 1, 1])
    grid, selection, per_task = threshold_grid_metrics(
        labels,
        probabilities,
        task_ids,
        [0.25, 0.5, 0.75],
        task_count=2,
        precision_floor=0.6,
        ece_bins=5,
    )
    assert len(grid) == 3 and len(per_task) == 2
    assert selection["selection_status"] == "precision_floor_satisfied"
    assert selection["threshold"] == pytest.approx(0.25)
    assert selection["task_macro_precision"] >= 0.6


def test_validation_bank_whitelist_hashes_and_final_seed_are_fail_closed(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_validation_bank(bank)
    manifest, files = audit_validation_bank(
        bank,
        benchmark="MT10",
        expected_validation_seed=VALIDATION_SEED,
        expected_benchmark_seed=VALIDATION_SEED,
        forbidden_final_seed=FINAL_SEED,
    )
    assert manifest["rows"] == 40
    assert len(files) == 10

    stale = bank / "task_00" / "failure_9999.npz"
    shutil.copyfile(files[0], stale)
    with pytest.raises(ValueError, match="(?:whitelist|inventory) mismatch"):
        audit_validation_bank(
            bank,
            benchmark="MT10",
            expected_validation_seed=VALIDATION_SEED,
            expected_benchmark_seed=VALIDATION_SEED,
            forbidden_final_seed=FINAL_SEED,
        )
    stale.unlink()
    files[0].write_bytes(files[0].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit_validation_bank(
            bank,
            benchmark="MT10",
            expected_validation_seed=VALIDATION_SEED,
            expected_benchmark_seed=VALIDATION_SEED,
            forbidden_final_seed=FINAL_SEED,
        )

    final_bank = tmp_path / "final-bank"
    _write_validation_bank(final_bank)
    with pytest.raises(ValueError, match="final-evaluation"):
        audit_validation_bank(
            final_bank,
            benchmark="MT10",
            expected_validation_seed=FINAL_SEED,
            expected_benchmark_seed=FINAL_SEED,
            forbidden_final_seed=FINAL_SEED,
        )


def test_checkpoint_requires_ordered_vocab_and_independent_training_manifest(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    validation_manifest = _write_validation_bank(bank)
    validation_digest = file_sha256(bank / "manifest.json")
    checkpoint = tmp_path / "detector.pt"
    _write_checkpoint(checkpoint, vocabulary=list(reversed(TASKS)))
    with pytest.raises(ValueError, match="vocabulary/order"):
        _validate_detector_checkpoint(
            checkpoint,
            benchmark="MT10",
            task_vocabulary=TASKS,
            validation_manifest=validation_manifest,
            validation_manifest_sha256=validation_digest,
            device="cpu",
        )

    _write_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["data_manifest_sha256"] = validation_digest
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="independent validation bank"):
        _validate_detector_checkpoint(
            checkpoint,
            benchmark="MT10",
            task_vocabulary=TASKS,
            validation_manifest=validation_manifest,
            validation_manifest_sha256=validation_digest,
            device="cpu",
        )


def test_raw_fixed_threshold_bank_and_mismatched_training_calibration_fail_closed(
    tmp_path: Path,
) -> None:
    raw_bank = tmp_path / "raw-bank"
    raw_manifest = _write_validation_bank(raw_bank)
    for key in (
        "label_calibration",
        "label_calibration_fingerprint_sha256",
        "label_calibration_source_sha256",
        "dataset_fingerprint_sha256",
    ):
        raw_manifest.pop(key)
    (raw_bank / "manifest.json").write_text(
        json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks label_calibration"):
        audit_validation_bank(
            raw_bank,
            benchmark="MT10",
            expected_validation_seed=VALIDATION_SEED,
            expected_benchmark_seed=VALIDATION_SEED,
            forbidden_final_seed=FINAL_SEED,
        )

    bank = tmp_path / "frozen-bank"
    validation_manifest = _write_validation_bank(bank)
    checkpoint = tmp_path / "detector.pt"
    _write_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    altered = dict(payload["label_calibration"])
    altered_thresholds = [dict(item) for item in altered["task_thresholds"]]
    altered_thresholds[0]["threshold"] = 1.6
    altered["task_thresholds"] = altered_thresholds
    payload["label_calibration"] = altered
    payload["label_calibration_fingerprint_sha256"] = _canonical_json_sha256(
        altered
    )
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="does not reuse this detector"):
        _validate_detector_checkpoint(
            checkpoint,
            benchmark="MT10",
            task_vocabulary=TASKS,
            validation_manifest=validation_manifest,
            validation_manifest_sha256=file_sha256(bank / "manifest.json"),
            device="cpu",
        )


def test_tuner_writes_reproducible_json_and_csv_without_final_bank(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    checkpoint = tmp_path / "detector.pt"
    _write_validation_bank(bank)
    _write_checkpoint(checkpoint)
    args = _args(bank, checkpoint)

    first = tune(args)
    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    second = tune(args)

    assert first["bank_role"] == "validation_only"
    assert first["final_bank_accessed"] is False
    assert first["dataset"]["task_count"] == 10
    assert first["dataset"]["trajectory_count"] == 10
    assert first["dataset"]["sample_count"] == 40
    assert first["selection"]["task_macro_precision"] >= 0.6
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    assert second == first
