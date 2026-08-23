#!/usr/bin/env python3
"""Deterministically relabel MT10/MT50 failure banks without simulation.

The raw ACT/expert disagreement traces in a failure-dataset v2 bank are an
immutable diagnostic signal.  This tool converts them into useful causal risk
labels using one threshold per task.  A training bank fits those thresholds at
a fixed quantile; validation and final banks are forbidden from fitting and
must load the frozen calibration from the training manifest.

Rewrites are transactional.  A journal stores the original manifest and
immutable-array digests before the first shard changes.  Re-running the same
command resumes safely after interruption and refuses any unexplained payload
change.  The collection provenance fingerprint is preserved while a separate
calibration and full-dataset fingerprint are added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_save_npz, atomic_write_json, file_sha256  # noqa: E402


FAILURE_SCHEMA = "reim-multitask-failures-v2"
FAILURE_DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
CALIBRATION_SCHEMA = "reim-task-conditional-risk-calibration-v1"
TRANSACTION_SCHEMA = "reim-failure-relabel-transaction-v1"
DEFAULT_QUANTILE = 0.90
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
MODES = {"fit-task-quantile", "frozen-task-thresholds"}
DATASET_ROLES = {"training", "validation", "final"}
JOURNAL_NAME = ".relabel_transaction.json"
_SHARD_PATTERN = re.compile(r"^task_(\d{2})/failure_(\d{4,})\.npz$")
_MUTABLE_ARRAYS = {
    "risk_events",
    "labels",
    "label_threshold",
    "label_calibration_mode",
    "label_calibration_source_sha256",
    "label_calibration_fingerprint_sha256",
    "label_prediction_horizon",
    "label_terminal_positive_horizon",
}


def _canonical_sha256(payload: Any) -> str:
    value = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_payload_sha256(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(set(payload) - _MUTABLE_ARRAYS):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_array_sha256(np.asarray(payload[key])).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _safe_relative_path(value: Any) -> str:
    text = str(value)
    path = Path(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != text
        or _SHARD_PATTERN.fullmatch(text) is None
    ):
        raise ValueError(f"Unsafe or non-canonical failure shard path {text!r}")
    return text


def _scalar(payload: Mapping[str, np.ndarray], key: str, *, path: Path) -> Any:
    if key not in payload:
        raise ValueError(f"{path}: missing scalar {key!r}")
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{path}: {key} must be scalar")
    return value.reshape(-1)[0]


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Failure shard must be a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read failure shard {path}: {exc}") from exc
    required = {
        "states",
        "raw_observations",
        "actions",
        "expert_actions",
        "action_disagreement_l1",
        "risk_events",
        "labels",
        "rewards",
        "success",
        "task_id",
        "task_name",
        "task_variant",
        "task_payload_sha256",
        "episode_seed",
        "schema_version",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    length = len(np.asarray(payload["action_disagreement_l1"]))
    shapes = {
        "states": np.asarray(payload["states"]).shape,
        "raw_observations": np.asarray(payload["raw_observations"]).shape,
        "actions": np.asarray(payload["actions"]).shape,
        "expert_actions": np.asarray(payload["expert_actions"]).shape,
        "action_disagreement_l1": np.asarray(payload["action_disagreement_l1"]).shape,
        "risk_events": np.asarray(payload["risk_events"]).shape,
        "labels": np.asarray(payload["labels"]).shape,
        "rewards": np.asarray(payload["rewards"]).shape,
    }
    if length <= 0 or shapes["action_disagreement_l1"] != (length,):
        raise ValueError(f"{path}: empty or invalid disagreement trace")
    if shapes["risk_events"] != (length,) or shapes["labels"] != (length,):
        raise ValueError(f"{path}: risk_events/labels shape mismatch")
    if shapes["rewards"] != (length,):
        raise ValueError(f"{path}: rewards shape mismatch")
    if shapes["raw_observations"] != (length, 39):
        raise ValueError(f"{path}: raw observations are not [T,39]")
    if shapes["actions"] != (length, 4) or shapes["expert_actions"] != (length, 4):
        raise ValueError(f"{path}: actions are not [T,4]")
    if shapes["states"][0] != length or shapes["states"][1] not in (49, 89):
        raise ValueError(f"{path}: states are not MT10/MT50 observations")
    for key in (
        "states",
        "raw_observations",
        "actions",
        "expert_actions",
        "action_disagreement_l1",
        "rewards",
    ):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError(f"{path}: non-finite {key}")
    disagreements = np.asarray(payload["action_disagreement_l1"])
    if np.any(disagreements < 0.0):
        raise ValueError(f"{path}: negative action disagreement")
    if str(_scalar(payload, "schema_version", path=path)) != FAILURE_SCHEMA:
        raise ValueError(f"{path}: unsupported failure shard schema")
    return payload


def _future_labels(events: np.ndarray, horizon: int) -> np.ndarray:
    events = np.asarray(events, dtype=np.bool_)
    if horizon < 0:
        raise ValueError("prediction_horizon must be non-negative")
    # Reverse cumulative window sum is O(T) and bit-for-bit deterministic.
    prefix = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(events, dtype=np.int64)]
    )
    indices = np.arange(len(events), dtype=np.int64)
    ends = np.minimum(len(events), indices + horizon + 1)
    return (prefix[ends] - prefix[indices]) > 0


def _validate_manifest_header(manifest: Mapping[str, Any]) -> tuple[str, list[str]]:
    if manifest.get("schema_version") != FAILURE_SCHEMA:
        raise ValueError("Relabeling requires a failure-dataset v2 manifest")
    if manifest.get("dataset_type") != FAILURE_DATASET_TYPE:
        raise ValueError("Manifest is not a behavioral-deviation failure bank")
    if manifest.get("complete") is not True:
        raise ValueError("Failure bank must be complete before calibration")
    benchmark = str(manifest.get("benchmark", "")).upper()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError("Failure manifest benchmark must be MT10 or MT50")
    vocabulary = manifest.get("task_vocabulary")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != SUPPORTED_BENCHMARKS[benchmark]
        or any(not isinstance(name, str) or not name for name in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("Failure manifest has an invalid ordered task vocabulary")
    if manifest.get("task_vocabulary_sha256") != _canonical_sha256(vocabulary):
        raise ValueError("Failure manifest task vocabulary hash mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Failure manifest lacks collection provenance")
    if manifest.get("provenance_fingerprint_sha256") != _canonical_sha256(provenance):
        raise ValueError("Failure manifest collection provenance fingerprint mismatch")
    return benchmark, list(vocabulary)


def _manifest_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    require_hashes: bool,
) -> list[tuple[dict[str, Any], Path, dict[str, np.ndarray]]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Failure manifest needs a non-empty files whitelist")
    expected: list[str] = []
    entries: list[tuple[dict[str, Any], Path, dict[str, np.ndarray]]] = []
    seen_slots: set[tuple[int, int]] = set()
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Manifest files[{index}] is not an object")
        relative = _safe_relative_path(raw_entry.get("file"))
        if relative in expected:
            raise ValueError(f"Duplicate failure manifest file {relative}")
        path = directory / relative
        if require_hashes and raw_entry.get("sha256") != file_sha256(path):
            raise ValueError(f"Failure shard SHA256 mismatch: {relative}")
        payload = _load_shard(path)
        task_id = int(_scalar(payload, "task_id", path=path))
        task_name = str(_scalar(payload, "task_name", path=path))
        match = _SHARD_PATTERN.fullmatch(relative)
        assert match is not None
        path_task_id = int(match.group(1))
        rollout_index = int(match.group(2))
        if not 0 <= task_id < len(manifest["task_vocabulary"]):
            raise ValueError(f"{path}: invalid task_id {task_id}")
        if task_id != path_task_id or manifest["task_vocabulary"][task_id] != task_name:
            raise ValueError(f"{path}: path/task metadata disagree with task vocabulary")
        slot = (task_id, rollout_index)
        if slot in seen_slots:
            raise ValueError(f"Duplicate task/rollout slot {slot}")
        seen_slots.add(slot)
        expected_metadata = {
            "task_id": task_id,
            "task_name": task_name,
            "rollout_index": rollout_index,
            "length": len(payload["labels"]),
            "success": bool(_scalar(payload, "success", path=path)),
        }
        for key, value in expected_metadata.items():
            if raw_entry.get(key) != value:
                raise ValueError(
                    f"{path}: manifest {key}={raw_entry.get(key)!r}, actual={value!r}"
                )
        expected.append(relative)
        entries.append((dict(raw_entry), path, payload))
    disk = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.glob("task_*/failure_*.npz")
        if path.is_file() or path.is_symlink()
    )
    if sorted(expected) != disk:
        missing = sorted(set(expected) - set(disk))
        stale = sorted(set(disk) - set(expected))
        raise ValueError(
            f"Failure bank whitelist mismatch: missing={missing[:5]}, stale={stale[:5]}"
        )
    ordered = sorted(
        entries,
        key=lambda item: (
            int(_scalar(item[2], "task_id", path=item[1])),
            int(_SHARD_PATTERN.fullmatch(item[0]["file"]).group(2)),  # type: ignore[union-attr]
        ),
    )
    if [entry[0]["file"] for entry in entries] != [entry[0]["file"] for entry in ordered]:
        raise ValueError("Failure manifest whitelist is not in canonical task/rollout order")
    if int(manifest.get("episodes", -1)) != len(entries):
        raise ValueError("Failure manifest episode count is inconsistent")
    if int(manifest.get("rows", -1)) != sum(len(payload["labels"]) for _, _, payload in entries):
        raise ValueError("Failure manifest row count is inconsistent")
    return entries


def _raw_source_sha256(
    inventory: Sequence[tuple[dict[str, Any], Path, dict[str, np.ndarray]]]
) -> str:
    return _canonical_sha256(
        [
            {
                "file": entry["file"],
                "task_id": int(_scalar(payload, "task_id", path=path)),
                "length": len(payload["action_disagreement_l1"]),
                "success": bool(_scalar(payload, "success", path=path)),
                "action_disagreement_sha256": _array_sha256(
                    np.asarray(payload["action_disagreement_l1"], dtype=np.float32)
                ),
            }
            for entry, path, payload in inventory
        ]
    )


def _fit_thresholds(
    inventory: Sequence[tuple[dict[str, Any], Path, dict[str, np.ndarray]]],
    vocabulary: Sequence[str],
    quantile: float,
) -> list[dict[str, Any]]:
    thresholds: list[dict[str, Any]] = []
    for task_id, task_name in enumerate(vocabulary):
        values = np.concatenate(
            [
                np.asarray(payload["action_disagreement_l1"], dtype=np.float64)
                for _, path, payload in inventory
                if int(_scalar(payload, "task_id", path=path)) == task_id
            ]
        )
        if len(values) == 0:
            raise ValueError(f"No calibration disagreements for task {task_name}")
        threshold = float(np.quantile(values, quantile, method="linear"))
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid fitted threshold for task {task_name}")
        thresholds.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "threshold": threshold,
                "samples": int(len(values)),
            }
        )
    return thresholds


def _validate_calibration(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    calibration = manifest.get("label_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Calibration source manifest lacks label_calibration")
    fingerprint = manifest.get("label_calibration_fingerprint_sha256")
    if fingerprint != _canonical_sha256(calibration):
        raise ValueError("Calibration source fingerprint mismatch")
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("Unsupported calibration source schema")
    if calibration.get("mode") != "fit-task-quantile":
        raise ValueError("Frozen thresholds must come from a fitted training bank")
    if calibration.get("dataset_role") != "training":
        raise ValueError("Frozen thresholds must come from dataset_role=training")
    return calibration


def _training_source_calibration(
    path: Path,
    *,
    benchmark: str,
    vocabulary: Sequence[str],
) -> tuple[dict[str, Any], str, str]:
    manifest = _read_json(path, description="training calibration manifest")
    source_benchmark, source_vocabulary = _validate_manifest_header(manifest)
    if source_benchmark != benchmark or source_vocabulary != list(vocabulary):
        raise ValueError("Training calibration benchmark/task vocabulary mismatch")
    calibration = dict(_validate_calibration(manifest))
    source_dir = path.parent.resolve()
    _manifest_inventory(source_dir, manifest, require_hashes=True)
    expected_dataset_fingerprint = _dataset_fingerprint(manifest)
    if manifest.get("dataset_fingerprint_sha256") != expected_dataset_fingerprint:
        raise ValueError("Training calibration dataset fingerprint mismatch")
    return calibration, file_sha256(path), str(
        manifest["label_calibration_fingerprint_sha256"]
    )


def _build_calibration(
    *,
    mode: str,
    role: str,
    quantile: float,
    benchmark: str,
    vocabulary: Sequence[str],
    inventory: Sequence[tuple[dict[str, Any], Path, dict[str, np.ndarray]]],
    prediction_horizon: int,
    terminal_positive_horizon: int,
    calibration_manifest: Path | None,
) -> dict[str, Any]:
    vocabulary_sha256 = _canonical_sha256(list(vocabulary))
    raw_source_sha256 = _raw_source_sha256(inventory)
    if mode == "fit-task-quantile":
        if role != "training":
            raise ValueError(
                "Validation/final banks cannot fit thresholds; use frozen-task-thresholds"
            )
        thresholds = _fit_thresholds(inventory, vocabulary, quantile)
        source = {
            "kind": "training_bank_raw_action_disagreement",
            "sha256": raw_source_sha256,
        }
        fitted_quantile: float | None = quantile
        quantile_method: str | None = "linear"
    else:
        if role == "training":
            raise ValueError("Training banks must fit task thresholds, not import them")
        if calibration_manifest is None:
            raise ValueError(
                "Validation/final frozen calibration requires --calibration-manifest"
            )
        source_calibration, source_manifest_sha, source_fingerprint = (
            _training_source_calibration(
                calibration_manifest,
                benchmark=benchmark,
                vocabulary=vocabulary,
            )
        )
        thresholds = [dict(item) for item in source_calibration["task_thresholds"]]
        source = {
            "kind": "frozen_training_calibration_manifest",
            "sha256": source_fingerprint,
            "manifest_sha256": source_manifest_sha,
            "manifest_path": str(calibration_manifest),
        }
        fitted_quantile = float(source_calibration["quantile"])
        quantile_method = str(source_calibration["quantile_method"])
    if len(thresholds) != len(vocabulary):
        raise ValueError("Calibration does not contain one threshold per task")
    for task_id, (task_name, item) in enumerate(zip(vocabulary, thresholds, strict=True)):
        if int(item.get("task_id", -1)) != task_id or item.get("task_name") != task_name:
            raise ValueError("Calibration task thresholds are not in official order")
        threshold = float(item.get("threshold", float("nan")))
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Calibration has invalid threshold for {task_name}")
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "mode": mode,
        "dataset_role": role,
        "benchmark": benchmark,
        "task_vocabulary_sha256": vocabulary_sha256,
        "comparison": "greater_than_or_equal",
        "quantile": fitted_quantile,
        "quantile_method": quantile_method,
        "prediction_horizon": prediction_horizon,
        "terminal_positive_horizon": terminal_positive_horizon,
        "terminal_rule": "OR final N steps when official episode success is false",
        "task_thresholds": thresholds,
        "calibration_source": source,
        "target_raw_disagreement_sha256": raw_source_sha256,
    }


def _desired_arrays(
    payload: Mapping[str, np.ndarray],
    *,
    threshold: float,
    calibration: Mapping[str, Any],
    calibration_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    disagreements = np.asarray(payload["action_disagreement_l1"], dtype=np.float32)
    events = disagreements >= np.float32(threshold)
    success = bool(np.asarray(payload["success"]).reshape(-1)[0])
    terminal_horizon = int(calibration["terminal_positive_horizon"])
    if not success and terminal_horizon > 0:
        events[max(0, len(events) - terminal_horizon) :] = True
    labels = _future_labels(events, int(calibration["prediction_horizon"]))
    metadata = {
        "label_threshold": np.asarray(threshold, dtype=np.float64),
        "label_calibration_mode": np.asarray(str(calibration["mode"])),
        "label_calibration_source_sha256": np.asarray(
            str(calibration["calibration_source"]["sha256"])
        ),
        "label_calibration_fingerprint_sha256": np.asarray(calibration_fingerprint),
        "label_prediction_horizon": np.asarray(
            calibration["prediction_horizon"], dtype=np.int32
        ),
        "label_terminal_positive_horizon": np.asarray(
            calibration["terminal_positive_horizon"], dtype=np.int32
        ),
    }
    return events.astype(np.bool_), labels.astype(np.bool_), metadata


def _already_desired(
    payload: Mapping[str, np.ndarray],
    events: np.ndarray,
    labels: np.ndarray,
    metadata: Mapping[str, np.ndarray],
) -> bool:
    if not np.array_equal(np.asarray(payload["risk_events"], dtype=np.bool_), events):
        return False
    if not np.array_equal(np.asarray(payload["labels"], dtype=np.bool_), labels):
        return False
    for key, expected in metadata.items():
        if key not in payload or not np.array_equal(np.asarray(payload[key]), expected):
            return False
    return True


def _dataset_fingerprint(manifest: Mapping[str, Any]) -> str:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Cannot fingerprint manifest without files")
    return _canonical_sha256(
        {
            "collection_provenance_fingerprint_sha256": manifest.get(
                "provenance_fingerprint_sha256"
            ),
            "label_calibration_fingerprint_sha256": manifest.get(
                "label_calibration_fingerprint_sha256"
            ),
            "files": [
                {"file": entry.get("file"), "sha256": entry.get("sha256")}
                for entry in files
            ],
        }
    )


def _final_manifest(
    source: Mapping[str, Any],
    *,
    inventory: Sequence[tuple[dict[str, Any], Path, dict[str, np.ndarray]]],
    output_dir: Path,
    calibration: Mapping[str, Any],
    calibration_fingerprint: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    manifest = dict(source)
    manifest["prediction_horizon"] = int(calibration["prediction_horizon"])
    manifest["terminal_positive_horizon"] = int(calibration["terminal_positive_horizon"])
    thresholds = {
        int(item["task_id"]): float(item["threshold"])
        for item in calibration["task_thresholds"]
    }
    new_entries: list[dict[str, Any]] = []
    per_task_accumulator: dict[int, dict[str, int]] = {
        index: {"rows": 0, "positive": 0, "risk_events": 0, "episodes": 0}
        for index in range(len(source["task_vocabulary"]))
    }
    for old_entry, source_path, _ in inventory:
        destination = output_dir / old_entry["file"]
        payload = _load_shard(destination)
        task_id = int(_scalar(payload, "task_id", path=destination))
        events, labels, metadata = _desired_arrays(
            payload,
            threshold=thresholds[task_id],
            calibration=calibration,
            calibration_fingerprint=calibration_fingerprint,
        )
        if not _already_desired(payload, events, labels, metadata):
            raise RuntimeError(f"Relabeled shard failed final verification: {destination}")
        entry = dict(old_entry)
        entry.update(
            {
                "positive_labels": int(labels.sum()),
                "risk_events": int(events.sum()),
                "label_threshold": thresholds[task_id],
                "label_calibration_fingerprint_sha256": calibration_fingerprint,
                "sha256": file_sha256(destination),
            }
        )
        new_entries.append(entry)
        accumulator = per_task_accumulator[task_id]
        accumulator["rows"] += len(labels)
        accumulator["positive"] += int(labels.sum())
        accumulator["risk_events"] += int(events.sum())
        accumulator["episodes"] += 1

    total_rows = sum(item["rows"] for item in per_task_accumulator.values())
    total_positive = sum(item["positive"] for item in per_task_accumulator.values())
    manifest["files"] = new_entries
    manifest["rows"] = total_rows
    manifest["positive_rate"] = total_positive / max(total_rows, 1)
    manifest["label_calibration"] = dict(calibration)
    manifest["label_calibration_fingerprint_sha256"] = calibration_fingerprint
    manifest["label_calibration_source_sha256"] = calibration["calibration_source"][
        "sha256"
    ]
    manifest["labeling_strategy"] = "task_conditional_quantile_calibrated_future_risk"
    manifest["calibration_input_manifest_sha256"] = input_manifest_sha256
    previous = source.get("label_calibration_fingerprint_sha256")
    history = list(source.get("label_calibration_history", []))
    if previous is not None and previous != calibration_fingerprint:
        history.append(
            {
                "label_calibration_fingerprint_sha256": previous,
                "positive_rate": source.get("positive_rate"),
            }
        )
    manifest["label_calibration_history"] = history
    per_task = dict(source.get("per_task", {}))
    threshold_by_id = {
        int(item["task_id"]): float(item["threshold"])
        for item in calibration["task_thresholds"]
    }
    for task_id, task_name in enumerate(source["task_vocabulary"]):
        values = per_task_accumulator[task_id]
        old = dict(per_task.get(task_name, {}))
        old.update(
            {
                "episodes": values["episodes"],
                "rows": values["rows"],
                "risk_events": values["risk_events"],
                "positive_rate": values["positive"] / max(values["rows"], 1),
                "label_threshold": threshold_by_id[task_id],
            }
        )
        per_task[task_name] = old
    manifest["per_task"] = per_task
    manifest["dataset_fingerprint_sha256"] = _dataset_fingerprint(manifest)
    manifest["complete"] = True
    return manifest


def relabel_failure_bank(
    data_dir: str | Path,
    *,
    mode: str = "fit-task-quantile",
    quantile: float = DEFAULT_QUANTILE,
    dataset_role: str = "training",
    prediction_horizon: int | None = None,
    terminal_positive_horizon: int | None = None,
    calibration_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
    _fail_after_shards: int | None = None,
) -> dict[str, Any]:
    """Fit/reuse task thresholds and transactionally rewrite risk labels."""

    mode = str(mode)
    role = str(dataset_role)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if role not in DATASET_ROLES:
        raise ValueError(f"dataset_role must be one of {sorted(DATASET_ROLES)}")
    quantile = float(quantile)
    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    if role != "training" and mode != "frozen-task-thresholds":
        raise ValueError("Validation/final banks may only use frozen-task-thresholds")
    if role == "training" and mode != "fit-task-quantile":
        raise ValueError("Training banks must use fit-task-quantile")

    source_dir = Path(data_dir).expanduser().resolve()
    destination_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else source_dir
    )
    source_manifest_path = source_dir / "manifest.json"
    destination_manifest_path = destination_dir / "manifest.json"
    journal_path = destination_dir / JOURNAL_NAME
    calibration_path = (
        Path(calibration_manifest).expanduser().resolve()
        if calibration_manifest is not None
        else None
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "source_dir": str(source_dir),
        "destination_dir": str(destination_dir),
        "mode": mode,
        "quantile": quantile,
        "dataset_role": role,
        "prediction_horizon": prediction_horizon,
        "terminal_positive_horizon": terminal_positive_horizon,
        "calibration_manifest": str(calibration_path) if calibration_path else None,
    }
    request_sha256 = _canonical_sha256(request)
    journal: dict[str, Any] | None = None
    if journal_path.exists():
        journal = _read_json(journal_path, description="relabel transaction journal")
        if journal.get("schema_version") != TRANSACTION_SCHEMA:
            raise ValueError("Unsupported relabel transaction journal schema")
        if journal.get("request_sha256") != request_sha256:
            raise ValueError("Existing relabel transaction belongs to another request")
        source_manifest = journal.get("source_manifest")
        if not isinstance(source_manifest, dict):
            raise ValueError("Relabel transaction lacks its source manifest snapshot")
        source_manifest_sha256 = str(journal.get("source_manifest_sha256", ""))
    else:
        if (
            destination_dir != source_dir
            and any(destination_dir.iterdir())
            and not destination_manifest_path.is_file()
        ):
            raise FileExistsError(
                f"Relabel output directory is not empty: {destination_dir}"
            )
        source_manifest = _read_json(
            source_manifest_path, description="failure source manifest"
        )
        source_manifest_sha256 = file_sha256(source_manifest_path)

    benchmark, vocabulary = _validate_manifest_header(source_manifest)
    source_inventory = _manifest_inventory(
        source_dir, source_manifest, require_hashes=journal is None
    )
    if prediction_horizon is None:
        prediction_horizon = int(source_manifest.get("prediction_horizon", 10))
    if terminal_positive_horizon is None:
        terminal_positive_horizon = int(
            source_manifest.get("terminal_positive_horizon", 25)
        )
    if prediction_horizon < 0 or terminal_positive_horizon < 0:
        raise ValueError("Label horizons must be non-negative")
    # Defaults resolved above are part of the exact resumable request.
    resolved_request = {
        **request,
        "prediction_horizon": prediction_horizon,
        "terminal_positive_horizon": terminal_positive_horizon,
    }
    resolved_request_sha256 = _canonical_sha256(resolved_request)
    if journal is not None and journal.get("resolved_request_sha256") != resolved_request_sha256:
        raise ValueError("Relabel transaction resolved parameters changed")

    calibration = _build_calibration(
        mode=mode,
        role=role,
        quantile=quantile,
        benchmark=benchmark,
        vocabulary=vocabulary,
        inventory=source_inventory,
        prediction_horizon=prediction_horizon,
        terminal_positive_horizon=terminal_positive_horizon,
        calibration_manifest=calibration_path,
    )
    calibration_fingerprint = _canonical_sha256(calibration)

    # Idempotent fast path still audits every hash and desired label.
    if journal is None and destination_manifest_path.is_file():
        current = _read_json(destination_manifest_path, description="failure manifest")
        if current.get("label_calibration_fingerprint_sha256") == calibration_fingerprint:
            current_inventory = _manifest_inventory(
                destination_dir, current, require_hashes=True
            )
            thresholds = {
                int(item["task_id"]): float(item["threshold"])
                for item in calibration["task_thresholds"]
            }
            for _, path, payload in current_inventory:
                task_id = int(_scalar(payload, "task_id", path=path))
                events, labels, metadata = _desired_arrays(
                    payload,
                    threshold=thresholds[task_id],
                    calibration=calibration,
                    calibration_fingerprint=calibration_fingerprint,
                )
                if not _already_desired(payload, events, labels, metadata):
                    raise ValueError(f"Calibrated shard labels are inconsistent: {path}")
            if current.get("dataset_fingerprint_sha256") != _dataset_fingerprint(current):
                raise ValueError("Calibrated dataset fingerprint mismatch")
            return current
        if destination_dir != source_dir:
            raise FileExistsError(
                "Relabel output already contains a manifest for a different "
                f"calibration request: {destination_manifest_path}"
            )

    immutable_records = {
        entry["file"]: {
            "original_sha256": entry["sha256"],
            "immutable_payload_sha256": _immutable_payload_sha256(payload),
            "action_disagreement_sha256": _array_sha256(
                np.asarray(payload["action_disagreement_l1"], dtype=np.float32)
            ),
        }
        for entry, _, payload in source_inventory
    }
    if journal is None:
        journal = {
            "schema_version": TRANSACTION_SCHEMA,
            "request_sha256": request_sha256,
            "resolved_request_sha256": resolved_request_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest": source_manifest,
            "calibration": calibration,
            "calibration_fingerprint_sha256": calibration_fingerprint,
            "immutable_records": immutable_records,
        }
        atomic_write_json(journal_path, journal)
    else:
        if journal.get("calibration_fingerprint_sha256") != calibration_fingerprint:
            raise ValueError("Relabel transaction calibration changed")
        if journal.get("calibration") != calibration:
            raise ValueError("Relabel transaction calibration payload changed")
        if journal.get("immutable_records") != immutable_records:
            # During an in-place resume, source_inventory reads modified shards;
            # immutable digests remain equal but old file hashes in entries do not.
            stored_records = journal.get("immutable_records")
            if not isinstance(stored_records, Mapping):
                raise ValueError("Relabel transaction immutable records are missing")
            for relative, current_record in immutable_records.items():
                stored = stored_records.get(relative)
                if not isinstance(stored, Mapping):
                    raise ValueError(f"Relabel transaction lacks {relative}")
                for key in ("immutable_payload_sha256", "action_disagreement_sha256"):
                    if stored.get(key) != current_record[key]:
                        raise ValueError(f"Immutable failure payload changed: {relative}")
            immutable_records = dict(stored_records)

    thresholds = {
        int(item["task_id"]): float(item["threshold"])
        for item in calibration["task_thresholds"]
    }
    rewritten = 0
    for entry, source_path, source_payload in source_inventory:
        relative = entry["file"]
        destination = destination_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _load_shard(destination) if destination.is_file() else source_payload
        record = immutable_records[relative]
        if _immutable_payload_sha256(payload) != record["immutable_payload_sha256"]:
            raise ValueError(f"Immutable failure payload changed: {relative}")
        if _array_sha256(
            np.asarray(payload["action_disagreement_l1"], dtype=np.float32)
        ) != record["action_disagreement_sha256"]:
            raise ValueError(f"Raw disagreement trace changed: {relative}")
        task_id = int(_scalar(payload, "task_id", path=destination))
        events, labels, metadata = _desired_arrays(
            payload,
            threshold=thresholds[task_id],
            calibration=calibration,
            calibration_fingerprint=calibration_fingerprint,
        )
        if not _already_desired(payload, events, labels, metadata):
            rewritten_payload = dict(payload)
            rewritten_payload["risk_events"] = events
            rewritten_payload["labels"] = labels
            rewritten_payload.update(metadata)
            atomic_save_npz(destination, **rewritten_payload)
            rewritten += 1
            if _fail_after_shards is not None and rewritten >= _fail_after_shards:
                raise RuntimeError("Injected interruption after shard rewrite")

    manifest = _final_manifest(
        source_manifest,
        inventory=source_inventory,
        output_dir=destination_dir,
        calibration=calibration,
        calibration_fingerprint=calibration_fingerprint,
        input_manifest_sha256=source_manifest_sha256,
    )
    atomic_write_json(destination_manifest_path, manifest)
    if journal_path.exists():
        journal_path.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=tuple(sorted(MODES)), default="fit-task-quantile")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--dataset-role", choices=tuple(sorted(DATASET_ROLES)), default="training")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--prediction-horizon", type=int)
    parser.add_argument("--terminal-positive-horizon", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = relabel_failure_bank(
        args.data_dir,
        mode=args.mode,
        quantile=args.quantile,
        dataset_role=args.dataset_role,
        prediction_horizon=args.prediction_horizon,
        terminal_positive_horizon=args.terminal_positive_horizon,
        calibration_manifest=args.calibration_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "benchmark": manifest["benchmark"],
                "dataset_role": manifest["label_calibration"]["dataset_role"],
                "quantile": manifest["label_calibration"]["quantile"],
                "positive_rate": manifest["positive_rate"],
                "label_calibration_fingerprint_sha256": manifest[
                    "label_calibration_fingerprint_sha256"
                ],
                "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
