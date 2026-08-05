"""Robust NPZ readers shared by the supervised REIM trainers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from data.io import file_sha256


STATE_KEYS = ("states", "observations", "obs", "state")
ACTION_KEYS = ("actions", "action", "expert_actions")
LABEL_KEYS = ("labels", "failure_labels", "failure_label", "failures", "failure", "y")
WINDOW_KEYS = ("windows", "state_sequences", "sequences", "history_states")
LENGTH_KEYS = ("valid_lengths", "lengths", "sequence_lengths")
MULTITASK_FAILURE_SCHEMA = "reim-multitask-failures-v2"
MULTITASK_FAILURE_DATASET_TYPE = "task_conditioned_behavioral_deviation_risk"
MULTITASK_CALIBRATION_SCHEMA = "reim-task-conditional-risk-calibration-v1"
MULTITASK_BENCHMARKS = {"MT10": 10, "MT50": 50}
_MULTITASK_SHARD_PATTERN = re.compile(r"^task_(\d{2})/failure_(\d{4,})\.npz$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DemonstrationData:
    states: np.ndarray
    actions: np.ndarray
    groups: np.ndarray
    files: tuple[Path, ...]


@dataclass(frozen=True)
class FailureData:
    windows: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    groups: np.ndarray
    files: tuple[Path, ...]


@dataclass(frozen=True)
class AuditedMultitaskFailureBank:
    """Cryptographically and semantically verified calibrated failure bank."""

    directory: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    whitelist: tuple[Path, ...]
    benchmark: str
    task_vocabulary: tuple[str, ...]
    calibration: dict[str, Any]
    calibration_fingerprint_sha256: str
    dataset_fingerprint_sha256: str


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return digest


def _future_risk_labels(events: np.ndarray, horizon: int) -> np.ndarray:
    events = np.asarray(events, dtype=np.bool_)
    prefix = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(events, dtype=np.int64)]
    )
    starts = np.arange(len(events), dtype=np.int64)
    stops = np.minimum(len(events), starts + horizon + 1)
    return (prefix[stops] - prefix[starts]) > 0


def _manifest_scalar(
    archive: np.lib.npyio.NpzFile, key: str, *, path: Path
) -> Any:
    if key not in archive.files:
        raise ValueError(f"{path}: missing calibrated scalar {key!r}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{path}: {key} must be scalar")
    return value.reshape(-1)[0]


def _safe_multitask_shard(value: Any) -> str:
    text = str(value)
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != text
        or _MULTITASK_SHARD_PATTERN.fullmatch(text) is None
    ):
        raise ValueError(f"Unsafe calibrated failure shard path {text!r}")
    return text


def is_multitask_failure_manifest(manifest: Mapping[str, Any]) -> bool:
    """Return true when a manifest claims any MT10/MT50 representation."""

    benchmark = str(manifest.get("benchmark", "")).upper()
    vocabulary = manifest.get("task_vocabulary")
    state_dim = manifest.get("state_dim")
    return (
        benchmark in MULTITASK_BENCHMARKS
        or isinstance(vocabulary, list)
        and len(vocabulary) in set(MULTITASK_BENCHMARKS.values())
        or state_dim in (49, 89)
    )


def audit_calibrated_multitask_failure_bank(
    path: str | Path,
    *,
    expected_role: str,
    expected_mode: str,
) -> AuditedMultitaskFailureBank:
    """Fail closed unless an MT10/MT50 calibrated bank is fully self-consistent.

    This validates both cryptographic inventory and labeling semantics.  Raw
    fixed-threshold v2 banks deliberately fail because they do not carry the
    required calibration block and per-shard calibration metadata.
    """

    directory = Path(path).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"Calibrated multi-task bank requires a regular manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse calibrated manifest {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Calibrated failure manifest must be a JSON object")
    if manifest.get("schema_version") != MULTITASK_FAILURE_SCHEMA:
        raise ValueError("Multi-task detector requires failure-dataset schema v2")
    if manifest.get("dataset_type") != MULTITASK_FAILURE_DATASET_TYPE:
        raise ValueError("Multi-task detector received the wrong dataset_type")
    if manifest.get("complete") is not True:
        raise ValueError("Calibrated multi-task failure bank is not complete")
    benchmark = str(manifest.get("benchmark", "")).upper()
    if benchmark not in MULTITASK_BENCHMARKS:
        raise ValueError("Calibrated failure benchmark must be MT10 or MT50")
    task_count = MULTITASK_BENCHMARKS[benchmark]
    task_vocabulary = manifest.get("task_vocabulary")
    if (
        not isinstance(task_vocabulary, list)
        or len(task_vocabulary) != task_count
        or any(not isinstance(name, str) or not name for name in task_vocabulary)
        or len(set(task_vocabulary)) != task_count
    ):
        raise ValueError("Calibrated manifest has an invalid ordered task vocabulary")
    vocabulary_fingerprint = _canonical_json_sha256(task_vocabulary)
    if manifest.get("task_vocabulary_sha256") != vocabulary_fingerprint:
        raise ValueError("Calibrated task vocabulary fingerprint mismatch")
    if int(manifest.get("state_dim", -1)) != 39 + task_count:
        raise ValueError("Calibrated manifest state_dim is not raw39 + task one-hot")
    if int(manifest.get("action_dim", -1)) != 4:
        raise ValueError("Calibrated manifest action_dim is not four")

    collection_provenance = manifest.get("provenance")
    if not isinstance(collection_provenance, Mapping):
        raise ValueError("Calibrated manifest lacks collection provenance")
    collection_fingerprint = _require_sha256(
        manifest.get("provenance_fingerprint_sha256"),
        field="provenance_fingerprint_sha256",
    )
    if collection_fingerprint != _canonical_json_sha256(collection_provenance):
        raise ValueError("Collection provenance fingerprint mismatch")
    if (
        str(collection_provenance.get("benchmark", "")).upper() != benchmark
        or collection_provenance.get("task_vocabulary") != task_vocabulary
        or collection_provenance.get("task_vocabulary_sha256")
        != vocabulary_fingerprint
    ):
        raise ValueError("Collection and calibrated task provenance disagree")

    calibration = manifest.get("label_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Multi-task bank lacks label_calibration")
    if calibration.get("schema_version") != MULTITASK_CALIBRATION_SCHEMA:
        raise ValueError("Unsupported task-conditional calibration schema")
    if calibration.get("mode") != expected_mode:
        raise ValueError(
            f"Calibration mode must be {expected_mode}, got {calibration.get('mode')!r}"
        )
    if calibration.get("dataset_role") != expected_role:
        raise ValueError(
            f"Calibration dataset_role must be {expected_role}, "
            f"got {calibration.get('dataset_role')!r}"
        )
    if (
        str(calibration.get("benchmark", "")).upper() != benchmark
        or calibration.get("task_vocabulary_sha256") != vocabulary_fingerprint
    ):
        raise ValueError("Calibration benchmark/task vocabulary mismatch")
    quantile = float(calibration.get("quantile", float("nan")))
    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("Calibration quantile q must lie strictly in (0,1)")
    if calibration.get("quantile_method") != "linear":
        raise ValueError("Calibration quantile_method must be linear")
    if calibration.get("comparison") != "greater_than_or_equal":
        raise ValueError("Calibration comparison must be greater_than_or_equal")
    try:
        prediction_horizon = int(calibration["prediction_horizon"])
        terminal_horizon = int(calibration["terminal_positive_horizon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Calibration label horizons are missing") from exc
    if prediction_horizon < 0 or terminal_horizon < 0:
        raise ValueError("Calibration label horizons must be non-negative")
    if (
        int(manifest.get("prediction_horizon", -1)) != prediction_horizon
        or int(manifest.get("terminal_positive_horizon", -1)) != terminal_horizon
    ):
        raise ValueError("Calibration horizons disagree with manifest diagnostics")
    thresholds = calibration.get("task_thresholds")
    if not isinstance(thresholds, list) or len(thresholds) != task_count:
        raise ValueError("Calibration must contain one task threshold per task")
    threshold_values: list[float] = []
    threshold_samples: list[int | None] = []
    for task_id, (task_name, item) in enumerate(
        zip(task_vocabulary, thresholds, strict=True)
    ):
        if not isinstance(item, Mapping):
            raise ValueError("Calibration task threshold must be an object")
        if item.get("task_id") != task_id or item.get("task_name") != task_name:
            raise ValueError("Calibration task thresholds are not in official order")
        threshold = float(item.get("threshold", float("nan")))
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid calibration threshold for {task_name}")
        threshold_values.append(threshold)
        samples = item.get("samples")
        threshold_samples.append(int(samples) if samples is not None else None)
    calibration_fingerprint = _require_sha256(
        manifest.get("label_calibration_fingerprint_sha256"),
        field="label_calibration_fingerprint_sha256",
    )
    if calibration_fingerprint != _canonical_json_sha256(calibration):
        raise ValueError("Label calibration fingerprint mismatch")
    calibration_source = calibration.get("calibration_source")
    if not isinstance(calibration_source, Mapping):
        raise ValueError("Calibration source metadata is missing")
    source_fingerprint = _require_sha256(
        calibration_source.get("sha256"), field="calibration_source.sha256"
    )
    if manifest.get("label_calibration_source_sha256") != source_fingerprint:
        raise ValueError("Calibration source fingerprint mismatch")
    if expected_mode == "fit-task-quantile" and calibration_source.get("kind") != (
        "training_bank_raw_action_disagreement"
    ):
        raise ValueError("Training calibration source must be raw action disagreement")
    if expected_mode == "frozen-task-thresholds" and calibration_source.get(
        "kind"
    ) != "frozen_training_calibration_manifest":
        raise ValueError("Validation calibration must reference a frozen training manifest")
    if expected_mode == "frozen-task-thresholds":
        _require_sha256(
            calibration_source.get("manifest_sha256"),
            field="calibration_source.manifest_sha256",
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Calibrated manifest needs a non-empty files whitelist")
    try:
        rollouts_per_task = int(manifest["rollouts_per_task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Calibrated rollouts_per_task is invalid") from exc
    if rollouts_per_task <= 0 or len(files) != task_count * rollouts_per_task:
        raise ValueError("Calibrated bank is not task balanced")

    whitelist: list[Path] = []
    raw_source_records: list[dict[str, Any]] = []
    task_rows = np.zeros(task_count, dtype=np.int64)
    task_positives = np.zeros(task_count, dtype=np.int64)
    total_rows = 0
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Calibrated files[{index}] is not an object")
        relative = _safe_multitask_shard(entry.get("file"))
        match = _MULTITASK_SHARD_PATTERN.fullmatch(relative)
        assert match is not None
        path_task_id = int(match.group(1))
        rollout_index = int(match.group(2))
        expected_task_id = index // rollouts_per_task
        expected_rollout = index % rollouts_per_task
        if path_task_id != expected_task_id or rollout_index != expected_rollout:
            raise ValueError("Calibrated files are not in canonical task/rollout order")
        path_value = directory / relative
        if path_value.is_symlink() or not path_value.is_file():
            raise ValueError(f"Calibrated shard must be a regular file: {path_value}")
        actual_digest = file_sha256(path_value)
        if entry.get("sha256") != actual_digest:
            raise ValueError(f"Calibrated shard SHA256 mismatch: {relative}")
        try:
            with np.load(path_value, allow_pickle=False) as archive:
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
                    "schema_version",
                    "label_threshold",
                    "label_calibration_mode",
                    "label_calibration_source_sha256",
                    "label_calibration_fingerprint_sha256",
                    "label_prediction_horizon",
                    "label_terminal_positive_horizon",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"{path_value}: missing calibrated arrays {sorted(missing)}"
                    )
                states = np.asarray(archive["states"], dtype=np.float32)
                raw = np.asarray(archive["raw_observations"], dtype=np.float32)
                actions = np.asarray(archive["actions"], dtype=np.float32)
                expert_actions = np.asarray(archive["expert_actions"], dtype=np.float32)
                disagreements = np.asarray(
                    archive["action_disagreement_l1"], dtype=np.float32
                )
                events = np.asarray(archive["risk_events"], dtype=np.bool_)
                labels = np.asarray(archive["labels"], dtype=np.bool_)
                rewards = np.asarray(archive["rewards"], dtype=np.float32)
                task_id = int(_manifest_scalar(archive, "task_id", path=path_value))
                task_name = str(
                    _manifest_scalar(archive, "task_name", path=path_value)
                )
                success = bool(_manifest_scalar(archive, "success", path=path_value))
                shard_schema = str(
                    _manifest_scalar(archive, "schema_version", path=path_value)
                )
                shard_threshold = float(
                    _manifest_scalar(archive, "label_threshold", path=path_value)
                )
                shard_mode = str(
                    _manifest_scalar(
                        archive, "label_calibration_mode", path=path_value
                    )
                )
                shard_source = str(
                    _manifest_scalar(
                        archive,
                        "label_calibration_source_sha256",
                        path=path_value,
                    )
                )
                shard_calibration = str(
                    _manifest_scalar(
                        archive,
                        "label_calibration_fingerprint_sha256",
                        path=path_value,
                    )
                )
                shard_prediction_horizon = int(
                    _manifest_scalar(
                        archive, "label_prediction_horizon", path=path_value
                    )
                )
                shard_terminal_horizon = int(
                    _manifest_scalar(
                        archive,
                        "label_terminal_positive_horizon",
                        path=path_value,
                    )
                )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(str(path_value)):
                raise
            raise ValueError(f"Cannot audit calibrated shard {path_value}: {exc}") from exc

        length = len(disagreements)
        expected_shapes = {
            "states": (length, 39 + task_count),
            "raw_observations": (length, 39),
            "actions": (length, 4),
            "expert_actions": (length, 4),
            "risk_events": (length,),
            "labels": (length,),
            "rewards": (length,),
        }
        actual_shapes = {
            "states": states.shape,
            "raw_observations": raw.shape,
            "actions": actions.shape,
            "expert_actions": expert_actions.shape,
            "risk_events": events.shape,
            "labels": labels.shape,
            "rewards": rewards.shape,
        }
        if length <= 0 or actual_shapes != expected_shapes:
            raise ValueError(
                f"{path_value}: calibrated trajectory shapes are invalid: {actual_shapes}"
            )
        if not all(
            np.isfinite(value).all()
            for value in (states, raw, actions, expert_actions, disagreements, rewards)
        ):
            raise ValueError(f"{path_value}: non-finite calibrated trajectory value")
        if np.any(disagreements < 0.0):
            raise ValueError(f"{path_value}: negative action disagreement")
        if task_id != expected_task_id or task_name != task_vocabulary[task_id]:
            raise ValueError(f"{path_value}: task metadata/order mismatch")
        one_hot = np.zeros(task_count, dtype=np.float32)
        one_hot[task_id] = 1.0
        if not np.array_equal(
            states[:, 39:], np.broadcast_to(one_hot, (length, task_count))
        ):
            raise ValueError(f"{path_value}: invalid task one-hot block")
        if (
            shard_schema != MULTITASK_FAILURE_SCHEMA
            or shard_mode != expected_mode
            or shard_source != source_fingerprint
            or shard_calibration != calibration_fingerprint
            or shard_prediction_horizon != prediction_horizon
            or shard_terminal_horizon != terminal_horizon
            or shard_threshold != threshold_values[task_id]
        ):
            raise ValueError(f"{path_value}: per-shard calibration provenance mismatch")
        recomputed_events = disagreements >= np.float32(threshold_values[task_id])
        if not success and terminal_horizon > 0:
            recomputed_events[max(0, length - terminal_horizon) :] = True
        recomputed_labels = _future_risk_labels(
            recomputed_events, prediction_horizon
        )
        if not np.array_equal(events, recomputed_events):
            raise ValueError(f"{path_value}: risk_events do not match calibrated threshold")
        if not np.array_equal(labels, recomputed_labels):
            raise ValueError(f"{path_value}: labels do not match future-risk horizon")
        expected_entry = {
            "task_id": task_id,
            "task_name": task_name,
            "rollout_index": rollout_index,
            "length": length,
            "success": success,
            "positive_labels": int(labels.sum()),
            "risk_events": int(events.sum()),
            "label_threshold": threshold_values[task_id],
            "label_calibration_fingerprint_sha256": calibration_fingerprint,
            "sha256": actual_digest,
        }
        mismatches = {
            key: (entry.get(key), value)
            for key, value in expected_entry.items()
            if entry.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{path_value}: manifest shard metadata mismatch {mismatches}")
        whitelist.append(path_value)
        task_rows[task_id] += length
        task_positives[task_id] += int(labels.sum())
        total_rows += length
        raw_source_records.append(
            {
                "file": relative,
                "task_id": task_id,
                "length": length,
                "success": success,
                "action_disagreement_sha256": _array_sha256(disagreements),
            }
        )

    disk_files = sorted(
        item.relative_to(directory).as_posix()
        for item in directory.rglob("*.npz")
        if item.is_file() or item.is_symlink()
    )
    listed_files = [item.relative_to(directory).as_posix() for item in whitelist]
    if disk_files != listed_files:
        missing = sorted(set(listed_files) - set(disk_files))
        stale = sorted(set(disk_files) - set(listed_files))
        raise ValueError(
            f"Calibrated manifest whitelist mismatch: missing={missing[:5]}, "
            f"stale={stale[:5]}"
        )
    if manifest.get("episodes") != len(files) or manifest.get("rows") != total_rows:
        raise ValueError("Calibrated manifest aggregate counts are inconsistent")
    expected_positive_rate = int(task_positives.sum()) / max(total_rows, 1)
    if not np.isclose(
        float(manifest.get("positive_rate", float("nan"))),
        expected_positive_rate,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Calibrated manifest positive_rate is inconsistent")
    per_task = manifest.get("per_task")
    if not isinstance(per_task, Mapping):
        raise ValueError("Calibrated manifest lacks per_task statistics")
    for task_id, task_name in enumerate(task_vocabulary):
        item = per_task.get(task_name)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing calibrated per_task statistics for {task_name}")
        expected_rate = int(task_positives[task_id]) / max(int(task_rows[task_id]), 1)
        if (
            int(item.get("rows", -1)) != int(task_rows[task_id])
            or not np.isclose(
                float(item.get("positive_rate", float("nan"))),
                expected_rate,
                rtol=0.0,
                atol=1e-15,
            )
            or float(item.get("label_threshold", float("nan")))
            != threshold_values[task_id]
        ):
            raise ValueError(f"Calibrated per_task statistics mismatch for {task_name}")
        if expected_mode == "fit-task-quantile" and threshold_samples[task_id] != int(
            task_rows[task_id]
        ):
            raise ValueError(f"Calibration sample count mismatch for {task_name}")

    raw_source_fingerprint = _canonical_json_sha256(raw_source_records)
    if calibration.get("target_raw_disagreement_sha256") != raw_source_fingerprint:
        raise ValueError("Target raw disagreement fingerprint mismatch")
    if expected_mode == "fit-task-quantile" and source_fingerprint != (
        raw_source_fingerprint
    ):
        raise ValueError("Training calibration source does not match raw disagreements")
    dataset_fingerprint = _canonical_json_sha256(
        {
            "collection_provenance_fingerprint_sha256": collection_fingerprint,
            "label_calibration_fingerprint_sha256": calibration_fingerprint,
            "files": [
                {"file": entry["file"], "sha256": entry["sha256"]}
                for entry in files
            ],
        }
    )
    stored_dataset_fingerprint = _require_sha256(
        manifest.get("dataset_fingerprint_sha256"),
        field="dataset_fingerprint_sha256",
    )
    if stored_dataset_fingerprint != dataset_fingerprint:
        raise ValueError("Calibrated dataset fingerprint mismatch")
    return AuditedMultitaskFailureBank(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        manifest=manifest,
        whitelist=tuple(whitelist),
        benchmark=benchmark,
        task_vocabulary=tuple(task_vocabulary),
        calibration=dict(calibration),
        calibration_fingerprint_sha256=str(calibration_fingerprint),
        dataset_fingerprint_sha256=dataset_fingerprint,
    )


def discover_npz(path: str | Path) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz file, got {source}")
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    files = sorted(source.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {source}")
    return files


def _first_key(archive: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    return next((key for key in keys if key in archive.files), None)


def _as_trajectories(array: np.ndarray) -> list[np.ndarray]:
    """Convert dense or object arrays to a list of [time, feature] arrays."""
    array = np.asarray(array)
    if array.dtype == object:
        return [
            np.asarray(item)
            if np.asarray(item).ndim > 1
            else np.asarray(item).reshape(1, -1)
            for item in array.reshape(-1)
        ]
    if array.ndim == 1:
        return [array.reshape(1, -1)]
    if array.ndim == 2:
        return [array]
    if array.ndim >= 3:
        return [array[index].reshape(-1, array.shape[-1]) for index in range(array.shape[0])]
    raise ValueError(f"Cannot interpret array with shape {array.shape} as trajectories.")


def load_demonstrations(path: str | Path) -> DemonstrationData:
    files = discover_npz(path)
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=True) as archive:
            state_key = _first_key(archive, STATE_KEYS)
            action_key = _first_key(archive, ACTION_KEYS)
            if state_key is None or action_key is None:
                continue
            state_trajectories = _as_trajectories(archive[state_key])
            action_trajectories = _as_trajectories(archive[action_key])
            if len(state_trajectories) != len(action_trajectories):
                # A flat file is still usable when both sides have the same rows.
                states = np.asarray(archive[state_key]).reshape(
                    -1, np.asarray(archive[state_key]).shape[-1]
                )
                actions = np.asarray(archive[action_key]).reshape(
                    -1, np.asarray(archive[action_key]).shape[-1]
                )
                state_trajectories, action_trajectories = [states], [actions]

            for states, actions in zip(state_trajectories, action_trajectories):
                states = np.asarray(states, dtype=np.float32)
                actions = np.asarray(actions, dtype=np.float32)
                count = min(len(states), len(actions))
                if count == 0:
                    group_id += 1
                    continue
                states, actions = states[:count], actions[:count]
                finite = np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
                states, actions = states[finite], actions[finite]
                if len(states):
                    all_states.append(states)
                    all_actions.append(actions)
                    all_groups.append(np.full(len(states), group_id, dtype=np.int64))
                group_id += 1

    if not all_states:
        raise ValueError(
            f"No valid state/action pairs found in {path}. "
            f"Expected keys {STATE_KEYS} and {ACTION_KEYS}."
        )
    state_dims = {array.shape[-1] for array in all_states}
    action_dims = {array.shape[-1] for array in all_actions}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(
            f"Inconsistent dimensions in demonstrations: states={state_dims}, "
            f"actions={action_dims}."
        )
    return DemonstrationData(
        states=np.concatenate(all_states).astype(np.float32, copy=False),
        actions=np.concatenate(all_actions).astype(np.float32, copy=False),
        groups=np.concatenate(all_groups),
        files=tuple(files),
    )


def make_causal_windows(
    states: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad causal histories so valid frames work with packed LSTMs."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"Expected [time, state_dim] states, got {states.shape}.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    windows = np.zeros(
        (len(states), sequence_length, states.shape[-1]), dtype=np.float32
    )
    lengths = np.empty(len(states), dtype=np.int64)
    for timestep in range(len(states)):
        start = max(0, timestep - sequence_length + 1)
        history = states[start : timestep + 1]
        lengths[timestep] = len(history)
        windows[timestep, : len(history)] = history
    return windows, lengths


def _labels_for_trajectory(labels: np.ndarray, index: int, count: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype == object:
        selected = np.asarray(labels.reshape(-1)[index]).reshape(-1)
    elif labels.ndim == 0:
        selected = np.repeat(labels.reshape(1), count)
    elif labels.ndim == 1:
        selected = labels.reshape(-1)
    else:
        selected = np.asarray(labels[index]).reshape(-1)
    if selected.size == 1 and count != 1:
        selected = np.repeat(selected, count)
    return selected[:count]


def _canonicalize_existing_windows(
    windows: np.ndarray,
    lengths: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 2:
        windows = windows[None, ...]
    if windows.ndim != 3:
        raise ValueError(f"Expected windows [samples,time,state], got {windows.shape}.")
    result = np.zeros(
        (len(windows), sequence_length, windows.shape[-1]), dtype=np.float32
    )
    canonical_lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if canonical_lengths.size == 1 and len(windows) > 1:
        canonical_lengths = np.repeat(canonical_lengths, len(windows))
    canonical_lengths = np.clip(canonical_lengths[: len(windows)], 1, sequence_length)
    for index, (window, length) in enumerate(zip(windows, canonical_lengths)):
        length = min(int(length), len(window), sequence_length)
        # Existing datasets usually left-pad histories. Select the side with
        # larger energy, then move valid frames to the front for pack_padded_sequence.
        if length < len(window):
            leading = window[:length]
            trailing = window[-length:]
            valid = trailing if np.linalg.norm(trailing) > np.linalg.norm(leading) else leading
        else:
            valid = window[-length:]
        result[index, :length] = valid[-length:]
    return result, canonical_lengths


def load_failure_data(
    path: str | Path,
    sequence_length: int = 10,
) -> FailureData:
    files = discover_npz(path)
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=True) as archive:
            label_key = _first_key(archive, LABEL_KEYS)
            state_key = _first_key(archive, STATE_KEYS)
            window_key = _first_key(archive, WINDOW_KEYS)
            if label_key is None or (state_key is None and window_key is None):
                continue
            raw_labels = archive[label_key]

            # States are preferred: rebuilding their causal histories guarantees
            # a consistent padding convention across generated and external data.
            if state_key is not None:
                trajectories = _as_trajectories(archive[state_key])
                for trajectory_index, states in enumerate(trajectories):
                    states = np.asarray(states, dtype=np.float32)
                    labels = _labels_for_trajectory(
                        raw_labels, trajectory_index, len(states)
                    )
                    count = min(len(states), len(labels))
                    if count == 0:
                        group_id += 1
                        continue
                    states = states[:count]
                    labels = labels[:count]
                    windows, lengths = make_causal_windows(states, sequence_length)
                    finite = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(labels)
                    windows = windows[finite]
                    lengths = lengths[finite]
                    labels = labels[finite]
                    if len(windows):
                        all_windows.append(windows)
                        all_lengths.append(lengths)
                        all_labels.append(labels.astype(np.float32))
                        all_groups.append(
                            np.full(len(windows), group_id, dtype=np.int64)
                        )
                    group_id += 1
                continue

            raw_windows = np.asarray(archive[window_key])
            if raw_windows.dtype == object:
                window_trajectories = [
                    np.asarray(item) for item in raw_windows.reshape(-1)
                ]
            elif raw_windows.ndim == 4:
                window_trajectories = [
                    raw_windows[index] for index in range(raw_windows.shape[0])
                ]
            else:
                window_trajectories = [raw_windows]
            length_key = _first_key(archive, LENGTH_KEYS)
            raw_lengths = (
                np.asarray(archive[length_key])
                if length_key is not None
                else np.asarray(sequence_length)
            )
            for trajectory_index, windows in enumerate(window_trajectories):
                windows = np.asarray(windows)
                count = len(windows) if windows.ndim >= 3 else 1
                labels = _labels_for_trajectory(raw_labels, trajectory_index, count)
                if raw_lengths.ndim <= 1:
                    lengths = raw_lengths.reshape(-1)
                else:
                    lengths = np.asarray(raw_lengths[trajectory_index]).reshape(-1)
                if lengths.size == 1:
                    lengths = np.repeat(lengths, count)
                windows, lengths = _canonicalize_existing_windows(
                    windows, lengths, sequence_length
                )
                count = min(len(windows), len(labels), len(lengths))
                windows, labels, lengths = (
                    windows[:count],
                    labels[:count],
                    lengths[:count],
                )
                finite = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(labels)
                windows, labels, lengths = (
                    windows[finite],
                    labels[finite],
                    lengths[finite],
                )
                if len(windows):
                    all_windows.append(windows)
                    all_labels.append(labels.astype(np.float32))
                    all_lengths.append(lengths.astype(np.int64))
                    all_groups.append(np.full(len(windows), group_id, dtype=np.int64))
                group_id += 1

    if not all_windows:
        raise ValueError(
            f"No valid failure examples found in {path}. Expected labels {LABEL_KEYS} "
            f"and either states {STATE_KEYS} or windows {WINDOW_KEYS}."
        )
    state_dims = {array.shape[-1] for array in all_windows}
    if len(state_dims) != 1:
        raise ValueError(f"Inconsistent state dimensions in failure data: {state_dims}.")
    labels = np.concatenate(all_labels).astype(np.float32, copy=False)
    # Labels may arrive as bool, integer, probability, or {-1,+1}.
    labels = (labels > 0.5).astype(np.float32)
    return FailureData(
        windows=np.concatenate(all_windows).astype(np.float32, copy=False),
        labels=labels,
        lengths=np.concatenate(all_lengths).astype(np.int64, copy=False),
        groups=np.concatenate(all_groups),
        files=tuple(files),
    )


def group_train_validation_split(
    groups: np.ndarray,
    validation_fraction: float,
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by trajectory to avoid temporal leakage between train and validation."""
    groups = np.asarray(groups)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    if len(unique_groups) >= 2:
        shuffled = rng.permutation(unique_groups)
        validation_count = int(round(len(shuffled) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(shuffled) - 1)
        validation_groups = shuffled[:validation_count]
        validation_mask = np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        indices = rng.permutation(len(groups))
        validation_count = min(
            max(int(round(len(indices) * validation_fraction)), 1),
            max(len(indices) - 1, 1),
        )
        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]

    if labels is not None and len(np.unique(labels)) > 1:
        # A group-wise split is preferred; only fall back to a stratified row
        # split when it accidentally removes an entire class from either side.
        train_classes = np.unique(labels[train_indices])
        validation_classes = np.unique(labels[validation_indices])
        if len(train_classes) < 2 or len(validation_classes) < 2:
            train_parts: list[np.ndarray] = []
            validation_parts: list[np.ndarray] = []
            for class_value in np.unique(labels):
                class_indices = rng.permutation(np.flatnonzero(labels == class_value))
                validation_count = min(
                    max(int(round(len(class_indices) * validation_fraction)), 1),
                    max(len(class_indices) - 1, 1),
                )
                validation_parts.append(class_indices[:validation_count])
                train_parts.append(class_indices[validation_count:])
            train_indices = rng.permutation(np.concatenate(train_parts))
            validation_indices = rng.permutation(np.concatenate(validation_parts))

    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Dataset is too small to create non-empty train/validation sets.")
    return train_indices, validation_indices
