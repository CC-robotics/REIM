#!/usr/bin/env python3
"""Recompute failure-monitor metrics before, at, and after event onset.

The detector's prospective label is positive when an event occurs in the
inclusive interval ``[t, t + horizon]``.  Consequently, the conventional
window-level confusion matrix mixes genuinely prospective samples
(``steps_to_failure_event > 0``) with current-event samples
(``steps_to_failure_event == 0``).  This audit reproduces the grouped
validation split used during training and reports those cases separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.io import file_sha256  # noqa: E402
from models.failure_detector import FailureDetector  # noqa: E402
from trainers.data import make_causal_windows  # noqa: E402
from utils.common import atomic_json_dump  # noqa: E402


LOGGER = logging.getLogger("audit_detector_pre_event")


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.bool_).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have identical shapes")
    tn = int(np.sum(~labels & ~predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    tp = int(np.sum(labels & predictions))
    samples = int(labels.size)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "samples": samples,
        "positive_samples": int(np.sum(labels)),
        "negative_samples": int(np.sum(~labels)),
        "predicted_positive_samples": int(np.sum(predictions)),
        "positive_fraction": float(np.mean(labels)) if samples else 0.0,
        "predicted_positive_fraction": (
            float(np.mean(predictions)) if samples else 0.0
        ),
        "accuracy": float((tn + tp) / max(samples, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _combined_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validation_file_ids(
    file_count: int, validation_fraction: float, seed: int
) -> np.ndarray:
    """Match ``group_train_validation_split`` for one trajectory per NPZ."""

    if file_count < 2:
        raise ValueError("at least two trajectory files are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    groups = np.arange(file_count, dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(groups)
    count = int(round(file_count * validation_fraction))
    count = min(max(count, 1), file_count - 1)
    return np.sort(shuffled[:count])


def _load_trajectory(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "states",
            "labels",
            "steps_to_failure_event",
            "event_labels",
            "forecast_horizon",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        states = np.asarray(archive["states"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.uint8).reshape(-1)
        offsets = np.asarray(
            archive["steps_to_failure_event"], dtype=np.int16
        ).reshape(-1)
        events = np.asarray(archive["event_labels"], dtype=np.uint8).reshape(-1)
        horizon = int(np.asarray(archive["forecast_horizon"]).item())
    if states.ndim != 2:
        raise ValueError(
            f"{path}: audit requires one [time,state] trajectory per NPZ, got {states.shape}"
        )
    if not (len(states) == len(labels) == len(offsets) == len(events)):
        raise ValueError(f"{path}: trajectory arrays have inconsistent lengths")
    if not np.array_equal(labels.astype(bool), offsets >= 0):
        raise ValueError(f"{path}: labels do not match prospective-event offsets")
    if not np.array_equal(events.astype(bool), offsets == 0):
        raise ValueError(f"{path}: event labels do not match zero event offsets")
    if np.any(offsets > horizon):
        raise ValueError(f"{path}: event offset exceeds stored forecast horizon")
    return states, labels, offsets, events, np.asarray(horizon)


def _write_offset_csv(
    path: Path,
    offsets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "event_offset",
        "phase",
        "samples",
        "predicted_positive_samples",
        "predicted_positive_rate",
        "mean_probability",
        "median_probability",
    )
    rows: list[dict[str, Any]] = []
    for offset in sorted(int(value) for value in np.unique(offsets)):
        mask = offsets == offset
        phase = (
            "no_event_within_horizon"
            if offset < 0
            else ("current_event" if offset == 0 else "strict_future_event")
        )
        rows.append(
            {
                "event_offset": offset,
                "phase": phase,
                "samples": int(np.sum(mask)),
                "predicted_positive_samples": int(np.sum(predictions[mask])),
                "predicted_positive_rate": float(np.mean(predictions[mask])),
                "mean_probability": float(np.mean(probabilities[mask])),
                "median_probability": float(np.median(probabilities[mask])),
            }
        )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    metrics_path = args.detector_metrics.resolve()
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no trajectory NPZ files found under {data_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"detector checkpoint not found: {checkpoint}")
    validation_ids = _validation_file_ids(
        len(files), args.validation_fraction, args.seed
    )
    validation_files = [files[int(index)] for index in validation_ids]
    device = torch.device(args.device)
    model = FailureDetector.from_checkpoint(checkpoint, map_location=device)
    model.eval()
    torch.set_num_threads(max(1, int(args.torch_threads)))

    probabilities_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    offsets_parts: list[np.ndarray] = []
    events_parts: list[np.ndarray] = []
    trajectory_records: list[dict[str, Any]] = []
    horizons: set[int] = set()
    with torch.inference_mode():
        for file_id, path in zip(validation_ids, validation_files):
            states, labels, offsets, events, horizon_value = _load_trajectory(path)
            horizon = int(horizon_value.item())
            horizons.add(horizon)
            windows, lengths = make_causal_windows(
                states, model.sequence_length
            )
            local_parts: list[np.ndarray] = []
            for start in range(0, len(windows), args.batch_size):
                stop = start + args.batch_size
                local_parts.append(
                    model.predict_proba(
                        windows[start:stop], lengths[start:stop]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            probabilities = np.concatenate(local_parts).astype(np.float64)
            probabilities_parts.append(probabilities)
            labels_parts.append(labels)
            offsets_parts.append(offsets)
            events_parts.append(events)
            trajectory_records.append(
                {
                    "file_id": int(file_id),
                    "path": path,
                    "probabilities": probabilities,
                    "event_labels": events,
                }
            )
    if len(horizons) != 1:
        raise ValueError(f"validation trajectories use multiple horizons: {horizons}")

    probabilities = np.concatenate(probabilities_parts)
    labels = np.concatenate(labels_parts).astype(np.bool_)
    offsets = np.concatenate(offsets_parts)
    events = np.concatenate(events_parts).astype(np.bool_)
    predictions = probabilities >= args.threshold
    prospective = _binary_metrics(labels, predictions)

    current_mask = offsets == 0
    future_mask = offsets > 0
    negative_mask = offsets < 0
    pre_event_mask = offsets != 0
    strict_pre_event = _binary_metrics(
        future_mask[pre_event_mask], predictions[pre_event_mask]
    )
    positive_count = int(np.sum(labels))
    current_count = int(np.sum(current_mask))
    future_count = int(np.sum(future_mask))

    event_trajectories = 0
    event_trajectories_detected_pre_event = 0
    non_event_trajectories = 0
    non_event_trajectories_with_alert = 0
    latest_alert_leads: list[int] = []
    for record in trajectory_records:
        local_probabilities = np.asarray(record["probabilities"])
        local_events = np.asarray(record["event_labels"], dtype=np.bool_)
        event_indices = np.flatnonzero(local_events)
        if event_indices.size:
            event_trajectories += 1
            onset = int(event_indices[0])
            alerts = np.flatnonzero(
                (local_probabilities >= args.threshold)
                & (np.arange(len(local_probabilities)) < onset)
            )
            if alerts.size:
                event_trajectories_detected_pre_event += 1
                latest_alert_leads.append(onset - int(alerts[-1]))
        else:
            non_event_trajectories += 1
            non_event_trajectories_with_alert += int(
                np.any(local_probabilities >= args.threshold)
            )

    trajectory_metrics = {
        "event_trajectories": event_trajectories,
        "event_trajectories_with_strict_pre_event_alert": (
            event_trajectories_detected_pre_event
        ),
        "strict_pre_event_detection_rate": float(
            event_trajectories_detected_pre_event / max(event_trajectories, 1)
        ),
        "latest_pre_event_alert_lead_steps": {
            "count": len(latest_alert_leads),
            "mean": (
                float(np.mean(latest_alert_leads)) if latest_alert_leads else None
            ),
            "median": (
                float(np.median(latest_alert_leads)) if latest_alert_leads else None
            ),
            "minimum": min(latest_alert_leads) if latest_alert_leads else None,
            "maximum": max(latest_alert_leads) if latest_alert_leads else None,
        },
        "non_event_trajectories": non_event_trajectories,
        "non_event_trajectories_with_any_alert": non_event_trajectories_with_alert,
        "non_event_trajectory_alert_rate": float(
            non_event_trajectories_with_alert / max(non_event_trajectories, 1)
        ),
    }

    if metrics_path.is_file():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        deployment = stored.get("deployment_threshold_metrics", {})
        expected_matrix = deployment.get("confusion_matrix")
        if expected_matrix != prospective["confusion_matrix"]:
            raise ValueError(
                "recomputed deployment confusion matrix does not match "
                f"{metrics_path}"
            )
        if not np.isclose(
            float(deployment.get("threshold", np.nan)),
            args.threshold,
            atol=1e-12,
        ):
            raise ValueError("stored detector deployment threshold differs from audit")

    _write_offset_csv(
        args.output_csv.resolve(), offsets, probabilities, predictions
    )
    manifest_path = data_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "artifact": "detector_pre_event_audit",
        "interpretation": (
            "The conventional prospective-label confusion matrix is dominated "
            "by samples observed at an event. Strict pre-event metrics exclude "
            "offset-zero windows and quantify future-event monitoring separately."
        ),
        "protocol": {
            "validation_split": "trajectory_grouped",
            "validation_fraction": args.validation_fraction,
            "split_seed": args.seed,
            "trajectory_files_total": len(files),
            "validation_trajectory_files": len(validation_files),
            "sequence_length": model.sequence_length,
            "forecast_horizon": next(iter(horizons)),
            "threshold": args.threshold,
            "target_interval": "[t, t + forecast_horizon]",
            "strict_pre_event_universe": (
                "offsets -1 and 1..horizon; offset 0 excluded"
            ),
        },
        "validation_samples": {
            "total": int(len(labels)),
            "prospective_positive": positive_count,
            "negative_no_event_within_horizon": int(np.sum(negative_mask)),
            "current_event_offset_zero": current_count,
            "strict_future_offset_one_to_horizon": future_count,
            "current_event_fraction_of_positive": float(
                current_count / max(positive_count, 1)
            ),
            "strict_future_fraction_of_positive": float(
                future_count / max(positive_count, 1)
            ),
        },
        "prospective_window_metrics_including_current_event": prospective,
        "current_event_window_recall": float(np.mean(predictions[current_mask])),
        "strict_future_positive_window_recall": float(
            np.mean(predictions[future_mask])
        ),
        "strict_pre_event_window_metrics": strict_pre_event,
        "trajectory_metrics": trajectory_metrics,
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "dataset_manifest": (
                str(manifest_path.resolve()) if manifest_path.is_file() else None
            ),
            "dataset_manifest_sha256": (
                file_sha256(manifest_path) if manifest_path.is_file() else None
            ),
            "validation_files_combined_sha256": _combined_sha256(validation_files),
            "detector_metrics": (
                str(metrics_path) if metrics_path.is_file() else None
            ),
            "detector_metrics_sha256": (
                file_sha256(metrics_path) if metrics_path.is_file() else None
            ),
        },
        "outputs": {
            "offset_csv": str(args.output_csv.resolve()),
        },
        "audit": {
            "raw_offsets_consistent_with_labels": True,
            "raw_events_consistent_with_zero_offsets": True,
            "stored_deployment_confusion_matrix_reproduced": metrics_path.is_file(),
            "feature_windows_rebuilt_causally_from_raw_states": True,
        },
    }
    atomic_json_dump(payload, args.output_json)
    LOGGER.info(
        "Saved pre-event audit: current-event positive share=%.4f, "
        "strict-pre-event trajectory detection=%.4f",
        payload["validation_samples"]["current_event_fraction_of_positive"],
        trajectory_metrics["strict_pre_event_detection_rate"],
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "datasets/failures"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/failure_detector.pt",
    )
    parser.add_argument(
        "--detector-metrics",
        type=Path,
        default=ROOT / "results/tables/detector_metrics.json",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results/tables/detector_pre_event_audit.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results/tables/detector_pre_event_by_offset.csv",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie in [0, 1]")
    if args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("--batch-size and --torch-threads must be positive")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    payload = audit(args)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
