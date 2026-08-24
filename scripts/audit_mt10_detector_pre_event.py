#!/usr/bin/env python3
"""Pre-event / lead-time audit for MT10/MT50 multitask detectors.

Multitask counterpart of scripts/audit_detector_pre_event.py (which covers
only the single-task PickPlace bank).  For each calibrated multitask
validation bank this audit:

- derives per-timestep event offsets from ``risk_events`` (0 = current event,
  k>0 = event starts in k steps, -1 = no event within the prediction horizon)
  and checks them against the stored prospective labels;
- reports the prospective confusion matrix split into strict pre-event
  (offset > 0) and current-event (offset == 0) components;
- reports trajectory-level early-warning metrics: strict pre-event detection
  rate over event trajectories, latest pre-event alert lead-time
  (mean/median/min/max), and the false-alert rate over non-event
  trajectories.

Used by the terminal-positive-horizon backfill ablation: each horizon's
detector is scored on its own relabelled validation bank at its own
precision-floor-selected threshold (never a mechanically shared threshold).

Usage:
  .venv/Scripts/python.exe scripts/audit_mt10_detector_pre_event.py \
      --benchmark MT10 \
      --validation-data datasets/mt10/failures_validation_horizon0_calibrated \
      --detector-checkpoint checkpoints/mt10/horizon_0/failure_detector.pt \
      --threshold 0.72 \
      --output-json results/tables/mt10_horizon0_pre_event_audit.json \
      --output-csv results/tables/mt10_horizon0_pre_event_by_offset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.io import atomic_write_json, file_sha256  # noqa: E402
from evaluation.tune_multitask_detector import (  # noqa: E402
    _binary_metrics,
    _infer_probabilities,
    _load_audited_failure_data,
    _protocol_bank_seeds,
    _validate_detector_checkpoint,
    audit_validation_bank,
)
from utils.common import (  # noqa: E402
    configure_logging,
    resolve_path,
    seed_everything,
    select_device,
)

LOGGER = logging.getLogger("audit_mt10_detector_pre_event")
AUDIT_SCHEMA = "reim-multitask-pre-event-audit-v1"


def _event_offsets(events: np.ndarray, prediction_horizon: int) -> np.ndarray:
    """Replicate the single-task steps_to_failure_event semantics.

    offset[t] = 0 when an event is active at t; otherwise the distance to the
    next event onset when it lies within [t+1, t+prediction_horizon]; else -1.
    """
    events = np.asarray(events, dtype=np.bool_).reshape(-1)
    length = events.size
    offsets = np.full(length, -1, dtype=np.int64)
    next_event = np.full(length, -1, dtype=np.int64)
    running = -1
    for index in range(length - 1, -1, -1):
        if events[index]:
            running = index
        next_event[index] = running
    for index in range(length):
        if events[index]:
            offsets[index] = 0
        elif next_event[index] >= 0:
            distance = int(next_event[index]) - index
            if distance <= prediction_horizon:
                offsets[index] = distance
    return offsets


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
    benchmark = str(args.benchmark).upper()
    seed_everything(int(args.seed), deterministic_torch=True)
    device = select_device(args.device)
    configure_logging("audit_mt10_detector_pre_event", args.log_file)
    validation_seed, validation_benchmark_seed, final_seed = _protocol_bank_seeds(
        args, benchmark
    )
    data_dir = resolve_path(args.validation_data).resolve()
    manifest, whitelist = audit_validation_bank(
        data_dir,
        benchmark=benchmark,
        expected_validation_seed=validation_seed,
        expected_benchmark_seed=validation_benchmark_seed,
        forbidden_final_seed=final_seed,
    )
    manifest_path = data_dir / "manifest.json"
    task_vocabulary = list(manifest["task_vocabulary"])
    checkpoint_path = resolve_path(args.detector_checkpoint).resolve()
    model, checkpoint, checkpoint_digest = _validate_detector_checkpoint(
        checkpoint_path,
        benchmark=benchmark,
        task_vocabulary=task_vocabulary,
        validation_manifest=manifest,
        validation_manifest_sha256=file_sha256(manifest_path),
        device=device,
    )
    sequence_length = int(checkpoint["sequence_length"])
    data, task_ids = _load_audited_failure_data(
        data_dir,
        whitelist,
        sequence_length=sequence_length,
        task_vocabulary=task_vocabulary,
    )
    probabilities = _infer_probabilities(
        model, data, batch_size=int(args.batch_size), device=str(device)
    )
    predictions = probabilities >= float(args.threshold)
    prediction_horizon = int(manifest["label_calibration"]["prediction_horizon"])

    probability_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = []
    trajectory_records: list[dict[str, Any]] = []
    label_mismatches = 0
    groups = np.asarray(data.groups, dtype=np.int64)
    for group_id in range(len(whitelist)):
        mask = groups == group_id
        shard_path = data.files[group_id]
        with np.load(shard_path, allow_pickle=False) as archive:
            events = np.asarray(archive["risk_events"], dtype=np.bool_).reshape(-1)
            stored_labels = np.asarray(archive["labels"], dtype=np.bool_).reshape(-1)
        local_probabilities = probabilities[mask]
        local_labels = np.asarray(data.labels[mask], dtype=np.bool_).reshape(-1)
        if local_probabilities.size != events.size:
            raise ValueError(
                f"{shard_path}: {local_probabilities.size} windows vs "
                f"{events.size} timesteps"
            )
        if not np.array_equal(local_labels, stored_labels):
            raise ValueError(f"{shard_path}: loader labels differ from shard labels")
        offsets = _event_offsets(events, prediction_horizon)
        derived_labels = (offsets >= 0) & (offsets <= prediction_horizon)
        label_mismatches += int(np.count_nonzero(derived_labels != stored_labels))
        probability_parts.append(local_probabilities)
        prediction_parts.append(predictions[mask])
        label_parts.append(stored_labels)
        offset_parts.append(offsets)
        trajectory_records.append(
            {
                "shard": str(shard_path.relative_to(ROOT)).replace("\\", "/"),
                "task_id": int(task_ids[mask][0]),
                "probabilities": local_probabilities,
                "events": events,
            }
        )

    all_probabilities = np.concatenate(probability_parts)
    all_predictions = np.concatenate(prediction_parts)
    all_labels = np.concatenate(label_parts)
    all_offsets = np.concatenate(offset_parts)

    prospective = _binary_metrics(all_labels, all_probabilities, float(args.threshold))
    current_mask = all_offsets == 0
    future_mask = all_offsets > 0
    pre_event_mask = all_offsets != 0
    strict_pre_event = _binary_metrics(
        future_mask[pre_event_mask].astype(np.int64),
        all_probabilities[pre_event_mask],
        float(args.threshold),
    )
    current_event_window_recall = float(
        np.mean(all_predictions[current_mask]) if np.any(current_mask) else 0.0
    )

    event_trajectories = 0
    event_trajectories_detected_pre_event = 0
    non_event_trajectories = 0
    non_event_trajectories_with_alert = 0
    latest_alert_leads: list[int] = []
    for record in trajectory_records:
        local_probabilities = np.asarray(record["probabilities"])
        local_events = np.asarray(record["events"], dtype=np.bool_)
        event_indices = np.flatnonzero(local_events)
        if event_indices.size:
            event_trajectories += 1
            onset = int(event_indices[0])
            alerts = np.flatnonzero(
                (local_probabilities >= float(args.threshold))
                & (np.arange(len(local_probabilities)) < onset)
            )
            if alerts.size:
                event_trajectories_detected_pre_event += 1
                latest_alert_leads.append(onset - int(alerts[-1]))
        else:
            non_event_trajectories += 1
            non_event_trajectories_with_alert += int(
                np.any(local_probabilities >= float(args.threshold))
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
            "mean": float(np.mean(latest_alert_leads)) if latest_alert_leads else None,
            "median": (
                float(np.median(latest_alert_leads)) if latest_alert_leads else None
            ),
            "minimum": min(latest_alert_leads) if latest_alert_leads else None,
            "maximum": max(latest_alert_leads) if latest_alert_leads else None,
        },
        "non_event_trajectories": non_event_trajectories,
        "non_event_trajectories_with_alert": non_event_trajectories_with_alert,
        "non_event_trajectory_alert_rate": float(
            non_event_trajectories_with_alert / max(non_event_trajectories, 1)
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "artifact": "multitask_detector_pre_event_audit",
        "benchmark": benchmark,
        "threshold": float(args.threshold),
        "prediction_horizon": prediction_horizon,
        "terminal_positive_horizon": int(
            manifest["label_calibration"]["terminal_positive_horizon"]
        ),
        "inputs": {
            "validation_data": str(data_dir.relative_to(ROOT)).replace("\\", "/"),
            "validation_manifest_sha256": file_sha256(manifest_path),
            "detector_checkpoint": str(checkpoint_path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "detector_checkpoint_sha256": checkpoint_digest,
        },
        "validation_samples": {
            "total": int(all_labels.size),
            "prospective_positive": int(np.sum(all_labels)),
            "current_event_offset_zero": int(np.sum(current_mask)),
            "strict_future_offset_one_to_horizon": int(np.sum(future_mask)),
            "negative_no_event_within_horizon": int(np.sum(all_offsets < 0)),
            "derived_label_mismatches": label_mismatches,
        },
        "prospective_window_metrics_including_current_event": prospective,
        "strict_pre_event_window_metrics": strict_pre_event,
        "current_event_window_recall": current_event_window_recall,
        "trajectory_metrics": trajectory_metrics,
    }
    atomic_write_json(args.output_json, payload)
    if args.output_csv:
        _write_offset_csv(args.output_csv, all_offsets, all_probabilities, all_predictions)
    LOGGER.info(
        "Saved multitask pre-event audit: strict-pre-event detection=%.4f, "
        "median lead=%s",
        trajectory_metrics["strict_pre_event_detection_rate"],
        trajectory_metrics["latest_pre_event_alert_lead_steps"]["median"],
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="MT10", choices=("MT10", "MT50"))
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--protocol-config")
    parser.add_argument("--validation-bank-seed", type=int)
    parser.add_argument("--validation-benchmark-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--log-file")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return audit(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
