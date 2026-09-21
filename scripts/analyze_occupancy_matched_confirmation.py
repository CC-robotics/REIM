"""Analyze a frozen occupancy-matched gate comparison.

The candidate and reference CSVs must come from the same confirmation bank.
The script checks paired episode IDs, reports task-macro success and the
episode-level recovery occupancy used by the validation search, and computes
task-stratified paired bootstrap intervals for the candidate-minus-reference
success difference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path, method: str | None = None) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if method is not None:
        rows = [row for row in rows if row["method"] == method]
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = row["paired_episode_id"]
        if episode_id in output:
            raise ValueError(f"duplicate paired_episode_id in {path}: {episode_id}")
        output[episode_id] = {
            "task": row["task_name"],
            "success": int(row["success"]),
            "steps": max(1, int(row["steps"])),
            "recovery_steps": int(row["recovery_steps_total"]),
        }
    if not output:
        raise ValueError(f"no rows loaded from {path}")
    return output


def _macro_success(rows: dict[str, dict[str, Any]]) -> float:
    by_task: dict[str, list[int]] = {}
    for row in rows.values():
        by_task.setdefault(row["task"], []).append(row["success"])
    return float(np.mean([np.mean(values) for values in by_task.values()]))


def _occupancy(rows: dict[str, dict[str, Any]]) -> float:
    return float(
        np.mean(
            [row["recovery_steps"] / row["steps"] for row in rows.values()]
        )
    )


def _bootstrap_delta(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, list[float]]:
    common = sorted(set(reference) & set(candidate))
    if set(reference) != set(candidate):
        raise ValueError("candidate and reference episode IDs do not match")
    by_task: dict[str, list[str]] = {}
    for episode_id in common:
        by_task.setdefault(reference[episode_id]["task"], []).append(episode_id)
    observed = float(_macro_success(candidate) - _macro_success(reference))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    task_names = sorted(by_task)
    for index in range(samples):
        task_deltas: list[float] = []
        for task in task_names:
            episode_ids = by_task[task]
            sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
            task_deltas.append(
                float(
                    np.mean([candidate[e]["success"] for e in sampled])
                    - np.mean([reference[e]["success"] for e in sampled])
                )
            )
        estimates[index] = float(np.mean(task_deltas))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return observed, [float(lower), float(upper)]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    reference = _load(args.reference, args.reference_method)
    candidate = _load(args.candidate, args.candidate_method)
    if set(reference) != set(candidate):
        raise ValueError("candidate and reference episode IDs do not match")
    delta, ci = _bootstrap_delta(
        reference,
        candidate,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    rescued = sum(
        candidate[e]["success"] == 1 and reference[e]["success"] == 0
        for e in candidate
    )
    harmed = sum(
        candidate[e]["success"] == 0 and reference[e]["success"] == 1
        for e in candidate
    )
    result = {
        "reference": {
            "path": str(args.reference.resolve()),
            "method": args.reference_method,
            "episodes": len(reference),
            "task_macro_success": _macro_success(reference),
            "mean_episode_recovery_occupancy": _occupancy(reference),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "method": args.candidate_method,
            "episodes": len(candidate),
            "task_macro_success": _macro_success(candidate),
            "mean_episode_recovery_occupancy": _occupancy(candidate),
        },
        "paired": {
            "rescued": int(rescued),
            "harmed": int(harmed),
            "delta_task_macro_success": float(delta),
            "delta_percentage_points": float(100.0 * delta),
            "bootstrap_95_ci": [float(value) for value in ci],
            "bootstrap_95_ci_percentage_points": [
                float(100.0 * value) for value in ci
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-method", default="MT-REIM")
    parser.add_argument(
        "--candidate-method", default="MT-ACT + Heuristic-Gated Learned Recovery"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2))


if __name__ == "__main__":
    main()
