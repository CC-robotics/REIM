#!/usr/bin/env python3
"""Summarize canonical seed 42 plus MT50 component seeds 43--45."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45)
CONDITIONS = (
    ("official_clean", "official_clean"),
    ("robustness_noise_10", "robustness_noise_10"),
    ("robustness_noise_40", "robustness_noise_40"),
)
REIM = "MT-REIM"
HEURISTIC = "MT-ACT + Heuristic-Gated Learned Recovery"


def _path(seed: int, condition: str, method: str) -> Path:
    if seed == 42:
        base = ROOT / "results" / "tables" / "confirmation_202660xx"
        return base / f"mt50_confirm_{condition}_episodes.csv"
    if method == "heuristic":
        return (
            ROOT
            / "results"
            / "tables"
            / f"mt50_seed{seed}_heuristic_confirm_{condition}_episodes.csv"
        )
    return (
        ROOT
        / "results"
        / "tables"
        / f"mt50_seed{seed}_confirm_{condition}_episodes.csv"
    )


def _load(path: Path, method: str) -> dict[str, tuple[str, int]]:
    values: dict[str, tuple[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] == method:
                values[row["paired_episode_id"]] = (
                    row["task_name"],
                    int(row["success"]),
                )
    return values


def _task_macro(values: dict[str, tuple[str, int]]) -> float:
    by_task: dict[str, list[int]] = {}
    for task, success in values.values():
        by_task.setdefault(task, []).append(success)
    if len(by_task) != 50 or any(len(rows) != 50 for rows in by_task.values()):
        raise ValueError("expected 50 MT50 tasks with 50 episodes each")
    return statistics.fmean(statistics.fmean(rows) for rows in by_task.values())


def main() -> None:
    rows: list[dict[str, object]] = []
    for label, condition in CONDITIONS:
        canonical = _path(42, condition, "reim")
        seed_success: dict[int, float] = {}
        heuristic_success: dict[int, float] = {}
        deltas: list[float] = []
        rescued: list[int] = []
        harmed: list[int] = []
        for seed in SEEDS:
            path = _path(seed, condition, "reim")
            heuristic_path = _path(seed, condition, "heuristic")
            if not path.is_file():
                raise FileNotFoundError(path)
            if not heuristic_path.is_file():
                raise FileNotFoundError(heuristic_path)
            reim = _load(path, REIM)
            heuristic = _load(heuristic_path, HEURISTIC)
            if set(reim) != set(heuristic):
                raise ValueError(f"paired episode IDs do not match for seed {seed}: {condition}")
            seed_success[seed] = _task_macro(reim)
            heuristic_success[seed] = _task_macro(heuristic)
            pairs = [(reim[key][1], heuristic[key][1]) for key in sorted(reim)]
            rescued.append(sum(r == 1 and h == 0 for r, h in pairs))
            harmed.append(sum(r == 0 and h == 1 for r, h in pairs))
            deltas.append(statistics.fmean(r - h for r, h in pairs))

        successes = [seed_success[seed] for seed in SEEDS]
        heuristic_successes = [heuristic_success[seed] for seed in SEEDS]
        row: dict[str, object] = {
            "condition": label,
        }
        row.update({f"reim_seed{seed}": round(seed_success[seed], 6) for seed in SEEDS})
        row.update(
            {
                f"heuristic_seed{seed}": round(heuristic_success[seed], 6)
                for seed in SEEDS
            }
        )
        row.update(
            {
                "reim_mean": round(statistics.fmean(successes), 6),
                "reim_sample_std": round(statistics.stdev(successes), 6),
                "heuristic_mean": round(statistics.fmean(heuristic_successes), 6),
                "heuristic_sample_std": round(
                    statistics.stdev(heuristic_successes), 6
                ),
                "delta_mean": round(statistics.fmean(deltas), 6),
                "delta_sample_std": round(statistics.stdev(deltas), 6),
                "delta_per_seed": "/".join(f"{value:.4f}" for value in deltas),
                "rescued_per_seed": "/".join(map(str, rescued)),
                "harmed_per_seed": "/".join(map(str, harmed)),
                "episodes_per_task": 50,
                "task_count": 50,
                "benchmark_seed": 20266050,
            }
        )
        rows.append(row)

    output = ROOT / "results" / "tables" / "mt50_four_seed_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
