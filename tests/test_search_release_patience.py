"""Unit tests for the REIM release-parameter grid search helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import search_release_patience as search


def _reim_row(
    task_id: int,
    *,
    success: int,
    interventions: int,
    recovery_success: int = 0,
    steps: int = 100,
    recovery_steps_total: int = 0,
    trigger_step: int = -1,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "success": success,
        "intervention_count": interventions,
        "recovery_success": recovery_success,
        "steps": steps,
        "recovery_steps_total": recovery_steps_total,
        "trigger_step": trigger_step,
    }


def test_aggregate_reim_metrics() -> None:
    rows = [
        _reim_row(0, success=1, interventions=0, steps=100),
        _reim_row(0, success=0, interventions=1, recovery_success=1, steps=200, recovery_steps_total=50, trigger_step=10),
        _reim_row(1, success=1, interventions=2, recovery_success=0, steps=150, recovery_steps_total=100, trigger_step=5),
        _reim_row(1, success=1, interventions=1, recovery_success=1, steps=120, trigger_step=8),
    ]
    result = search._aggregate_reim(rows, task_count=2)
    assert result["episodes"] == 4
    assert result["micro_success"] == pytest.approx(0.75)
    assert result["task_macro_success"] == pytest.approx(0.75)
    assert result["intervened_episode_rate"] == pytest.approx(0.75)
    assert result["interventions_per_episode_mean"] == pytest.approx(1.0)
    assert result["interventions_per_episode_median"] == pytest.approx(1.0)
    # Episodes with recovery_success >= 1 among the 3 intervened episodes.
    assert result["release_rate"] == pytest.approx(2 / 3)
    assert result["median_trigger_step"] == pytest.approx(8.0)
    # (0 + 0.25 + 100/150 + 0) / 4, rounded to the metric's 6 decimals
    assert result["recovery_occupancy_mean"] == pytest.approx(
        round((0 + 0.25 + 100 / 150 + 0) / 4, 6)
    )
    assert result["mean_steps"] == pytest.approx(142.5)


def test_aggregate_reim_no_intervention() -> None:
    rows = [
        _reim_row(0, success=1, interventions=0, steps=80),
        _reim_row(1, success=0, interventions=0, steps=90),
    ]
    result = search._aggregate_reim(rows, task_count=2)
    assert result["intervened_episode_rate"] == 0.0
    assert result["release_rate"] == 0.0
    assert result["median_trigger_step"] == -1
    assert result["recovery_occupancy_mean"] == 0.0


def test_aggregate_reference_macro_uses_all_tasks() -> None:
    rows = [
        {"task_id": 0, "success": 1, "steps": 100},
        {"task_id": 0, "success": 1, "steps": 100},
        {"task_id": 1, "success": 0, "steps": 200},
        {"task_id": 1, "success": 0, "steps": 200},
    ]
    result = search._aggregate_reference(rows, task_count=2)
    assert result["micro_success"] == pytest.approx(0.5)
    assert result["task_macro_success"] == pytest.approx(0.5)
    assert result["mean_steps"] == pytest.approx(150.0)


def test_aggregate_empty_cell_fails_closed() -> None:
    with pytest.raises(RuntimeError):
        search._aggregate_reim([], task_count=2)


def test_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    rows = [
        {
            "condition": "official_clean",
            "noise_level": 0.0,
            "release_threshold": 0.05,
            "release_patience": 1,
            "task_macro_success": 0.9,
        }
    ]
    search._write_csv(path, search.GRID_FIELDS, rows)
    loaded = search._read_csv(path)
    assert len(loaded) == 1
    assert loaded[0]["condition"] == "official_clean"
    assert float(loaded[0]["release_threshold"]) == 0.05
    assert int(float(loaded[0]["release_patience"])) == 1
    assert float(loaded[0]["task_macro_success"]) == pytest.approx(0.9)


def test_parser_defaults() -> None:
    parser = search._build_parser()
    args = parser.parse_args(
        [
            "--benchmark",
            "MT10",
            "--act-checkpoint",
            "act.pt",
            "--detector-checkpoint",
            "detector.pt",
            "--recovery-checkpoint",
            "recovery.pt",
        ]
    )
    assert args.benchmark_seed is None  # resolved in main()
    assert args.threshold is None  # resolved in main()
    assert args.release_thresholds == [0.05, 0.10, 0.15, 0.20, 0.30]
    assert args.release_patiences == [1, 3, 5, 10]
    assert args.noise_levels == [0.0, 0.1]
    assert args.episodes_per_task == 20
    assert args.min_recovery_steps == 5
    assert args.intervention_cooldown == 10


def test_default_validation_seeds_and_thresholds() -> None:
    assert search.DEFAULT_VALIDATION_SEEDS == {"MT10": 20264010, "MT50": 20264050}
    assert search.DEFAULT_TRIGGER_THRESHOLDS == {"MT10": 0.65, "MT50": 0.64}


def test_json_output_shape(tmp_path: Path) -> None:
    from data.io import atomic_write_json

    payload = {
        "protocol": {"schema_version": search.SCHEMA_VERSION},
        "results": {"official_clean": {"grid": {}}},
    }
    path = tmp_path / "search.json"
    atomic_write_json(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["protocol"]["schema_version"] == search.SCHEMA_VERSION
