"""Tests for the equal-budget recovery-policy seed diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.model_seed_stability import (
    ANALYSIS_ID,
    PROJECT_ROOT,
    _read_episodes,
    _wilson_interval,
)


def _write_episode_fixture(path: Path, *, noise_field: str) -> None:
    fieldnames = [
        "episode",
        "seed",
        "backend",
        "success",
        "steps",
        "elapsed_seconds",
        "recovery_attempts",
        "recovery_successes",
        "detector_triggers",
        "recovery_steps",
        "failure_probability_max",
        "Profile",
        noise_field,
        "Benchmark Eligible",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(2):
            writer.writerow(
                {
                    "episode": episode,
                    "seed": 100 + episode,
                    "backend": "metaworld",
                    "success": episode == 0,
                    "steps": 20 + episode,
                    "elapsed_seconds": 0.1,
                    "recovery_attempts": 1,
                    "recovery_successes": int(episode == 0),
                    "detector_triggers": 1,
                    "recovery_steps": 10,
                    "failure_probability_max": 0.4,
                    "Profile": "custom",
                    noise_field: 0.2,
                    "Benchmark Eligible": False,
                }
            )


@pytest.mark.parametrize("noise_field", ["Noise Level", "noise_level"])
def test_episode_reader_accepts_audited_noise_header_versions(
    tmp_path: Path,
    noise_field: str,
) -> None:
    path = tmp_path / "episodes.csv"
    _write_episode_fixture(path, noise_field=noise_field)

    episodes = _read_episodes(
        path,
        expected_episode_count=2,
        expected_seed_start=100,
        expected_noise_level=0.2,
    )

    assert episodes.seeds == (100, 101)
    assert episodes.successes.tolist() == [True, False]
    assert episodes.recovery_attempts == 2
    assert episodes.recovery_successes == 1


def test_wilson_interval_matches_measured_seed43_interval() -> None:
    lower, upper = _wilson_interval(162, 200)

    assert lower == pytest.approx(0.7499876129240015)
    assert upper == pytest.approx(0.8583282847220932)


def test_equal_budget_artifact_has_strict_recovery_only_scope() -> None:
    path = PROJECT_ROOT / "results/tables/model_seed_stability.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["analysis_id"] == ANALYSIS_ID
    assert payload["benchmark_eligible"] is False
    assert payload["protocol"]["varied_component"] == (
        "PPO recovery policy training seed only"
    )
    assert "not a three-seed full-system result" in payload["disclaimer"]
    assert payload["equal_budget_and_frozen_config_validation"][
        "all_actual_training_timesteps_at_least_500000"
    ]
    models = payload["models"]
    assert [model["recovery_training_seed"] for model in models] == [42, 43, 44]
    for model in models:
        assert model["training_actual_timesteps"] >= 500_000
        checkpoint = Path(model["checkpoint"])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == model["checkpoint_sha256"]


def test_table_and_figure_are_explicitly_non_primary() -> None:
    table = (
        PROJECT_ROOT / "paper_assets/Table4_model_seed_stability.tex"
    ).read_text(encoding="utf-8")
    assert "Legacy PPO negative-control seed stability" in table
    assert "Only the PPO training seed changes" in table
    assert "unreferenced development diagnostic is not part of" in table
    assert "does not represent three independently trained full REIM systems" in table
    for relative in (
        "results/figures/model_seed_stability.png",
        "results/figures/model_seed_stability.pdf",
        "paper_assets/Figure6_model_seed_stability.png",
        "paper_assets/Figure6_model_seed_stability.pdf",
    ):
        assert (PROJECT_ROOT / relative).is_file()
