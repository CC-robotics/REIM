"""Tests for the publication-result gate and frozen recovery semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.update_paper_results import (
    FINAL_RECOVERY_DEFINITION,
    PaperResultError,
    _paired_gain,
    _render_macros,
    _validate_condition,
)


EPISODE_BANK_SHA256 = "a" * 64
EPISODE_BANK_FILE_SHA256 = "b" * 64
EPISODE_SPECIFICATION_SHA256 = "c" * 64
METAWORLD_TASK_SHA256 = "d" * 64
REIM_METHOD = "REIM (ACT + Detector + Recovery)"
HEURISTIC_RECOVERY_METHOD = "ACT + Heuristic Recovery"


def _summary(
    definition: str, *, method: str = REIM_METHOD
) -> dict[str, object]:
    return {
        "Method": method,
        "Success Rate": 1.0,
        "Success CI Lower": 0.2,
        "Success CI Upper": 1.0,
        "Recovery Rate": 1.0,
        "Recovery CI Lower": 0.2,
        "Recovery CI Upper": 1.0,
        "Average Steps": 10.0,
        "Episodes": 1,
        "Successes": 1,
        "Recovery Attempts": 1,
        "Recovery Successes": 1,
        "Detector Triggers": 1,
        "Backend": "metaworld",
        "Benchmark Eligible": "True",
        "Profile": "full",
        "Noise Level": 0.2,
        "Recovery Definition": definition,
        "Intervened Episodes": 1,
        "Evaluation Seed Start": 9_000_001,
        "Evaluation Seed End": 9_000_001,
        "Episode Bank SHA256": EPISODE_BANK_SHA256,
        "Episode Bank File SHA256": EPISODE_BANK_FILE_SHA256,
        "CRN Episode Specifications Verified": "True",
    }


def _raw(*, method: str = REIM_METHOD) -> dict[str, object]:
    return {
        "method": method,
        "episode": 0,
        "seed": 9_000_001,
        "backend": "metaworld",
        "success": "True",
        "steps": 10,
        "recovery_attempts": 1,
        "recovery_successes": 1,
        "recovery_definition": FINAL_RECOVERY_DEFINITION,
        "detector_triggers": 1,
        "Profile": "full",
        "noise_level": 0.2,
        "episode_specification_sha256": EPISODE_SPECIFICATION_SHA256,
        "episode_bank_sha256": EPISODE_BANK_SHA256,
        "metaworld_task_sha256": METAWORLD_TASK_SHA256,
        "retry_specification_sha256s": "",
        "retry_task_sha256s": "",
    }


def test_final_recovery_definition_accepts_active_controller_completion() -> None:
    _validate_condition(
        summary=_summary(FINAL_RECOVERY_DEFINITION),
        raw=[_raw()],
        method=REIM_METHOD,
        episodes=1,
        noise=0.2,
        expected_seeds={9_000_001},
    )


def test_legacy_recovery_definition_is_rejected() -> None:
    with pytest.raises(PaperResultError, match="Recovery Definition"):
        _validate_condition(
            summary=_summary("risk_clear_return_to_ACT_per_intervention"),
            raw=[_raw()],
            method=REIM_METHOD,
            episodes=1,
            noise=0.2,
            expected_seeds={9_000_001},
        )


def test_heuristic_recovery_label_uses_active_controller_completion() -> None:
    _validate_condition(
        summary=_summary(
            FINAL_RECOVERY_DEFINITION,
            method=HEURISTIC_RECOVERY_METHOD,
        ),
        raw=[_raw(method=HEURISTIC_RECOVERY_METHOD)],
        method=HEURISTIC_RECOVERY_METHOD,
        episodes=1,
        noise=0.2,
        expected_seeds={9_000_001},
    )


def test_macro_renderer_preserves_protocol_and_enables_final_switch() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "paper_assets"
        / "reim_macros.tex"
    ).read_text(encoding="utf-8")
    rendered = _render_macros(source, {"FinalREIMGain": r"+12.0\,pp"})
    assert r"\REIMFinalResultstrue" in rendered
    assert r"\newcommand{\FinalREIMGain}{+12.0\,pp}" in rendered
    assert r"\newcommand{\FinalControllerTrigger}{0.20}" in rendered
    assert r"\newcommand{\PPOEpochsPerUpdate}{5}" in rendered


def test_paired_rescued_and_harmed_counts_match_success_gain(tmp_path: Path) -> None:
    paired = {
        "ACT": {1: False, 2: False, 3: True, 4: True},
        "REIM (ACT + Detector + Recovery)": {
            1: True,
            2: True,
            3: False,
            4: True,
        },
    }
    gain, low, high, rescued, harmed = _paired_gain(
        paired=paired,
        comparison_path=tmp_path / "absent_comparison.csv",
        baseline_reim={"Success Rate": 0.75},
        evaluation_seed=1,
    )
    assert gain == pytest.approx(0.25)
    assert low <= gain <= high
    assert rescued == 2
    assert harmed == 1
