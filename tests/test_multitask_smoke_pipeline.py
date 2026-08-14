"""Tests for the isolated smoke profile of the multi-task pipeline runner."""

from __future__ import annotations

import pytest

from scripts.run_multitask_pipeline import (
    PipelineConfigurationError,
    STAGES,
    _contexts_from_args,
    _plan_payload,
    build_parser,
    build_plan,
    build_stage_commands,
    load_context,
)


ENV_DRIVING_STAGES = {
    "collect_demos",
    "generate_failures",
    "generate_failure_validation",
    "collect_recovery",
    "evaluate_clean",
    "audit_banks",
}


def _smoke_context():
    return load_context("MT10", profile="smoke")


def test_smoke_context_uses_isolated_roots() -> None:
    context = _smoke_context()
    assert context.profile == "smoke"
    assert context.backend == "toy"
    for path in (
        context.config_path,
        context.act_config_path,
        context.mlp_config_path,
        context.detector_config_path,
    ):
        assert path.is_file(), path
        assert path.parent.name == "multitask"
        assert path.name.startswith("smoke")
    assert context.datasets_root.match("*/datasets/smoke/multitask")
    assert context.checkpoints_root.match("*/checkpoints/smoke/multitask")
    assert context.tables_dir.match("*/results/smoke/multitask/tables")
    assert context.audits_dir.match("*/results/smoke/multitask/audits")
    assert context.log_dir.match("*/results/smoke/multitask/logs")


def test_smoke_profile_rejects_mt50_and_both() -> None:
    with pytest.raises(PipelineConfigurationError):
        load_context("MT50", profile="smoke")
    parser = build_parser()
    args = parser.parse_args(["all", "both", "--smoke"])
    with pytest.raises(PipelineConfigurationError):
        _contexts_from_args(args)


def test_smoke_plan_covers_all_stages_with_toy_backend() -> None:
    context = _smoke_context()
    commands = build_stage_commands(context, "all")
    stages = [command.stage for command in commands]
    for stage in STAGES:
        assert stage in stages
    for command in commands:
        argv = list(command.argv)
        if command.stage in ENV_DRIVING_STAGES or command.stage.startswith(
            "evaluate_disturbed"
        ):
            assert "--backend" in argv, command.label
            assert argv[argv.index("--backend") + 1] == "toy"
        # No smoke command may touch the production mt1/mt10 artifact trees.
        for value in argv[2:]:
            normalized = value.replace("\\", "/")
            assert "datasets/mt10" not in normalized, command.label
            assert "checkpoints/mt10" not in normalized, command.label
            assert "datasets/demonstrations" not in normalized, command.label
            assert "results/tables" not in normalized, command.label
            assert "results/audits" not in normalized, command.label
            assert "results/figures" not in normalized, command.label
            assert "results/logs" not in normalized, command.label
    plan = build_plan([context], "all")
    payload = _plan_payload(context, "all", plan)
    assert payload["profile"] == "smoke"
    assert payload["backend"] == "toy"


def test_full_profile_plan_is_unchanged_by_smoke_additions() -> None:
    context = load_context("MT10")
    assert context.profile == "full"
    assert context.backend == "metaworld"
    commands = build_stage_commands(context, "all")
    for command in commands:
        assert "--backend" not in command.argv
    payload = _plan_payload(context, "all", commands)
    assert payload["profile"] == "full"
    assert payload["backend"] == "metaworld"
    assert "datasets/mt10/failures" in payload["artifacts"]["failure_training_raw"]
    assert payload["artifacts"]["detector_threshold_tuning"]["json"] == (
        "results/tables/mt10_detector_threshold.json"
    )
