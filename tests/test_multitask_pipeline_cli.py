from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_multitask_pipeline import (
    EXPECTED_DISTURBANCE_LEVELS,
    PIPELINE_SCHEMA_VERSION,
    PROJECT_ROOT,
    PipelineCommandError,
    PipelineConfigurationError,
    StageCommand,
    _plan_payload,
    build_parser,
    build_stage_commands,
    execute_commands,
    load_context,
    main,
)


def _option(command: StageCommand, name: str) -> str:
    index = command.argv.index(name)
    return command.argv[index + 1]


@pytest.mark.parametrize("benchmark,state_dim", [("MT10", 49), ("MT50", 89)])
def test_context_loads_official_benchmark_configs(benchmark: str, state_dim: int) -> None:
    context = load_context(benchmark, python="python-for-test")
    assert context.config["state_dim"] == state_dim
    assert context.config["max_episode_steps"] == 500
    assert context.config["disturbance"]["object_position_noise"] is False
    assert context.mlp_config["benchmark"] == benchmark


def test_cli_is_dry_run_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["collect_demos", "MT10", "--python", "python-for-test"]) == 0
    output = capsys.readouterr()
    assert "mode=DRY-RUN" in output.out
    assert "no collection, training, evaluation, or output writes occurred" in output.out
    assert "results/tables/mt10_detector_threshold.json" in output.out
    assert "still use config thresholds" in output.out


def test_stage_subcommands_are_explicit() -> None:
    parser = build_parser()
    args = parser.parse_args(["train_mlp", "MT50"])
    assert args.stage == "train_mlp"
    assert args.benchmark == "MT50"
    assert args.execute is False


def test_clean_protocol_excludes_retry() -> None:
    context = load_context("MT10", python="python-for-test")
    commands = build_stage_commands(context, "evaluate_clean")
    assert len(commands) == 1
    argv = commands[0].argv
    method_index = argv.index("--methods")
    episode_index = argv.index("--episodes-per-task")
    methods = argv[method_index + 1 : episode_index]
    assert methods == ("mlp_bc", "act", "heuristic_recovery", "reim")
    assert "act_retry" not in argv
    assert _option(commands[0], "--mlp-checkpoint") == (
        "checkpoints/mt10/seed_42/mlp_bc.pt"
    )
    assert _option(commands[0], "--noise-level") == "0"
    assert _option(commands[0], "--max-steps") == "500"
    assert _option(commands[0], "--release-threshold") == "0.15"
    assert _option(commands[0], "--release-patience") == "5"
    assert _option(commands[0], "--min-recovery-steps") == "5"
    assert _option(commands[0], "--intervention-cooldown") == "10"


def test_disturbed_protocol_has_exact_five_levels_and_no_retry() -> None:
    context = load_context("MT50", python="python-for-test")
    commands = build_stage_commands(context, "evaluate_disturbed")
    levels = tuple(float(_option(command, "--noise-level")) for command in commands)
    assert levels == EXPECTED_DISTURBANCE_LEVELS
    assert all("act_retry" not in command.argv for command in commands)
    assert all("mlp_bc" in command.argv for command in commands)
    assert all("--mlp-checkpoint" in command.argv for command in commands)
    assert [command.label for command in commands] == [
        "evaluate_disturbed_noise_00",
        "evaluate_disturbed_noise_10",
        "evaluate_disturbed_noise_20",
        "evaluate_disturbed_noise_30",
        "evaluate_disturbed_noise_40",
    ]


def test_all_stage_order_and_existing_cli_paths() -> None:
    context = load_context("MT10", python="python-for-test")
    commands = build_stage_commands(context, "all")
    assert [command.stage for command in commands[:12]] == [
        "collect_demos",
        "train_act",
        "train_mlp",
        "generate_failures",
        "calibrate_failures",
        "train_detector",
        "generate_failure_validation",
        "calibrate_failure_validation",
        "tune_detector",
        "collect_recovery",
        "train_recovery",
        "evaluate_clean",
    ]
    assert [command.stage for command in commands[12:]] == ["evaluate_disturbed"] * 5
    assert len(commands) == 17
    for command in commands:
        entrypoint = Path(command.argv[1])
        assert (PROJECT_ROOT / entrypoint).is_file()


def test_failure_collection_and_task_conditional_calibration_paths_are_separate() -> None:
    context = load_context("MT10", python="python-for-test")
    generated = build_stage_commands(context, "generate_failures")[0]
    assert _option(generated, "--output-dir") == "datasets/mt10/failures"
    assert _option(generated, "--rollouts-per-task") == "50"
    assert _option(generated, "--benchmark-seed") == "20262010"
    assert _option(generated, "--seed") == "42"

    calibrated = build_stage_commands(context, "calibrate_failures")[0]
    assert calibrated.argv[1] == "scripts/relabel_multitask_failures.py"
    assert _option(calibrated, "--data-dir") == "datasets/mt10/failures"
    assert _option(calibrated, "--output-dir") == (
        "datasets/mt10/failures_calibrated"
    )
    assert _option(calibrated, "--mode") == "fit-task-quantile"
    assert _option(calibrated, "--quantile") == "0.9"
    assert _option(calibrated, "--dataset-role") == "training"
    assert _option(calibrated, "--prediction-horizon") == "10"
    assert _option(calibrated, "--terminal-positive-horizon") == "25"


def test_validation_bank_is_independent_and_reuses_frozen_training_calibration() -> None:
    context = load_context("MT50", python="python-for-test")
    generated = build_stage_commands(context, "generate_failure_validation")[0]
    assert _option(generated, "--output-dir") == (
        "datasets/mt50/failures_validation"
    )
    assert _option(generated, "--rollouts-per-task") == "20"
    assert _option(generated, "--benchmark-seed") == "20264050"
    assert _option(generated, "--seed") == "20264050"
    assert _option(generated, "--benchmark-seed") != "20262050"
    assert _option(generated, "--benchmark-seed") != "20265050"

    calibrated = build_stage_commands(
        context, "calibrate_failure_validation"
    )[0]
    assert _option(calibrated, "--data-dir") == (
        "datasets/mt50/failures_validation"
    )
    assert _option(calibrated, "--output-dir") == (
        "datasets/mt50/failures_validation_calibrated"
    )
    assert _option(calibrated, "--mode") == "frozen-task-thresholds"
    assert _option(calibrated, "--dataset-role") == "validation"
    assert _option(calibrated, "--calibration-manifest") == (
        "datasets/mt50/failures_calibrated/manifest.json"
    )


def test_detector_tuning_has_validation_only_inputs_and_explicit_outputs() -> None:
    context = load_context("MT10", python="python-for-test")
    tuned = build_stage_commands(context, "tune_detector")[0]
    assert tuned.argv[1] == "evaluation/tune_multitask_detector.py"
    assert _option(tuned, "--validation-data") == (
        "datasets/mt10/failures_validation_calibrated"
    )
    assert _option(tuned, "--detector-checkpoint") == (
        "checkpoints/mt10/seed_42/failure_detector.pt"
    )
    assert _option(tuned, "--protocol-config") == "configs/multitask/mt10.yaml"
    assert _option(tuned, "--validation-bank-seed") == "20264010"
    assert _option(tuned, "--validation-benchmark-seed") == "20264010"
    assert _option(tuned, "--output-json") == (
        "results/tables/mt10_detector_threshold.json"
    )
    assert _option(tuned, "--output-csv") == (
        "results/tables/mt10_detector_threshold_grid.csv"
    )


def test_v3_plan_records_tuned_artifact_without_applying_it() -> None:
    context = load_context("MT10", python="python-for-test")
    commands = build_stage_commands(context, "all")
    payload = _plan_payload(context, "all", commands)
    assert payload["schema_version"] == PIPELINE_SCHEMA_VERSION
    assert PIPELINE_SCHEMA_VERSION == "reim-multitask-pipeline-v3"
    assert payload["artifacts"]["failure_training_raw"] == (
        "datasets/mt10/failures"
    )
    assert payload["artifacts"]["failure_training_calibrated"] == (
        "datasets/mt10/failures_calibrated"
    )
    tuned = payload["artifacts"]["detector_threshold_tuning"]
    assert tuned["json"] == "results/tables/mt10_detector_threshold.json"
    assert tuned["csv"] == "results/tables/mt10_detector_threshold_grid.csv"
    assert tuned["consumption"] == "record_only_not_applied_by_this_pipeline_schema"
    assert payload["configured_runtime_thresholds"] == {
        "source": "benchmark_config_not_tuned_artifact",
        "deployment_threshold": 0.3,
        "collection_threshold": 0.2,
        "release_threshold": 0.15,
    }


def test_execute_commands_stops_on_first_failure(tmp_path: Path) -> None:
    marker = tmp_path / "must_not_exist"
    commands = [
        StageCommand(
            stage="test",
            label="failure",
            argv=(sys.executable, "-c", "raise SystemExit(7)"),
            log_path=tmp_path / "failure.log",
        ),
        StageCommand(
            stage="test",
            label="later",
            argv=(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ),
            log_path=tmp_path / "later.log",
        ),
    ]
    with pytest.raises(PipelineCommandError, match="exit status 7"):
        execute_commands(commands, cwd=tmp_path)
    assert not marker.exists()


def test_resume_only_targets_existing_training_state(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "multitask"
    config_dir.mkdir(parents=True)
    for name in (
        "mt10.yaml",
        "mt10_act.yaml",
        "mt10_mlp.yaml",
        "mt10_detector.yaml",
    ):
        shutil.copy2(PROJECT_ROOT / "configs" / "multitask" / name, config_dir / name)
    context = load_context(
        "MT10", root=tmp_path, python="python-for-test", resume=True
    )
    fresh = build_stage_commands(context, "train_recovery")[0]
    assert "--resume" not in fresh.argv

    mlp_latest = tmp_path / "checkpoints" / "mt10" / "seed_42" / "mlp_bc_latest.pt"
    mlp_latest.parent.mkdir(parents=True)
    mlp_latest.touch()
    resumed_mlp = build_stage_commands(context, "train_mlp")[0]
    assert resumed_mlp.argv[-2:] == ("--resume", str(mlp_latest))

    # The production path cannot be touched by this unit test, so exercise the
    # same fail-safe contract with a standalone command marker assertion.
    marker = tmp_path / "recovery_latest.pt"
    marker.touch()
    argv = ["python", "trainer.py"]
    from scripts.run_multitask_pipeline import _append_resume_if_present

    _append_resume_if_present(argv, requested=True, marker=marker, explicit_path=True)
    assert argv[-2:] == ["--resume", str(marker)]


def test_context_rejects_validation_seed_leakage_and_detector_path_drift(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs" / "multitask"
    config_dir.mkdir(parents=True)
    names = (
        "mt10.yaml",
        "mt10_act.yaml",
        "mt10_mlp.yaml",
        "mt10_detector.yaml",
    )
    for name in names:
        shutil.copy2(PROJECT_ROOT / "configs" / "multitask" / name, config_dir / name)

    benchmark_path = config_dir / "mt10.yaml"
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    benchmark["banks"]["validation"] = benchmark["banks"]["failure_training"]
    benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
    with pytest.raises(PipelineConfigurationError, match="pairwise distinct"):
        load_context("MT10", root=tmp_path, python="python-for-test")

    shutil.copy2(
        PROJECT_ROOT / "configs" / "multitask" / "mt10.yaml", benchmark_path
    )
    detector_path = config_dir / "mt10_detector.yaml"
    detector = yaml.safe_load(detector_path.read_text(encoding="utf-8"))
    detector["data_dir"] = "datasets/mt10/failures"
    detector_path.write_text(yaml.safe_dump(detector), encoding="utf-8")
    with pytest.raises(PipelineConfigurationError, match="calibrated training bank"):
        load_context("MT10", root=tmp_path, python="python-for-test")


def test_shell_wrapper_is_syntax_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(PROJECT_ROOT / "run_multitask.sh")],
        check=True,
        capture_output=True,
        text=True,
    )
