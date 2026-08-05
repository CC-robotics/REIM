from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_multitask_pipeline import (
    BANK_AUDIT_SCHEMA_VERSION,
    BANK_ROLES,
    EXPECTED_DISTURBANCE_LEVELS,
    PIPELINE_SCHEMA_VERSION,
    PROJECT_ROOT,
    TUNER_SCHEMA_VERSION,
    PipelineCommandError,
    PipelineConfigurationError,
    StageCommand,
    _canonical_json_sha256,
    _materialize_command,
    _plan_payload,
    _sha256,
    build_parser,
    build_stage_commands,
    execute_commands,
    load_context,
    main,
)


def _option(command: StageCommand, name: str) -> str:
    index = command.argv.index(name)
    return command.argv[index + 1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_mt10_configs(root: Path) -> None:
    config_dir = root / "configs" / "multitask"
    config_dir.mkdir(parents=True)
    for name in (
        "mt10.yaml",
        "mt10_act.yaml",
        "mt10_mlp.yaml",
        "mt10_detector.yaml",
    ):
        shutil.copy2(PROJECT_ROOT / "configs" / "multitask" / name, config_dir / name)


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
    assert "consumed at runtime by evaluation" in output.out
    assert "separate preregistered early-coverage threshold" in output.out
    assert "missing or stale artifacts fail closed" in output.out


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
    assert _option(commands[0], "--threshold") == (
        "<validated-threshold:results/tables/mt10_detector_threshold.json>"
    )
    assert commands[0].threshold_binding is not None


def test_disturbed_protocol_has_exact_five_levels_and_no_retry() -> None:
    context = load_context("MT50", python="python-for-test")
    commands = build_stage_commands(context, "evaluate_disturbed")
    levels = tuple(float(_option(command, "--noise-level")) for command in commands)
    assert levels == EXPECTED_DISTURBANCE_LEVELS
    assert all("act_retry" not in command.argv for command in commands)
    assert all("mlp_bc" in command.argv for command in commands)
    assert all("--mlp-checkpoint" in command.argv for command in commands)
    assert all(command.threshold_binding is not None for command in commands)
    assert all(
        _option(command, "--threshold")
        == "<validated-threshold:results/tables/mt50_detector_threshold.json>"
        for command in commands
    )
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
    assert [command.stage for command in commands[12:17]] == [
        "evaluate_disturbed"
    ] * 5
    assert commands[-1].stage == "audit_banks"
    assert len(commands) == 18
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


def test_recovery_collection_uses_separate_early_coverage_threshold_and_budget() -> None:
    context = load_context("MT10", python="python-for-test")
    command = build_stage_commands(context, "collect_recovery")[0]
    assert _option(command, "--threshold") == "0.2"
    assert _option(command, "--max-attempts-multiplier") == "20"
    assert command.threshold_binding is None


def test_recovery_attempt_budget_falls_back_for_legacy_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "multitask"
    config_dir.mkdir(parents=True)
    for name in (
        "mt10.yaml",
        "mt10_act.yaml",
        "mt10_mlp.yaml",
        "mt10_detector.yaml",
    ):
        shutil.copy2(PROJECT_ROOT / "configs" / "multitask" / name, config_dir / name)
    benchmark_path = config_dir / "mt10.yaml"
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    benchmark["data"].pop("recovery_max_attempts_multiplier", None)
    benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
    context = load_context("MT10", root=tmp_path, python="python-for-test")
    command = build_stage_commands(context, "collect_recovery")[0]
    assert _option(command, "--max-attempts-multiplier") == "5"


def test_bank_audit_stage_consumes_all_five_materialized_banks() -> None:
    context = load_context("MT50", python="python-for-test")
    command = build_stage_commands(context, "audit_banks")[0]
    assert command.argv[1] == "scripts/audit_multitask_banks.py"
    assert _option(command, "--benchmark") == "MT50"
    assert _option(command, "--demonstrations") == (
        "datasets/mt50/demonstrations/manifest.json"
    )
    assert _option(command, "--failure-train") == (
        "datasets/mt50/failures/manifest.json"
    )
    assert _option(command, "--failure-validation") == (
        "datasets/mt50/failures_validation/manifest.json"
    )
    assert _option(command, "--recovery") == "datasets/mt50/recovery/manifest.json"
    assert _option(command, "--final-evaluation-sidecar") == (
        "results/tables/mt50_clean_episodes.csv.run.json"
    )
    assert _option(command, "--final-evaluation-csv") == (
        "results/tables/mt50_clean_episodes.csv"
    )
    assert _option(command, "--output-json") == (
        "results/audits/mt50_bank_separation.json"
    )


def test_v5_plan_records_gated_audits_and_runtime_threshold_binding() -> None:
    context = load_context("MT10", python="python-for-test")
    commands = build_stage_commands(context, "all")
    payload = _plan_payload(context, "all", commands)
    assert payload["schema_version"] == PIPELINE_SCHEMA_VERSION
    assert PIPELINE_SCHEMA_VERSION == "reim-multitask-pipeline-v5"
    assert payload["artifacts"]["failure_training_raw"] == (
        "datasets/mt10/failures"
    )
    assert payload["artifacts"]["failure_training_calibrated"] == (
        "datasets/mt10/failures_calibrated"
    )
    tuned = payload["artifacts"]["detector_threshold_tuning"]
    assert tuned["json"] == "results/tables/mt10_detector_threshold.json"
    assert tuned["csv"] == "results/tables/mt10_detector_threshold_grid.csv"
    assert tuned["consumption"] == "fail_closed_runtime_binding"
    assert payload["artifacts"]["bank_separation_audit"] == {
        "json": "results/audits/mt10_bank_separation.json",
        "schema_version": BANK_AUDIT_SCHEMA_VERSION,
        "producer_stage": "audit_banks",
        "publication_gate_requirement": True,
        "write_mode": "atomic_after_complete_pass",
    }
    assert payload["runtime_threshold_binding"] == {
        "source": "validation_tuner_json",
        "artifact": "results/tables/mt10_detector_threshold.json",
        "schema_version": TUNER_SCHEMA_VERSION,
        "consumers": ["evaluate_clean", "evaluate_disturbed"],
        "missing_or_stale_artifact": "fail_closed",
        "configured_deployment_fallback_enabled": False,
        "release_threshold": 0.15,
    }
    assert payload["recovery_collection_threshold"] == {
        "source": "preregistered_benchmark_config",
        "value": 0.2,
        "purpose": "early_coverage_training_collection",
        "distinct_from_frozen_deployment_gate": True,
    }
    command_payloads = {item["label"]: item for item in payload["commands"]}
    assert command_payloads["collect_recovery"]["threshold_dependency"] is None
    assert command_payloads["evaluate_clean"]["threshold_dependency"] == {
        "artifact": "results/tables/mt10_detector_threshold.json",
        "placeholder": (
            "<validated-threshold:results/tables/mt10_detector_threshold.json>"
        ),
        "resolution": "validated_immediately_before_subprocess",
    }
    assert command_payloads["audit_banks"]["threshold_dependency"] is None


def _make_valid_threshold_contract(
    tmp_path: Path,
) -> tuple[StageCommand, Path, dict[str, object], Path]:
    _copy_mt10_configs(tmp_path)
    context = load_context("MT10", root=tmp_path, python="python-for-test")
    banks = context.config["banks"]
    vocabulary_digest = "1" * 64
    task_thresholds = [
        {
            "samples": 100,
            "task_id": 0,
            "task_name": "reach-v3",
            "threshold": 0.75,
        }
    ]
    training_calibration = {
        "dataset_role": "training",
        "mode": "fit-task-quantile",
        "prediction_horizon": 10,
        "quantile": 0.9,
        "task_thresholds": task_thresholds,
        "terminal_positive_horizon": 25,
    }
    training_calibration_fingerprint = _canonical_json_sha256(
        training_calibration
    )
    training_manifest_path = (
        tmp_path / "datasets" / "mt10" / "failures_calibrated" / "manifest.json"
    )
    _write_json(
        training_manifest_path,
        {
            "benchmark": "MT10",
            "benchmark_seed": banks["failure_training"],
            "complete": True,
            "label_calibration": training_calibration,
            "label_calibration_fingerprint_sha256": (
                training_calibration_fingerprint
            ),
            "task_vocabulary_sha256": vocabulary_digest,
        },
    )
    training_manifest_digest = _sha256(training_manifest_path)

    validation_provenance = {
        "benchmark_seed": banks["validation"],
        "collection_seed": banks["validation"],
    }
    validation_provenance_fingerprint = _canonical_json_sha256(
        validation_provenance
    )
    validation_calibration = {
        "calibration_source": {
            "kind": "frozen_training_calibration_manifest",
            "manifest_sha256": training_manifest_digest,
            "sha256": training_calibration_fingerprint,
        },
        "dataset_role": "validation",
        "mode": "frozen-task-thresholds",
        "prediction_horizon": 10,
        "quantile": 0.9,
        "task_thresholds": task_thresholds,
        "terminal_positive_horizon": 25,
    }
    validation_calibration_fingerprint = _canonical_json_sha256(
        validation_calibration
    )
    validation_manifest_path = (
        tmp_path
        / "datasets"
        / "mt10"
        / "failures_validation_calibrated"
        / "manifest.json"
    )
    _write_json(
        validation_manifest_path,
        {
            "benchmark": "MT10",
            "benchmark_seed": banks["validation"],
            "complete": True,
            "label_calibration": validation_calibration,
            "label_calibration_fingerprint_sha256": (
                validation_calibration_fingerprint
            ),
            "label_calibration_source_sha256": (
                training_calibration_fingerprint
            ),
            "provenance": validation_provenance,
            "provenance_fingerprint_sha256": validation_provenance_fingerprint,
            "seed": banks["validation"],
            "task_vocabulary_sha256": vocabulary_digest,
        },
    )
    validation_manifest_digest = _sha256(validation_manifest_path)

    detector_path = (
        tmp_path / "checkpoints" / "mt10" / "seed_42" / "failure_detector.pt"
    )
    detector_path.parent.mkdir(parents=True)
    detector_path.write_bytes(b"detector-checkpoint-for-provenance-test")
    tuner_path = (
        tmp_path / "results" / "tables" / "mt10_detector_threshold.json"
    )
    tuner_payload: dict[str, object] = {
        "bank_role": "validation_only",
        "benchmark": "MT10",
        "final_bank_accessed": False,
        "outputs": {
            "csv": str(tuner_path.with_suffix(".csv")),
            "json": str(tuner_path),
        },
        "provenance": {
            "detector_checkpoint": str(detector_path),
            "detector_checkpoint_sha256": _sha256(detector_path),
            "detector_training_manifest_sha256": training_manifest_digest,
            "reserved_final_evaluation_seed": banks["final_evaluation"],
            "seed": context.config["model_seed"],
            "task_vocabulary_sha256": vocabulary_digest,
            "validation_benchmark_seed": banks["validation"],
            "validation_collection_seed": banks["validation"],
            "validation_manifest": str(validation_manifest_path),
            "validation_manifest_sha256": validation_manifest_digest,
            "validation_provenance_fingerprint_sha256": (
                validation_provenance_fingerprint
            ),
        },
        "schema_version": TUNER_SCHEMA_VERSION,
        "selection": {"threshold": 0.69},
        "threshold_grid": [{"threshold": 0.69}],
    }
    _write_json(tuner_path, tuner_payload)
    planned = build_stage_commands(context, "evaluate_clean")[0]
    return planned, tuner_path, tuner_payload, detector_path


def test_runtime_binding_materializes_validated_tuner_threshold(
    tmp_path: Path,
) -> None:
    planned, _, _, _ = _make_valid_threshold_contract(tmp_path)
    assert planned.threshold_binding is not None
    assert _option(planned, "--threshold").startswith("<validated-threshold:")
    materialized = _materialize_command(planned)
    assert materialized.threshold_binding is None
    assert _option(materialized, "--threshold") == "0.69"


def test_evaluation_dry_plan_does_not_require_tuner_artifact(tmp_path: Path) -> None:
    _copy_mt10_configs(tmp_path)
    context = load_context("MT10", root=tmp_path, python="python-for-test")
    command = build_stage_commands(context, "evaluate_clean")[0]
    assert not (
        tmp_path / "results" / "tables" / "mt10_detector_threshold.json"
    ).exists()
    assert _option(command, "--threshold").startswith("<validated-threshold:")
    with pytest.raises(PipelineConfigurationError, match="missing"):
        _materialize_command(command)


def test_execute_reads_tuner_threshold_after_plan_construction(tmp_path: Path) -> None:
    planned, tuner_path, payload, _ = _make_valid_threshold_contract(tmp_path)
    assert planned.threshold_binding is not None
    payload["selection"] = {"threshold": 0.71}
    payload["threshold_grid"] = [{"threshold": 0.71}]
    _write_json(tuner_path, payload)
    marker = tmp_path / "resolved_threshold.txt"
    command = StageCommand(
        stage="test",
        label="runtime_threshold",
        argv=(
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
            ),
            str(marker),
            planned.threshold_binding.placeholder,
        ),
        log_path=tmp_path / "runtime_threshold.log",
        threshold_binding=planned.threshold_binding,
    )
    execute_commands([command], cwd=tmp_path)
    assert marker.read_text(encoding="utf-8") == "0.71"


def test_runtime_binding_fails_closed_for_missing_or_stale_provenance(
    tmp_path: Path,
) -> None:
    planned, tuner_path, payload, detector_path = _make_valid_threshold_contract(
        tmp_path
    )
    original = json.loads(json.dumps(payload))

    tuner_path.unlink()
    with pytest.raises(PipelineConfigurationError, match="missing"):
        _materialize_command(planned)

    stale_seed = json.loads(json.dumps(original))
    stale_seed["provenance"]["validation_collection_seed"] += 1
    _write_json(tuner_path, stale_seed)
    with pytest.raises(PipelineConfigurationError, match="validation_collection_seed"):
        _materialize_command(planned)

    _write_json(tuner_path, original)
    detector_path.write_bytes(b"changed-after-tuning")
    with pytest.raises(PipelineConfigurationError, match="detector_checkpoint_sha256"):
        _materialize_command(planned)


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


@pytest.mark.parametrize("colliding_role", BANK_ROLES[1:])
def test_context_rejects_collisions_across_all_five_bank_seeds(
    tmp_path: Path, colliding_role: str
) -> None:
    _copy_mt10_configs(tmp_path)
    benchmark_path = tmp_path / "configs" / "multitask" / "mt10.yaml"
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    benchmark["banks"][colliding_role] = benchmark["banks"]["demonstrations"]
    benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
    with pytest.raises(PipelineConfigurationError, match="all be pairwise distinct"):
        load_context("MT10", root=tmp_path, python="python-for-test")


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
