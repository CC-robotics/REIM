#!/usr/bin/env python3
"""Plan or execute the reproducible REIM MT10/MT50 experiment pipeline.

The command is deliberately dry-run by default.  Expensive collection,
training, and evaluation only start when ``--execute`` is supplied.  Each
stage delegates to the existing, provenance-aware project CLI rather than
reimplementing any learning or evaluation logic here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCHEMA_VERSION = "reim-multitask-pipeline-v5"
TUNER_SCHEMA_VERSION = "reim-multitask-detector-threshold-v1"
BANK_AUDIT_SCHEMA_VERSION = "reim-multitask-bank-separation-audit-v1"
BENCHMARKS = ("MT10", "MT50")
BANK_ROLES = (
    "demonstrations",
    "failure_training",
    "recovery_training",
    "validation",
    "final_evaluation",
)
STAGES = (
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
    "evaluate_disturbed",
    "audit_banks",
)
EXPECTED_DISTURBANCE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)


class PipelineConfigurationError(ValueError):
    """Raised when benchmark configuration cannot define a safe run."""


class PipelineCommandError(RuntimeError):
    """Raised immediately after a child stage returns a non-zero status."""

    def __init__(self, command: "StageCommand", returncode: int) -> None:
        super().__init__(
            f"Stage {command.label!r} failed with exit status {returncode}. "
            f"See {command.log_path}."
        )
        self.command = command
        self.returncode = int(returncode)


@dataclass(frozen=True)
class TunedThresholdBinding:
    """Fail-closed provenance contract for one validation-selected threshold."""

    placeholder: str
    artifact_path: Path
    benchmark: str
    root: Path
    failure_training_seed: int
    validation_seed: int
    final_evaluation_seed: int
    model_seed: int
    detector_checkpoint: Path
    calibrated_training_manifest: Path
    calibrated_validation_manifest: Path
    calibration_quantile: float
    prediction_horizon: int
    terminal_positive_horizon: int
    release_threshold: float


@dataclass(frozen=True)
class StageCommand:
    """One fail-fast subprocess in a pipeline plan."""

    stage: str
    label: str
    argv: tuple[str, ...]
    log_path: Path
    threshold_binding: TunedThresholdBinding | None = None

    def shell_line(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class PipelineContext:
    """Validated configuration and artifact locations for one benchmark."""

    benchmark: str
    root: Path
    python: str
    config_path: Path
    act_config_path: Path
    mlp_config_path: Path
    detector_config_path: Path
    config: Mapping[str, Any]
    act_config: Mapping[str, Any]
    mlp_config: Mapping[str, Any]
    detector_config: Mapping[str, Any]
    resume: bool
    device: str | None
    log_dir: Path

    @property
    def slug(self) -> str:
        return self.benchmark.lower()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PipelineConfigurationError(f"Expected a YAML mapping in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PipelineConfigurationError(
            "Threshold provenance is not canonical-JSON serializable"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256_value(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PipelineConfigurationError(
            f"Tuned-threshold field {field} is not a lowercase SHA-256 digest"
        )
    return text


def _read_json_artifact(path: Path, *, label: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise PipelineConfigurationError(
            f"Required {label} is missing or not a regular non-symlink file: {path}"
        )
    try:
        payload = json.loads(expanded.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineConfigurationError(f"Cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PipelineConfigurationError(f"{label} must contain a JSON object: {path}")
    return payload


def _recorded_path(value: Any, *, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PipelineConfigurationError(
            f"Tuned-threshold field {field} must be a non-empty path string"
        )
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_tuned_threshold(binding: TunedThresholdBinding) -> float:
    """Validate a tuner artifact and its complete calibration/checkpoint chain."""

    artifact_source = binding.artifact_path.expanduser()
    artifact_path = artifact_source.resolve()
    payload = _read_json_artifact(
        artifact_source, label=f"{binding.benchmark} validation tuner JSON"
    )
    expected_top_level = {
        "schema_version": TUNER_SCHEMA_VERSION,
        "benchmark": binding.benchmark,
        "bank_role": "validation_only",
    }
    for field, expected in expected_top_level.items():
        if payload.get(field) != expected:
            raise PipelineConfigurationError(
                f"Tuned-threshold {field}={payload.get(field)!r}, expected {expected!r}"
            )
    if payload.get("final_bank_accessed") is not False:
        raise PipelineConfigurationError(
            "Tuned-threshold final_bank_accessed must be the boolean false"
        )

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise PipelineConfigurationError("Tuned-threshold JSON lacks selection metadata")
    raw_threshold = selection.get("threshold")
    if isinstance(raw_threshold, bool):
        raise PipelineConfigurationError(
            "Tuned-threshold selection.threshold must be numeric"
        )
    try:
        threshold = float(raw_threshold)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineConfigurationError(
            "Tuned-threshold selection.threshold must be numeric"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise PipelineConfigurationError(
            "Tuned-threshold selection.threshold must be finite and in [0, 1]"
        )
    if threshold <= binding.release_threshold:
        raise PipelineConfigurationError(
            "Validation-selected detector threshold must exceed the configured "
            f"release threshold ({threshold} <= {binding.release_threshold})"
        )
    threshold_grid = payload.get("threshold_grid")
    if not isinstance(threshold_grid, list) or not any(
        isinstance(row, Mapping)
        and isinstance(row.get("threshold"), (int, float))
        and not isinstance(row.get("threshold"), bool)
        and math.isclose(float(row["threshold"]), threshold, rel_tol=0.0, abs_tol=1e-12)
        for row in threshold_grid
    ):
        raise PipelineConfigurationError(
            "Validation-selected threshold is absent from the recorded threshold grid"
        )

    training_source = binding.calibrated_training_manifest.expanduser()
    validation_source = binding.calibrated_validation_manifest.expanduser()
    training_path = training_source.resolve()
    validation_path = validation_source.resolve()
    training_manifest = _read_json_artifact(
        training_source, label="calibrated failure-training manifest"
    )
    validation_manifest = _read_json_artifact(
        validation_source, label="calibrated failure-validation manifest"
    )
    detector_source = binding.detector_checkpoint.expanduser()
    detector_path = detector_source.resolve()
    if detector_source.is_symlink() or not detector_source.is_file():
        raise PipelineConfigurationError(
            f"Detector checkpoint is missing or not a regular non-symlink file: {detector_path}"
        )
    training_digest = _sha256(training_path)
    validation_digest = _sha256(validation_path)
    detector_digest = _sha256(detector_path)

    for label, manifest, expected_seed in (
        ("training", training_manifest, binding.failure_training_seed),
        ("validation", validation_manifest, binding.validation_seed),
    ):
        if manifest.get("complete") is not True:
            raise PipelineConfigurationError(
                f"Calibrated {label} manifest is not marked complete"
            )
        if str(manifest.get("benchmark", "")).upper() != binding.benchmark:
            raise PipelineConfigurationError(
                f"Calibrated {label} manifest benchmark mismatch"
            )
        try:
            manifest_seed = int(manifest["benchmark_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineConfigurationError(
                f"Calibrated {label} manifest lacks a valid benchmark_seed"
            ) from exc
        if manifest_seed != expected_seed:
            raise PipelineConfigurationError(
                f"Calibrated {label} benchmark_seed={manifest_seed}, "
                f"expected {expected_seed}"
            )

    validation_provenance = validation_manifest.get("provenance")
    if not isinstance(validation_provenance, Mapping):
        raise PipelineConfigurationError(
            "Calibrated validation manifest lacks structured provenance"
        )
    validation_provenance_fingerprint = _require_sha256_value(
        validation_manifest.get("provenance_fingerprint_sha256"),
        field="validation_manifest.provenance_fingerprint_sha256",
    )
    if _canonical_json_sha256(validation_provenance) != validation_provenance_fingerprint:
        raise PipelineConfigurationError(
            "Calibrated validation manifest provenance fingerprint mismatch"
        )
    for field in ("collection_seed", "benchmark_seed"):
        try:
            stored_seed = int(validation_provenance[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineConfigurationError(
                f"Calibrated validation provenance lacks a valid {field}"
            ) from exc
        if stored_seed != binding.validation_seed:
            raise PipelineConfigurationError(
                f"Calibrated validation provenance {field}={stored_seed}, "
                f"expected {binding.validation_seed}"
            )
    try:
        top_level_validation_seed = int(validation_manifest["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineConfigurationError(
            "Calibrated validation manifest lacks a valid collection seed"
        ) from exc
    if top_level_validation_seed != binding.validation_seed:
        raise PipelineConfigurationError(
            "Calibrated validation collection seed does not match the registered bank"
        )

    training_calibration = training_manifest.get("label_calibration")
    validation_calibration = validation_manifest.get("label_calibration")
    if not isinstance(training_calibration, Mapping) or not isinstance(
        validation_calibration, Mapping
    ):
        raise PipelineConfigurationError(
            "Training/validation manifests must contain structured label calibration"
        )
    expected_calibration = {
        "quantile": binding.calibration_quantile,
        "prediction_horizon": binding.prediction_horizon,
        "terminal_positive_horizon": binding.terminal_positive_horizon,
    }
    for field, expected in expected_calibration.items():
        if training_calibration.get(field) != expected or validation_calibration.get(
            field
        ) != expected:
            raise PipelineConfigurationError(
                f"Calibration field {field} does not match the registered protocol"
            )
    if (
        training_calibration.get("mode") != "fit-task-quantile"
        or training_calibration.get("dataset_role") != "training"
        or validation_calibration.get("mode") != "frozen-task-thresholds"
        or validation_calibration.get("dataset_role") != "validation"
    ):
        raise PipelineConfigurationError(
            "Validation tuner must consume frozen task thresholds fitted on training only"
        )
    training_calibration_fingerprint = _require_sha256_value(
        training_manifest.get("label_calibration_fingerprint_sha256"),
        field="training_manifest.label_calibration_fingerprint_sha256",
    )
    validation_calibration_fingerprint = _require_sha256_value(
        validation_manifest.get("label_calibration_fingerprint_sha256"),
        field="validation_manifest.label_calibration_fingerprint_sha256",
    )
    if _canonical_json_sha256(training_calibration) != training_calibration_fingerprint:
        raise PipelineConfigurationError(
            "Training label-calibration fingerprint mismatch"
        )
    if _canonical_json_sha256(validation_calibration) != validation_calibration_fingerprint:
        raise PipelineConfigurationError(
            "Validation label-calibration fingerprint mismatch"
        )
    calibration_source = validation_calibration.get("calibration_source")
    if not isinstance(calibration_source, Mapping):
        raise PipelineConfigurationError(
            "Validation label calibration lacks a frozen training source"
        )
    expected_source = {
        "kind": "frozen_training_calibration_manifest",
        "manifest_sha256": training_digest,
        "sha256": training_calibration_fingerprint,
    }
    for field, expected in expected_source.items():
        if calibration_source.get(field) != expected:
            raise PipelineConfigurationError(
                f"Validation calibration source {field} does not match training provenance"
            )
    if validation_manifest.get(
        "label_calibration_source_sha256"
    ) != training_calibration_fingerprint:
        raise PipelineConfigurationError(
            "Validation calibration source fingerprint does not match training"
        )
    if validation_calibration.get("task_thresholds") != training_calibration.get(
        "task_thresholds"
    ):
        raise PipelineConfigurationError(
            "Validation task thresholds are not the frozen training thresholds"
        )
    training_vocabulary = _require_sha256_value(
        training_manifest.get("task_vocabulary_sha256"),
        field="training_manifest.task_vocabulary_sha256",
    )
    if validation_manifest.get("task_vocabulary_sha256") != training_vocabulary:
        raise PipelineConfigurationError(
            "Training and validation task-vocabulary provenance differs"
        )

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PipelineConfigurationError("Tuned-threshold JSON lacks provenance metadata")
    expected_provenance = {
        "validation_manifest_sha256": validation_digest,
        "validation_provenance_fingerprint_sha256": validation_provenance_fingerprint,
        "validation_collection_seed": binding.validation_seed,
        "validation_benchmark_seed": binding.validation_seed,
        "reserved_final_evaluation_seed": binding.final_evaluation_seed,
        "detector_checkpoint_sha256": detector_digest,
        "detector_training_manifest_sha256": training_digest,
        "task_vocabulary_sha256": training_vocabulary,
        "seed": binding.model_seed,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise PipelineConfigurationError(
                f"Tuned-threshold provenance {field}={provenance.get(field)!r}, "
                f"expected {expected!r}"
            )
    recorded_validation = _recorded_path(
        provenance.get("validation_manifest"),
        root=binding.root,
        field="provenance.validation_manifest",
    )
    recorded_detector = _recorded_path(
        provenance.get("detector_checkpoint"),
        root=binding.root,
        field="provenance.detector_checkpoint",
    )
    if recorded_validation != validation_path or recorded_detector != detector_path:
        raise PipelineConfigurationError(
            "Tuned-threshold provenance points to unexpected validation/checkpoint paths"
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or _recorded_path(
        outputs.get("json"), root=binding.root, field="outputs.json"
    ) != artifact_path:
        raise PipelineConfigurationError(
            "Tuned-threshold outputs.json does not identify the consumed artifact"
        )
    return threshold


def _materialize_command(command: StageCommand) -> StageCommand:
    binding = command.threshold_binding
    if binding is None:
        return command
    if command.argv.count(binding.placeholder) != 1:
        raise PipelineConfigurationError(
            f"Stage {command.label!r} has an invalid tuned-threshold placeholder contract"
        )
    threshold = _as_cli_number(_resolve_tuned_threshold(binding))
    argv = tuple(
        threshold if value == binding.placeholder else value for value in command.argv
    )
    return replace(command, argv=argv, threshold_binding=None)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise PipelineConfigurationError(f"Configuration field {key!r} must be a mapping")
    return value


def _integer(
    parent: Mapping[str, Any], key: str, *, minimum: int = 0
) -> int:
    value = parent.get(key)
    if isinstance(value, bool):
        raise PipelineConfigurationError(f"Configuration field {key!r} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineConfigurationError(
            f"Configuration field {key!r} must be an integer"
        ) from exc
    if result != value or result < minimum:
        raise PipelineConfigurationError(
            f"Configuration field {key!r} must be an integer >= {minimum}"
        )
    return result


def _number(
    parent: Mapping[str, Any], key: str, *, minimum: float = 0.0
) -> float:
    value = parent.get(key)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineConfigurationError(
            f"Configuration field {key!r} must be numeric"
        ) from exc
    if not result >= minimum:
        raise PipelineConfigurationError(
            f"Configuration field {key!r} must be >= {minimum}"
        )
    return result


def _path_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _command_path(root: Path, path: Path) -> str:
    """Prefer stable repo-relative arguments while allowing external configs."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _as_cli_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _noise_tag(value: float) -> str:
    return f"{int(round(value * 100)):02d}"


def _validate_methods(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PipelineConfigurationError(f"evaluation.{field} must be a non-empty list")
    methods = tuple(str(item) for item in value)
    required = ("mlp_bc", "act", "heuristic_recovery", "reim")
    allowed = set(required)
    unexpected = set(methods).difference(allowed)
    if unexpected:
        raise PipelineConfigurationError(
            f"evaluation.{field} contains non-primary methods {sorted(unexpected)}; "
            "retry is appendix-only and cannot enter this pipeline"
        )
    if len(set(methods)) != len(methods):
        raise PipelineConfigurationError(f"evaluation.{field} contains duplicates")
    if methods != required:
        raise PipelineConfigurationError(
            f"evaluation.{field} must be ordered as {list(required)}"
        )
    return methods


def load_context(
    benchmark: str,
    *,
    root: Path = PROJECT_ROOT,
    python: str = sys.executable,
    config_path: Path | None = None,
    act_config_path: Path | None = None,
    mlp_config_path: Path | None = None,
    detector_config_path: Path | None = None,
    resume: bool = False,
    device: str | None = None,
    log_dir: Path | None = None,
) -> PipelineContext:
    """Load and validate all configuration required for one benchmark."""

    benchmark = benchmark.upper()
    if benchmark not in BENCHMARKS:
        raise PipelineConfigurationError(f"Unsupported benchmark: {benchmark}")
    root = root.expanduser().resolve()
    slug = benchmark.lower()
    config_path = (
        config_path.expanduser().resolve()
        if config_path is not None
        else root / "configs" / "multitask" / f"{slug}.yaml"
    )
    act_config_path = (
        act_config_path.expanduser().resolve()
        if act_config_path is not None
        else root / "configs" / "multitask" / f"{slug}_act.yaml"
    )
    mlp_config_path = (
        mlp_config_path.expanduser().resolve()
        if mlp_config_path is not None
        else root / "configs" / "multitask" / f"{slug}_mlp.yaml"
    )
    detector_config_path = (
        detector_config_path.expanduser().resolve()
        if detector_config_path is not None
        else root / "configs" / "multitask" / f"{slug}_detector.yaml"
    )
    config = _read_yaml(config_path)
    act_config = _read_yaml(act_config_path)
    mlp_config = _read_yaml(mlp_config_path)
    detector_config = _read_yaml(detector_config_path)
    stored_benchmark = str(config.get("benchmark", "")).upper()
    if stored_benchmark != benchmark:
        raise PipelineConfigurationError(
            f"{config_path} declares benchmark={stored_benchmark!r}, expected {benchmark!r}"
        )
    expected_tasks = 10 if benchmark == "MT10" else 50
    if _integer(config, "task_count", minimum=1) != expected_tasks:
        raise PipelineConfigurationError(
            f"{benchmark} must declare exactly {expected_tasks} tasks"
        )
    if _integer(config, "state_dim", minimum=1) != 39 + expected_tasks:
        raise PipelineConfigurationError(
            f"{benchmark} must use raw39 plus a {expected_tasks}-way task one-hot"
        )
    if _integer(config, "action_dim", minimum=1) != 4:
        raise PipelineConfigurationError("Meta-World manipulation actions must be 4D")
    if _integer(config, "max_episode_steps", minimum=1) != 500:
        raise PipelineConfigurationError("Official MT10/MT50 evaluation uses 500 steps")
    if str(mlp_config.get("benchmark", "")).upper() != benchmark:
        raise PipelineConfigurationError(
            f"{mlp_config_path} does not declare benchmark={benchmark}"
        )
    if str(_mapping(mlp_config, "model").get("policy_type", "")).upper() != "MLP_BC":
        raise PipelineConfigurationError(
            f"{mlp_config_path} must configure policy_type=MLP_BC"
        )
    act_data_dir = _path_from_root(root, str(act_config.get("data_dir", "")))
    mlp_data_dir = _path_from_root(root, str(mlp_config.get("data_dir", "")))
    if act_data_dir != mlp_data_dir:
        raise PipelineConfigurationError(
            "ACT and MLP baselines must consume the identical demonstration bank"
        )

    disturbance = _mapping(config, "disturbance")
    levels_raw = disturbance.get("levels")
    if not isinstance(levels_raw, list):
        raise PipelineConfigurationError("disturbance.levels must be a list")
    levels = tuple(round(float(item), 10) for item in levels_raw)
    if levels != EXPECTED_DISTURBANCE_LEVELS:
        raise PipelineConfigurationError(
            "disturbance.levels must be exactly [0.0, 0.1, 0.2, 0.3, 0.4]"
        )
    evaluation = _mapping(config, "evaluation")
    _validate_methods(evaluation.get("clean_methods"), field="clean_methods")
    _validate_methods(evaluation.get("disturbed_methods"), field="disturbed_methods")
    if bool(disturbance.get("object_position_noise", True)):
        raise PipelineConfigurationError(
            "MT10/MT50 must not use the PickPlace-only object teleportation disturbance"
        )

    banks = _mapping(config, "banks")
    bank_seeds = {
        role: _integer(banks, role, minimum=0)
        for role in BANK_ROLES
    }
    if len(set(bank_seeds.values())) != len(BANK_ROLES):
        collisions: dict[int, list[str]] = {}
        for role, seed in bank_seeds.items():
            collisions.setdefault(seed, []).append(role)
        repeated = {
            seed: roles for seed, roles in collisions.items() if len(roles) > 1
        }
        raise PipelineConfigurationError(
            "demonstrations, failure_training, recovery_training, validation, "
            "and final_evaluation bank seeds must all be pairwise distinct; "
            f"collisions={repeated}"
        )
    data = _mapping(config, "data")
    _integer(data, "validation_rollouts_per_task", minimum=1)
    if "recovery_max_attempts_multiplier" in data:
        _integer(data, "recovery_max_attempts_multiplier", minimum=1)
    failure_labels = _mapping(config, "failure_labels")
    if failure_labels.get("calibration_mode") != "task_conditional_quantile":
        raise PipelineConfigurationError(
            "failure_labels.calibration_mode must be task_conditional_quantile"
        )
    calibration_quantile = _number(
        failure_labels, "calibration_quantile", minimum=0.0
    )
    if not 0.0 < calibration_quantile < 1.0:
        raise PipelineConfigurationError(
            "failure_labels.calibration_quantile must lie strictly between 0 and 1"
        )
    expected_detector_data = (
        root / "datasets" / slug / "failures_calibrated"
    ).resolve()
    configured_detector_data = _path_from_root(
        root, str(detector_config.get("data_dir", ""))
    )
    if configured_detector_data != expected_detector_data:
        raise PipelineConfigurationError(
            "Detector data_dir must be the suite's calibrated training bank: "
            f"{expected_detector_data}"
        )

    resolved_log_dir = (
        log_dir.expanduser().resolve()
        if log_dir is not None
        else root / "results" / "logs" / "multitask" / slug
    )
    return PipelineContext(
        benchmark=benchmark,
        root=root,
        python=str(python),
        config_path=config_path,
        act_config_path=act_config_path,
        mlp_config_path=mlp_config_path,
        detector_config_path=detector_config_path,
        config=config,
        act_config=act_config,
        mlp_config=mlp_config,
        detector_config=detector_config,
        resume=bool(resume),
        device=device,
        log_dir=resolved_log_dir,
    )


def _append_resume_if_present(
    argv: list[str], *, requested: bool, marker: Path, explicit_path: bool = False
) -> None:
    if requested and marker.exists():
        argv.append("--resume")
        if explicit_path:
            argv.append(str(marker))


def _stage_command(
    context: PipelineContext,
    stage: str,
    label: str,
    argv: Sequence[str],
    *,
    threshold_binding: TunedThresholdBinding | None = None,
) -> StageCommand:
    return StageCommand(
        stage=stage,
        label=label,
        argv=tuple(str(value) for value in argv),
        log_path=context.log_dir / f"{label}.log",
        threshold_binding=threshold_binding,
    )


def build_stage_commands(context: PipelineContext, stage: str) -> list[StageCommand]:
    """Resolve a stage to the exact existing project CLI invocation(s)."""

    if stage not in (*STAGES, "all"):
        raise PipelineConfigurationError(f"Unknown stage: {stage}")
    if stage == "all":
        commands: list[StageCommand] = []
        for child_stage in STAGES:
            commands.extend(build_stage_commands(context, child_stage))
        return commands

    config = context.config
    banks = _mapping(config, "banks")
    data = _mapping(config, "data")
    labels = _mapping(config, "failure_labels")
    disturbance = _mapping(config, "disturbance")
    recovery = _mapping(config, "recovery")
    evaluation = _mapping(config, "evaluation")
    model_seed = _integer(config, "model_seed", minimum=0)
    max_steps = _integer(config, "max_episode_steps", minimum=1)
    demos_dir = _path_from_root(context.root, str(context.act_config["data_dir"]))
    raw_failures_dir = context.root / "datasets" / context.slug / "failures"
    calibrated_failures_dir = _path_from_root(
        context.root, str(context.detector_config["data_dir"])
    )
    validation_failures_dir = (
        context.root / "datasets" / context.slug / "failures_validation"
    )
    calibrated_validation_failures_dir = (
        context.root
        / "datasets"
        / context.slug
        / "failures_validation_calibrated"
    )
    recovery_dir = context.root / "datasets" / context.slug / "recovery"
    act_checkpoint = _path_from_root(
        context.root, str(context.act_config["checkpoint"])
    )
    mlp_checkpoint = _path_from_root(
        context.root, str(context.mlp_config["checkpoint"])
    )
    detector_checkpoint = _path_from_root(
        context.root, str(context.detector_config["checkpoint"])
    )
    recovery_checkpoint = (
        context.root
        / "checkpoints"
        / context.slug
        / f"seed_{model_seed}"
        / "recovery.pt"
    )
    tuned_threshold_json = (
        context.root
        / "results"
        / "tables"
        / f"{context.slug}_detector_threshold.json"
    )
    tuned_threshold_csv = (
        context.root
        / "results"
        / "tables"
        / f"{context.slug}_detector_threshold_grid.csv"
    )
    action_scale = _number(disturbance, "action_std_scale")
    observation_scale = _number(disturbance, "observation_std_scale")
    training_noise = float(disturbance.get("training_level", 0.2))
    if training_noise < 0.0:
        raise PipelineConfigurationError("disturbance.training_level must be non-negative")
    device = context.device or str(context.act_config.get("device", "auto"))
    script = lambda value: _command_path(context.root, context.root / value)
    artifact = lambda value: _command_path(context.root, value)
    release_threshold = _number(recovery, "release_threshold")
    threshold_placeholder = (
        f"<validated-threshold:{artifact(tuned_threshold_json)}>"
    )
    threshold_binding = TunedThresholdBinding(
        placeholder=threshold_placeholder,
        artifact_path=tuned_threshold_json,
        benchmark=context.benchmark,
        root=context.root,
        failure_training_seed=_integer(banks, "failure_training", minimum=0),
        validation_seed=_integer(banks, "validation", minimum=0),
        final_evaluation_seed=_integer(banks, "final_evaluation", minimum=0),
        model_seed=model_seed,
        detector_checkpoint=detector_checkpoint,
        calibrated_training_manifest=calibrated_failures_dir / "manifest.json",
        calibrated_validation_manifest=(
            calibrated_validation_failures_dir / "manifest.json"
        ),
        calibration_quantile=_number(labels, "calibration_quantile"),
        prediction_horizon=_integer(labels, "prediction_horizon", minimum=1),
        terminal_positive_horizon=_integer(
            labels, "terminal_positive_horizon", minimum=1
        ),
        release_threshold=release_threshold,
    )

    if stage == "collect_demos":
        argv = [
            context.python,
            script("scripts/collect_multitask_demonstrations.py"),
            "--benchmark",
            context.benchmark,
            "--episodes-per-task",
            str(_integer(data, "demonstrations_per_task", minimum=1)),
            "--output",
            artifact(demos_dir),
            "--seed",
            str(_integer(banks, "demonstrations", minimum=0)),
            "--max-attempts",
            str(_integer(data, "max_attempts_per_success", minimum=1)),
        ]
        if context.resume:
            argv.append("--resume")
        return [_stage_command(context, stage, stage, argv)]

    if stage == "train_act":
        argv = [
            context.python,
            script("trainers/train_bc.py"),
            "--config",
            _command_path(context.root, context.act_config_path),
        ]
        if context.device is not None:
            argv.extend(("--device", context.device))
        act_latest = _path_from_root(
            context.root, str(context.act_config["latest_checkpoint"])
        )
        _append_resume_if_present(
            argv,
            requested=context.resume,
            marker=act_latest,
            explicit_path=True,
        )
        return [_stage_command(context, stage, stage, argv)]

    if stage == "train_mlp":
        argv = [
            context.python,
            script("trainers/train_multitask_mlp.py"),
            "--config",
            _command_path(context.root, context.mlp_config_path),
        ]
        if context.device is not None:
            argv.extend(("--device", context.device))
        mlp_latest = _path_from_root(
            context.root, str(context.mlp_config["latest_checkpoint"])
        )
        _append_resume_if_present(
            argv,
            requested=context.resume,
            marker=mlp_latest,
            explicit_path=True,
        )
        return [_stage_command(context, stage, stage, argv)]

    if stage == "generate_failures":
        argv = [
            context.python,
            script("scripts/generate_multitask_failures.py"),
            "--benchmark",
            context.benchmark,
            "--benchmark-seed",
            str(_integer(banks, "failure_training", minimum=0)),
            "--act-checkpoint",
            artifact(act_checkpoint),
            "--output-dir",
            artifact(raw_failures_dir),
            "--rollouts-per-task",
            str(_integer(data, "failure_rollouts_per_task", minimum=1)),
            "--max-steps",
            str(max_steps),
            "--noise-level",
            _as_cli_number(training_noise),
            "--action-std-scale",
            _as_cli_number(action_scale),
            "--observation-std-scale",
            _as_cli_number(observation_scale),
            "--disagreement-threshold",
            _as_cli_number(_number(labels, "expert_action_disagreement_l1")),
            "--prediction-horizon",
            str(_integer(labels, "prediction_horizon", minimum=1)),
            "--terminal-positive-horizon",
            str(_integer(labels, "terminal_positive_horizon", minimum=1)),
            "--seed",
            str(model_seed),
            "--device",
            device,
            "--log-file",
            artifact(context.log_dir / "generate_failures_internal.log"),
        ]
        if context.resume:
            argv.append("--resume")
        return [_stage_command(context, stage, stage, argv)]

    if stage == "calibrate_failures":
        argv = [
            context.python,
            script("scripts/relabel_multitask_failures.py"),
            "--data-dir",
            artifact(raw_failures_dir),
            "--output-dir",
            artifact(calibrated_failures_dir),
            "--mode",
            "fit-task-quantile",
            "--quantile",
            _as_cli_number(_number(labels, "calibration_quantile")),
            "--dataset-role",
            "training",
            "--prediction-horizon",
            str(_integer(labels, "prediction_horizon", minimum=1)),
            "--terminal-positive-horizon",
            str(_integer(labels, "terminal_positive_horizon", minimum=1)),
        ]
        return [_stage_command(context, stage, stage, argv)]

    if stage == "train_detector":
        argv = [
            context.python,
            script("trainers/train_detector.py"),
            "--config",
            _command_path(context.root, context.detector_config_path),
        ]
        if context.device is not None:
            argv.extend(("--device", context.device))
        detector_latest = _path_from_root(
            context.root, str(context.detector_config["latest_checkpoint"])
        )
        _append_resume_if_present(
            argv,
            requested=context.resume,
            marker=detector_latest,
            explicit_path=True,
        )
        return [_stage_command(context, stage, stage, argv)]

    if stage == "generate_failure_validation":
        validation_seed = _integer(banks, "validation", minimum=0)
        argv = [
            context.python,
            script("scripts/generate_multitask_failures.py"),
            "--benchmark",
            context.benchmark,
            "--benchmark-seed",
            str(validation_seed),
            "--act-checkpoint",
            artifact(act_checkpoint),
            "--output-dir",
            artifact(validation_failures_dir),
            "--rollouts-per-task",
            str(_integer(data, "validation_rollouts_per_task", minimum=1)),
            "--max-steps",
            str(max_steps),
            "--noise-level",
            _as_cli_number(training_noise),
            "--action-std-scale",
            _as_cli_number(action_scale),
            "--observation-std-scale",
            _as_cli_number(observation_scale),
            "--disagreement-threshold",
            _as_cli_number(_number(labels, "expert_action_disagreement_l1")),
            "--prediction-horizon",
            str(_integer(labels, "prediction_horizon", minimum=1)),
            "--terminal-positive-horizon",
            str(_integer(labels, "terminal_positive_horizon", minimum=1)),
            "--seed",
            str(validation_seed),
            "--device",
            device,
            "--log-file",
            artifact(context.log_dir / "generate_failure_validation_internal.log"),
        ]
        if context.resume:
            argv.append("--resume")
        return [_stage_command(context, stage, stage, argv)]

    if stage == "calibrate_failure_validation":
        argv = [
            context.python,
            script("scripts/relabel_multitask_failures.py"),
            "--data-dir",
            artifact(validation_failures_dir),
            "--output-dir",
            artifact(calibrated_validation_failures_dir),
            "--mode",
            "frozen-task-thresholds",
            "--quantile",
            _as_cli_number(_number(labels, "calibration_quantile")),
            "--dataset-role",
            "validation",
            "--calibration-manifest",
            artifact(calibrated_failures_dir / "manifest.json"),
            "--prediction-horizon",
            str(_integer(labels, "prediction_horizon", minimum=1)),
            "--terminal-positive-horizon",
            str(_integer(labels, "terminal_positive_horizon", minimum=1)),
        ]
        return [_stage_command(context, stage, stage, argv)]

    if stage == "tune_detector":
        validation_seed = _integer(banks, "validation", minimum=0)
        argv = [
            context.python,
            script("evaluation/tune_multitask_detector.py"),
            "--benchmark",
            context.benchmark,
            "--validation-data",
            artifact(calibrated_validation_failures_dir),
            "--detector-checkpoint",
            artifact(detector_checkpoint),
            "--protocol-config",
            _command_path(context.root, context.config_path),
            "--validation-bank-seed",
            str(validation_seed),
            "--validation-benchmark-seed",
            str(validation_seed),
            "--seed",
            str(model_seed),
            "--device",
            device,
            "--output-json",
            artifact(tuned_threshold_json),
            "--output-csv",
            artifact(tuned_threshold_csv),
            "--log-file",
            artifact(context.log_dir / "tune_detector_internal.log"),
        ]
        return [_stage_command(context, stage, stage, argv)]

    if stage == "collect_recovery":
        recovery_attempts_field = (
            "recovery_max_attempts_multiplier"
            if "recovery_max_attempts_multiplier" in data
            else "max_attempts_per_success"
        )
        argv = [
            context.python,
            script("scripts/collect_multitask_recovery.py"),
            "--benchmark",
            context.benchmark,
            "--benchmark-seed",
            str(_integer(banks, "recovery_training", minimum=0)),
            "--act-checkpoint",
            artifact(act_checkpoint),
            "--detector-checkpoint",
            artifact(detector_checkpoint),
            "--output-dir",
            artifact(recovery_dir),
            "--target-per-task",
            str(_integer(data, "recovery_rollouts_per_task", minimum=1)),
            "--max-attempts-multiplier",
            str(_integer(data, recovery_attempts_field, minimum=1)),
            "--max-steps",
            str(max_steps),
            "--threshold",
            _as_cli_number(_number(recovery, "collection_threshold")),
            "--noise-level",
            _as_cli_number(training_noise),
            "--action-std-scale",
            _as_cli_number(action_scale),
            "--observation-std-scale",
            _as_cli_number(observation_scale),
            "--seed",
            str(model_seed),
            "--device",
            device,
            "--log-file",
            artifact(context.log_dir / "collect_recovery_internal.log"),
        ]
        _append_resume_if_present(
            argv,
            requested=context.resume,
            marker=recovery_dir / "manifest.json",
        )
        return [_stage_command(context, stage, stage, argv)]

    if stage == "train_recovery":
        training = config.get("recovery_training", {})
        if not isinstance(training, Mapping):
            raise PipelineConfigurationError("recovery_training must be a mapping")
        history = context.root / "results" / "tables" / f"{context.slug}_recovery_training.csv"
        curve = context.root / "results" / "figures" / f"{context.slug}_recovery_training.png"
        summary = context.root / "results" / "tables" / f"{context.slug}_recovery_training.json"
        hidden_dims = training.get("hidden_dims", (512, 512, 256))
        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise PipelineConfigurationError("recovery_training.hidden_dims must be a list")
        argv = [
            context.python,
            script("trainers/train_multitask_recovery.py"),
            "--benchmark",
            context.benchmark,
            "--data-dir",
            artifact(recovery_dir),
            "--output",
            artifact(recovery_checkpoint),
            "--history",
            artifact(history),
            "--curve",
            artifact(curve),
            "--summary",
            artifact(summary),
            "--log-file",
            artifact(context.log_dir / "train_recovery_internal.log"),
            "--seed",
            str(model_seed),
            "--device",
            device,
            "--epochs",
            str(int(training.get("epochs", 60))),
            "--batch-size",
            str(int(training.get("batch_size", 1024))),
            "--learning-rate",
            str(float(training.get("learning_rate", 1e-3))),
            "--weight-decay",
            str(float(training.get("weight_decay", 1e-5))),
            "--validation-fraction",
            str(float(training.get("validation_fraction", 0.2))),
            "--hidden-dims",
            *(str(int(value)) for value in hidden_dims),
            "--state-noise-std",
            str(float(training.get("state_noise_std", 0.005))),
            "--grad-clip",
            str(float(training.get("grad_clip", 1.0))),
            "--patience",
            str(int(training.get("patience", 12))),
            "--min-delta",
            str(float(training.get("min_delta", 1e-6))),
            "--num-workers",
            str(int(training.get("num_workers", 0))),
        ]
        recovery_latest = recovery_checkpoint.with_name(
            recovery_checkpoint.stem + "_latest" + recovery_checkpoint.suffix
        )
        _append_resume_if_present(
            argv,
            requested=context.resume,
            marker=recovery_latest,
            explicit_path=True,
        )
        return [_stage_command(context, stage, stage, argv)]

    if stage == "audit_banks":
        clean_csv = (
            context.root / "results" / "tables" / f"{context.slug}_clean_episodes.csv"
        )
        audit_json = (
            context.root
            / "results"
            / "audits"
            / f"{context.slug}_bank_separation.json"
        )
        argv = [
            context.python,
            script("scripts/audit_multitask_banks.py"),
            "--benchmark",
            context.benchmark,
            "--demonstrations",
            artifact(demos_dir / "manifest.json"),
            "--failure-train",
            artifact(raw_failures_dir / "manifest.json"),
            "--failure-validation",
            artifact(validation_failures_dir / "manifest.json"),
            "--recovery",
            artifact(recovery_dir / "manifest.json"),
            "--final-evaluation-sidecar",
            artifact(clean_csv.with_suffix(clean_csv.suffix + ".run.json")),
            "--final-evaluation-csv",
            artifact(clean_csv),
            "--output-json",
            artifact(audit_json),
        ]
        return [_stage_command(context, stage, stage, argv)]

    clean_methods = _validate_methods(
        evaluation.get("clean_methods"), field="clean_methods"
    )
    disturbed_methods = _validate_methods(
        evaluation.get("disturbed_methods"), field="disturbed_methods"
    )
    episodes = _integer(evaluation, "episodes_per_task", minimum=1)
    release_patience = _integer(recovery, "release_patience", minimum=1)
    min_recovery_steps = _integer(recovery, "min_recovery_steps", minimum=1)
    intervention_cooldown = _integer(
        recovery, "intervention_cooldown", minimum=0
    )
    recovery_budget = _integer(
        recovery, "max_consecutive_steps_within_episode", minimum=1
    )
    eval_seed = _integer(banks, "final_evaluation", minimum=0)

    def evaluation_command(
        *,
        condition: str,
        methods: tuple[str, ...],
        noise_level: float,
        label: str,
        output_stem: str,
    ) -> StageCommand:
        output_csv = context.root / "results" / "tables" / f"{output_stem}_episodes.csv"
        output_summary = context.root / "results" / "tables" / f"{output_stem}_summary.json"
        argv = [
            context.python,
            script("evaluation/evaluate_multitask.py"),
            "--benchmark",
            context.benchmark,
            "--condition",
            condition,
            "--benchmark-seed",
            str(eval_seed),
            "--act-checkpoint",
            artifact(act_checkpoint),
            "--mlp-checkpoint",
            artifact(mlp_checkpoint),
            "--detector-checkpoint",
            artifact(detector_checkpoint),
            "--recovery-checkpoint",
            artifact(recovery_checkpoint),
            "--output-csv",
            artifact(output_csv),
            "--output-summary",
            artifact(output_summary),
            "--methods",
            *methods,
            "--episodes-per-task",
            str(episodes),
            "--max-steps",
            str(max_steps),
            "--noise-level",
            _as_cli_number(noise_level),
            "--action-std-scale",
            _as_cli_number(action_scale),
            "--observation-std-scale",
            _as_cli_number(observation_scale),
            "--threshold",
            threshold_placeholder,
            "--release-threshold",
            _as_cli_number(release_threshold),
            "--release-patience",
            str(release_patience),
            "--min-recovery-steps",
            str(min_recovery_steps),
            "--intervention-cooldown",
            str(intervention_cooldown),
            "--recovery-budget",
            str(recovery_budget),
            "--seed",
            str(model_seed),
            "--device",
            device,
            "--log-file",
            artifact(context.log_dir / f"{label}_internal.log"),
        ]
        if context.resume:
            argv.append("--resume")
        return _stage_command(
            context,
            stage,
            label,
            argv,
            threshold_binding=threshold_binding,
        )

    if stage == "evaluate_clean":
        return [
            evaluation_command(
                condition="official_clean",
                methods=clean_methods,
                noise_level=0.0,
                label="evaluate_clean",
                output_stem=f"{context.slug}_clean",
            )
        ]

    levels = tuple(float(value) for value in disturbance["levels"])
    return [
        evaluation_command(
            condition=f"robustness_noise_{_noise_tag(level)}",
            methods=disturbed_methods,
            noise_level=level,
            label=f"evaluate_disturbed_noise_{_noise_tag(level)}",
            output_stem=f"{context.slug}_disturbed_noise_{_noise_tag(level)}",
        )
        for level in levels
    ]


def build_plan(contexts: Iterable[PipelineContext], stage: str) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for context in contexts:
        commands.extend(build_stage_commands(context, stage))
    return commands


def _plan_payload(
    context: PipelineContext, stage: str, commands: Sequence[StageCommand]
) -> dict[str, Any]:
    own_commands = [command for command in commands if command.log_path.parent == context.log_dir]
    calibrated_training = _path_from_root(
        context.root, str(context.detector_config["data_dir"])
    )
    tuned_json = (
        context.root
        / "results"
        / "tables"
        / f"{context.slug}_detector_threshold.json"
    )
    tuned_csv = (
        context.root
        / "results"
        / "tables"
        / f"{context.slug}_detector_threshold_grid.csv"
    )
    bank_audit_json = (
        context.root
        / "results"
        / "audits"
        / f"{context.slug}_bank_separation.json"
    )
    recovery = _mapping(context.config, "recovery")
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "benchmark": context.benchmark,
        "stage": stage,
        "resume_requested": context.resume,
        "configs": {
            "benchmark": {
                "path": _command_path(context.root, context.config_path),
                "sha256": _sha256(context.config_path),
            },
            "act": {
                "path": _command_path(context.root, context.act_config_path),
                "sha256": _sha256(context.act_config_path),
            },
            "mlp": {
                "path": _command_path(context.root, context.mlp_config_path),
                "sha256": _sha256(context.mlp_config_path),
            },
            "detector": {
                "path": _command_path(context.root, context.detector_config_path),
                "sha256": _sha256(context.detector_config_path),
            },
        },
        "artifacts": {
            "failure_training_raw": _command_path(
                context.root,
                context.root / "datasets" / context.slug / "failures",
            ),
            "failure_training_calibrated": _command_path(
                context.root, calibrated_training
            ),
            "failure_validation_raw": _command_path(
                context.root,
                context.root
                / "datasets"
                / context.slug
                / "failures_validation",
            ),
            "failure_validation_calibrated": _command_path(
                context.root,
                context.root
                / "datasets"
                / context.slug
                / "failures_validation_calibrated",
            ),
            "detector_threshold_tuning": {
                "json": _command_path(context.root, tuned_json),
                "csv": _command_path(context.root, tuned_csv),
                "consumption": "fail_closed_runtime_binding",
            },
            "bank_separation_audit": {
                "json": _command_path(context.root, bank_audit_json),
                "schema_version": BANK_AUDIT_SCHEMA_VERSION,
                "producer_stage": "audit_banks",
                "publication_gate_requirement": True,
                "write_mode": "atomic_after_complete_pass",
            },
        },
        "runtime_threshold_binding": {
            "source": "validation_tuner_json",
            "artifact": _command_path(context.root, tuned_json),
            "schema_version": TUNER_SCHEMA_VERSION,
            "consumers": [
                "evaluate_clean",
                "evaluate_disturbed",
            ],
            "missing_or_stale_artifact": "fail_closed",
            "configured_deployment_fallback_enabled": False,
            "release_threshold": _number(recovery, "release_threshold"),
        },
        "recovery_collection_threshold": {
            "source": "preregistered_benchmark_config",
            "value": _number(recovery, "collection_threshold"),
            "purpose": "early_coverage_training_collection",
            "distinct_from_frozen_deployment_gate": True,
        },
        "commands": [
            {
                "stage": command.stage,
                "label": command.label,
                "argv": list(command.argv),
                "log": _command_path(context.root, command.log_path),
                "threshold_dependency": (
                    None
                    if command.threshold_binding is None
                    else {
                        "artifact": _command_path(
                            context.root, command.threshold_binding.artifact_path
                        ),
                        "placeholder": command.threshold_binding.placeholder,
                        "resolution": "validated_immediately_before_subprocess",
                    }
                ),
            }
            for command in own_commands
        ],
    }


def _write_plan(context: PipelineContext, stage: str, commands: Sequence[StageCommand]) -> None:
    context.log_dir.mkdir(parents=True, exist_ok=True)
    destination = context.log_dir / "pipeline_plan.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_plan_payload(context, stage, commands), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def execute_commands(commands: Sequence[StageCommand], *, cwd: Path) -> None:
    """Execute sequentially, streaming output and stopping at first failure."""

    for index, planned_command in enumerate(commands, start=1):
        command = _materialize_command(planned_command)
        if planned_command.threshold_binding is not None:
            print(
                f"[{index}/{len(commands)}] validated threshold artifact: "
                f"{_command_path(cwd, planned_command.threshold_binding.artifact_path)}"
            )
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(commands)}] EXEC {command.label}: {command.shell_line()}")
        with command.log_path.open("a", encoding="utf-8") as log_handle:
            started = datetime.now(timezone.utc).isoformat()
            log_handle.write(f"\n=== {started} ===\n$ {command.shell_line()}\n")
            log_handle.flush()
            process = subprocess.Popen(
                command.argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                log_handle.flush()
            returncode = process.wait()
            log_handle.write(f"=== exit_status={returncode} ===\n")
        if returncode != 0:
            raise PipelineCommandError(command, returncode)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("benchmark", choices=(*BENCHMARKS, "both"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the printed commands. Without this flag the runner is dry-run only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume collectors/evaluations and existing latest trainer checkpoints.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for child stages.")
    parser.add_argument("--device", help="Override the configured device for all applicable stages.")
    parser.add_argument("--config", type=Path, help="Benchmark YAML (single benchmark only).")
    parser.add_argument("--act-config", type=Path, help="ACT YAML (single benchmark only).")
    parser.add_argument("--mlp-config", type=Path, help="MLP YAML (single benchmark only).")
    parser.add_argument(
        "--detector-config", type=Path, help="Detector YAML (single benchmark only)."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Log directory; with benchmark=both, MT10/MT50 subdirectories are added.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run-first MT10/MT50 orchestration. Use a stage subcommand and "
            "add --execute only after inspecting the resolved commands."
        )
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in (*STAGES, "all"):
        subparser = subparsers.add_parser(stage, help=f"Plan or execute {stage}.")
        _add_common_arguments(subparser)
    return parser


def _contexts_from_args(args: argparse.Namespace) -> list[PipelineContext]:
    benchmarks = BENCHMARKS if args.benchmark == "both" else (args.benchmark,)
    if len(benchmarks) > 1 and any(
        value is not None
        for value in (args.config, args.act_config, args.mlp_config, args.detector_config)
    ):
        raise PipelineConfigurationError(
            "Custom config paths require a single benchmark; run MT10 and MT50 separately"
        )
    contexts = []
    for benchmark in benchmarks:
        log_dir = args.log_dir
        if log_dir is not None and len(benchmarks) > 1:
            log_dir = log_dir / benchmark.lower()
        contexts.append(
            load_context(
                benchmark,
                python=args.python,
                config_path=args.config,
                act_config_path=args.act_config,
                mlp_config_path=args.mlp_config,
                detector_config_path=args.detector_config,
                resume=args.resume,
                device=args.device,
                log_dir=log_dir,
            )
        )
    return contexts


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contexts = _contexts_from_args(args)
        commands = build_plan(contexts, args.stage)
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(
            f"REIM multi-task pipeline | mode={mode} | stage={args.stage} | "
            f"benchmarks={','.join(context.benchmark for context in contexts)}"
        )
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {command.label}: {command.shell_line()}")
        for context in contexts:
            tuned_json = (
                context.root
                / "results"
                / "tables"
                / f"{context.slug}_detector_threshold.json"
            )
            print(
                f"{context.benchmark} tuned detector artifact: "
                f"{_command_path(context.root, tuned_json)} "
                "(validated and consumed at runtime by evaluation; recovery collection "
                "uses its separate preregistered early-coverage threshold; "
                "missing or stale artifacts fail closed)"
            )
        if not args.execute:
            print("Dry-run only: no collection, training, evaluation, or output writes occurred.")
            return 0
        for context in contexts:
            _write_plan(context, args.stage, commands)
        execute_commands(commands, cwd=PROJECT_ROOT)
        print("Pipeline stage(s) completed successfully.")
        return 0
    except (OSError, PipelineConfigurationError, PipelineCommandError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
