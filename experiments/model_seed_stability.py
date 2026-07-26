#!/usr/bin/env python3
"""Summarize paired recovery-model seed stability from raw episodes.

This analysis is deliberately separate from the primary benchmark.  It uses
one fixed set of evaluation episodes for ACT and three independently trained
recovery policies, validates the raw records, and reports paired success gains
with deterministic bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
from statistics import NormalDist
import sys
from typing import Any, Sequence
import zipfile

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_reim import _atomic_write_csv  # noqa: E402
from utils.common import atomic_json_dump  # noqa: E402


LOGGER = logging.getLogger("reim.model_seed_stability")
ANALYSIS_ID = "equal_budget_500k_n3"
RESULT_SCOPE = "equal_budget_500k_model_seed_stability_not_primary_benchmark"
EXPECTED_TRAINING_SEEDS = (42, 43, 44)
MINIMUM_TRAINING_BUDGET = 500_000
EXPECTED_CONFIG_SHA256 = (
    "9b75b6318c287725b4e977b9f1c66c1da6edc0ccde00cfa338f493e71cf16f63"
)
EXPECTED_TRAIN_BANK_SHA256 = (
    "b338c9b2c5e6f880f0166897b07e7ec3fee763f579b9446ea8fd334b58239e2d"
)
EXPECTED_VALIDATION_BANK_SHA256 = (
    "88d3b54dba8844cfad435beb1c991d409b2a0d0dcb4d6246255bb92c4d4c02fa"
)
EXPECTED_ACT_CHECKPOINT_SHA256 = (
    "da3ea384f469a290472920cc0b15575b671ea8febead6af72cc2832e8fbaba33"
)
EXPECTED_DETECTOR_CHECKPOINT_SHA256 = (
    "62c93c5250a5cb50fcf31fee6d00f17da13f773d44e8a7adc7a5406b92c22d34"
)
EXPECTED_SELECTED_CHECKPOINT_SHA256 = {
    42: "55d92acfdd98061e198c4e27fac882e735f5ef4a1336c578e520f1ee8df0ffe7",
    43: "23ccadbc6eef8be9cd557202c319274d9ad519b96f6e93049a7aebd9034b6656",
    44: "dbc5cb5c787d7b48568ef91e207633622f24ec4c9081cfd0b6d4b871ecf86746",
}
FROZEN_TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "torch_num_threads": 4,
    "n_envs": 4,
    "n_steps": 1024,
    "batch_size": 512,
    "n_epochs": 5,
    "learning_rate": 3e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.0001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.01,
    "eval_freq": 10_000,
    "eval_episodes": 50,
    "checkpoint_freq": 25_000,
    "minimum_selection_timesteps": 10_000,
}
DEFAULT_MODEL_RUNS = {
    42: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed42_canonical_seed7000042_episodes.csv",
    43: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed43_500k_seed7000042_episodes.csv",
    44: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed44_500k_seed7000042_episodes.csv",
}
DEFAULT_CHECKPOINTS = {
    42: PROJECT_ROOT / "checkpoints/recovery_trigger_seed42_500k.zip",
    43: PROJECT_ROOT / "checkpoints/recovery_trigger_seed43_500k.zip",
    44: PROJECT_ROOT / "checkpoints/recovery_trigger_seed44_500k.zip",
}
DEFAULT_TRAINING_METRICS = {
    seed: PROJECT_ROOT
    / f"results/tables/recovery_trigger_seed{seed}_500k_metrics.json"
    for seed in EXPECTED_TRAINING_SEEDS
}
DEFAULT_EVALUATION_AUDITS = {
    42: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed42_canonical_seed7000042_audit.json",
    43: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed43_500k_seed7000042_audit.json",
    44: PROJECT_ROOT
    / "results/tables/model_seeds/"
    "reim_seed44_500k_seed7000042_audit.json",
}
DEFAULT_CONFIG = PROJECT_ROOT / "configs/ppo_trigger.yaml"
DEFAULT_TRAIN_BANK = PROJECT_ROOT / "datasets/recovery_starts/train.npz"
DEFAULT_VALIDATION_BANK = (
    PROJECT_ROOT / "datasets/recovery_starts/validation.npz"
)
DEFAULT_ACT_CHECKPOINT = PROJECT_ROOT / "checkpoints/bc_policy.pt"
DEFAULT_DETECTOR_CHECKPOINT = PROJECT_ROOT / "checkpoints/failure_detector.pt"


@dataclass(frozen=True)
class EpisodeSet:
    """Validated episode-level outcomes for one controller."""

    path: Path
    seeds: tuple[int, ...]
    successes: np.ndarray
    recovery_attempts: int
    recovery_successes: int


@dataclass(frozen=True)
class TrainingRunAudit:
    """Verified equal-budget provenance for one recovery training run."""

    seed: int
    metrics_path: Path
    metrics_sha256: str
    target_timesteps: int
    starting_timesteps: int
    actual_additional_timesteps: int
    trained_timesteps: int
    selected_model_timesteps: int
    selected_model_type: str
    selected_checkpoint: Path
    selected_checkpoint_sha256: str
    final_checkpoint: Path
    final_checkpoint_sha256: str
    validation_history_records: int
    config_sha256: str
    train_bank_sha256: str
    validation_bank_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _recorded_path(value: Any, *, source: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: invalid recorded path {value!r}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_close(
    actual: Any,
    expected: float,
    *,
    field: str,
    source: Path,
) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid numeric {field}={actual!r}") from exc
    if not math.isfinite(value) or not np.isclose(
        value,
        expected,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{source}: {field}={value!r}, expected frozen value {expected!r}"
        )


def _validate_frozen_config(
    config_path: Path,
    *,
    train_bank_path: Path,
    validation_bank_path: Path,
) -> str:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_sha256 = _sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            f"{config_path}: SHA256 {config_sha256} differs from the frozen "
            f"equal-budget config {EXPECTED_CONFIG_SHA256}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    training = config.get("training")
    recovery = config.get("recovery")
    if not isinstance(training, dict) or not isinstance(recovery, dict):
        raise ValueError(f"{config_path}: missing training/recovery mappings")
    for field, expected in FROZEN_TRAINING_CONFIG.items():
        _require_close(
            training.get(field),
            float(expected),
            field=f"training.{field}",
            source=config_path,
        )
    if training.get("policy") != "MlpPolicy":
        raise ValueError(f"{config_path}: frozen policy must be MlpPolicy")
    if training.get("selection_metric") != "task_success":
        raise ValueError(
            f"{config_path}: frozen selection metric must be task_success"
        )
    if training.get("select_best_model") is not True:
        raise ValueError(f"{config_path}: best-model selection must be enabled")
    net_arch = (training.get("policy_kwargs") or {}).get("net_arch")
    if list(net_arch or ()) != [256, 256]:
        raise ValueError(f"{config_path}: frozen policy net_arch must be [256, 256]")
    recorded_train_bank = _recorded_path(
        recovery.get("start_state_dataset"),
        source=config_path,
    )
    recorded_validation_bank = _recorded_path(
        recovery.get("validation_start_state_dataset"),
        source=config_path,
    )
    if recorded_train_bank != train_bank_path.resolve():
        raise ValueError(f"{config_path}: unexpected training snapshot bank")
    if recorded_validation_bank != validation_bank_path.resolve():
        raise ValueError(f"{config_path}: unexpected validation snapshot bank")
    return config_sha256


def _read_sb3_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("data") as handle:
                payload = json.load(handle)
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid Stable-Baselines3 checkpoint") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: invalid Stable-Baselines3 metadata")
    return payload


def _validate_ppo_metadata(
    checkpoint: Path,
    *,
    expected_timesteps: int,
) -> None:
    data = _read_sb3_data(checkpoint)
    if int(data.get("num_timesteps", -1)) != expected_timesteps:
        raise ValueError(
            f"{checkpoint}: num_timesteps={data.get('num_timesteps')} differs "
            f"from metrics value {expected_timesteps}"
        )
    ppo_expected = {
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 512,
        "n_epochs": 5,
        "learning_rate": 3e-5,
        "target_kl": 0.01,
        "ent_coef": 0.0001,
    }
    for field, expected in ppo_expected.items():
        _require_close(
            data.get(field),
            float(expected),
            field=field,
            source=checkpoint,
        )
    policy_kwargs = data.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict) or list(
        policy_kwargs.get("net_arch") or ()
    ) != [256, 256]:
        raise ValueError(f"{checkpoint}: policy net_arch is not frozen [256, 256]")


def _validate_training_run(
    *,
    seed: int,
    metrics_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    config_path: Path,
    config_sha256: str,
    train_bank_path: Path,
    validation_bank_path: Path,
) -> TrainingRunAudit:
    metrics = _load_json(metrics_path)
    if int(metrics.get("seed", -1)) != seed:
        raise ValueError(f"{metrics_path}: training seed mismatch")
    if str(metrics.get("backend", "")).lower() != "metaworld":
        raise ValueError(f"{metrics_path}: backend must be metaworld")
    target_timesteps = int(metrics.get("target_timesteps", -1))
    trained_timesteps = int(metrics.get("trained_timesteps", -1))
    starting_timesteps = int(metrics.get("starting_timesteps", -1))
    additional_timesteps = int(metrics.get("additional_timesteps", -1))
    if target_timesteps != MINIMUM_TRAINING_BUDGET:
        raise ValueError(
            f"{metrics_path}: target_timesteps must equal "
            f"{MINIMUM_TRAINING_BUDGET}"
        )
    if trained_timesteps < MINIMUM_TRAINING_BUDGET:
        raise ValueError(
            f"{metrics_path}: actual trained timesteps {trained_timesteps} "
            "do not reach the equal budget"
        )
    rollout_quantum = (
        int(FROZEN_TRAINING_CONFIG["n_envs"])
        * int(FROZEN_TRAINING_CONFIG["n_steps"])
    )
    if trained_timesteps >= MINIMUM_TRAINING_BUDGET + rollout_quantum:
        raise ValueError(
            f"{metrics_path}: budget overshoot exceeds one PPO rollout"
        )
    if additional_timesteps != trained_timesteps - starting_timesteps:
        raise ValueError(f"{metrics_path}: inconsistent additional_timesteps")
    if metrics.get("selected_model_type") != "validation_best":
        raise ValueError(f"{metrics_path}: selected model is not validation_best")
    if metrics.get("selection_metric") != "task_success":
        raise ValueError(f"{metrics_path}: selection metric is not task_success")
    if int(metrics.get("minimum_selection_timesteps", -1)) != 10_000:
        raise ValueError(
            f"{metrics_path}: minimum selection timesteps are not frozen"
        )
    if int(metrics.get("episodes", -1)) != 50:
        raise ValueError(f"{metrics_path}: validation must contain 50 episodes")
    if str(metrics.get("device", "")).lower() != "cpu":
        raise ValueError(f"{metrics_path}: equal-budget training device is not CPU")
    if int(metrics.get("torch_num_threads", -1)) != 4:
        raise ValueError(f"{metrics_path}: torch_num_threads must equal 4")
    if _recorded_path(metrics.get("config"), source=metrics_path) != config_path:
        raise ValueError(f"{metrics_path}: recorded config path mismatch")

    reward = metrics.get("reward")
    validation_reward = metrics.get("validation_reward")
    if not isinstance(reward, dict) or not isinstance(validation_reward, dict):
        raise ValueError(f"{metrics_path}: missing reward provenance")
    if (
        _recorded_path(
            reward.get("recovery_start_dataset"),
            source=metrics_path,
        )
        != train_bank_path
    ):
        raise ValueError(f"{metrics_path}: training snapshot bank mismatch")
    if (
        _recorded_path(
            validation_reward.get("recovery_start_dataset"),
            source=metrics_path,
        )
        != validation_bank_path
    ):
        raise ValueError(f"{metrics_path}: validation snapshot bank mismatch")

    selected_checkpoint = _recorded_path(
        metrics.get("checkpoint"),
        source=metrics_path,
    )
    if selected_checkpoint != checkpoint_path:
        raise ValueError(f"{metrics_path}: selected checkpoint path mismatch")
    selected_checkpoint_sha256 = _sha256(checkpoint_path)
    if selected_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"{checkpoint_path}: SHA256 {selected_checkpoint_sha256} differs "
            f"from audited value {expected_checkpoint_sha256}"
        )
    selected_model_timesteps = int(
        metrics.get("selected_model_timesteps", -1)
    )
    _validate_ppo_metadata(
        checkpoint_path,
        expected_timesteps=selected_model_timesteps,
    )

    final_checkpoint = _recorded_path(
        metrics.get("final_training_checkpoint"),
        source=metrics_path,
    )
    _validate_ppo_metadata(
        final_checkpoint,
        expected_timesteps=trained_timesteps,
    )
    final_checkpoint_sha256 = _sha256(final_checkpoint)

    history = metrics.get("task_success_selection_history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{metrics_path}: empty validation selection history")
    history_timesteps = [int(item["timesteps"]) for item in history]
    if history_timesteps != sorted(history_timesteps):
        raise ValueError(f"{metrics_path}: validation history is not sorted")
    if len(history_timesteps) != len(set(history_timesteps)):
        raise ValueError(
            f"{metrics_path}: validation history contains duplicate timesteps"
        )
    numeric_history_fields = (
        "failure_rate",
        "mean_episode_reward",
        "mean_episode_steps",
        "recovery_rate",
        "std_episode_reward",
        "success_rate",
    )
    for item in history:
        if not all(
            math.isfinite(float(item[field]))
            for field in numeric_history_fields
        ):
            raise ValueError(
                f"{metrics_path}: non-finite validation-history metric"
            )
    selected_history = {
        int(item["timesteps"])
        for item in history
        if bool(item.get("selected"))
    }
    if selected_model_timesteps not in selected_history:
        raise ValueError(
            f"{metrics_path}: selected timestep is absent from selection history"
        )

    return TrainingRunAudit(
        seed=seed,
        metrics_path=metrics_path,
        metrics_sha256=_sha256(metrics_path),
        target_timesteps=target_timesteps,
        starting_timesteps=starting_timesteps,
        actual_additional_timesteps=additional_timesteps,
        trained_timesteps=trained_timesteps,
        selected_model_timesteps=selected_model_timesteps,
        selected_model_type=str(metrics["selected_model_type"]),
        selected_checkpoint=selected_checkpoint,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        final_checkpoint=final_checkpoint,
        final_checkpoint_sha256=final_checkpoint_sha256,
        validation_history_records=len(history),
        config_sha256=config_sha256,
        train_bank_sha256=_sha256(train_bank_path),
        validation_bank_sha256=_sha256(validation_bank_path),
    )


def _parse_bool(value: str, *, field: str, path: Path) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{path}: invalid {field} value {value!r}")


def _read_episodes(
    path: Path,
    *,
    expected_episode_count: int,
    expected_seed_start: int,
    expected_noise_level: float,
) -> EpisodeSet:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    required = {
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
    }
    missing = required - fieldnames
    if missing:
        raise ValueError(f"{path}: missing raw fields {sorted(missing)}")
    noise_fields = [
        field for field in ("Noise Level", "noise_level") if field in fieldnames
    ]
    if len(noise_fields) != 1:
        raise ValueError(
            f"{path}: expected exactly one noise-level field, found "
            f"{noise_fields}"
        )
    noise_field = noise_fields[0]
    if len(rows) != expected_episode_count:
        raise ValueError(
            f"{path}: expected {expected_episode_count} rows, found {len(rows)}"
        )
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if str(value).strip().lower() in {"nan", "+nan", "-nan"}:
                raise ValueError(
                    f"{path}: literal NaN at row {row_index}, field {field}"
                )

    numeric_fields = (
        "episode",
        "seed",
        "steps",
        "elapsed_seconds",
        "recovery_attempts",
        "recovery_successes",
        "detector_triggers",
        "recovery_steps",
        "failure_probability_max",
        noise_field,
    )
    for row_index, row in enumerate(rows):
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: non-numeric {field} at row {row_index}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"{path}: non-finite {field} at row {row_index}"
                )
        if str(row["backend"]).lower() != "metaworld":
            raise ValueError(f"{path}: row {row_index} is not Meta-World")
        if str(row["Profile"]).strip().lower() != "custom":
            raise ValueError(
                f"{path}: row {row_index} must be a custom development run"
            )
        benchmark_value = row.get("Benchmark Eligible")
        if benchmark_value not in (None, "") and _parse_bool(
            str(benchmark_value),
            field="Benchmark Eligible",
            path=path,
        ):
            raise ValueError(
                f"{path}: model-seed diagnostics cannot be benchmark eligible"
            )
        if not np.isclose(float(row[noise_field]), expected_noise_level):
            raise ValueError(
                f"{path}: row {row_index} noise differs from "
                f"{expected_noise_level}"
            )

    episodes = tuple(int(float(row["episode"])) for row in rows)
    expected_episodes = tuple(range(expected_episode_count))
    if episodes != expected_episodes:
        raise ValueError(f"{path}: episode indices are not exactly ordered")
    seeds = tuple(int(float(row["seed"])) for row in rows)
    expected_seeds = tuple(
        range(expected_seed_start, expected_seed_start + expected_episode_count)
    )
    if seeds != expected_seeds:
        raise ValueError(f"{path}: evaluation seeds are not the expected range")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate evaluation seeds")

    successes = np.asarray(
        [
            _parse_bool(row["success"], field="success", path=path)
            for row in rows
        ],
        dtype=np.bool_,
    )
    attempts = sum(int(float(row["recovery_attempts"])) for row in rows)
    recovered = sum(int(float(row["recovery_successes"])) for row in rows)
    if attempts < 0 or recovered < 0 or recovered > attempts:
        raise ValueError(f"{path}: invalid aggregate recovery counts")
    return EpisodeSet(
        path=path,
        seeds=seeds,
        successes=successes,
        recovery_attempts=attempts,
        recovery_successes=recovered,
    )


def _validate_evaluation_audit(
    *,
    path: Path,
    training_seed: int,
    model: EpisodeSet,
    expected_checkpoint_sha256: str,
    expected_act_checkpoint_sha256: str,
    expected_detector_checkpoint_sha256: str,
    expected_episode_count: int,
    expected_seed_start: int,
    expected_noise_level: float,
) -> str:
    payload = _load_json(path)
    if int(payload.get("recovery_training_seed", -1)) != training_seed:
        raise ValueError(f"{path}: recovery training seed mismatch")
    if str(payload.get("backend", "")).lower() != "metaworld":
        raise ValueError(f"{path}: backend must be metaworld")
    if int(payload.get("episodes", -1)) != expected_episode_count:
        raise ValueError(f"{path}: episode count mismatch")
    if int(payload.get("evaluation_seed_start", -1)) != expected_seed_start:
        raise ValueError(f"{path}: evaluation seed start mismatch")
    if int(payload.get("evaluation_seed_end", -1)) != (
        expected_seed_start + expected_episode_count - 1
    ):
        raise ValueError(f"{path}: evaluation seed end mismatch")

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: missing evaluation protocol")
    expected_protocol = {
        "noise_level": expected_noise_level,
        "max_steps": 200,
        "failure_threshold": 0.2,
        "recovery_exit_threshold": 0.0,
        "recovery_budget": 150,
        "recovery_min_steps": 150,
        "recovery_clear_steps": 200,
    }
    for field, expected in expected_protocol.items():
        _require_close(
            protocol.get(field),
            float(expected),
            field=f"protocol.{field}",
            source=path,
        )

    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, dict):
        raise ValueError(f"{path}: missing checkpoint SHA256 provenance")
    if checkpoint_sha256.get("act") != expected_act_checkpoint_sha256:
        raise ValueError(f"{path}: ACT checkpoint SHA256 mismatch")
    if (
        checkpoint_sha256.get("failure_detector")
        != expected_detector_checkpoint_sha256
    ):
        raise ValueError(f"{path}: detector checkpoint SHA256 mismatch")
    if expected_checkpoint_sha256 not in checkpoint_sha256.values():
        raise ValueError(f"{path}: recovery checkpoint SHA256 mismatch")

    output_sha256 = payload.get("output_sha256")
    if not isinstance(output_sha256, dict):
        raise ValueError(f"{path}: missing output SHA256 provenance")
    if output_sha256.get("episodes") != _sha256(model.path):
        raise ValueError(f"{path}: raw episode CSV SHA256 mismatch")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path}: missing recomputed results")
    successes = int(model.successes.sum())
    if int(results.get("successes", -1)) != successes:
        raise ValueError(f"{path}: audited success count mismatch")
    if int(results.get("recovery_attempts", -1)) != model.recovery_attempts:
        raise ValueError(f"{path}: audited recovery attempts mismatch")
    if int(results.get("recovery_successes", -1)) != model.recovery_successes:
        raise ValueError(f"{path}: audited recovery successes mismatch")
    return _sha256(path)


def _paired_bootstrap_ci(
    act_successes: np.ndarray,
    model_successes: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        act_successes.size,
        size=(samples, act_successes.size),
    )
    gains = (
        model_successes[indices].mean(axis=1)
        - act_successes[indices].mean(axis=1)
    ) * 100.0
    lower, upper = np.quantile(gains, (0.025, 0.975))
    return float(lower), float(upper)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts for Wilson interval")
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def _exact_mcnemar_p_value(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if discordant == 0:
        return 1.0
    tail = min(rescued, harmed)
    probability = sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2.0**discordant)
    return float(min(1.0, 2.0 * probability))


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("sample standard deviation requires at least two seeds")
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def summarize(
    *,
    act_path: Path,
    model_paths: dict[int, Path],
    checkpoint_paths: dict[int, Path],
    metrics_paths: dict[int, Path],
    evaluation_audit_paths: dict[int, Path],
    config_path: Path,
    train_bank_path: Path,
    validation_bank_path: Path,
    act_checkpoint_path: Path,
    detector_checkpoint_path: Path,
    episodes: int,
    evaluation_seed_start: int,
    noise_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tuple(sorted(model_paths)) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            "model runs must contain exactly recovery training seeds 42, 43, 44"
        )
    if tuple(sorted(checkpoint_paths)) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            "checkpoints must contain exactly recovery training seeds 42, 43, 44"
        )
    if tuple(sorted(metrics_paths)) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            "training metrics must contain exactly recovery seeds 42, 43, 44"
        )
    if tuple(sorted(evaluation_audit_paths)) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            "evaluation audits must contain exactly recovery seeds 42, 43, 44"
        )
    if episodes != 200:
        raise ValueError(
            "the equal-budget development diagnostic requires exactly 200 "
            "episodes per recovery seed"
        )
    if evaluation_seed_start != 7_000_042:
        raise ValueError(
            "the equal-budget development diagnostic requires evaluation "
            "seeds 7000042..7000241"
        )
    if not np.isclose(noise_level, 0.2):
        raise ValueError(
            "the equal-budget development diagnostic requires noise_level=0.2"
        )

    config_path = config_path.resolve()
    train_bank_path = train_bank_path.resolve()
    validation_bank_path = validation_bank_path.resolve()
    act_checkpoint_path = act_checkpoint_path.resolve()
    detector_checkpoint_path = detector_checkpoint_path.resolve()
    config_sha256 = _validate_frozen_config(
        config_path,
        train_bank_path=train_bank_path,
        validation_bank_path=validation_bank_path,
    )
    train_bank_sha256 = _sha256(train_bank_path)
    validation_bank_sha256 = _sha256(validation_bank_path)
    if train_bank_sha256 != EXPECTED_TRAIN_BANK_SHA256:
        raise ValueError("training snapshot bank SHA256 differs from frozen bank")
    if validation_bank_sha256 != EXPECTED_VALIDATION_BANK_SHA256:
        raise ValueError(
            "validation snapshot bank SHA256 differs from frozen bank"
        )
    act_checkpoint_sha256 = _sha256(act_checkpoint_path)
    detector_checkpoint_sha256 = _sha256(detector_checkpoint_path)
    if act_checkpoint_sha256 != EXPECTED_ACT_CHECKPOINT_SHA256:
        raise ValueError("fixed ACT checkpoint SHA256 mismatch")
    if detector_checkpoint_sha256 != EXPECTED_DETECTOR_CHECKPOINT_SHA256:
        raise ValueError("fixed detector checkpoint SHA256 mismatch")

    training_audits: dict[int, TrainingRunAudit] = {}
    for training_seed in EXPECTED_TRAINING_SEEDS:
        training_audits[training_seed] = _validate_training_run(
            seed=training_seed,
            metrics_path=metrics_paths[training_seed].resolve(),
            checkpoint_path=checkpoint_paths[training_seed].resolve(),
            expected_checkpoint_sha256=EXPECTED_SELECTED_CHECKPOINT_SHA256[
                training_seed
            ],
            config_path=config_path,
            config_sha256=config_sha256,
            train_bank_path=train_bank_path,
            validation_bank_path=validation_bank_path,
        )

    act = _read_episodes(
        act_path,
        expected_episode_count=episodes,
        expected_seed_start=evaluation_seed_start,
        expected_noise_level=noise_level,
    )
    if act.recovery_attempts or act.recovery_successes:
        raise ValueError("ACT reference unexpectedly contains recovery attempts")

    model_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for training_seed in EXPECTED_TRAINING_SEEDS:
        model = _read_episodes(
            model_paths[training_seed],
            expected_episode_count=episodes,
            expected_seed_start=evaluation_seed_start,
            expected_noise_level=noise_level,
        )
        if model.seeds != act.seeds:
            raise ValueError(
                f"training seed {training_seed} is not paired with ACT"
            )
        checkpoint = checkpoint_paths[training_seed]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        training = training_audits[training_seed]
        evaluation_audit_sha256 = _validate_evaluation_audit(
            path=evaluation_audit_paths[training_seed].resolve(),
            training_seed=training_seed,
            model=model,
            expected_checkpoint_sha256=training.selected_checkpoint_sha256,
            expected_act_checkpoint_sha256=act_checkpoint_sha256,
            expected_detector_checkpoint_sha256=detector_checkpoint_sha256,
            expected_episode_count=episodes,
            expected_seed_start=evaluation_seed_start,
            expected_noise_level=noise_level,
        )

        act_successes = act.successes
        model_successes = model.successes
        rescued = int(np.count_nonzero(~act_successes & model_successes))
        harmed = int(np.count_nonzero(act_successes & ~model_successes))
        success_rate = float(model_successes.mean())
        recovery_rate = (
            float(model.recovery_successes / model.recovery_attempts)
            if model.recovery_attempts
            else 0.0
        )
        success_ci_lower, success_ci_upper = _wilson_interval(
            int(model_successes.sum()),
            episodes,
        )
        recovery_ci_lower, recovery_ci_upper = _wilson_interval(
            model.recovery_successes,
            model.recovery_attempts,
        )
        paired_gain = float(
            100.0 * (model_successes.mean() - act_successes.mean())
        )
        ci_lower, ci_upper = _paired_bootstrap_ci(
            act_successes,
            model_successes,
            samples=bootstrap_samples,
            seed=bootstrap_seed + training_seed,
        )
        result = {
            "recovery_training_seed": training_seed,
            "training_target_timesteps": training.target_timesteps,
            "training_starting_timesteps": training.starting_timesteps,
            "training_actual_additional_timesteps": (
                training.actual_additional_timesteps
            ),
            "training_actual_timesteps": training.trained_timesteps,
            "selected_model_timesteps": training.selected_model_timesteps,
            "selected_model_type": training.selected_model_type,
            "training_metrics": str(training.metrics_path),
            "training_metrics_sha256": training.metrics_sha256,
            "checkpoint": str(training.selected_checkpoint),
            "checkpoint_sha256": training.selected_checkpoint_sha256,
            "final_training_checkpoint": str(training.final_checkpoint),
            "final_training_checkpoint_sha256": (
                training.final_checkpoint_sha256
            ),
            "validation_history_records": training.validation_history_records,
            "evaluation_audit": str(
                evaluation_audit_paths[training_seed].resolve()
            ),
            "evaluation_audit_sha256": evaluation_audit_sha256,
            "raw_episodes": str(model.path.resolve()),
            "raw_episodes_sha256": _sha256(model.path),
            "episodes": episodes,
            "successes": int(model_successes.sum()),
            "success_rate": success_rate,
            "success_wilson_95_ci": [
                success_ci_lower,
                success_ci_upper,
            ],
            "recovery_attempts": model.recovery_attempts,
            "recovery_successes": model.recovery_successes,
            "recovery_rate": recovery_rate,
            "recovery_wilson_95_ci": [
                recovery_ci_lower,
                recovery_ci_upper,
            ],
            "paired_gain_vs_act_percentage_points": paired_gain,
            "paired_gain_bootstrap_95_ci_percentage_points": [
                ci_lower,
                ci_upper,
            ],
            "rescued": rescued,
            "harmed": harmed,
            "exact_mcnemar_p_value": _exact_mcnemar_p_value(
                rescued,
                harmed,
            ),
        }
        model_results.append(result)
        csv_rows.append(
            {
                "Analysis ID": ANALYSIS_ID,
                "Row Type": "training_seed",
                "Recovery Training Seed": training_seed,
                "Training Target Steps": training.target_timesteps,
                "Actual Trained Steps": training.trained_timesteps,
                "Selected Checkpoint Steps": training.selected_model_timesteps,
                "Checkpoint": checkpoint.name,
                "Checkpoint SHA256": training.selected_checkpoint_sha256,
                "Success Rate": success_rate,
                "Success CI Lower": success_ci_lower,
                "Success CI Upper": success_ci_upper,
                "Success Rate Sample Std": "",
                "Recovery Rate": recovery_rate,
                "Recovery CI Lower": recovery_ci_lower,
                "Recovery CI Upper": recovery_ci_upper,
                "Recovery Rate Sample Std": "",
                "Paired Gain vs ACT (pp)": paired_gain,
                "Paired Gain Sample Std (pp)": "",
                "Paired Gain CI Lower (pp)": ci_lower,
                "Paired Gain CI Upper (pp)": ci_upper,
                "Rescued": rescued,
                "Harmed": harmed,
                "Exact McNemar p": _exact_mcnemar_p_value(rescued, harmed),
                "Successes": int(model_successes.sum()),
                "Recovery Attempts": model.recovery_attempts,
                "Recovery Successes": model.recovery_successes,
                "Episodes": episodes,
                "Evaluation Seed Start": evaluation_seed_start,
                "Evaluation Seed End": evaluation_seed_start + episodes - 1,
                "Noise Level": noise_level,
                "Training Seed Count": 1,
                "Result Scope": RESULT_SCOPE,
                "Benchmark Eligible": False,
                "Fixed ACT SHA256": act_checkpoint_sha256,
                "Fixed Detector SHA256": detector_checkpoint_sha256,
                "Frozen Config SHA256": config_sha256,
            }
        )

    success_summary = _sample_summary(
        [item["success_rate"] for item in model_results]
    )
    recovery_summary = _sample_summary(
        [item["recovery_rate"] for item in model_results]
    )
    gain_summary = _sample_summary(
        [
            item["paired_gain_vs_act_percentage_points"]
            for item in model_results
        ]
    )
    aggregate = {
        "n_training_seeds": len(model_results),
        "training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "success_rate_mean": success_summary["mean"],
        "success_rate_sample_std": success_summary["sample_std"],
        "recovery_rate_mean": recovery_summary["mean"],
        "recovery_rate_sample_std": recovery_summary["sample_std"],
        "paired_gain_vs_act_percentage_points_mean": gain_summary["mean"],
        "paired_gain_vs_act_percentage_points_sample_std": gain_summary[
            "sample_std"
        ],
    }
    csv_rows.append(
        {
            "Analysis ID": ANALYSIS_ID,
            "Row Type": "aggregate_mean_sample_std",
            "Recovery Training Seed": "mean_n3",
            "Training Target Steps": MINIMUM_TRAINING_BUDGET,
            "Actual Trained Steps": "all>=500000",
            "Selected Checkpoint Steps": "",
            "Checkpoint": "",
            "Checkpoint SHA256": "",
            "Success Rate": success_summary["mean"],
            "Success CI Lower": "",
            "Success CI Upper": "",
            "Success Rate Sample Std": success_summary["sample_std"],
            "Recovery Rate": recovery_summary["mean"],
            "Recovery CI Lower": "",
            "Recovery CI Upper": "",
            "Recovery Rate Sample Std": recovery_summary["sample_std"],
            "Paired Gain vs ACT (pp)": gain_summary["mean"],
            "Paired Gain Sample Std (pp)": gain_summary["sample_std"],
            "Paired Gain CI Lower (pp)": "",
            "Paired Gain CI Upper (pp)": "",
            "Rescued": "",
            "Harmed": "",
            "Exact McNemar p": "",
            "Successes": "",
            "Recovery Attempts": "",
            "Recovery Successes": "",
            "Episodes": episodes,
            "Evaluation Seed Start": evaluation_seed_start,
            "Evaluation Seed End": evaluation_seed_start + episodes - 1,
            "Noise Level": noise_level,
            "Training Seed Count": len(model_results),
            "Result Scope": RESULT_SCOPE,
            "Benchmark Eligible": False,
            "Fixed ACT SHA256": act_checkpoint_sha256,
            "Fixed Detector SHA256": detector_checkpoint_sha256,
            "Frozen Config SHA256": config_sha256,
        }
    )

    payload = {
        "schema_version": 2,
        "analysis_id": ANALYSIS_ID,
        "result_scope": RESULT_SCOPE,
        "benchmark_eligible": False,
        "disclaimer": (
            "Equal-budget recovery-policy seed stability on 200 paired "
            "development episodes per seed. ACT and the failure detector are "
            "fixed; only the PPO recovery training seed changes. This is not "
            "a three-seed full-system result and is not the primary benchmark."
        ),
        "protocol": {
            "backend": "metaworld",
            "training_budget_minimum_actual_env_steps": (
                MINIMUM_TRAINING_BUDGET
            ),
            "varied_component": "PPO recovery policy training seed only",
            "fixed_components": [
                "ACT checkpoint",
                "failure-detector checkpoint",
                "PPO configuration",
                "recovery training snapshot bank",
                "recovery validation snapshot bank",
                "paired evaluation episodes",
            ],
            "episodes_per_training_seed": episodes,
            "evaluation_seed_start": evaluation_seed_start,
            "evaluation_seed_end": evaluation_seed_start + episodes - 1,
            "noise_level": noise_level,
            "failure_threshold": 0.2,
            "recovery_exit_threshold": 0.0,
            "recovery_budget": 150,
            "recovery_min_steps": 150,
            "recovery_clear_steps": 200,
            "max_steps": 200,
        },
        "equal_budget_and_frozen_config_validation": {
            "all_actual_training_timesteps_at_least_500000": all(
                audit.trained_timesteps >= MINIMUM_TRAINING_BUDGET
                for audit in training_audits.values()
            ),
            "all_targets_equal_500000": all(
                audit.target_timesteps == MINIMUM_TRAINING_BUDGET
                for audit in training_audits.values()
            ),
            "all_selected_checkpoints_match_audited_sha256": True,
            "all_selected_and_final_checkpoint_internal_timesteps_match_metrics": True,
            "all_validation_histories_sorted_unique_and_finite": True,
            "config": str(config_path),
            "config_sha256": config_sha256,
            "frozen_training_config": FROZEN_TRAINING_CONFIG,
            "training_snapshot_bank": str(train_bank_path),
            "training_snapshot_bank_sha256": train_bank_sha256,
            "validation_snapshot_bank": str(validation_bank_path),
            "validation_snapshot_bank_sha256": validation_bank_sha256,
        },
        "bootstrap": {
            "type": "paired_episode_nonparametric",
            "samples": bootstrap_samples,
            "base_seed": bootstrap_seed,
            "per_model_seed": "base_seed + recovery_training_seed",
            "confidence_level": 0.95,
        },
        "act_reference": {
            "raw_episodes": str(act.path.resolve()),
            "raw_episodes_sha256": _sha256(act.path),
            "checkpoint": str(act_checkpoint_path),
            "checkpoint_sha256": act_checkpoint_sha256,
            "failure_detector_checkpoint": str(detector_checkpoint_path),
            "failure_detector_checkpoint_sha256": detector_checkpoint_sha256,
            "episodes": episodes,
            "successes": int(act.successes.sum()),
            "success_rate": float(act.successes.mean()),
            "success_wilson_95_ci": list(
                _wilson_interval(int(act.successes.sum()), episodes)
            ),
        },
        "models": model_results,
        "aggregate_mean_and_sample_std": aggregate,
        "raw_validation": {
            "expected_training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "continuous_exact_ordered_evaluation_seed_range": True,
            "paired_with_same_act_episode_order": True,
            "unique_evaluation_seeds_per_run": episodes,
            "no_literal_nan": True,
            "core_numeric_fields_finite": True,
            "summaries_recomputed_from_raw": True,
            "training_metrics_checked": True,
            "checkpoint_sha256_checked": True,
            "evaluation_audit_sha256_checked": True,
            "benchmark_eligible": False,
        },
    }
    return csv_rows, payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _render_latex_table(payload: dict[str, Any]) -> str:
    models = payload["models"]
    aggregate = payload["aggregate_mean_and_sample_std"]
    act_rate = 100.0 * payload["act_reference"]["success_rate"]
    rows: list[str] = []
    for item in models:
        gain_ci = item[
            "paired_gain_bootstrap_95_ci_percentage_points"
        ]
        rows.append(
            " & ".join(
                (
                    str(item["recovery_training_seed"]),
                    f"{item['training_actual_timesteps']:,}",
                    f"{item['selected_model_timesteps']:,}",
                    f"{100.0 * item['success_rate']:.1f}",
                    f"{100.0 * item['recovery_rate']:.1f}",
                    (
                        f"{item['paired_gain_vs_act_percentage_points']:+.1f} "
                        f"[{gain_ci[0]:.1f}, {gain_ci[1]:.1f}]"
                    ),
                    f"{item['rescued']}/{item['harmed']}",
                )
            )
            + r" \\"
        )
    rows.append(r"\midrule")
    rows.append(
        " & ".join(
            (
                r"Mean $\pm$ s.d.",
                r"all $\geq$500k",
                "--",
                (
                    f"{100.0 * aggregate['success_rate_mean']:.1f} "
                    f"$\\pm$ {100.0 * aggregate['success_rate_sample_std']:.1f}"
                ),
                (
                    f"{100.0 * aggregate['recovery_rate_mean']:.1f} "
                    f"$\\pm$ {100.0 * aggregate['recovery_rate_sample_std']:.1f}"
                ),
                (
                    f"{aggregate['paired_gain_vs_act_percentage_points_mean']:+.1f} "
                    f"$\\pm$ "
                    f"{aggregate['paired_gain_vs_act_percentage_points_sample_std']:.1f}"
                ),
                "--",
            )
        )
        + r" \\"
    )
    body = "\n".join(rows)
    return (
        "% Auto-generated from measured equal-budget episode CSVs by "
        "experiments/model_seed_stability.py.\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Legacy PPO negative-control seed stability with the ACT "
        "policy and failure detector fixed. Only the PPO training seed "
        "changes. Each row uses 200 paired Meta-World development "
        "episodes (noise level 0.2; seeds 7000042--7000241). This unreferenced "
        "development diagnostic is not part of REIM's supervised recovery "
        "method or primary benchmark. "
        "Success and recovery are percentages; $\\Delta$ is the paired "
        f"success gain over the fixed ACT reference ({act_rate:.1f}\\%).}}\n"
        "\\label{tab:model-seed-stability}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4.2pt}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Recovery seed & Actual steps & Selected step & Success "
        "(\\%) & Recovery (\\%) & $\\Delta$ vs. ACT [95\\% CI] (pp) "
        "& Rescued/Harmed \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\vspace{1mm}\n"
        "\\begin{minipage}{0.98\\textwidth}\n"
        "\\footnotesize Recovery rate is task success while recovery is active "
        "per intervention. Confidence intervals for $\\Delta$ use 20,000 "
        "paired episode bootstrap samples; s.d. is the sample standard "
        "deviation across the three recovery seeds. This table does not "
        "represent three independently trained full REIM systems.\n"
        "\\end{minipage}\n"
        "\\end{table*}\n"
    )


def _save_figure_pair(fig: Any, output_png: Path) -> tuple[Path, Path]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    temporary_png = output_png.with_name(f".{output_png.stem}.tmp.png")
    temporary_pdf = output_pdf.with_name(f".{output_pdf.stem}.tmp.pdf")
    metadata = {
        "Title": "Equal-budget recovery-policy seed stability",
        "Author": "REIM experiment pipeline",
        "Software": "Matplotlib",
    }
    fig.savefig(
        temporary_png,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    fig.savefig(
        temporary_pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": metadata["Title"],
            "Author": metadata["Author"],
            "Creator": metadata["Software"],
            "CreationDate": None,
            "ModDate": None,
        },
    )
    temporary_png.replace(output_png)
    temporary_pdf.replace(output_pdf)
    return output_png, output_pdf


def _copy_figure_pair(source_png: Path, destination_png: Path) -> None:
    destination_png.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (source_png, destination_png),
        (source_png.with_suffix(".pdf"), destination_png.with_suffix(".pdf")),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def _plot_seed_stability(
    payload: dict[str, Any],
    *,
    output_png: Path,
    paper_output_png: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for the stability figure") from exc

    blue = "#356A9A"
    teal = "#4B8F8C"
    orange = "#D97935"
    ink = "#25313C"
    gray = "#75808A"
    light_gray = "#E9EDF0"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": gray,
            "axes.linewidth": 0.8,
            "xtick.color": ink,
            "ytick.color": ink,
            "text.color": ink,
            "axes.labelcolor": ink,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    models = payload["models"]
    aggregate = payload["aggregate_mean_and_sample_std"]
    seeds = [int(item["recovery_training_seed"]) for item in models]
    x = np.arange(len(seeds), dtype=np.float64)
    success = 100.0 * np.asarray(
        [float(item["success_rate"]) for item in models]
    )
    recovery = 100.0 * np.asarray(
        [float(item["recovery_rate"]) for item in models]
    )
    success_ci = np.asarray(
        [item["success_wilson_95_ci"] for item in models],
        dtype=np.float64,
    ).T * 100.0
    recovery_ci = np.asarray(
        [item["recovery_wilson_95_ci"] for item in models],
        dtype=np.float64,
    ).T * 100.0
    gain = np.asarray(
        [item["paired_gain_vs_act_percentage_points"] for item in models],
        dtype=np.float64,
    )
    gain_ci = np.asarray(
        [
            item["paired_gain_bootstrap_95_ci_percentage_points"]
            for item in models
        ],
        dtype=np.float64,
    ).T

    fig, (ax_rate, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    offset = 0.12
    ax_rate.errorbar(
        x - offset,
        success,
        yerr=np.vstack((success - success_ci[0], success_ci[1] - success)),
        fmt="o",
        color=blue,
        markersize=5.5,
        elinewidth=1.2,
        capsize=2.5,
        label="Task success",
        zorder=3,
    )
    ax_rate.errorbar(
        x + offset,
        recovery,
        yerr=np.vstack(
            (recovery - recovery_ci[0], recovery_ci[1] - recovery)
        ),
        fmt="s",
        color=teal,
        markersize=5.0,
        elinewidth=1.2,
        capsize=2.5,
        label="Recovery success",
        zorder=3,
    )
    act_rate = 100.0 * float(payload["act_reference"]["success_rate"])
    ax_rate.axhline(
        act_rate,
        color=gray,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label=f"Fixed ACT ({act_rate:.1f}%)",
        zorder=1,
    )
    actual_steps = [
        int(item["training_actual_timesteps"]) for item in models
    ]
    ax_rate.set_xticks(
        x,
        [f"Seed {seed}\n{steps / 1000.0:.1f}k steps"
         for seed, steps in zip(seeds, actual_steps)],
    )
    ax_rate.set_ylabel("Rate (%)")
    ax_rate.set_ylim(50.0, 95.0)
    ax_rate.set_title("(a) Outcomes across recovery seeds", loc="left")
    ax_rate.grid(axis="y", color=light_gray, linewidth=0.7, zorder=0)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.legend(
        frameon=False,
        fontsize=7.4,
        loc="lower left",
        ncol=1,
    )

    y = np.arange(len(seeds), dtype=np.float64)
    gain_mean = float(
        aggregate["paired_gain_vs_act_percentage_points_mean"]
    )
    gain_std = float(
        aggregate["paired_gain_vs_act_percentage_points_sample_std"]
    )
    ax_gain.axvspan(
        gain_mean - gain_std,
        gain_mean + gain_std,
        color=orange,
        alpha=0.11,
        linewidth=0,
        label=r"Seed mean $\pm$ s.d.",
        zorder=0,
    )
    ax_gain.axvline(
        gain_mean,
        color=orange,
        linewidth=1.1,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax_gain.axvline(0.0, color=gray, linewidth=0.9, zorder=1)
    ax_gain.errorbar(
        gain,
        y,
        xerr=np.vstack((gain - gain_ci[0], gain_ci[1] - gain)),
        fmt="o",
        color=blue,
        markersize=5.5,
        elinewidth=1.3,
        capsize=2.5,
        zorder=3,
    )
    for row_index, item in enumerate(models):
        ax_gain.text(
            16.7,
            row_index,
            f"R/H {item['rescued']}/{item['harmed']}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=ink,
        )
    ax_gain.set_yticks(y, [f"Seed {seed}" for seed in seeds])
    ax_gain.invert_yaxis()
    ax_gain.set_xlim(-1.0, 19.2)
    ax_gain.set_xlabel("Paired success gain vs. ACT (pp)")
    ax_gain.set_title("(b) Episode-paired gain (95% CI)", loc="left")
    ax_gain.grid(axis="x", color=light_gray, linewidth=0.7, zorder=0)
    ax_gain.spines["top"].set_visible(False)
    ax_gain.spines["right"].set_visible(False)
    ax_gain.legend(frameon=False, fontsize=7.4, loc="upper left")

    fig.text(
        0.5,
        0.012,
        (
            "Fixed ACT + detector  |  only recovery seed varies  |  "
            "n=200 paired episodes/seed  |  development diagnostic, "
            "not the primary benchmark"
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=orange,
        weight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0), w_pad=2.0)
    _save_figure_pair(fig, output_png)
    plt.close(fig)
    _copy_figure_pair(output_png, paper_output_png)


def _parse_seed_path_specs(
    specifications: Sequence[str] | None,
    defaults: dict[int, Path],
) -> dict[int, Path]:
    if not specifications:
        return dict(defaults)
    parsed: dict[int, Path] = {}
    for specification in specifications:
        seed_text, separator, path_text = specification.partition("=")
        if not separator or not path_text:
            raise ValueError(
                f"invalid SEED=PATH specification: {specification!r}"
            )
        seed = int(seed_text)
        if seed in parsed:
            raise ValueError(f"duplicate recovery training seed: {seed}")
        path = Path(path_text).expanduser()
        parsed[seed] = (
            path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--act-episodes",
        type=Path,
        default=PROJECT_ROOT
        / "results/tables/model_seeds/act_seed7000042_episodes.csv",
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="SEED=PATH",
        help=(
            "Raw REIM episode CSV for a training seed; repeat for 42, 43, 44. "
            "The isolated model_seeds outputs are used by default."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="SEED=PATH",
        help=(
            "Recovery checkpoint for a training seed; repeat for 42, 43, 44. "
            "The audited equal-budget 500k checkpoints are used by default."
        ),
    )
    parser.add_argument(
        "--training-metrics",
        action="append",
        metavar="SEED=PATH",
        help=(
            "Recovery training metrics JSON for a seed; repeat for 42, 43, "
            "44. The equal-budget 500k metrics are used by default."
        ),
    )
    parser.add_argument(
        "--evaluation-audit",
        action="append",
        metavar="SEED=PATH",
        help=(
            "Independent 200-episode evaluation audit JSON for a recovery "
            "seed; repeat for 42, 43, 44."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--train-bank",
        type=Path,
        default=DEFAULT_TRAIN_BANK,
    )
    parser.add_argument(
        "--validation-bank",
        type=Path,
        default=DEFAULT_VALIDATION_BANK,
    )
    parser.add_argument(
        "--act-checkpoint",
        type=Path,
        default=DEFAULT_ACT_CHECKPOINT,
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=DEFAULT_DETECTOR_CHECKPOINT,
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--evaluation-seed-start", type=int, default=7000042)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7000042)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results/tables/model_seed_stability.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results/tables/model_seed_stability.json",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=PROJECT_ROOT / "paper_assets/Table4_model_seed_stability.tex",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_ROOT / "results/figures/model_seed_stability.png",
    )
    parser.add_argument(
        "--paper-output-figure",
        type=Path,
        default=PROJECT_ROOT
        / "paper_assets/Figure6_model_seed_stability.png",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    model_paths = _parse_seed_path_specs(args.run, DEFAULT_MODEL_RUNS)
    checkpoint_paths = _parse_seed_path_specs(
        args.checkpoint,
        DEFAULT_CHECKPOINTS,
    )
    metrics_paths = _parse_seed_path_specs(
        args.training_metrics,
        DEFAULT_TRAINING_METRICS,
    )
    evaluation_audit_paths = _parse_seed_path_specs(
        args.evaluation_audit,
        DEFAULT_EVALUATION_AUDITS,
    )
    csv_rows, payload = summarize(
        act_path=args.act_episodes.resolve(),
        model_paths=model_paths,
        checkpoint_paths=checkpoint_paths,
        metrics_paths=metrics_paths,
        evaluation_audit_paths=evaluation_audit_paths,
        config_path=args.config.resolve(),
        train_bank_path=args.train_bank.resolve(),
        validation_bank_path=args.validation_bank.resolve(),
        act_checkpoint_path=args.act_checkpoint.resolve(),
        detector_checkpoint_path=args.detector_checkpoint.resolve(),
        episodes=args.episodes,
        evaluation_seed_start=args.evaluation_seed_start,
        noise_level=args.noise_level,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_write_csv(args.output_csv, csv_rows)
    atomic_json_dump(payload, args.output_json)
    _atomic_text(args.output_table, _render_latex_table(payload))
    _plot_seed_stability(
        payload,
        output_png=args.output_figure,
        paper_output_png=args.paper_output_figure,
    )
    LOGGER.info(
        "Saved equal-budget non-primary recovery-seed stability analysis to "
        "%s, %s, %s, and %s",
        args.output_csv,
        args.output_json,
        args.output_table,
        args.output_figure,
    )
    return payload


if __name__ == "__main__":
    main()
