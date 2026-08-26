#!/usr/bin/env python3
"""Audit and summarize the recovery-policy training-design ablation.

Five recovery checkpoints are evaluated on the same 200 Meta-World seeds:

1. exact online trigger states + expert actor warm start + task-selected model;
2. approximate disturbed starts + expert actor warm start + task-selected model;
3. exact online trigger states + random actor initialization + task-selected model;
4. the expert-actor warm start before any PPO environment interaction; and
5. the unselected 500k endpoint of the full training run.

The script refuses summary-only inputs.  It recomputes every rate from ordered
per-episode CSVs, checks checkpoint/config/metrics provenance, and then writes
CSV, JSON, a compact LaTeX table, and a publication figure.
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
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_reim import _atomic_write_csv  # noqa: E402
from utils.common import atomic_json_dump  # noqa: E402


LOGGER = logging.getLogger("reim.training_design_ablation")
SCOPE = "paired_training_design_ablation_not_primary_benchmark"


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    start_label: str
    actor_label: str
    checkpoint_label: str
    raw_episodes: Path
    checkpoint: Path
    metrics: Path
    config: Path
    use_training_endpoint: bool = False
    pretrain_only: bool = False


@dataclass(frozen=True)
class EpisodeSet:
    path: Path
    seeds: tuple[int, ...]
    successes: np.ndarray
    steps: np.ndarray
    recovery_attempts: np.ndarray
    recovery_successes: np.ndarray


DEFAULT_CONDITIONS = (
    Condition(
        key="full_selected",
        label="Full REIM",
        start_label="Online",
        actor_label="Expert",
        checkpoint_label="Task-selected",
        raw_episodes=ROOT
        / "results/tables/training_design/"
        "full_selected_seed7300042_episodes.csv",
        checkpoint=ROOT / "checkpoints/recovery_trigger_seed42.zip",
        metrics=ROOT
        / "results/tables/recovery_trigger_seed42_500k_metrics.json",
        config=ROOT / "configs/ppo_trigger.yaml",
    ),
    Condition(
        key="approximate_starts",
        label="Approximate starts",
        start_label="Approx.",
        actor_label="Expert",
        checkpoint_label="Task-selected",
        raw_episodes=ROOT
        / "results/tables/training_design/"
        "approximate_starts_seed7300042_episodes.csv",
        checkpoint=ROOT / "checkpoints/recovery_ablation_approximate.zip",
        metrics=ROOT
        / "results/tables/recovery_ablation_approximate_metrics.json",
        config=ROOT / "configs/ppo_approximate_starts.yaml",
    ),
    Condition(
        key="no_warmstart",
        label="No actor warm start",
        start_label="Online",
        actor_label="Random",
        checkpoint_label="Task-selected",
        raw_episodes=ROOT
        / "results/tables/training_design/"
        "no_warmstart_seed7300042_episodes.csv",
        checkpoint=ROOT / "checkpoints/recovery_ablation_scratch.zip",
        metrics=ROOT
        / "results/tables/recovery_ablation_scratch_metrics.json",
        config=ROOT / "configs/ppo_trigger_scratch.yaml",
    ),
    Condition(
        key="expert_only",
        label="Actor warm start only",
        start_label="Online",
        actor_label="Expert",
        checkpoint_label="No PPO",
        raw_episodes=ROOT
        / "results/tables/training_design/"
        "expert_only_seed7300042_episodes.csv",
        checkpoint=ROOT
        / "checkpoints/recovery_ablation_warmstart_only.zip",
        metrics=ROOT
        / "results/tables/recovery_ablation_warmstart_only_metrics.json",
        config=ROOT / "configs/ppo_trigger.yaml",
        pretrain_only=True,
    ),
    Condition(
        key="full_endpoint",
        label="500k endpoint",
        start_label="Online",
        actor_label="Expert",
        checkpoint_label="Final endpoint",
        raw_episodes=ROOT
        / "results/tables/training_design/"
        "full_endpoint_seed7300042_episodes.csv",
        checkpoint=ROOT
        / "checkpoints/recovery_trigger_seed42_500k_final.zip",
        metrics=ROOT
        / "results/tables/recovery_trigger_seed42_500k_metrics.json",
        config=ROOT / "configs/ppo_trigger.yaml",
        use_training_endpoint=True,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_bool(value: str, *, path: Path, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{path}: invalid {field}={value!r}")


def _noise_value(row: Mapping[str, str], path: Path) -> float:
    for key in ("Noise Level", "noise_level"):
        if key in row:
            return float(row[key])
    raise ValueError(f"{path}: missing noise-level field")


def _read_episodes(
    path: Path,
    *,
    expected_episodes: int,
    expected_seed_start: int,
    expected_noise: float,
) -> EpisodeSet:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    required = {
        "episode",
        "seed",
        "backend",
        "success",
        "steps",
        "recovery_attempts",
        "recovery_successes",
    }
    missing = required - fields
    if missing:
        raise ValueError(f"{path}: missing raw fields {sorted(missing)}")
    if len(rows) != expected_episodes:
        raise ValueError(
            f"{path}: expected {expected_episodes} rows, found {len(rows)}"
        )
    expected_seeds = tuple(
        range(expected_seed_start, expected_seed_start + expected_episodes)
    )
    episodes = tuple(int(float(row["episode"])) for row in rows)
    seeds = tuple(int(float(row["seed"])) for row in rows)
    if episodes != tuple(range(expected_episodes)):
        raise ValueError(f"{path}: episode indices are not ordered 0..n-1")
    if seeds != expected_seeds:
        raise ValueError(f"{path}: evaluation seed range/order mismatch")
    if len(set(seeds)) != expected_episodes:
        raise ValueError(f"{path}: evaluation seeds are not unique")

    successes: list[bool] = []
    steps: list[int] = []
    attempts: list[int] = []
    recovered: list[int] = []
    for row_index, row in enumerate(rows):
        if str(row["backend"]).strip().lower() != "metaworld":
            raise ValueError(f"{path}: row {row_index} is not Meta-World")
        if not math.isclose(
            _noise_value(row, path),
            expected_noise,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: row {row_index} noise mismatch")
        for field, value in row.items():
            if str(value).strip().lower() in {"nan", "+nan", "-nan"}:
                raise ValueError(
                    f"{path}: literal NaN at row {row_index}, field {field}"
                )
        row_steps = int(float(row["steps"]))
        row_attempts = int(float(row["recovery_attempts"]))
        row_recovered = int(float(row["recovery_successes"]))
        if (
            row_steps < 0
            or row_attempts < 0
            or row_recovered < 0
            or row_recovered > row_attempts
        ):
            raise ValueError(f"{path}: invalid counts at row {row_index}")
        successes.append(_parse_bool(row["success"], path=path, field="success"))
        steps.append(row_steps)
        attempts.append(row_attempts)
        recovered.append(row_recovered)
    return EpisodeSet(
        path=path,
        seeds=seeds,
        successes=np.asarray(successes, dtype=np.bool_),
        steps=np.asarray(steps, dtype=np.int64),
        recovery_attempts=np.asarray(attempts, dtype=np.int64),
        recovery_successes=np.asarray(recovered, dtype=np.int64),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _actual_training_steps(
    metrics: Mapping[str, Any],
    path: Path,
    *,
    pretrain_only: bool,
) -> int:
    try:
        trained = int(metrics["trained_timesteps"])
        target = int(metrics["target_timesteps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing training-step provenance") from exc
    if pretrain_only:
        if target != 0 or trained != 0:
            raise ValueError(
                f"{path}: actor-only control must contain zero PPO steps"
            )
        return trained
    if target < 500_000 or trained < target:
        raise ValueError(
            f"{path}: equal-budget control did not reach 500k "
            f"(trained={trained}, target={target})"
        )
    return trained


def _validate_design_configs(configs: Mapping[str, Mapping[str, Any]]) -> None:
    full = configs["full_selected"]
    approximate = configs["approximate_starts"]
    scratch = configs["no_warmstart"]
    if full.get("training") != approximate.get("training"):
        raise ValueError("approximate-start control changes PPO hyperparameters")
    if full.get("training") != scratch.get("training"):
        raise ValueError("no-warm-start control changes PPO hyperparameters")
    if full.get("reward") != approximate.get("reward"):
        raise ValueError("approximate-start control changes the reward")
    if full.get("reward") != scratch.get("reward"):
        raise ValueError("no-warm-start control changes the reward")

    full_expert = full.get("expert_pretraining", {})
    approximate_expert = approximate.get("expert_pretraining", {})
    scratch_expert = scratch.get("expert_pretraining", {})
    if full_expert.get("enabled") is not True:
        raise ValueError("full method must enable expert actor warm start")
    if approximate_expert != full_expert:
        raise ValueError("approximate-start control changes actor warm start")
    if scratch_expert.get("enabled") is not False:
        raise ValueError("no-warm-start control must disable expert pretraining")
    for key, value in full_expert.items():
        if key != "enabled" and scratch_expert.get(key) != value:
            raise ValueError(
                f"no-warm-start control changes inactive expert field {key}"
            )

    full_recovery = full.get("recovery", {})
    scratch_recovery = scratch.get("recovery", {})
    if scratch_recovery != full_recovery:
        raise ValueError("no-warm-start control changes recovery starts")
    approximate_recovery = approximate.get("recovery", {})
    if full_recovery.get("start_state_dataset") is None:
        raise ValueError("full method lacks exact online start-state dataset")
    if full_recovery.get("initialize_with_failure_states") is not False:
        raise ValueError("full method must restore exact start states")
    if approximate_recovery.get("initialize_with_failure_states") is not True:
        raise ValueError("approximate control must use disturbed warmup starts")
    if approximate_recovery.get("start_state_dataset") is not None:
        raise ValueError("approximate control unexpectedly restores train snapshots")
    if (
        approximate_recovery.get("validation_start_state_dataset")
        != full_recovery.get("validation_start_state_dataset")
    ):
        raise ValueError("controls do not share the validation snapshot bank")
    unchanged_recovery_keys = {
        "successful_starts_only",
        "max_recovery_steps",
        "recovered_distance",
        "relift_height",
        "terminate_on_recovery",
    }
    for key in unchanged_recovery_keys:
        if approximate_recovery.get(key) != full_recovery.get(key):
            raise ValueError(f"approximate control changes recovery field {key}")


def _bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, first.size, size=(samples, first.size))
    differences = (
        first[indices].mean(axis=1) - second[indices].mean(axis=1)
    ) * 100.0
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def _mcnemar(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if discordant == 0:
        return 1.0
    tail = min(rescued, harmed)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2.0**discordant)
    return float(min(1.0, 2.0 * probability))


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def summarize(
    *,
    act_raw_path: Path,
    conditions: Sequence[Condition],
    episodes: int,
    seed_start: int,
    noise_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    act = _read_episodes(
        act_raw_path,
        expected_episodes=episodes,
        expected_seed_start=seed_start,
        expected_noise=noise_level,
    )
    if int(act.recovery_attempts.sum()) != 0:
        raise ValueError("ACT reference unexpectedly contains interventions")
    if tuple(condition.key for condition in conditions) != (
        "full_selected",
        "approximate_starts",
        "no_warmstart",
        "expert_only",
        "full_endpoint",
    ):
        raise ValueError("training-design conditions are incomplete or reordered")

    configs = {
        condition.key: _read_yaml(condition.config)
        for condition in conditions
        if condition.key != "full_endpoint"
    }
    _validate_design_configs(configs)
    full_episode_set: EpisodeSet | None = None
    rows: list[dict[str, Any]] = []
    condition_payloads: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        episode_set = _read_episodes(
            condition.raw_episodes,
            expected_episodes=episodes,
            expected_seed_start=seed_start,
            expected_noise=noise_level,
        )
        if episode_set.seeds != act.seeds:
            raise ValueError(f"{condition.key}: raw episodes are not paired")
        if not condition.checkpoint.is_file():
            raise FileNotFoundError(condition.checkpoint)
        metrics = _read_json(condition.metrics)
        actual_steps = _actual_training_steps(
            metrics,
            condition.metrics,
            pretrain_only=condition.pretrain_only,
        )
        if str(metrics.get("backend", "")).lower() != "metaworld":
            raise ValueError(f"{condition.metrics}: training backend mismatch")
        if int(metrics.get("seed", -1)) != 42:
            raise ValueError(f"{condition.metrics}: recovery seed is not 42")
        metrics_config = Path(str(metrics.get("config", ""))).expanduser()
        if not metrics_config.is_absolute():
            metrics_config = (ROOT / metrics_config).resolve()
        if metrics_config.resolve() != condition.config.resolve():
            raise ValueError(f"{condition.metrics}: config provenance mismatch")
        metrics_checkpoint_key = (
            "final_training_checkpoint"
            if condition.use_training_endpoint
            else "checkpoint"
        )
        metrics_checkpoint = Path(
            str(metrics.get(metrics_checkpoint_key, ""))
        ).expanduser()
        if not metrics_checkpoint.is_absolute():
            metrics_checkpoint = (ROOT / metrics_checkpoint).resolve()
        if not metrics_checkpoint.is_file():
            raise ValueError(
                f"{condition.metrics}: missing {metrics_checkpoint_key}"
            )
        if _sha256(metrics_checkpoint) != _sha256(condition.checkpoint):
            raise ValueError(
                f"{condition.key}: checkpoint bytes do not match metrics"
            )
        if condition.use_training_endpoint:
            selected_steps = actual_steps
        else:
            try:
                selected_steps = int(metrics["selected_model_timesteps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{condition.metrics}: missing selected model step"
                ) from exc
            if condition.pretrain_only:
                if selected_steps != 0:
                    raise ValueError(
                        f"{condition.metrics}: actor-only step must be zero"
                    )
            elif selected_steps <= 0 or selected_steps > actual_steps:
                raise ValueError(
                    f"{condition.metrics}: invalid selected model step"
                )

        successes = int(episode_set.successes.sum())
        success_rate = successes / episodes
        attempts = int(episode_set.recovery_attempts.sum())
        recovery_successes = int(episode_set.recovery_successes.sum())
        recovery_rate = recovery_successes / attempts if attempts else 0.0
        intervened = int(np.count_nonzero(episode_set.recovery_attempts))
        intervention_rate = intervened / episodes
        gain = float(
            100.0
            * (episode_set.successes.mean(dtype=np.float64)
               - act.successes.mean(dtype=np.float64))
        )
        gain_ci = _bootstrap_difference(
            episode_set.successes,
            act.successes,
            samples=bootstrap_samples,
            seed=bootstrap_seed + condition_index,
        )
        rescued = int(
            np.count_nonzero(~act.successes & episode_set.successes)
        )
        harmed = int(
            np.count_nonzero(act.successes & ~episode_set.successes)
        )
        success_ci = _wilson(successes, episodes)
        if condition.key == "full_selected":
            full_episode_set = episode_set
            delta_full = 0.0
            delta_full_ci = (0.0, 0.0)
        else:
            if full_episode_set is None:
                raise RuntimeError("full condition must be processed first")
            delta_full = float(
                100.0
                * (
                    episode_set.successes.mean(dtype=np.float64)
                    - full_episode_set.successes.mean(dtype=np.float64)
                )
            )
            delta_full_ci = _bootstrap_difference(
                episode_set.successes,
                full_episode_set.successes,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 100 + condition_index,
            )
        result = {
            "condition": condition.key,
            "label": condition.label,
            "start_distribution": condition.start_label,
            "actor_initialization": condition.actor_label,
            "checkpoint_choice": condition.checkpoint_label,
            "checkpoint": str(condition.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(condition.checkpoint),
            "raw_episodes": str(condition.raw_episodes.resolve()),
            "raw_episodes_sha256": _sha256(condition.raw_episodes),
            "metrics": str(condition.metrics.resolve()),
            "metrics_sha256": _sha256(condition.metrics),
            "config": str(condition.config.resolve()),
            "config_sha256": _sha256(condition.config),
            "actual_training_steps": actual_steps,
            "checkpoint_training_steps": selected_steps,
            "episodes": episodes,
            "successes": successes,
            "success_rate": success_rate,
            "success_wilson_95_ci": list(success_ci),
            "recovery_attempts": attempts,
            "recovery_successes": recovery_successes,
            "recovery_rate": recovery_rate,
            "intervened_episodes": intervened,
            "intervention_rate": intervention_rate,
            "average_steps": float(episode_set.steps.mean()),
            "paired_gain_vs_act_percentage_points": gain,
            "paired_gain_vs_act_bootstrap_95_ci": list(gain_ci),
            "paired_delta_vs_full_percentage_points": delta_full,
            "paired_delta_vs_full_bootstrap_95_ci": list(delta_full_ci),
            "rescued_vs_act": rescued,
            "harmed_vs_act": harmed,
            "exact_mcnemar_p_value_vs_act": _mcnemar(rescued, harmed),
        }
        condition_payloads.append(result)
        rows.append(
            {
                "Condition": condition.label,
                "Start Distribution": condition.start_label,
                "Actor Initialization": condition.actor_label,
                "Checkpoint Choice": condition.checkpoint_label,
                "Checkpoint Steps": selected_steps,
                "Actual Training Steps": actual_steps,
                "Success Rate": success_rate,
                "Success CI Lower": success_ci[0],
                "Success CI Upper": success_ci[1],
                "Recovery Rate": recovery_rate,
                "Intervention Rate": intervention_rate,
                "Average Steps": float(episode_set.steps.mean()),
                "Paired Gain vs ACT (pp)": gain,
                "Gain CI Lower (pp)": gain_ci[0],
                "Gain CI Upper (pp)": gain_ci[1],
                "Paired Delta vs Full (pp)": delta_full,
                "Delta vs Full CI Lower (pp)": delta_full_ci[0],
                "Delta vs Full CI Upper (pp)": delta_full_ci[1],
                "Episodes": episodes,
                "Evaluation Seed Start": seed_start,
                "Evaluation Seed End": seed_start + episodes - 1,
                "Noise Level": noise_level,
                "Result Scope": SCOPE,
                "Benchmark Eligible": False,
            }
        )

    payload = {
        "schema_version": 1,
        "result_scope": SCOPE,
        "benchmark_eligible": False,
        "disclaimer": (
            "Focused recovery-training controls on one fixed ACT/detector "
            "pair and one recovery-training seed; not the primary benchmark."
        ),
        "protocol": {
            "backend": "metaworld",
            "episodes_per_condition": episodes,
            "evaluation_seed_start": seed_start,
            "evaluation_seed_end": seed_start + episodes - 1,
            "noise_level": noise_level,
            "failure_threshold": 0.2,
            "recovery_exit_threshold": 0.0,
            "recovery_budget": 150,
            "recovery_min_steps": 150,
            "recovery_clear_steps": 200,
            "max_steps": 200,
        },
        "act_reference": {
            "raw_episodes": str(act.path.resolve()),
            "raw_episodes_sha256": _sha256(act.path),
            "successes": int(act.successes.sum()),
            "success_rate": float(act.successes.mean()),
            "average_steps": float(act.steps.mean()),
        },
        "conditions": condition_payloads,
        "audit": {
            "all_ppo_runs_reached_at_least_500k_steps": True,
            "expert_actor_only_checkpoint_has_zero_ppo_steps": True,
            "same_ordered_paired_episode_seeds": True,
            "summaries_recomputed_from_raw": True,
            "no_literal_nan": True,
            "training_and_reward_hyperparameters_matched": True,
            "validation_snapshot_bank_matched": True,
            "checkpoint_and_input_sha256_recorded": True,
        },
        "bootstrap": {
            "type": "paired_episode_nonparametric",
            "samples": bootstrap_samples,
            "base_seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
    }
    return rows, payload


def _plot(path: Path, payload: Mapping[str, Any]) -> None:
    conditions = payload["conditions"]
    labels = [
        "Full\nselected",
        "Approx.\nstarts",
        "No warm\nstart",
        "Warm start\nonly",
        "500k\nendpoint",
    ]
    successes = np.asarray(
        [100.0 * row["success_rate"] for row in conditions]
    )
    success_low = np.asarray(
        [100.0 * row["success_wilson_95_ci"][0] for row in conditions]
    )
    success_high = np.asarray(
        [100.0 * row["success_wilson_95_ci"][1] for row in conditions]
    )
    recovery = np.asarray(
        [100.0 * row["recovery_rate"] for row in conditions]
    )
    interventions = np.asarray(
        [100.0 * row["intervention_rate"] for row in conditions]
    )
    gains = np.asarray(
        [row["paired_gain_vs_act_percentage_points"] for row in conditions]
    )
    gain_low = np.asarray(
        [row["paired_gain_vs_act_bootstrap_95_ci"][0] for row in conditions]
    )
    gain_high = np.asarray(
        [row["paired_gain_vs_act_bootstrap_95_ci"][1] for row in conditions]
    )
    colors = ["#2563EB", "#D97706", "#94A3B8", "#059669", "#0F172A"]
    x = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#CBD5E1",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    axes[0].bar(
        x,
        successes,
        color=colors,
        width=0.68,
        yerr=np.vstack((successes - success_low, success_high - successes)),
        capsize=2.5,
        error_kw={"linewidth": 0.8},
    )
    act_rate = 100.0 * payload["act_reference"]["success_rate"]
    axes[0].axhline(
        act_rate,
        color="#343A40",
        linewidth=1.0,
        linestyle="--",
        label=f"ACT {act_rate:.1f}%",
    )
    axes[0].set_title("(a) End-to-end task success")
    axes[0].set_ylabel("Success rate (%) ↑")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")

    width = 0.34
    axes[1].bar(
        x - width / 2,
        recovery,
        width=width,
        color="#2563EB",
        label="Intervention outcome",
    )
    axes[1].bar(
        x + width / 2,
        interventions,
        width=width,
        color="#D97706",
        label="Episodes intervened",
    )
    axes[1].set_title("(b) Recovery behavior")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].legend(frameon=False, fontsize=6.8, loc="upper right")

    axes[2].errorbar(
        x,
        gains,
        yerr=np.vstack((gains - gain_low, gain_high - gains)),
        fmt="o",
        markersize=5.5,
        color="#2563EB",
        ecolor="#94A3B8",
        capsize=3,
        linewidth=1.1,
    )
    axes[2].axhline(0.0, color="#0F172A", linewidth=0.9, linestyle="--")
    axes[2].set_title("(c) Paired gain over ACT")
    axes[2].set_ylabel("Success difference (pp) ↑")

    for axis in axes:
        axis.set_xticks(x, labels, fontsize=7.5)
        axis.grid(axis="y", color="#E9EEF3", linewidth=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.005,
        "Same 200 paired Meta-World seeds • 20% disturbance • "
        "PPO controls trained for ≥500k steps • actor-only control: 0 PPO steps",
        ha="center",
        fontsize=7,
        color="#65737E",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%")


def _write_latex(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Recovery training-design controls on the same 200 paired",
        r"  Meta-World seeds at 20\% disturbance. PPO controls receive at least",
        r"  500k environment steps; the actor-only row receives zero PPO steps.",
        r"  ``Online'' denotes full trigger-state reset; the endpoint row disables",
        r"  task-success checkpoint selection.}",
        r"  \label{tab:training-design}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3.2pt}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Condition & Start & Actor & Checkpoint & Success $\uparrow$ \\",
        r"    \midrule",
    ]
    for row in payload["conditions"]:
        lines.append(
            "    "
            + _latex_escape(row["label"])
            + " & "
            + _latex_escape(row["start_distribution"])
            + " & "
            + _latex_escape(row["actor_initialization"])
            + " & "
            + _latex_escape(row["checkpoint_choice"])
            + f" & {100.0 * row['success_rate']:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--act-episodes",
        type=Path,
        default=ROOT
        / "results/tables/training_design/act_seed7300042_episodes.csv",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--evaluation-seed-start", type=int, default=7_300_042)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7_300_042)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results/tables/training_design_ablation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results/tables/training_design_ablation.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results/figures/training_design_ablation.png",
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=ROOT / "paper_assets/Figure7_training_design.png",
    )
    parser.add_argument(
        "--latex-table",
        type=Path,
        default=ROOT / "paper_assets/Table5_training_design.tex",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    rows, payload = summarize(
        act_raw_path=args.act_episodes.resolve(),
        conditions=DEFAULT_CONDITIONS,
        episodes=args.episodes,
        seed_start=args.evaluation_seed_start,
        noise_level=args.noise_level,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_write_csv(args.output_csv, rows)
    atomic_json_dump(payload, args.output_json)
    _plot(args.figure, payload)
    args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
    args.paper_figure.write_bytes(args.figure.read_bytes())
    args.paper_figure.with_suffix(".pdf").write_bytes(
        args.figure.with_suffix(".pdf").read_bytes()
    )
    _write_latex(args.latex_table, payload)
    LOGGER.info(
        "Saved audited training-design ablation to %s, %s, %s, and %s",
        args.output_csv,
        args.output_json,
        args.figure,
        args.latex_table,
    )
    return payload


if __name__ == "__main__":
    main()
