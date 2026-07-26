#!/usr/bin/env python3
"""Train the PPO recovery policy with REIM reward shaping."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.metaworld_pickplace import REIMPickPlaceEnv  # noqa: E402
from models.recovery_policy import RecoveryRewardWrapper  # noqa: E402
from utils.common import (  # noqa: E402
    atomic_json_dump,
    configure_logging,
    load_yaml,
    resolve_path,
    seed_everything,
    select_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_trigger.yaml")
    parser.add_argument("--env-config", default="configs/environment.yaml")
    parser.add_argument("--output", "--checkpoint", dest="checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        help="Override the run-specific periodic/best-checkpoint directory.",
    )
    parser.add_argument("--tensorboard-log", help="Override TensorBoard log directory.")
    parser.add_argument("--monitor-dir", help="Override Monitor CSV directory.")
    parser.add_argument("--curve-path", help="Override the training-curve output path.")
    parser.add_argument("--metrics-path", help="Override the metrics JSON output path.")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--torch-threads",
        type=int,
        help=(
            "CPU threads used by PyTorch. Small PPO MLPs are commonly faster "
            "with a bounded thread pool than with every host core."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=("metaworld", "toy"))
    parser.add_argument("--state-mode", choices=("semantic", "raw"))
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume PATH, or choose the final/latest PPO checkpoint automatically.",
    )
    parser.add_argument(
        "--pretrain-only",
        action="store_true",
        help=(
            "Run the configured expert actor warm start, save the resulting "
            "zero-PPO-step policy, evaluate it on the validation bank, and exit."
        ),
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _latest_ppo_checkpoint(checkpoint_dir: Path, output_path: Path) -> Path | None:
    training_endpoint = output_path.with_name(
        f"{output_path.stem}_final{output_path.suffix}"
    )
    if training_endpoint.exists():
        return training_endpoint
    candidates = list(checkpoint_dir.glob("recovery_*_steps.zip"))
    if not candidates:
        # The public output may be validation-best and is therefore the last
        # resort for legacy runs that predate endpoint preservation.
        return output_path if output_path.exists() else None

    def step_number(path: Path) -> int:
        match = re.search(r"_(\d+)_steps$", path.stem)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_number)


def _isolate_resumed_tensorboard_log(model: Any, tensorboard_log: Path) -> None:
    """Redirect a loaded SB3 model away from its serialized parent log path."""

    model.tensorboard_log = str(tensorboard_log)


def _wrapper_kwargs(
    ppo_config: dict[str, Any],
    environment_config: dict[str, Any],
) -> dict[str, Any]:
    reward = dict(ppo_config.get("reward", {}))
    recovery = dict(ppo_config.get("recovery", {}))
    disturbance = dict(environment_config.get("disturbance", {}))
    return {
        "task_success_reward": float(reward.get("task_success", 10.0)),
        "recovery_reward": float(reward.get("successful_recovery", 5.0)),
        "failure_penalty": float(reward.get("terminal_failure", -10.0)),
        "time_penalty": float(reward.get("time_penalty", -0.01)),
        "base_reward_scale": float(reward.get("base_reward_scale", 0.0)),
        "distance_progress_scale": float(
            reward.get("distance_progress_scale", 2.0)
        ),
        "reach_progress_scale": float(reward.get("reach_progress_scale", 0.0)),
        "lift_progress_scale": float(reward.get("lift_progress_scale", 0.0)),
        "initialize_with_failure_states": bool(
            recovery.get("initialize_with_failure_states", True)
        ),
        "initialization_disturbance": float(
            recovery.get(
                "initialization_disturbance",
                disturbance.get("object_noise_magnitude", 0.04),
            )
        ),
        "recovered_distance": float(recovery.get("recovered_distance", 0.08)),
        "max_recovery_steps": int(recovery.get("max_recovery_steps", 50)),
        "warmup_min_steps": int(recovery.get("warmup_min_steps", 45)),
        "warmup_max_steps": int(recovery.get("warmup_max_steps", 45)),
        "relift_height": float(recovery.get("relift_height", 0.03)),
        "terminate_on_recovery": bool(
            recovery.get("terminate_on_recovery", True)
        ),
        "recovery_start_dataset": (
            str(resolve_path(recovery["start_state_dataset"]))
            if recovery.get("start_state_dataset")
            else None
        ),
        "successful_starts_only": bool(
            recovery.get("successful_starts_only", True)
        ),
    }


def make_environment_factory(
    *,
    environment_config: dict[str, Any],
    wrapper_kwargs: dict[str, Any],
    seed: int,
    backend: str | None,
    state_mode: str | None,
    monitor_file: Path | None,
    append_monitor: bool = False,
) -> Callable[[], Any]:
    def factory() -> Any:
        from stable_baselines3.common.monitor import Monitor

        env = REIMPickPlaceEnv(
            config=environment_config,
            seed=seed,
            backend=backend,
            state_mode=state_mode,
        )
        wrapped = RecoveryRewardWrapper(env, **wrapper_kwargs)
        if monitor_file is not None:
            monitor_file.parent.mkdir(parents=True, exist_ok=True)
        return Monitor(
            wrapped,
            filename=str(monitor_file) if monitor_file is not None else None,
            override_existing=not append_monitor,
            info_keywords=("success", "recovery_success", "failure"),
        )

    return factory


def evaluate_recovery(
    model: Any,
    environment_factory: Callable[[], Any],
    episodes: int,
) -> dict[str, Any]:
    environment = environment_factory()
    successes = 0
    recoveries = 0
    failures = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    try:
        for _ in tqdm(range(episodes), desc="PPO evaluation", unit="episode"):
            observation, _ = environment.reset()
            terminated = truncated = False
            total_reward = 0.0
            steps = 0
            episode_success = False
            episode_recovery = False
            episode_failure = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    action
                )
                total_reward += float(reward)
                steps += 1
                episode_success |= bool(info.get("success", False))
                episode_recovery |= bool(
                    info.get("recovery_success", info.get("recovered", False))
                )
                episode_failure |= bool(info.get("failure", False))
            successes += int(episode_success)
            recoveries += int(episode_recovery)
            failures += int(episode_failure)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
    finally:
        environment.close()
    return {
        "episodes": int(episodes),
        "success_rate": float(successes / max(episodes, 1)),
        "recovery_rate": float(recoveries / max(episodes, 1)),
        "failure_rate": float(failures / max(episodes, 1)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "mean_episode_steps": float(np.mean(episode_lengths)),
    }


def pretrain_recovery_actor(
    model: Any,
    *,
    dataset_path: Path,
    validation_dataset_path: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    observation_noise_std: float,
    seed: int,
    logger: Any,
) -> dict[str, Any]:
    """Warm-start PPO's actor from successful scripted recovery actions.

    This is deliberately actor-only supervision. PPO subsequently optimizes
    the same policy on the task reward from exact online ACT trigger states;
    benchmark execution contains no scripted-expert calls.
    """

    import torch
    import torch.nn.functional as functional

    def load_demonstrations(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Recovery demonstration dataset missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            schema = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
            if schema != "reim-recovery-starts-v1":
                raise ValueError(
                    f"Unsupported recovery demonstration schema {schema!r}: {path}"
                )
            states = np.asarray(archive["demo_states"], dtype=np.float32)
            actions = np.asarray(archive["demo_actions"], dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
            raise ValueError(
                f"Invalid recovery demonstrations in {path}: "
                f"states={states.shape}, actions={actions.shape}"
            )
        return states, actions

    train_states, train_actions = load_demonstrations(dataset_path)
    if validation_dataset_path is not None:
        validation_states, validation_actions = load_demonstrations(
            validation_dataset_path
        )
    else:
        generator = np.random.default_rng(seed)
        order = generator.permutation(len(train_states))
        validation_count = max(1, int(round(0.1 * len(order))))
        validation_indices = order[:validation_count]
        training_indices = order[validation_count:]
        validation_states = train_states[validation_indices]
        validation_actions = train_actions[validation_indices]
        train_states = train_states[training_indices]
        train_actions = train_actions[training_indices]
    if not len(train_states) or not len(validation_states):
        raise ValueError("Recovery pretraining requires non-empty train/validation data.")

    device = model.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, Any] | None = None
    model.policy.set_training_mode(True)

    def actor_loss(states: np.ndarray, actions: np.ndarray) -> Any:
        observations = torch.as_tensor(states, dtype=torch.float32, device=device)
        targets = torch.as_tensor(actions, dtype=torch.float32, device=device)
        distribution = model.policy.get_distribution(observations).distribution
        means = distribution.mean
        return functional.smooth_l1_loss(means, targets)

    for epoch in tqdm(
        range(1, epochs + 1), desc="Recovery actor pretraining", unit="epoch"
    ):
        order = rng.permutation(len(train_states))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            states = train_states[indices].copy()
            if observation_noise_std > 0.0:
                states += rng.normal(
                    0.0, observation_noise_std, size=states.shape
                ).astype(np.float32)
            loss = actor_loss(states, train_actions[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            validation_loss = float(
                actor_loss(validation_states, validation_actions).detach().cpu()
            )
        train_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.policy.state_dict())
    if best_state is not None:
        model.policy.load_state_dict(best_state)
    model.policy.set_training_mode(False)
    logger.info(
        "Recovery actor pretraining complete: train_samples=%d val_samples=%d "
        "best_val_loss=%.6f",
        len(train_states),
        len(validation_states),
        best_validation_loss,
    )
    return {
        "enabled": True,
        "dataset": str(dataset_path),
        "validation_dataset": (
            str(validation_dataset_path)
            if validation_dataset_path is not None
            else None
        ),
        "train_samples": int(len(train_states)),
        "validation_samples": int(len(validation_states)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "observation_noise_std": float(observation_noise_std),
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def _plot_monitor_files(
    monitor_dir: Path,
    curve_path: Path,
    *,
    evaluation_archive: Path | None = None,
    selected_timestep: int | None = None,
    phase_boundary_timestep: int | None = None,
) -> None:
    rewards: list[float] = []
    x_label = "Episode"
    for path in sorted(monitor_dir.glob("train_*.monitor.csv")):
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            del first_line
            reader = csv.DictReader(handle)
            rewards.extend(float(row["r"]) for row in reader if row.get("r"))
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        window = min(50, len(rewards))
        smoothed = np.convolve(
            rewards, np.ones(window, dtype=np.float64) / window, mode="valid"
        )
        axis.plot(episodes, rewards, alpha=0.22, color="#4c78a8", label="Episode")
        axis.plot(
            episodes[window - 1 :],
            smoothed,
            lw=2,
            color="#e45756",
            label=f"{window}-episode mean",
        )
        plotted = True
    elif evaluation_archive is not None and evaluation_archive.is_file():
        # A no-op resume must not erase the scientific training curve.  SB3's
        # evaluation archive is durable even when newly constructed Monitor
        # wrappers contain no episodes, so it is also a useful lower-variance
        # publication view of PPO progress.
        archive = np.load(evaluation_archive)
        timesteps = np.asarray(archive["timesteps"], dtype=np.int64)
        evaluations = np.asarray(archive["results"], dtype=np.float64)
        means = evaluations.mean(axis=1)
        standard_errors = evaluations.std(axis=1) / np.sqrt(
            max(evaluations.shape[1], 1)
        )
        axis.plot(
            timesteps,
            means,
            lw=2,
            color="#4c78a8",
            label="Validation mean",
        )
        axis.fill_between(
            timesteps,
            means - standard_errors,
            means + standard_errors,
            color="#4c78a8",
            alpha=0.18,
            linewidth=0,
            label="±1 standard error",
        )
        if selected_timestep is not None and timesteps.size:
            selected_index = int(
                np.argmin(np.abs(timesteps - int(selected_timestep)))
            )
            axis.scatter(
                [timesteps[selected_index]],
                [means[selected_index]],
                marker="D",
                s=34,
                color="#D97935",
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
                label=f"Selected checkpoint ({int(selected_timestep):,})",
            )
        if phase_boundary_timestep is not None:
            axis.axvline(
                int(phase_boundary_timestep),
                color="#4B8F8C",
                linestyle="--",
                linewidth=1.2,
                label=f"Reward curriculum ({int(phase_boundary_timestep):,})",
            )
        x_label = "Environment steps"
        plotted = True
    axis.set(
        xlabel=x_label,
        ylabel="Shaped return",
        title="PPO Recovery Training",
    )
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=320)
    figure.savefig(curve_path.with_suffix(".pdf"))
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:
        raise ImportError(
            "stable-baselines3 is required. Install dependencies with ./setup.sh."
        ) from error

    class EpisodeSignalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episode_successes: list[float] = []
            self.episode_recoveries: list[float] = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if done:
                    self.episode_successes.append(float(bool(info.get("success", False))))
                    self.episode_recoveries.append(
                        float(bool(info.get("recovery_success", False)))
                    )
            if self.episode_successes:
                self.logger.record(
                    "recovery/rolling_success_rate",
                    float(np.mean(self.episode_successes[-100:])),
                )
                self.logger.record(
                    "recovery/rolling_recovery_rate",
                    float(np.mean(self.episode_recoveries[-100:])),
                )
            return True

    class TaskSuccessEvalCallback(BaseCallback):
        """Select PPO checkpoints by held-out task success, not shaped return."""

        def __init__(
            self,
            *,
            environment_factory: Callable[[], Any],
            eval_freq: int,
            episodes: int,
            save_path: Path,
            history_path: Path,
            minimum_selection_timesteps: int = 0,
        ) -> None:
            super().__init__()
            self.environment_factory = environment_factory
            self.eval_freq = max(int(eval_freq), 1)
            self.episodes = int(episodes)
            self.save_path = save_path
            self.history_path = history_path
            self.minimum_selection_timesteps = max(
                int(minimum_selection_timesteps), 0
            )
            self.best_score = (-1.0, -1.0, float("-inf"))
            self.history: list[dict[str, Any]] = []

        @staticmethod
        def _score(metrics: dict[str, Any]) -> tuple[float, float, float]:
            # Task completion is lexicographically dominant. Recovery signal
            # and return break ties only between equal success rates.
            return (
                float(metrics["success_rate"]),
                float(metrics["recovery_rate"]),
                float(metrics["mean_episode_reward"]),
            )

        def _record(
            self, model: Any, metrics: dict[str, Any], timestep: int
        ) -> None:
            row = {"timesteps": int(timestep), **metrics}
            self.history.append(row)
            score = self._score(metrics)
            selection_eligible = (
                int(timestep) >= self.minimum_selection_timesteps
            )
            row["selection_eligible"] = selection_eligible
            if selection_eligible and score > self.best_score:
                self.best_score = score
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(self.save_path))
                row["selected"] = True
            else:
                row["selected"] = False
            atomic_json_dump(
                {
                    "selection_metric": "task_success",
                    "minimum_selection_timesteps": (
                        self.minimum_selection_timesteps
                    ),
                    "best_score": list(self.best_score),
                    "history": self.history,
                },
                self.history_path,
            )

        def record_initial(self, model: Any, metrics: dict[str, Any]) -> None:
            self._record(model, metrics, int(model.num_timesteps))

        def restore_history(self, max_timestep: int) -> None:
            """Restore task-success selection state for a true continuation."""

            if not self.history_path.is_file() or not self.save_path.is_file():
                return
            with self.history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = [
                dict(row)
                for row in payload.get("history", [])
                if isinstance(row, dict)
                and int(row.get("timesteps", -1)) <= int(max_timestep)
            ]
            eligible = [
                row
                for row in rows
                if bool(
                    row.get(
                        "selection_eligible",
                        int(row.get("timesteps", -1))
                        >= self.minimum_selection_timesteps,
                    )
                )
            ]
            if not eligible:
                return
            self.history = rows
            self.best_score = max(self._score(row) for row in eligible)

        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq:
                return True
            metrics = evaluate_recovery(
                self.model, self.environment_factory, self.episodes
            )
            self._record(self.model, metrics, int(self.num_timesteps))
            self.logger.record(
                "recovery_validation/task_success",
                float(metrics["success_rate"]),
            )
            self.logger.record(
                "recovery_validation/recovery_rate",
                float(metrics["recovery_rate"]),
            )
            return True

    config = load_yaml(args.config)
    environment_config = load_yaml(args.env_config)
    training = dict(config.get("training", {}))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device_name = select_device(args.device or config.get("device", "auto"))
    torch_threads = int(
        args.torch_threads
        if args.torch_threads is not None
        else training.get("torch_num_threads", 0)
    )
    if torch_threads < 0:
        raise ValueError("torch_num_threads must be non-negative.")
    if device_name == "cpu" and torch_threads:
        import torch

        torch.set_num_threads(torch_threads)
        # Inter-op parallelism gives no benefit for the small sequential MLP
        # graph and can otherwise multiply host contention across evaluations.
        try:
            torch.set_num_interop_threads(min(torch_threads, 2))
        except RuntimeError:
            # PyTorch permits setting this value only before inter-op work has
            # started. A resumed process may already have initialized the pool.
            pass
    seed_everything(seed)
    logger = configure_logging("train_recovery", "results/logs/train_recovery.log")

    output_path = resolve_path(
        args.checkpoint or config.get("checkpoint", "checkpoints/recovery_policy.zip")
    )
    if output_path.suffix != ".zip":
        output_path = output_path.with_suffix(".zip")
    checkpoint_dir = resolve_path(
        args.checkpoint_dir
        or config.get("checkpoint_dir", "checkpoints/ppo")
    )
    tensorboard_log = resolve_path(
        args.tensorboard_log
        or config.get("tensorboard_log", "results/logs/ppo")
    )
    monitor_dir = resolve_path(
        args.monitor_dir
        or config.get("monitor_dir", "results/logs/ppo_monitor")
    )
    curve_path = resolve_path(
        args.curve_path
        or config.get("curve_path", "results/figures/recovery_training_curve.png")
    )
    metrics_path = resolve_path(
        args.metrics_path
        or config.get("metrics_path", "results/tables/recovery_metrics.json")
    )
    for directory in (checkpoint_dir, tensorboard_log, monitor_dir):
        directory.mkdir(parents=True, exist_ok=True)

    total_timesteps = int(
        args.total_timesteps
        if args.total_timesteps is not None
        else training.get("total_timesteps", 500_000)
    )
    n_envs = int(args.n_envs if args.n_envs is not None else training.get("n_envs", 1))
    n_steps = int(
        args.n_steps if args.n_steps is not None else training.get("n_steps", 2048)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 256)
    )
    n_epochs = int(
        args.n_epochs if args.n_epochs is not None else training.get("n_epochs", 10)
    )
    eval_frequency = int(
        args.eval_freq if args.eval_freq is not None else training.get("eval_freq", 10_000)
    )
    checkpoint_frequency = int(
        args.checkpoint_freq
        if args.checkpoint_freq is not None
        else training.get("checkpoint_freq", 25_000)
    )
    eval_episodes = int(
        args.eval_episodes
        if args.eval_episodes is not None
        else training.get("eval_episodes", 20)
    )
    progress_bar = (
        bool(args.progress_bar)
        if args.progress_bar is not None
        else bool(training.get("progress_bar", True))
    )
    if min(total_timesteps, n_envs, n_steps, batch_size, n_epochs, eval_episodes) <= 0:
        raise ValueError("PPO training counts must all be positive.")

    wrapper_kwargs = _wrapper_kwargs(config, environment_config)
    evaluation_wrapper_kwargs = copy.deepcopy(wrapper_kwargs)
    recovery_config = dict(config.get("recovery", {}))
    validation_start_dataset = recovery_config.get(
        "validation_start_state_dataset"
    )
    if validation_start_dataset:
        evaluation_wrapper_kwargs["recovery_start_dataset"] = str(
            resolve_path(validation_start_dataset)
        )
    resume_requested = args.resume is not None or bool(config.get("resume"))
    if args.pretrain_only and resume_requested:
        raise ValueError("--pretrain-only cannot be combined with --resume")
    train_factories = [
        make_environment_factory(
            environment_config=environment_config,
            wrapper_kwargs=wrapper_kwargs,
            seed=seed + index,
            backend=args.backend,
            state_mode=args.state_mode,
            monitor_file=monitor_dir / f"train_{index}.monitor.csv",
            append_monitor=resume_requested,
        )
        for index in range(n_envs)
    ]
    train_environment = DummyVecEnv(train_factories)
    evaluation_factory = make_environment_factory(
        environment_config=environment_config,
        wrapper_kwargs=evaluation_wrapper_kwargs,
        seed=seed + 10_000,
        backend=args.backend,
        state_mode=args.state_mode,
        monitor_file=None,
    )
    evaluation_environment = DummyVecEnv([evaluation_factory])

    resume_value = args.resume
    if resume_value is None and config.get("resume"):
        resume_value = str(config["resume"])
    resume_path: Path | None = None
    if resume_value == "auto":
        resume_path = _latest_ppo_checkpoint(checkpoint_dir, output_path)
        if resume_path is None:
            raise FileNotFoundError(
                f"No PPO checkpoint found in {checkpoint_dir} or at {output_path}."
            )
    elif resume_value:
        resume_path = resolve_path(resume_value)
        if resume_path.suffix != ".zip":
            resume_path = resume_path.with_suffix(".zip")
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    pretraining_metrics: dict[str, Any] = {"enabled": False}
    if resume_path is not None:
        model = PPO.load(
            str(resume_path), env=train_environment, device=device_name
        )
        # SB3 serializes the original TensorBoard directory inside the model.
        # A resumed run with an explicit run-specific directory must not append
        # events to the parent run (for example, ``*_v2``).  Rebind it after
        # loading so checkpoint, Monitor, and TensorBoard artifacts all remain
        # isolated under the requested continuation lineage.
        _isolate_resumed_tensorboard_log(model, tensorboard_log)
        logger.info(
            "Resumed PPO from %s at %d environment steps.",
            resume_path,
            model.num_timesteps,
        )
    else:
        policy_kwargs = dict(training.get("policy_kwargs", {}))
        model = PPO(
            policy=str(training.get("policy", "MlpPolicy")),
            env=train_environment,
            learning_rate=float(
                args.learning_rate
                if args.learning_rate is not None
                else training.get("learning_rate", 3e-4)
            ),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=float(training.get("gamma", 0.99)),
            gae_lambda=float(training.get("gae_lambda", 0.95)),
            clip_range=float(training.get("clip_range", 0.2)),
            ent_coef=float(training.get("ent_coef", 0.01)),
            vf_coef=float(training.get("vf_coef", 0.5)),
            max_grad_norm=float(training.get("max_grad_norm", 0.5)),
            target_kl=(
                float(training["target_kl"])
                if training.get("target_kl") is not None
                else None
            ),
            tensorboard_log=str(tensorboard_log),
            policy_kwargs=policy_kwargs or None,
            verbose=1,
            seed=seed,
            device=device_name,
        )
        pretraining = dict(config.get("expert_pretraining", {}))
        if bool(pretraining.get("enabled", False)):
            dataset_value = pretraining.get(
                "dataset", recovery_config.get("start_state_dataset")
            )
            if not dataset_value:
                raise ValueError(
                    "expert_pretraining.enabled requires a dataset or "
                    "recovery.start_state_dataset."
                )
            validation_value = pretraining.get(
                "validation_dataset",
                recovery_config.get("validation_start_state_dataset"),
            )
            pretraining_metrics = pretrain_recovery_actor(
                model,
                dataset_path=resolve_path(dataset_value),
                validation_dataset_path=(
                    resolve_path(validation_value) if validation_value else None
                ),
                epochs=int(pretraining.get("epochs", 30)),
                batch_size=int(pretraining.get("batch_size", 512)),
                learning_rate=float(pretraining.get("learning_rate", 3e-4)),
                observation_noise_std=float(
                    pretraining.get("observation_noise_std", 0.005)
                ),
                seed=seed,
                logger=logger,
            )

    if args.pretrain_only:
        if not bool(pretraining_metrics.get("enabled", False)):
            train_environment.close()
            evaluation_environment.close()
            raise ValueError(
                "--pretrain-only requires expert_pretraining.enabled=true"
            )
        try:
            evaluation_metrics = evaluate_recovery(
                model, evaluation_factory, eval_episodes
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
        finally:
            train_environment.close()
            evaluation_environment.close()
        evaluation_metrics.update(
            {
                "checkpoint": str(output_path),
                "selected_model_type": "expert_actor_only",
                "selected_model_timesteps": int(model.num_timesteps),
                "final_training_checkpoint": None,
                "validation_best_checkpoint": None,
                "target_timesteps": 0,
                "starting_timesteps": 0,
                "additional_timesteps": 0,
                "trained_timesteps": int(model.num_timesteps),
                "resume_checkpoint": None,
                "config": str(resolve_path(args.config)),
                "backend": args.backend
                or environment_config.get("environment", {}).get(
                    "backend", "metaworld"
                ),
                "state_mode": args.state_mode
                or environment_config.get("environment", {}).get(
                    "state_mode", "semantic"
                ),
                "reward": wrapper_kwargs,
                "validation_reward": evaluation_wrapper_kwargs,
                "expert_pretraining": pretraining_metrics,
                "selection_metric": "none_pretrain_only",
                "minimum_selection_timesteps": None,
                "task_success_selection_history": None,
                "seed": seed,
                "device": device_name,
                "torch_num_threads": (
                    torch_threads if device_name == "cpu" else None
                ),
            }
        )
        if int(model.num_timesteps) != 0:
            raise RuntimeError(
                "pretrain-only checkpoint unexpectedly contains PPO steps"
            )
        atomic_json_dump(evaluation_metrics, metrics_path)
        logger.info(
            "Expert actor-only control complete: success=%.3f recovery=%.3f; %s",
            evaluation_metrics["success_rate"],
            evaluation_metrics["recovery_rate"],
            output_path,
        )
        return evaluation_metrics

    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_frequency // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="recovery",
    )
    selection_metric = str(
        training.get("selection_metric", "mean_reward")
    ).lower()
    eval_callback: Any | None = None
    task_eval_callback: TaskSuccessEvalCallback | None = None
    if selection_metric == "task_success":
        task_eval_callback = TaskSuccessEvalCallback(
            environment_factory=evaluation_factory,
            eval_freq=max(eval_frequency // n_envs, 1),
            episodes=eval_episodes,
            save_path=checkpoint_dir / "best" / "best_model.zip",
            history_path=checkpoint_dir / "evaluations" / "task_success.json",
            minimum_selection_timesteps=int(
                training.get("minimum_selection_timesteps", 0)
            ),
        )
        if resume_path is not None:
            task_eval_callback.restore_history(int(model.num_timesteps))
    elif selection_metric == "mean_reward":
        eval_callback = EvalCallback(
            evaluation_environment,
            best_model_save_path=str(checkpoint_dir / "best"),
            log_path=str(checkpoint_dir / "evaluations"),
            eval_freq=max(eval_frequency // n_envs, 1),
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
    else:
        raise ValueError(
            "training.selection_metric must be 'task_success' or 'mean_reward'"
        )
    evaluation_archive = checkpoint_dir / "evaluations" / "evaluations.npz"
    if (
        eval_callback is not None
        and resume_path is not None
        and evaluation_archive.is_file()
    ):
        # SB3 initializes EvalCallback history and best_mean_reward from
        # scratch.  Restore only observations no newer than the checkpoint
        # being resumed so a continuation cannot overwrite a genuinely better
        # validation model or erase the earlier learning curve.
        with np.load(evaluation_archive) as historical:
            timesteps = np.asarray(historical["timesteps"], dtype=np.int64)
            keep = timesteps <= int(model.num_timesteps)
            if keep.any():
                results = np.asarray(historical["results"], dtype=np.float64)[keep]
                lengths = np.asarray(historical["ep_lengths"], dtype=np.int64)[keep]
                eval_callback.evaluations_timesteps = timesteps[keep].tolist()
                eval_callback.evaluations_results = results.tolist()
                eval_callback.evaluations_length = lengths.tolist()
                if (
                    "successes" in historical.files
                    and hasattr(eval_callback, "evaluations_successes")
                ):
                    successes = np.asarray(historical["successes"], dtype=object)[keep]
                    eval_callback.evaluations_successes = successes.tolist()
                eval_callback.best_mean_reward = float(
                    np.max(results.mean(axis=1))
                )
                logger.info(
                    "Restored %d PPO validation points; historical best=%.3f.",
                    int(keep.sum()),
                    eval_callback.best_mean_reward,
                )
    if task_eval_callback is not None:
        initial_validation = evaluate_recovery(
            model, evaluation_factory, eval_episodes
        )
        task_eval_callback.record_initial(model, initial_validation)
    signal_callback = EpisodeSignalCallback()
    callbacks = CallbackList(
        [
            checkpoint_callback,
            task_eval_callback if task_eval_callback is not None else eval_callback,
            signal_callback,
        ]
    )
    completed_steps = int(model.num_timesteps)
    remaining_steps = max(total_timesteps - completed_steps, 0)
    logger.info(
        "PPO recovery: target=%d completed=%d remaining=%d, n_envs=%d, "
        "observation=%s action=%s backend=%s device=%s",
        total_timesteps,
        completed_steps,
        remaining_steps,
        n_envs,
        train_environment.observation_space.shape,
        train_environment.action_space.shape,
        args.backend
        or environment_config.get("environment", {}).get("backend", "metaworld"),
        device_name,
    )
    trained_timesteps = 0
    try:
        if remaining_steps:
            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
                reset_num_timesteps=resume_path is None,
                progress_bar=progress_bar,
            )
        trained_timesteps = int(model.num_timesteps)
        final_training_path = output_path.with_name(
            f"{output_path.stem}_final{output_path.suffix}"
        )
        model.save(str(final_training_path))
        model.save(str(output_path))
    finally:
        train_environment.close()
        evaluation_environment.close()

    selected_model = model
    selected_model_type = "final_training_step"
    best_model_path = checkpoint_dir / "best" / "best_model.zip"
    if bool(training.get("select_best_model", True)) and best_model_path.is_file():
        temporary_selected = output_path.with_name(
            f".{output_path.stem}.selected.tmp{output_path.suffix}"
        )
        shutil.copy2(best_model_path, temporary_selected)
        temporary_selected.replace(output_path)
        selected_model = PPO.load(str(output_path), device=device_name)
        selected_model_type = "validation_best"

    evaluation_metrics = evaluate_recovery(
        selected_model, evaluation_factory, eval_episodes
    )
    evaluation_metrics.update(
        {
            "checkpoint": str(output_path),
            "selected_model_type": selected_model_type,
            "selected_model_timesteps": int(selected_model.num_timesteps),
            "final_training_checkpoint": str(final_training_path),
            "validation_best_checkpoint": (
                str(best_model_path) if best_model_path.is_file() else None
            ),
            "target_timesteps": total_timesteps,
            "starting_timesteps": completed_steps,
            "additional_timesteps": max(trained_timesteps - completed_steps, 0),
            "trained_timesteps": trained_timesteps,
            "resume_checkpoint": str(resume_path) if resume_path is not None else None,
            "config": str(resolve_path(args.config)),
            "backend": args.backend
            or environment_config.get("environment", {}).get("backend", "metaworld"),
            "state_mode": args.state_mode
            or environment_config.get("environment", {}).get("state_mode", "semantic"),
            "reward": wrapper_kwargs,
            "validation_reward": evaluation_wrapper_kwargs,
            "expert_pretraining": pretraining_metrics,
            "selection_metric": selection_metric,
            "minimum_selection_timesteps": int(
                training.get("minimum_selection_timesteps", 0)
            ),
            "task_success_selection_history": (
                task_eval_callback.history
                if task_eval_callback is not None
                else None
            ),
            "seed": seed,
            "device": device_name,
            "torch_num_threads": torch_threads if device_name == "cpu" else None,
        }
    )
    atomic_json_dump(evaluation_metrics, metrics_path)
    _plot_monitor_files(
        monitor_dir,
        curve_path,
        evaluation_archive=checkpoint_dir / "evaluations" / "evaluations.npz",
        selected_timestep=int(selected_model.num_timesteps),
        phase_boundary_timestep=completed_steps if completed_steps > 0 else None,
    )
    logger.info(
        "PPO complete: success=%.3f recovery=%.3f mean_reward=%.3f; %s",
        evaluation_metrics["success_rate"],
        evaluation_metrics["recovery_rate"],
        evaluation_metrics["mean_episode_reward"],
        output_path,
    )
    return evaluation_metrics


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
