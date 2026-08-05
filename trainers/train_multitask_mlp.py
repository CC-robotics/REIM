#!/usr/bin/env python3
"""Train the shared task-conditioned MLP-BC benchmark baseline.

This trainer accepts only the balanced MT10/MT50 demonstration format emitted
by ``scripts/collect_multitask_demonstrations.py``.  Every trajectory is
verified against the manifest whitelist and SHA-256 digest before training.
The split is stratified by task at trajectory granularity, preventing adjacent
states from the same rollout from leaking into validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import file_sha256  # noqa: E402
from models.bc_policy import MLPBCPolicy  # noqa: E402
from utils.common import (  # noqa: E402
    atomic_json_dump,
    capture_rng_state,
    configure_logging,
    load_yaml,
    resolve_path,
    restore_rng_state,
    seed_everything,
    select_device,
)


RAW_OBSERVATION_DIM = 39
ACTION_DIM = 4
SUPPORTED_BENCHMARKS = {"MT10": 10, "MT50": 50}
DEMONSTRATION_SCHEMA = "reim-multitask-demonstrations-v1"
DEMONSTRATION_DATASET_TYPE = "balanced_multitask_scripted_expert_demonstrations"
TRAINING_SCHEMA = "reim-multitask-mlp-training-v1"


@dataclass(frozen=True)
class MultitaskDemonstrations:
    """Validated trajectories in official ordered-task representation."""

    states: tuple[np.ndarray, ...]
    actions: tuple[np.ndarray, ...]
    task_ids: tuple[int, ...]
    relative_paths: tuple[str, ...]
    manifest_sha256: str
    benchmark: str
    metaworld_version: str
    task_vocabulary: tuple[str, ...]
    task_vocabulary_sha256: str
    schema_version: str
    dataset_type: str


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        value = torch.load(path, map_location=map_location)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping checkpoint at {path}.")
    return value


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _task_vocabulary_sha256(task_vocabulary: Sequence[str]) -> str:
    serialized = json.dumps(
        list(task_vocabulary), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _split_sha256(train_paths: Sequence[str], validation_paths: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "train": sorted(train_paths),
            "validation": sorted(validation_paths),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    array = np.asarray(archive[key])
    if array.size != 1:
        raise ValueError(f"NPZ field {key!r} must be scalar, got {array.shape}.")
    return array.reshape(-1)[0]


def _manifest_entries(
    data_dir: Path, manifest: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any]]], set[str]]:
    raw_entries = manifest.get("trajectories")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Demonstration manifest needs a non-empty trajectories list.")
    entries: list[tuple[str, dict[str, Any]]] = []
    expected: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("Every demonstration trajectory entry must be a mapping.")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe demonstration manifest path: {relative}")
        name = relative.as_posix()
        if name in expected:
            raise ValueError(f"Duplicate demonstration manifest path: {name}")
        path = data_dir / relative
        try:
            path.resolve().relative_to(data_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Demonstration path escapes the dataset: {relative}") from exc
        expected.add(name)
        entries.append((name, entry))

    actual = {
        path.relative_to(data_dir).as_posix() for path in data_dir.rglob("*.npz")
    }
    if actual != expected:
        missing = sorted(expected - actual)
        stale = sorted(actual - expected)
        raise ValueError(
            "Demonstration shards do not match the manifest whitelist: "
            f"missing={missing}, stale={stale}"
        )
    return sorted(entries, key=lambda item: item[0]), expected


def load_multitask_demonstrations(
    data_dir: str | Path,
    *,
    benchmark: str,
) -> MultitaskDemonstrations:
    """Load and cryptographically validate one official multi-task dataset."""

    root = Path(data_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Demonstration manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Demonstration manifest must contain a JSON object.")
    benchmark_name = str(benchmark).upper()
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError("benchmark must be MT10 or MT50.")
    task_count = SUPPORTED_BENCHMARKS[benchmark_name]
    required_manifest = {
        "benchmark": benchmark_name,
        "schema_version": DEMONSTRATION_SCHEMA,
        "dataset_type": DEMONSTRATION_DATASET_TYPE,
        "complete": True,
        "task_count": task_count,
    }
    disagreements = {
        key: (manifest.get(key), expected)
        for key, expected in required_manifest.items()
        if manifest.get(key) != expected
    }
    if disagreements:
        raise ValueError(f"Invalid demonstration manifest metadata: {disagreements}")
    task_vocabulary = manifest.get("task_vocabulary")
    if (
        not isinstance(task_vocabulary, list)
        or len(task_vocabulary) != task_count
        or any(not isinstance(name, str) or not name for name in task_vocabulary)
        or len(set(task_vocabulary)) != task_count
    ):
        raise ValueError(f"{benchmark_name} needs {task_count} unique ordered task names.")
    vocabulary_hash = _task_vocabulary_sha256(task_vocabulary)
    if manifest.get("task_vocabulary_sha256") != vocabulary_hash:
        raise ValueError("Demonstration task vocabulary SHA-256 is invalid.")

    observation_schema = manifest.get("observation_schema", {})
    expected_state_dim = RAW_OBSERVATION_DIM + task_count
    if observation_schema.get("raw_observations", {}).get("shape") != [
        "T",
        RAW_OBSERVATION_DIM,
    ]:
        raise ValueError("Manifest raw observation schema is not [T,39].")
    if observation_schema.get("states", {}).get("shape") != [
        "T",
        expected_state_dim,
    ]:
        raise ValueError(f"Manifest state schema is not [T,{expected_state_dim}].")
    if observation_schema.get("actions", {}).get("shape") != ["T", ACTION_DIM]:
        raise ValueError("Manifest action schema is not [T,4].")

    entries, expected_paths = _manifest_entries(root, manifest)
    statistics = manifest.get("statistics", {})
    if int(statistics.get("successful_trajectories", -1)) != len(entries):
        raise ValueError("Manifest successful trajectory count is inconsistent.")
    expected_per_task = int(manifest.get("protocol", {}).get("episodes_per_task", -1))
    if expected_per_task <= 0 or len(entries) != expected_per_task * task_count:
        raise ValueError("Manifest is not an equal per-task demonstration collection.")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    relative_paths: list[str] = []
    seen_trajectories: set[tuple[int, int]] = set()
    task_yields = np.zeros(task_count, dtype=np.int64)
    total_transitions = 0
    for relative_name, entry in entries:
        path = root / relative_name
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise ValueError(f"Demonstration shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "states",
                "raw_observations",
                "actions",
                "rewards",
                "success",
                "task_id",
                "task_name",
                "task_variant",
                "seed",
                "trajectory_index",
                "attempt_index",
                "schema_version",
                "benchmark",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path}: missing required keys {sorted(missing)}")
            state = np.asarray(archive["states"], dtype=np.float32)
            raw = np.asarray(archive["raw_observations"], dtype=np.float32)
            action = np.asarray(archive["actions"], dtype=np.float32)
            rewards = np.asarray(archive["rewards"], dtype=np.float32).reshape(-1)
            task_id = int(_scalar(archive, "task_id"))
            task_name = str(_scalar(archive, "task_name"))
            task_variant = int(_scalar(archive, "task_variant"))
            episode_seed = int(_scalar(archive, "seed"))
            trajectory_index = int(_scalar(archive, "trajectory_index"))
            attempt_index = int(_scalar(archive, "attempt_index"))
            schema_version = str(_scalar(archive, "schema_version"))
            shard_benchmark = str(_scalar(archive, "benchmark")).upper()
            success = bool(_scalar(archive, "success"))
        length = len(state)
        if state.shape != (length, expected_state_dim):
            raise ValueError(
                f"{path}: expected states [T,{expected_state_dim}], got {state.shape}."
            )
        if raw.shape != (length, RAW_OBSERVATION_DIM):
            raise ValueError(f"{path}: expected raw observations [T,39], got {raw.shape}.")
        if action.shape != (length, ACTION_DIM) or rewards.shape != (length,):
            raise ValueError(f"{path}: invalid action/reward trajectory shapes.")
        if length <= 0:
            raise ValueError(f"{path}: empty trajectory.")
        if not success:
            raise ValueError(f"{path}: unsuccessful trajectory in expert dataset.")
        if schema_version != DEMONSTRATION_SCHEMA or shard_benchmark != benchmark_name:
            raise ValueError(f"{path}: shard schema/benchmark disagrees with manifest.")
        if not 0 <= task_id < task_count or task_vocabulary[task_id] != task_name:
            raise ValueError(f"{path}: task ID/name disagrees with ordered vocabulary.")
        if not 0 <= task_variant < 50:
            raise ValueError(f"{path}: task_variant must be one of the 50 official goals.")
        if not (
            np.isfinite(state).all()
            and np.isfinite(raw).all()
            and np.isfinite(action).all()
            and np.isfinite(rewards).all()
        ):
            raise ValueError(f"{path}: non-finite trajectory value.")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError(f"{path}: action target leaves [-1,1].")
        if not np.array_equal(state[:, :RAW_OBSERVATION_DIM], raw):
            raise ValueError(f"{path}: state does not preserve raw39 exactly.")
        one_hot = np.zeros(task_count, dtype=np.float32)
        one_hot[task_id] = 1.0
        if not np.array_equal(
            state[:, RAW_OBSERVATION_DIM:],
            np.broadcast_to(one_hot, (length, task_count)),
        ):
            raise ValueError(f"{path}: invalid official task one-hot block.")
        entry_values = {
            "task_id": task_id,
            "task_name": task_name,
            "task_variant": task_variant,
            "trajectory_index": trajectory_index,
            "attempt_index": attempt_index,
            "seed": episode_seed,
            "length": length,
            "return": float(rewards.sum(dtype=np.float64)),
            "success": True,
        }
        disagreements = {
            key: (entry.get(key), value)
            for key, value in entry_values.items()
            if entry.get(key) != value
        }
        if disagreements:
            raise ValueError(f"{path}: shard disagrees with manifest: {disagreements}")
        identity = (task_id, trajectory_index)
        if identity in seen_trajectories:
            raise ValueError(f"Duplicate task/trajectory identity: {identity}")
        seen_trajectories.add(identity)
        task_yields[task_id] += 1
        total_transitions += length
        states.append(state)
        actions.append(action)
        task_ids.append(task_id)
        relative_paths.append(relative_name)

    if not np.all(task_yields == expected_per_task):
        raise ValueError(f"Per-task trajectory yields are imbalanced: {task_yields.tolist()}")
    if int(statistics.get("total_transitions", -1)) != total_transitions:
        raise ValueError("Manifest total transition count is inconsistent.")
    if set(relative_paths) != expected_paths:
        raise RuntimeError("Internal manifest whitelist validation failed.")
    return MultitaskDemonstrations(
        states=tuple(states),
        actions=tuple(actions),
        task_ids=tuple(task_ids),
        relative_paths=tuple(relative_paths),
        manifest_sha256=file_sha256(manifest_path),
        benchmark=benchmark_name,
        metaworld_version=str(manifest.get("metaworld_version", "unknown")),
        task_vocabulary=tuple(task_vocabulary),
        task_vocabulary_sha256=vocabulary_hash,
        schema_version=DEMONSTRATION_SCHEMA,
        dataset_type=DEMONSTRATION_DATASET_TYPE,
    )


def stratified_trajectory_split(
    task_ids: Sequence[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Place trajectories from every task in both deterministic split sides."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    values = np.asarray(task_ids, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("task_ids must be a non-empty one-dimensional sequence.")
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    validation: list[int] = []
    for task_id in sorted(np.unique(values).tolist()):
        indices = np.flatnonzero(values == task_id)
        if len(indices) < 2:
            raise ValueError(
                f"Task {task_id} needs at least two trajectories for a stratified split."
            )
        shuffled = rng.permutation(indices)
        count = min(
            max(int(round(len(shuffled) * validation_fraction)), 1),
            len(shuffled) - 1,
        )
        validation.extend(int(index) for index in shuffled[:count])
        train.extend(int(index) for index in shuffled[count:])
    return sorted(train), sorted(validation)


def _flatten(
    dataset: MultitaskDemonstrations, trajectory_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([dataset.states[index] for index in trajectory_indices]).astype(
            np.float32, copy=False
        ),
        np.concatenate([dataset.actions[index] for index in trajectory_indices]).astype(
            np.float32, copy=False
        ),
    )


def _save_history(
    history: list[dict[str, float]], csv_path: Path, curve_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_mse",
                "validation_mse",
                "train_mae",
                "validation_mae",
                "learning_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(history)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    if history:
        epochs = [row["epoch"] for row in history]
        axis.plot(
            epochs,
            [row["train_mse"] for row in history],
            color="#356A9A",
            linewidth=1.8,
            label="Train",
        )
        axis.plot(
            epochs,
            [row["validation_mse"] for row in history],
            color="#D97935",
            linewidth=1.8,
            label="Validation",
        )
    axis.set(xlabel="Epoch", ylabel="Action MSE", title="Shared multi-task MLP-BC")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=240)
    figure.savefig(curve_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model", {}))
    if str(model.get("policy_type", "MLP_BC")).upper().replace("-", "_") != "MLP_BC":
        raise ValueError("train_multitask_mlp.py requires model.policy_type=MLP_BC.")
    hidden_dims = tuple(int(width) for width in model.get("hidden_dims", (256, 256)))
    if not hidden_dims or any(width <= 0 for width in hidden_dims):
        raise ValueError("model.hidden_dims must contain positive widths.")
    return {"hidden_dims": hidden_dims}


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    training = dict(config.get("training", {}))
    benchmark = str(args.benchmark or config.get("benchmark", "MT10")).upper()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError("benchmark must be MT10 or MT50.")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device_name = select_device(args.device or config.get("device", "auto"))
    seed_everything(seed)
    logger = configure_logging(
        f"train_{benchmark.lower()}_mlp",
        config.get("log_path", f"results/logs/{benchmark.lower()}_mlp_training.log"),
    )

    data_dir = resolve_path(args.data_dir or config.get("data_dir", ""))
    output_path = resolve_path(
        args.checkpoint
        or config.get("checkpoint", f"checkpoints/{benchmark.lower()}/seed_{seed}/mlp_bc.pt")
    )
    latest_path = resolve_path(
        config.get(
            "latest_checkpoint",
            f"checkpoints/{benchmark.lower()}/seed_{seed}/mlp_bc_latest.pt",
        )
    )
    curve_path = resolve_path(
        config.get("curve_path", f"results/figures/{benchmark.lower()}_mlp_training.png")
    )
    history_path = resolve_path(
        config.get("history_path", f"results/tables/{benchmark.lower()}_mlp_training.csv")
    )
    summary_path = resolve_path(
        config.get("summary_path", f"results/tables/{benchmark.lower()}_mlp_training.json")
    )
    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 100))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 512)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate", 3e-4)
    )
    validation_fraction = float(training.get("validation_fraction", 0.1))
    num_workers = int(training.get("num_workers", 0))
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or num_workers < 0:
        raise ValueError("epochs/batch_size/learning_rate must be positive.")

    resume_value = args.resume
    if resume_value is None and config.get("resume"):
        resume_value = str(config["resume"])
    resume_path = (
        latest_path
        if resume_value == "auto"
        else resolve_path(resume_value)
        if resume_value
        else None
    )
    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = _torch_load(resume_path, "cpu")

    dataset = load_multitask_demonstrations(data_dir, benchmark=benchmark)
    train_groups, validation_groups = stratified_trajectory_split(
        dataset.task_ids, validation_fraction, seed
    )
    train_paths = [dataset.relative_paths[index] for index in train_groups]
    validation_paths = [dataset.relative_paths[index] for index in validation_groups]
    split_hash = _split_sha256(train_paths, validation_paths)
    train_states, train_actions = _flatten(dataset, train_groups)
    validation_states, validation_actions = _flatten(dataset, validation_groups)
    state_dim, action_dim = train_states.shape[1], train_actions.shape[1]
    expected_state_dim = RAW_OBSERVATION_DIM + SUPPORTED_BENCHMARKS[benchmark]
    if state_dim != expected_state_dim or action_dim != ACTION_DIM:
        raise ValueError(
            f"Unexpected {benchmark} dimensions: state={state_dim}, action={action_dim}."
        )

    state_mean = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        train_states.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    # Task identity is categorical, not a continuous physical feature.
    state_mean[RAW_OBSERVATION_DIM:] = 0.0
    state_std[RAW_OBSERVATION_DIM:] = 1.0
    model_config = _model_config(config)

    if resume_checkpoint is not None:
        checkpoint_type = str(resume_checkpoint.get("policy_type", "")).upper().replace(
            "-", "_"
        )
        if checkpoint_type != "MLP_BC":
            raise ValueError("Resume checkpoint is not an MLP-BC policy.")
        checks = {
            "training schema": (
                resume_checkpoint.get("training_schema"),
                TRAINING_SCHEMA,
            ),
            "seed": (int(resume_checkpoint.get("seed", -1)), seed),
            "benchmark": (resume_checkpoint.get("benchmark"), benchmark),
            "manifest": (
                resume_checkpoint.get("data_manifest_sha256"),
                dataset.manifest_sha256,
            ),
            "vocabulary": (
                resume_checkpoint.get("task_vocabulary_sha256"),
                dataset.task_vocabulary_sha256,
            ),
            "split": (resume_checkpoint.get("split_sha256"), split_hash),
            "state_dim": (int(resume_checkpoint.get("state_dim", -1)), state_dim),
            "action_dim": (int(resume_checkpoint.get("action_dim", -1)), action_dim),
        }
        disagreements = {
            name: pair for name, pair in checks.items() if pair[0] != pair[1]
        }
        if disagreements:
            raise ValueError(f"Resume checkpoint provenance changed: {disagreements}")
        if list(resume_checkpoint.get("task_vocabulary", [])) != list(
            dataset.task_vocabulary
        ):
            raise ValueError("Resume checkpoint ordered task vocabulary changed.")
        if resume_checkpoint.get("train_trajectories") != train_paths or resume_checkpoint.get(
            "validation_trajectories"
        ) != validation_paths:
            raise ValueError("Resume checkpoint trajectory split changed.")
        stored_config = dict(resume_checkpoint.get("model_config", {}))
        if tuple(stored_config.get("hidden_dims", ())) != model_config["hidden_dims"]:
            raise ValueError("Resume checkpoint MLP architecture changed.")
        state_dict = resume_checkpoint.get("model_state_dict", {})
        if "state_mean" in state_dict and not np.allclose(
            np.asarray(state_dict["state_mean"]), state_mean, atol=0.0, rtol=0.0
        ):
            raise ValueError("Resume checkpoint normalization mean changed.")
        if "state_std" in state_dict and not np.allclose(
            np.asarray(state_dict["state_std"]), state_std, atol=0.0, rtol=0.0
        ):
            raise ValueError("Resume checkpoint normalization scale changed.")

    train_dataset = TensorDataset(
        torch.from_numpy(train_states), torch.from_numpy(train_actions)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_states), torch.from_numpy(validation_actions)
    )
    generator = torch.Generator().manual_seed(seed)
    if resume_checkpoint is not None and resume_checkpoint.get(
        "dataloader_generator_state"
    ) is not None:
        generator.set_state(resume_checkpoint["dataloader_generator_state"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device_name.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_name.startswith("cuda"),
    )

    model = MLPBCPolicy(state_dim, action_dim, **model_config).to(device_name)
    model.set_normalization(state_mean, state_std)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        best_validation_loss = float(
            resume_checkpoint.get("best_validation_loss", best_validation_loss)
        )
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(resume_checkpoint.get("history", []))
        restore_rng_state(resume_checkpoint.get("rng_state"))
        logger.info("Resumed %s at epoch %d.", resume_path, start_epoch)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "%s shared MLP-BC: %d/%d train/validation transitions, %d/%d "
        "trajectories, state=%d action=%d parameters=%d device=%s",
        benchmark,
        len(train_dataset),
        len(validation_dataset),
        len(train_groups),
        len(validation_groups),
        state_dim,
        action_dim,
        parameter_count,
        device_name,
    )

    mse = nn.MSELoss()
    mae = nn.L1Loss()
    grad_clip = float(training.get("grad_clip_norm", 5.0))
    patience = int(training.get("patience", 20))
    checkpoint_every = int(training.get("checkpoint_every", 5))
    epoch_bar = tqdm(
        range(start_epoch, epochs),
        initial=start_epoch,
        total=epochs,
        desc=f"{benchmark} MLP-BC",
        unit="epoch",
    )
    for epoch in epoch_bar:
        phase_metrics: dict[str, tuple[float, float]] = {}
        for phase, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(phase == "train")
            mse_sum = 0.0
            mae_sum = 0.0
            sample_count = 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for states_batch, actions_batch in loader:
                    states_batch = states_batch.to(device_name, non_blocking=True)
                    actions_batch = actions_batch.to(device_name, non_blocking=True)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    predictions = model(states_batch)
                    loss = mse(predictions, actions_batch)
                    batch_mae = mae(predictions, actions_batch)
                    if phase == "train":
                        loss.backward()
                        if grad_clip > 0.0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    count = len(states_batch)
                    mse_sum += float(loss.detach()) * count
                    mae_sum += float(batch_mae.detach()) * count
                    sample_count += count
            phase_metrics[phase] = (
                mse_sum / max(sample_count, 1),
                mae_sum / max(sample_count, 1),
            )

        row = {
            "epoch": float(epoch + 1),
            "train_mse": phase_metrics["train"][0],
            "validation_mse": phase_metrics["validation"][0],
            "train_mae": phase_metrics["train"][1],
            "validation_mae": phase_metrics["validation"][1],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        improved = row["validation_mse"] < best_validation_loss - 1e-12
        if improved:
            best_validation_loss = row["validation_mse"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload: dict[str, Any] = {
            "format_version": 2,
            "training_schema": TRAINING_SCHEMA,
            "policy_type": "MLP_BC",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_config": model.model_config,
            "state_mean": state_mean,
            "state_std": state_std,
            "normalization_scope": "raw39_only_task_onehot_unchanged",
            "loss": "mean_squared_error",
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "seed": seed,
            "benchmark": benchmark,
            "metaworld_version": dataset.metaworld_version,
            "task_vocabulary": list(dataset.task_vocabulary),
            "task_vocabulary_sha256": dataset.task_vocabulary_sha256,
            "data_manifest_sha256": dataset.manifest_sha256,
            "data_schema_version": dataset.schema_version,
            "dataset_type": dataset.dataset_type,
            "split_sha256": split_hash,
            "train_trajectories": train_paths,
            "validation_trajectories": validation_paths,
            "dataloader_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(),
            "config": config,
        }
        _atomic_torch_save(payload, latest_path)
        if improved:
            _atomic_torch_save(payload, output_path)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            snapshot = latest_path.parent / "mlp_bc" / f"epoch_{epoch + 1:04d}.pt"
            _atomic_torch_save(payload, snapshot)
        _save_history(history, history_path, curve_path)
        epoch_bar.set_postfix(
            train=f"{row['train_mse']:.5f}", val=f"{row['validation_mse']:.5f}"
        )
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping after %d non-improving epochs.", patience)
            break

    if not output_path.exists() and latest_path.exists():
        _atomic_torch_save(_torch_load(latest_path), output_path)
    task_train_counts = {
        dataset.task_vocabulary[task_id]: int(
            sum(dataset.task_ids[index] == task_id for index in train_groups)
        )
        for task_id in range(len(dataset.task_vocabulary))
    }
    task_validation_counts = {
        dataset.task_vocabulary[task_id]: int(
            sum(dataset.task_ids[index] == task_id for index in validation_groups)
        )
        for task_id in range(len(dataset.task_vocabulary))
    }
    summary = {
        "training_schema": TRAINING_SCHEMA,
        "policy_type": "MLP_BC",
        "benchmark": benchmark,
        "samples": int(len(train_dataset) + len(validation_dataset)),
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(validation_dataset)),
        "trajectories": len(dataset.states),
        "train_trajectories": len(train_groups),
        "validation_trajectories": len(validation_groups),
        "per_task_train_trajectories": task_train_counts,
        "per_task_validation_trajectories": task_validation_counts,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": list(model.hidden_dims),
        "parameters": parameter_count,
        "epochs_completed": int(history[-1]["epoch"] if history else start_epoch),
        "best_validation_mse": float(best_validation_loss),
        "best_validation_mae": float(
            min((row["validation_mae"] for row in history), default=float("nan"))
        ),
        "checkpoint": str(output_path),
        "latest_checkpoint": str(latest_path),
        "curve": str(curve_path),
        "history": str(history_path),
        "device": device_name,
        "seed": seed,
        "loss": "mean_squared_error",
        "normalization_scope": "raw39_only_task_onehot_unchanged",
        "task_vocabulary": list(dataset.task_vocabulary),
        "task_vocabulary_sha256": dataset.task_vocabulary_sha256,
        "data_manifest_sha256": dataset.manifest_sha256,
        "split_sha256": split_hash,
    }
    atomic_json_dump(summary, summary_path)
    logger.info(
        "%s MLP-BC complete: best validation MSE %.6f; checkpoint=%s",
        benchmark,
        best_validation_loss,
        output_path,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an auditable shared MLP-BC baseline on MT10 or MT50."
    )
    parser.add_argument("--config", default="configs/multitask/mt10_mlp.yaml")
    parser.add_argument("--benchmark", choices=tuple(SUPPORTED_BENCHMARKS), default=None)
    parser.add_argument("--data-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from PATH, or latest_checkpoint when PATH is omitted.",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
