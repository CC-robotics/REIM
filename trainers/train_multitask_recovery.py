#!/usr/bin/env python3
"""Train a task-conditioned recovery actor from successful expert continuations.

The dataset is expected to contain one NPZ shard per recovery continuation.  A
shared MLP is trained for the whole MT10 or MT50 suite, while inverse-frequency
sampling prevents long or easy tasks from dominating the update stream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import file_sha256
from models.imitation_recovery_policy import ImitationRecoveryPolicy
from utils.common import (
    atomic_json_dump,
    capture_rng_state,
    configure_logging,
    restore_rng_state,
    seed_everything,
    select_device,
)


LOGGER = logging.getLogger("reim.train_multitask_recovery")
SCHEMA_VERSION = "reim-multitask-recovery-training-v1"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_dataset(
    data_dir: Path,
    manifest: dict[str, Any],
    task_vocabulary: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[Path]]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Recovery manifest must contain a non-empty files list")
    expected_relative_paths: list[str] = []
    entry_by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("Every recovery manifest file entry must be a mapping")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe recovery manifest path: {relative}")
        relative_name = relative.as_posix()
        if relative_name in entry_by_path:
            raise ValueError(f"Duplicate recovery manifest file: {relative_name}")
        expected_relative_paths.append(relative_name)
        entry_by_path[relative_name] = entry
    actual_relative_paths = sorted(
        path.relative_to(data_dir).as_posix() for path in data_dir.rglob("*.npz")
    )
    if sorted(expected_relative_paths) != actual_relative_paths:
        missing = sorted(set(expected_relative_paths) - set(actual_relative_paths))
        stale = sorted(set(actual_relative_paths) - set(expected_relative_paths))
        raise ValueError(
            "Recovery shards do not match the manifest whitelist: "
            f"missing={missing}, stale={stale}"
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    files: list[Path] = []
    seen_episode_seeds: set[int] = set()
    expected_state_dim = 39 + len(task_vocabulary)
    for relative_name in sorted(expected_relative_paths):
        entry = entry_by_path[relative_name]
        path = data_dir / relative_name
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Recovery shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "states",
                "actions",
                "success",
                "task_id",
                "task_name",
                "episode_seed",
                "schema_version",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path}: missing required keys {sorted(missing)}")
            success = bool(np.asarray(archive["success"]).reshape(-1)[0])
            if not success:
                raise ValueError(f"{path}: manifest lists an unsuccessful continuation")
            state = np.asarray(archive["states"], dtype=np.float32)
            action = np.asarray(archive["actions"], dtype=np.float32)
            task_id = int(np.asarray(archive["task_id"]).reshape(-1)[0])
            task_name = str(np.asarray(archive["task_name"]).reshape(-1)[0])
            episode_seed = int(np.asarray(archive["episode_seed"]).reshape(-1)[0])
            schema_version = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
            if state.shape != (len(action), expected_state_dim) or action.shape[1:] != (4,):
                raise ValueError(
                    f"{path}: expected states [T,{expected_state_dim}] and actions [T,4], "
                    f"got {state.shape}/{action.shape}"
                )
            if len(state) == 0:
                raise ValueError(f"{path}: empty recovery continuation")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{path}: non-finite state/action values")
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError(f"{path}: actions leave the Meta-World action bounds")
            if not 0 <= task_id < len(task_vocabulary):
                raise ValueError(f"{path}: invalid task_id {task_id}")
            if task_vocabulary[task_id] != task_name:
                raise ValueError(f"{path}: task_id/task_name disagree with ordered vocabulary")
            one_hot = np.zeros(len(task_vocabulary), dtype=np.float32)
            one_hot[task_id] = 1.0
            if not np.array_equal(
                state[:, 39:], np.broadcast_to(one_hot, (len(state), len(one_hot)))
            ):
                raise ValueError(f"{path}: invalid task one-hot block")
            if int(entry.get("task_id", task_id)) != task_id or str(
                entry.get("task_name", task_name)
            ) != task_name:
                raise ValueError(f"{path}: shard metadata disagrees with manifest")
            if int(entry.get("length", len(state))) != len(state):
                raise ValueError(f"{path}: shard length disagrees with manifest")
            if schema_version != str(manifest.get("schema_version")):
                raise ValueError(f"{path}: schema version disagrees with manifest")
            if episode_seed in seen_episode_seeds:
                raise ValueError(f"{path}: duplicate episode_seed {episode_seed}")
            seen_episode_seeds.add(episode_seed)
            states.append(state)
            actions.append(action)
            task_ids.append(task_id)
            files.append(path)
    if not states:
        raise ValueError(f"No successful recovery continuations found in {data_dir}")
    state_dims = {array.shape[1] for array in states}
    action_dims = {array.shape[1] for array in actions}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(
            f"Inconsistent recovery dimensions: states={state_dims}, actions={action_dims}"
        )
    return states, actions, task_ids, files


def _stratified_group_split(
    task_ids: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for task_id in sorted(set(task_ids)):
        indices = np.asarray(
            [index for index, value in enumerate(task_ids) if value == task_id],
            dtype=np.int64,
        )
        indices = rng.permutation(indices)
        if len(indices) == 1:
            train.extend(indices.tolist())
            continue
        count = min(max(int(round(len(indices) * validation_fraction)), 1), len(indices) - 1)
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    if not validation:
        raise ValueError("At least two continuation shards are needed for validation")
    return sorted(train), sorted(validation)


def _flatten(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    task_ids: list[int],
    groups: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([states[index] for index in groups]).astype(np.float32),
        np.concatenate([actions[index] for index in groups]).astype(np.float32),
        np.concatenate(
            [
                np.full(len(states[index]), task_ids[index], dtype=np.int64)
                for index in groups
            ]
        ),
    )


def _save_history(history: list[dict[str, float]], csv_path: Path, curve_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "train_loss", "validation_loss")
        )
        writer.writeheader()
        writer.writerows(history)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    if history:
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
        axis.plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="Validation",
        )
    axis.set(xlabel="Epoch", ylabel="Smooth-L1", title="Multi-task recovery training")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=220)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = select_device(args.device)
    logger = configure_logging(
        "train_multitask_recovery", args.log_file or f"results/logs/{args.benchmark.lower()}_recovery.log"
    )
    data_dir = Path(args.data_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recovery dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = file_sha256(manifest_path)
    if str(manifest.get("benchmark", "")).upper() != args.benchmark.upper():
        raise ValueError("Dataset benchmark does not match --benchmark")
    if manifest.get("dataset_type") != "online_detector_triggered_expert_continuations":
        raise ValueError("Unsupported recovery dataset type")
    task_vocabulary = list(manifest.get("task_vocabulary", []))
    expected_tasks = 10 if args.benchmark.upper() == "MT10" else 50
    if len(task_vocabulary) != expected_tasks:
        raise ValueError(
            f"{args.benchmark} requires {expected_tasks} ordered tasks; found {len(task_vocabulary)}"
        )
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(task_vocabulary, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("task_vocabulary_sha256") != vocabulary_sha256:
        raise ValueError("Recovery manifest task vocabulary hash is invalid")
    if int(manifest.get("state_dim", -1)) != 39 + expected_tasks:
        raise ValueError("Recovery manifest state_dim is invalid")
    if int(manifest.get("action_dim", -1)) != 4:
        raise ValueError("Recovery manifest action_dim is invalid")

    states, actions, group_task_ids, files = _load_dataset(
        data_dir, manifest, task_vocabulary
    )
    train_groups, validation_groups = _stratified_group_split(
        group_task_ids, args.validation_fraction, args.seed
    )
    train_states, train_actions, train_task_ids = _flatten(
        states, actions, group_task_ids, train_groups
    )
    validation_states, validation_actions, _ = _flatten(
        states, actions, group_task_ids, validation_groups
    )
    state_dim, action_dim = train_states.shape[1], train_actions.shape[1]
    if state_dim != 39 + expected_tasks or action_dim != 4:
        raise ValueError(
            f"Unexpected {args.benchmark} data dimensions: state={state_dim}, action={action_dim}"
        )

    mean = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_states.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    # Preserve exact binary task indicators; normalization applies only to raw39.
    mean[39:] = 0.0
    std[39:] = 1.0
    model = ImitationRecoveryPolicy(
        state_dim,
        action_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation="tanh",
        observation_mean=mean,
        observation_std=std,
        provenance={
            "schema_version": SCHEMA_VERSION,
            "benchmark": args.benchmark.upper(),
            "task_vocabulary": task_vocabulary,
            "task_vocabulary_sha256": vocabulary_sha256,
            "dataset_manifest_sha256": manifest_sha256,
            "source_training": {"algorithm": "task_balanced_smooth_l1", "num_timesteps": 0},
        },
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.SmoothL1Loss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_states), torch.from_numpy(train_actions)
    )
    counts = np.bincount(train_task_ids, minlength=expected_tasks).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training recovery data has no rows for task IDs {missing}")
    weights = 1.0 / counts[train_task_ids]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_states), torch.from_numpy(validation_actions)
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    latest = output.with_name(output.stem + "_latest" + output.suffix)
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_loss = float("inf")
    stale = 0
    if args.resume:
        resume_path = latest if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        # Generator/RNG states must remain CPU ByteTensors. Optimizer state is
        # migrated to parameter devices by ``load_state_dict``.
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        required_resume_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "seed",
            "dataset_manifest_sha256",
            "train_files",
            "validation_files",
            "sampler_generator_state",
            "rng_state",
        }
        missing_resume = required_resume_keys.difference(checkpoint)
        if missing_resume:
            raise ValueError(
                "--resume requires a recovery training-state checkpoint, not a "
                f"deployment policy; missing {sorted(missing_resume)}"
            )
        requested_hidden_dims = list(map(int, args.hidden_dims))
        expected_train_files = [files[index].relative_to(data_dir).as_posix() for index in train_groups]
        expected_validation_files = [
            files[index].relative_to(data_dir).as_posix() for index in validation_groups
        ]
        resume_mismatches = []
        for label, stored, requested in (
            ("benchmark", checkpoint.get("benchmark"), args.benchmark.upper()),
            ("seed", int(checkpoint.get("seed", -1)), args.seed),
            ("dataset manifest", checkpoint.get("dataset_manifest_sha256"), manifest_sha256),
            ("task vocabulary", checkpoint.get("task_vocabulary"), task_vocabulary),
            ("task vocabulary hash", checkpoint.get("task_vocabulary_sha256"), vocabulary_sha256),
            ("hidden dimensions", checkpoint.get("hidden_dims"), requested_hidden_dims),
            ("train split", checkpoint.get("train_files"), expected_train_files),
            ("validation split", checkpoint.get("validation_files"), expected_validation_files),
        ):
            if stored != requested:
                resume_mismatches.append(f"{label}: stored={stored!r}, requested={requested!r}")
        if resume_mismatches:
            raise ValueError("Incompatible recovery resume: " + "; ".join(resume_mismatches))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_validation_loss", best_loss))
        stale = int(checkpoint.get("epochs_without_improvement", 0))
        generator.set_state(checkpoint["sampler_generator_state"])
        restore_rng_state(checkpoint["rng_state"])

    raw_dim = 39
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc=f"{args.benchmark} recovery"):
        model.train()
        train_sum = 0.0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            if args.state_noise_std > 0:
                noisy = batch_states.clone()
                noisy[:, :raw_dim] += torch.randn_like(noisy[:, :raw_dim]) * args.state_noise_std
            else:
                noisy = batch_states
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.mean_action(noisy), batch_actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch_states)
        model.eval()
        validation_sum = 0.0
        with torch.inference_mode():
            for batch_states, batch_actions in validation_loader:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                validation_sum += float(
                    loss_function(model.mean_action(batch_states), batch_actions)
                ) * len(batch_states)
        train_loss = train_sum / len(train_dataset)
        validation_loss = validation_sum / len(validation_states)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        improved = validation_loss < best_loss - args.min_delta
        if improved:
            best_loss = validation_loss
            stale = 0
            model.provenance["best_epoch"] = epoch
            model.provenance["best_validation_loss"] = best_loss
            model.save(output)
        else:
            stale += 1
        _atomic_torch_save(
            {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_validation_loss": best_loss,
                "epochs_without_improvement": stale,
                "benchmark": args.benchmark.upper(),
                "seed": args.seed,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dims": list(map(int, args.hidden_dims)),
                "task_vocabulary": task_vocabulary,
                "task_vocabulary_sha256": vocabulary_sha256,
                "dataset_manifest_sha256": manifest_sha256,
                "train_files": [
                    files[index].relative_to(data_dir).as_posix()
                    for index in train_groups
                ],
                "validation_files": [
                    files[index].relative_to(data_dir).as_posix()
                    for index in validation_groups
                ],
                "sampler_generator_state": generator.get_state(),
                "rng_state": capture_rng_state(),
            },
            latest,
        )
        logger.info(
            "epoch=%d train=%.6f validation=%.6f best=%.6f",
            epoch,
            train_loss,
            validation_loss,
            best_loss,
        )
        if stale >= args.patience:
            break

    history_path = Path(args.history).expanduser().resolve()
    curve_path = Path(args.curve).expanduser().resolve()
    _save_history(history, history_path, curve_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": args.benchmark.upper(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "task_count": expected_tasks,
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_rows": len(train_states),
        "validation_rows": len(validation_states),
        "best_validation_loss": best_loss,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "checkpoint": str(output),
        "checkpoint_sha256": file_sha256(output),
        "dataset_manifest_sha256": manifest_sha256,
        "source_file_count": len(files),
        "task_vocabulary": task_vocabulary,
    }
    atomic_json_dump(summary, args.summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(512, 512, 256))
    parser.add_argument("--state-noise-std", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from PATH, or the latest training-state checkpoint if omitted.",
    )
    return parser


def main() -> None:
    summary = train(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
