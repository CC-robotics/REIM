#!/usr/bin/env python3
"""Train the state-based ACT imitation policy from expert trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.bc_policy import ACTPolicy  # noqa: E402
from data.io import file_sha256  # noqa: E402
from trainers.data import (  # noqa: E402
    DemonstrationData,
    group_train_validation_split,
    load_demonstrations,
)
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


class ActionChunkDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Map each trajectory state to its padded future expert action chunk."""

    def __init__(
        self,
        data: DemonstrationData,
        indices: np.ndarray,
        chunk_size: int,
    ) -> None:
        self.states = data.states
        self.actions = data.actions
        self.groups = data.groups
        self.indices = np.asarray(indices, dtype=np.int64)
        self.chunk_size = int(chunk_size)
        self.allowed = np.zeros(len(data.states), dtype=bool)
        self.allowed[self.indices] = True

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        group = self.groups[index]
        chunk = np.zeros(
            (self.chunk_size, self.actions.shape[-1]), dtype=np.float32
        )
        padding_mask = np.ones(self.chunk_size, dtype=bool)
        valid_count = 0
        for offset in range(self.chunk_size):
            future_index = index + offset
            if (
                future_index >= len(self.actions)
                or self.groups[future_index] != group
                or not self.allowed[future_index]
            ):
                break
            chunk[offset] = self.actions[future_index]
            padding_mask[offset] = False
            valid_count += 1
        if valid_count == 0:
            raise RuntimeError("Every ACT sample must contain its current expert action.")
        return (
            torch.from_numpy(self.states[index]),
            torch.from_numpy(chunk),
            torch.from_numpy(padding_mask),
        )


def _torch_load(path: Path, map_location: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_history(
    history: list[dict[str, float]],
    csv_path: Path,
    curve_path: Path,
) -> None:
    fields = (
        "epoch",
        "train_loss",
        "validation_loss",
        "train_l1",
        "validation_l1",
        "train_kl",
        "validation_kl",
        "learning_rate",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    curve_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    if history:
        epochs = [row["epoch"] for row in history]
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in history],
            label="Train",
            lw=1.8,
            color="#356A9A",
        )
        axes[0].plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="Validation",
            lw=1.8,
            color="#D97935",
        )
        axes[1].plot(
            epochs,
            [row["train_l1"] for row in history],
            label="L1 reconstruction",
            color="#356A9A",
            lw=1.8,
        )
        axes[1].plot(
            epochs,
            [row["train_kl"] for row in history],
            label="KL divergence",
            color="#D97935",
            lw=1.8,
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel=r"$\mathcal{L}_{L1}+\beta\mathcal{L}_{KL}$",
        title="ACT objective",
        yscale="log",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Loss component",
        title="Training components",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=320)
    figure.savefig(curve_path.with_suffix(".pdf"))
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bc.yaml")
    parser.add_argument("--data", "--data-dir", dest="data_dir")
    parser.add_argument("--output", "--checkpoint", dest="checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--nheads", "--n-heads", dest="nheads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--decoder-layers", type=int)
    parser.add_argument("--dim-feedforward", type=int)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from PATH, or latest_checkpoint when PATH is omitted.",
    )
    return parser


def _model_config(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    source = dict(config.get("model", {}))
    values: dict[str, Any] = {
        "chunk_size": 20,
        "hidden_dim": 256,
        "latent_dim": 32,
        "nheads": int(source.get("n_heads", source.get("nheads", 8))),
        "encoder_layers": 3,
        "decoder_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "temporal_ensemble": True,
        "ensemble_decay": 0.05,
        "action_scale": 1.0,
    }
    for key in values:
        config_key = "n_heads" if key == "nheads" else key
        if config_key in source:
            values[key] = source[config_key]
        elif key in source:
            values[key] = source[key]
    for key in (
        "chunk_size",
        "hidden_dim",
        "latent_dim",
        "nheads",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
    ):
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    return values


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    training = dict(config.get("training", {}))
    if str(config.get("model", {}).get("policy_type", "ACT")).upper() != "ACT":
        raise ValueError("train_bc.py trains ACT; use an explicit ablation trainer for MLP.")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device_name = select_device(args.device or config.get("device", "auto"))
    seed_everything(seed)
    logger = configure_logging("train_act", "results/logs/train_act.log")
    attention_backend = str(training.get("attention_backend", "math")).lower()
    if device_name.startswith("cuda"):
        if attention_backend == "math":
            # The cuDNN SDPA backward kernel in some CUDA 12.8 development
            # stacks over-subscribes resources on Blackwell GPUs. The exact
            # PyTorch math backend is slower per operation but supports large
            # physical batches reliably and is recorded in every checkpoint.
            for function_name in (
                "enable_flash_sdp",
                "enable_mem_efficient_sdp",
                "enable_cudnn_sdp",
            ):
                function = getattr(torch.backends.cuda, function_name, None)
                if function is not None:
                    function(False)
            torch.backends.cuda.enable_math_sdp(True)
        elif attention_backend != "default":
            raise ValueError("training.attention_backend must be 'math' or 'default'.")
        logger.info("ACT scaled-dot-product attention backend=%s", attention_backend)

    data_path = resolve_path(args.data_dir or config.get("data_dir", "datasets/demonstrations"))
    data_manifest_path = data_path / "manifest.json"
    data_manifest_sha256 = (
        file_sha256(data_manifest_path) if data_manifest_path.is_file() else None
    )
    data_manifest = (
        json.loads(data_manifest_path.read_text(encoding="utf-8"))
        if data_manifest_path.is_file()
        else {}
    )
    task_vocabulary = list(data_manifest.get("task_vocabulary", []))
    task_vocabulary_sha256 = data_manifest.get("task_vocabulary_sha256")
    benchmark_name = data_manifest.get("benchmark")
    output_path = resolve_path(
        args.checkpoint or config.get("checkpoint", "checkpoints/bc_policy.pt")
    )
    latest_path = resolve_path(
        config.get("latest_checkpoint", "checkpoints/bc_policy_latest.pt")
    )
    curve_path = resolve_path(
        config.get("curve_path", "results/figures/act_training_curve.png")
    )
    history_path = resolve_path(
        config.get("history_path", "results/tables/act_training_history.csv")
    )
    summary_path = resolve_path(
        config.get("summary_path", "results/tables/bc_training_summary.json")
    )

    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 200))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 256)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate", 1e-4)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else training.get("num_workers", 0)
    )
    kl_weight = float(
        args.kl_weight
        if args.kl_weight is not None
        else training.get("kl_weight", 10.0)
    )
    if epochs <= 0 or batch_size <= 0 or kl_weight < 0:
        raise ValueError("epochs/batch_size must be positive and kl_weight non-negative.")

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
    resume_checkpoint = None
    model_config = _model_config(config, args)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        # RNG and DataLoader generator snapshots must remain CPU ByteTensors.
        resume_checkpoint = _torch_load(resume_path, "cpu")
        if str(resume_checkpoint.get("policy_type", "")).upper() != "ACT":
            raise ValueError("Resume checkpoint is not an ACT policy.")
        model_config = dict(resume_checkpoint.get("model_config", model_config))

    data = load_demonstrations(data_path)
    states, actions = data.states, data.actions
    if np.max(np.abs(actions)) > 1.0001:
        clipped_fraction = float(np.mean(np.abs(actions) > 1.0))
        logger.warning(
            "Clipping %.3f%% of expert action components to [-1,1].",
            100.0 * clipped_fraction,
        )
        actions = np.clip(actions, -1.0, 1.0)
        data = DemonstrationData(states, actions, data.groups, data.files)
    state_dim, action_dim = states.shape[-1], actions.shape[-1]
    if task_vocabulary and state_dim != 39 + len(task_vocabulary):
        raise ValueError(
            "Demonstration task vocabulary is incompatible with state_dim: "
            f"{len(task_vocabulary)} tasks require {39 + len(task_vocabulary)}, "
            f"found {state_dim}."
        )
    train_indices, validation_indices = group_train_validation_split(
        data.groups,
        float(training.get("validation_fraction", 0.1)),
        seed,
    )
    state_mean = states[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        states[train_indices].std(axis=0, dtype=np.float64).astype(np.float32),
        1e-6,
    )

    if resume_checkpoint is not None:
        if int(resume_checkpoint.get("seed", seed)) != seed:
            raise ValueError("Resume checkpoint seed does not match --seed.")
        stored_manifest_hash = resume_checkpoint.get("data_manifest_sha256")
        if (
            stored_manifest_hash is not None
            and stored_manifest_hash != data_manifest_sha256
        ):
            raise ValueError("Resume checkpoint demonstration manifest has changed.")
        if int(resume_checkpoint.get("state_dim", state_dim)) != state_dim:
            raise ValueError("Resume checkpoint state_dim does not match demonstrations.")
        if int(resume_checkpoint.get("action_dim", action_dim)) != action_dim:
            raise ValueError("Resume checkpoint action_dim does not match demonstrations.")
        if resume_checkpoint.get("task_vocabulary", task_vocabulary) != task_vocabulary:
            raise ValueError("Resume checkpoint task vocabulary has changed.")
        if (
            resume_checkpoint.get("task_vocabulary_sha256", task_vocabulary_sha256)
            != task_vocabulary_sha256
        ):
            raise ValueError("Resume checkpoint task vocabulary hash has changed.")
        stored_train_groups = resume_checkpoint.get("train_group_ids")
        stored_validation_groups = resume_checkpoint.get("validation_group_ids")
        if stored_train_groups is not None and set(stored_train_groups) != set(
            np.unique(data.groups[train_indices]).tolist()
        ):
            raise ValueError("Resume checkpoint ACT training split has changed.")
        if stored_validation_groups is not None and set(
            stored_validation_groups
        ) != set(np.unique(data.groups[validation_indices]).tolist()):
            raise ValueError("Resume checkpoint ACT validation split has changed.")
    chunk_size = int(model_config["chunk_size"])
    train_dataset = ActionChunkDataset(data, train_indices, chunk_size)
    validation_dataset = ActionChunkDataset(data, validation_indices, chunk_size)
    generator = torch.Generator().manual_seed(seed)
    effective_train_batch_size = min(batch_size, len(train_dataset))
    drop_last_batch = bool(training.get("drop_last_batch", True)) and len(
        train_dataset
    ) > effective_train_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_name.startswith("cuda"),
        generator=generator,
        drop_last=drop_last_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_name.startswith("cuda"),
    )

    model = ACTPolicy(state_dim, action_dim, **model_config).to(device_name)
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
        model.load_state_dict(
            resume_checkpoint.get(
                "model_state_dict", resume_checkpoint.get("state_dict", resume_checkpoint)
            )
        )
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
        if resume_checkpoint.get("dataloader_generator_state") is not None:
            generator.set_state(resume_checkpoint["dataloader_generator_state"])
        restore_rng_state(resume_checkpoint.get("rng_state"))
        logger.info("Resumed %s at epoch %d.", resume_path, start_epoch)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "ACT data: %d transitions/%d trajectories, state=%d action=%d chunk=%d; "
        "train=%d validation=%d; parameters=%d device=%s",
        len(states),
        len(np.unique(data.groups)),
        state_dim,
        action_dim,
        chunk_size,
        len(train_dataset),
        len(validation_dataset),
        parameter_count,
        device_name,
    )

    patience = int(training.get("patience", 40))
    grad_clip = float(training.get("grad_clip_norm", 5.0))
    checkpoint_every = int(training.get("checkpoint_every", 5))
    epoch_bar = tqdm(
        range(start_epoch, epochs),
        initial=start_epoch,
        total=epochs,
        desc="ACT",
        unit="epoch",
    )
    for epoch in epoch_bar:
        phase_metrics: dict[str, dict[str, float]] = {}
        for phase, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(phase == "train")
            sums = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
            count = 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for batch_states, action_chunks, padding_mask in loader:
                    batch_states = batch_states.to(device_name, non_blocking=True)
                    action_chunks = action_chunks.to(device_name, non_blocking=True)
                    padding_mask = padding_mask.to(device_name, non_blocking=True)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    output = model(batch_states, action_chunks, padding_mask)
                    if not isinstance(output, dict):
                        raise RuntimeError("ACT training must return posterior outputs.")
                    losses = model.loss(
                        output, action_chunks, padding_mask, kl_weight=kl_weight
                    )
                    if phase == "train":
                        losses["loss"].backward()
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    batch_count = len(batch_states)
                    for name in sums:
                        sums[name] += float(losses[name].detach()) * batch_count
                    count += batch_count
            phase_metrics[phase] = {
                name: value / max(count, 1) for name, value in sums.items()
            }

        row = {
            "epoch": epoch + 1,
            "train_loss": phase_metrics["train"]["loss"],
            "validation_loss": phase_metrics["validation"]["loss"],
            "train_l1": phase_metrics["train"]["l1"],
            "validation_l1": phase_metrics["validation"]["l1"],
            "train_kl": phase_metrics["train"]["kl"],
            "validation_kl": phase_metrics["validation"]["kl"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        improved = row["validation_loss"] < best_validation_loss - 1e-10
        if improved:
            best_validation_loss = row["validation_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload: dict[str, Any] = {
            "format_version": 2,
            "policy_type": "ACT",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": model.chunk_size,
            "model_config": model.model_config,
            "state_mean": state_mean,
            "state_std": state_std,
            "reconstruction_loss": "l1",
            "kl_weight": kl_weight,
            "attention_backend": attention_backend,
            "drop_last_batch": drop_last_batch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "seed": seed,
            "benchmark": benchmark_name,
            "task_vocabulary": task_vocabulary,
            "task_vocabulary_sha256": task_vocabulary_sha256,
            "data_manifest_sha256": data_manifest_sha256,
            "train_group_ids": np.unique(data.groups[train_indices]).tolist(),
            "validation_group_ids": np.unique(
                data.groups[validation_indices]
            ).tolist(),
            "dataloader_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(),
            "config": config,
        }
        _atomic_torch_save(payload, latest_path)
        if improved:
            _atomic_torch_save(payload, output_path)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            snapshot = latest_path.parent / "act" / f"epoch_{epoch + 1:04d}.pt"
            _atomic_torch_save(payload, snapshot)
        _save_history(history, history_path, curve_path)
        epoch_bar.set_postfix(
            train=f"{row['train_loss']:.4f}",
            val=f"{row['validation_loss']:.4f}",
            l1=f"{row['validation_l1']:.4f}",
            kl=f"{row['validation_kl']:.4f}",
        )
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping after %d non-improving epochs.", patience)
            break

    if not output_path.exists() and latest_path.exists():
        _atomic_torch_save(_torch_load(latest_path, "cpu"), output_path)
    summary = {
        "policy_type": "ACT",
        "samples": int(len(states)),
        "trajectories": int(len(np.unique(data.groups))),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "chunk_size": int(chunk_size),
        "parameters": int(parameter_count),
        "epochs_completed": int(history[-1]["epoch"] if history else start_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_l1": float(
            min((row["validation_l1"] for row in history), default=float("nan"))
        ),
        "checkpoint": str(output_path),
        "latest_checkpoint": str(latest_path),
        "curve": str(curve_path),
        "device": device_name,
        "seed": seed,
        "attention_backend": attention_backend,
        "drop_last_batch": drop_last_batch,
        "benchmark": benchmark_name,
        "task_vocabulary": task_vocabulary,
        "task_vocabulary_sha256": task_vocabulary_sha256,
    }
    atomic_json_dump(summary, summary_path)
    logger.info(
        "ACT complete: best validation objective %.6f; checkpoint=%s",
        best_validation_loss,
        output_path,
    )
    return summary


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
