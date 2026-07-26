#!/usr/bin/env python3
"""Train the causal LSTM failure detector from generated failure NPZ files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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

from models.failure_detector import FailureDetector  # noqa: E402
from data.io import file_sha256  # noqa: E402
from trainers.data import group_train_validation_split, load_failure_data  # noqa: E402
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


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int64)
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    total = max(len(labels), 1)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "samples": int(len(labels)),
        "positive_fraction": float(labels.mean()) if len(labels) else 0.0,
    }


def _save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#18212b",
                fontsize=12,
            )
    axis.set_xticks((0, 1), labels=("Normal", "Failure"))
    axis.set_yticks((0, 1), labels=("Normal", "Failure"))
    axis.set(xlabel="Predicted label", ylabel="True label", title="Failure Detector")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _save_history(
    history: list[dict[str, float]],
    csv_path: Path,
    curve_path: Path,
) -> None:
    fields = (
        "epoch",
        "train_loss",
        "validation_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
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
            epochs, [row["train_loss"] for row in history], label="Train", lw=2
        )
        axes[0].plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="Validation",
            lw=2,
        )
        for name in ("accuracy", "precision", "recall", "f1"):
            axes[1].plot(epochs, [row[name] for row in history], label=name.title())
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE", title="Detector objective")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0.0, 1.02), title="Validation")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(curve_path, dpi=220)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--data", "--data-dir", dest="data_dir")
    parser.add_argument("--output", "--checkpoint", dest="checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from PATH, or latest_checkpoint when PATH is omitted.",
    )
    return parser


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    model_config = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    device_name = select_device(args.device or config.get("device", "auto"))
    seed_everything(seed)
    logger = configure_logging("train_detector", "results/logs/train_detector.log")

    data_path = resolve_path(args.data_dir or config.get("data_dir", "datasets/failures"))
    data_manifest_path = data_path / "manifest.json"
    data_manifest_sha256 = (
        file_sha256(data_manifest_path) if data_manifest_path.is_file() else None
    )
    output_path = resolve_path(
        args.checkpoint or config.get("checkpoint", "checkpoints/failure_detector.pt")
    )
    latest_path = resolve_path(
        config.get("latest_checkpoint", "checkpoints/failure_detector_latest.pt")
    )
    curve_path = resolve_path(
        config.get("curve_path", "results/figures/detector_training_curve.png")
    )
    confusion_path = resolve_path(
        config.get("confusion_matrix_path", "results/figures/confusion_matrix.png")
    )
    history_path = resolve_path(
        config.get("history_path", "results/tables/detector_training_history.csv")
    )
    metrics_path = resolve_path(
        config.get("metrics_path", "results/tables/detector_metrics.json")
    )

    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 50))
    batch_size = int(
        args.batch_size if args.batch_size is not None else training.get("batch_size", 512)
    )
    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else training.get("learning_rate", 5e-4)
    )
    sequence_length = int(
        args.sequence_length
        if args.sequence_length is not None
        else model_config.get("sequence_length", 10)
    )
    threshold = float(
        args.threshold if args.threshold is not None else training.get("threshold", 0.5)
    )
    if epochs <= 0 or batch_size <= 0 or sequence_length <= 0:
        raise ValueError("epochs, batch_size, and sequence_length must be positive.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")

    data = load_failure_data(data_path, sequence_length)
    train_indices, validation_indices = group_train_validation_split(
        data.groups,
        float(training.get("validation_fraction", 0.2)),
        seed,
        labels=data.labels,
    )
    time_indices = np.arange(sequence_length)[None, :]
    valid_mask = time_indices < data.lengths[train_indices, None]
    valid_states = data.windows[train_indices][valid_mask]
    state_mean = valid_states.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        valid_states.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )

    def make_dataset(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(data.windows[indices]),
            torch.from_numpy(data.labels[indices]),
            torch.from_numpy(data.lengths[indices]),
        )

    train_dataset = make_dataset(train_indices)
    validation_dataset = make_dataset(validation_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        pin_memory=device_name.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(batch_size, len(validation_dataset)),
        shuffle=False,
        pin_memory=device_name.startswith("cuda"),
    )

    hidden_size = int(
        args.hidden_size
        if args.hidden_size is not None
        else model_config.get("hidden_size", model_config.get("hidden_dim", 128))
    )
    detector_kwargs = {
        "hidden_dim": hidden_size,
        "num_layers": int(model_config.get("num_layers", 1)),
        "mlp_hidden": int(model_config.get("mlp_hidden", 64)),
        "dropout": float(model_config.get("dropout", 0.1)),
        "sequence_length": sequence_length,
    }
    model = FailureDetector(data.windows.shape[-1], **detector_kwargs).to(device_name)
    model.set_normalization(state_mean, state_std)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )

    positive_count = float(data.labels[train_indices].sum())
    negative_count = float(len(train_indices) - positive_count)
    requested_positive_weight = training.get("positive_weight", "auto")
    if str(requested_positive_weight).lower() == "auto":
        positive_weight = negative_count / positive_count if positive_count else 1.0
    else:
        positive_weight = float(requested_positive_weight)
    positive_weight = max(float(positive_weight), 1e-6)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device_name)
    )

    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    resume_value = args.resume
    if resume_value is None and config.get("resume"):
        resume_value = str(config["resume"])
    if resume_value:
        resume_path = latest_path if resume_value == "auto" else resolve_path(resume_value)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = _torch_load(resume_path, device_name)
        if int(checkpoint.get("seed", seed)) != seed:
            raise ValueError("Resume detector seed does not match --seed.")
        stored_manifest_hash = checkpoint.get("data_manifest_sha256")
        if (
            stored_manifest_hash is not None
            and stored_manifest_hash != data_manifest_sha256
        ):
            raise ValueError("Resume detector failure-data manifest has changed.")
        if int(checkpoint.get("state_dim", model.state_dim)) != model.state_dim:
            raise ValueError("Resume detector state_dim does not match the dataset.")
        checkpoint_model_config = dict(checkpoint.get("model_config", detector_kwargs))
        architecture_fields = (
            "hidden_dim",
            "num_layers",
            "mlp_hidden",
            "sequence_length",
        )
        for field in architecture_fields:
            if (
                field in checkpoint_model_config
                and checkpoint_model_config[field] != detector_kwargs[field]
            ):
                raise ValueError(
                    f"Resume detector {field}={checkpoint_model_config[field]} does "
                    f"not match requested {detector_kwargs[field]}."
                )
        stored_train_groups = checkpoint.get("train_group_ids")
        stored_validation_groups = checkpoint.get("validation_group_ids")
        if stored_train_groups is not None and set(stored_train_groups) != set(
            np.unique(data.groups[train_indices]).tolist()
        ):
            raise ValueError("Resume detector training split has changed.")
        if stored_validation_groups is not None and set(
            stored_validation_groups
        ) != set(np.unique(data.groups[validation_indices]).tolist()):
            raise ValueError("Resume detector validation split has changed.")
        model.load_state_dict(
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        )
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", best_validation_loss)
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(checkpoint.get("history", []))
        if checkpoint.get("dataloader_generator_state") is not None:
            generator.set_state(checkpoint["dataloader_generator_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        logger.info("Resumed %s at epoch %d.", resume_path, start_epoch)

    logger.info(
        "Failure data: %d windows from %d trajectories, state=%d sequence=%d, "
        "positive=%.2f%%, train=%d validation=%d, pos_weight=%.3f, device=%s",
        len(data.labels),
        len(np.unique(data.groups)),
        model.state_dim,
        sequence_length,
        100.0 * float(data.labels.mean()),
        len(train_dataset),
        len(validation_dataset),
        positive_weight,
        device_name,
    )

    patience = int(training.get("patience", 10))
    grad_clip = float(training.get("grad_clip_norm", 5.0))
    checkpoint_every = int(training.get("checkpoint_every", 5))
    final_metrics: dict[str, Any] = {}
    epoch_bar = tqdm(
        range(start_epoch, epochs),
        initial=start_epoch,
        total=epochs,
        desc="Failure detector",
        unit="epoch",
    )
    for epoch in epoch_bar:
        model.train()
        train_sum, train_count = 0.0, 0
        for windows, labels, lengths in train_loader:
            windows = windows.to(device_name, non_blocking=True)
            labels = labels.to(device_name, non_blocking=True)
            lengths = lengths.to(device_name, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(windows, lengths), labels)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_sum += float(loss.detach()) * len(windows)
            train_count += len(windows)

        model.eval()
        validation_sum, validation_count = 0.0, 0
        probability_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for windows, labels, lengths in validation_loader:
                windows = windows.to(device_name, non_blocking=True)
                labels_device = labels.to(device_name, non_blocking=True)
                lengths = lengths.to(device_name, non_blocking=True)
                logits = model(windows, lengths)
                loss = loss_function(logits, labels_device)
                validation_sum += float(loss) * len(windows)
                validation_count += len(windows)
                probability_batches.append(torch.sigmoid(logits).cpu().numpy())
                label_batches.append(labels.numpy())
        train_loss = train_sum / max(train_count, 1)
        validation_loss = validation_sum / max(validation_count, 1)
        selected_labels = np.concatenate(label_batches)
        selected_probabilities = np.concatenate(probability_batches)
        final_metrics = classification_metrics(
            selected_labels,
            selected_probabilities,
            threshold,
        )
        deployment_threshold = float(
            training.get("deployment_threshold", 0.8)
        )
        if not 0.0 <= deployment_threshold <= 1.0:
            raise ValueError("deployment_threshold must be between zero and one.")
        final_metrics["deployment_threshold_metrics"] = classification_metrics(
            selected_labels,
            selected_probabilities,
            deployment_threshold,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        improved = validation_loss < best_validation_loss - 1e-10
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload: dict[str, Any] = {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": model.state_dim,
            "hidden_dim": model.hidden_dim,
            "hidden_size": model.hidden_dim,
            "num_layers": model.num_layers,
            "mlp_hidden": model.mlp_hidden,
            "dropout": model.dropout_probability,
            "sequence_length": model.sequence_length,
            "model_config": detector_kwargs,
            "state_mean": state_mean,
            "state_std": state_std,
            "positive_weight": positive_weight,
            "threshold": threshold,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "validation_metrics": final_metrics,
            "seed": seed,
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
            snapshot = latest_path.parent / "detector" / f"epoch_{epoch + 1:04d}.pt"
            _atomic_torch_save(payload, snapshot)
        _save_history(history, history_path, curve_path)
        _save_confusion_matrix(final_metrics, confusion_path)
        metrics_payload = {
            **final_metrics,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "epoch": epoch + 1,
            "checkpoint": str(output_path),
            "confusion_matrix_figure": str(confusion_path),
        }
        atomic_json_dump(metrics_payload, metrics_path)
        epoch_bar.set_postfix(
            loss=f"{validation_loss:.4f}",
            accuracy=f"{final_metrics['accuracy']:.3f}",
            f1=f"{final_metrics['f1']:.3f}",
        )
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping after %d non-improving epochs.", patience)
            break

    if not output_path.exists() and latest_path.exists():
        _atomic_torch_save(_torch_load(latest_path, "cpu"), output_path)
    # Report the selected (lowest validation BCE) checkpoint, not merely the
    # last epoch. This also makes a no-op resume at the target epoch useful.
    if output_path.exists():
        selected_checkpoint = _torch_load(output_path, device_name)
        model.load_state_dict(
            selected_checkpoint.get(
                "model_state_dict",
                selected_checkpoint.get("state_dict", selected_checkpoint),
            )
        )
        model.eval()
        validation_sum, validation_count = 0.0, 0
        probability_batches = []
        label_batches = []
        with torch.inference_mode():
            for windows, labels, lengths in validation_loader:
                windows = windows.to(device_name, non_blocking=True)
                labels_device = labels.to(device_name, non_blocking=True)
                lengths = lengths.to(device_name, non_blocking=True)
                logits = model(windows, lengths)
                validation_sum += float(loss_function(logits, labels_device)) * len(
                    windows
                )
                validation_count += len(windows)
                probability_batches.append(torch.sigmoid(logits).cpu().numpy())
                label_batches.append(labels.numpy())
        selected_validation_loss = validation_sum / max(validation_count, 1)
        selected_labels = np.concatenate(label_batches)
        selected_probabilities = np.concatenate(probability_batches)
        final_metrics = classification_metrics(
            selected_labels,
            selected_probabilities,
            threshold,
        )
        deployment_threshold = float(
            training.get("deployment_threshold", 0.8)
        )
        if not 0.0 <= deployment_threshold <= 1.0:
            raise ValueError("deployment_threshold must be between zero and one.")
        final_metrics["deployment_threshold_metrics"] = classification_metrics(
            selected_labels,
            selected_probabilities,
            deployment_threshold,
        )
        final_metrics.update(
            {
                "validation_loss": selected_validation_loss,
                "best_validation_loss": float(
                    selected_checkpoint.get(
                        "best_validation_loss", selected_validation_loss
                    )
                ),
                "epoch": int(selected_checkpoint.get("epoch", -1)) + 1,
                "checkpoint": str(output_path),
                "confusion_matrix_figure": str(confusion_path),
            }
        )
        _save_confusion_matrix(final_metrics, confusion_path)
        atomic_json_dump(final_metrics, metrics_path)
    logger.info(
        "Detector complete: accuracy=%.4f precision=%.4f recall=%.4f F1=%.4f; %s",
        final_metrics.get("accuracy", float("nan")),
        final_metrics.get("precision", float("nan")),
        final_metrics.get("recall", float("nan")),
        final_metrics.get("f1", float("nan")),
        output_path,
    )
    return final_metrics


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
