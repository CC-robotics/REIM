from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data.io import file_sha256
from models.bc_policy import MLPBCPolicy
from trainers.train_multitask_mlp import (
    DEMONSTRATION_DATASET_TYPE,
    DEMONSTRATION_SCHEMA,
    build_parser,
    load_multitask_demonstrations,
    stratified_trajectory_split,
    train,
)


TASKS = [f"task-{index:02d}-v3" for index in range(10)]


def _vocabulary_hash() -> str:
    value = json.dumps(TASKS, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _write_dataset(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    trajectories: list[dict[str, object]] = []
    total_transitions = 0
    for task_id, task_name in enumerate(TASKS):
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[task_id] = 1.0
        for trajectory_index in range(2):
            length = 3
            raw = np.full(
                (length, 39), task_id + 0.1 * trajectory_index, dtype=np.float32
            )
            states = np.concatenate(
                [raw, np.broadcast_to(one_hot, (length, 10))], axis=1
            ).astype(np.float32)
            actions = np.tanh(raw[:, :4] * 0.1).astype(np.float32)
            path = root / f"task{task_id:02d}_trajectory{trajectory_index:02d}.npz"
            np.savez_compressed(
                path,
                states=states,
                raw_observations=raw,
                actions=actions,
                rewards=np.ones(length, dtype=np.float32),
                success=np.asarray(True),
                task_id=np.asarray(task_id),
                task_name=np.asarray(task_name),
                task_variant=np.asarray(trajectory_index),
                seed=np.asarray(1000 + 10 * task_id + trajectory_index),
                trajectory_index=np.asarray(trajectory_index),
                attempt_index=np.asarray(trajectory_index),
                schema_version=np.asarray(DEMONSTRATION_SCHEMA),
                benchmark=np.asarray("MT10"),
            )
            trajectories.append(
                {
                    "file": path.name,
                    "sha256": file_sha256(path),
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_variant": trajectory_index,
                    "trajectory_index": trajectory_index,
                    "attempt_index": trajectory_index,
                    "seed": 1000 + 10 * task_id + trajectory_index,
                    "length": length,
                    "return": float(length),
                    "success": True,
                }
            )
            total_transitions += length
    manifest: dict[str, object] = {
        "schema_version": DEMONSTRATION_SCHEMA,
        "dataset_type": DEMONSTRATION_DATASET_TYPE,
        "benchmark": "MT10",
        "seed": 123,
        "metaworld_version": "3.1.1",
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _vocabulary_hash(),
        "task_count": 10,
        "complete": True,
        "protocol": {"episodes_per_task": 2},
        "observation_schema": {
            "raw_observations": {"shape": ["T", 39]},
            "states": {"shape": ["T", 49]},
            "actions": {"shape": ["T", 4]},
        },
        "statistics": {
            "successful_trajectories": len(trajectories),
            "total_transitions": total_transitions,
        },
        "trajectories": trajectories,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def test_mlp_checkpoint_round_trip_and_provenance(tmp_path: Path) -> None:
    model = MLPBCPolicy(49, 4, hidden_dims=(16, 8))
    mean = np.linspace(-1.0, 1.0, 49, dtype=np.float32)
    std = np.linspace(0.5, 1.5, 49, dtype=np.float32)
    model.set_normalization(mean, std)
    state = np.linspace(-2.0, 2.0, 49, dtype=np.float32)
    expected = model.act(state)
    path = tmp_path / "mlp.pt"
    torch.save(
        {
            "format_version": 2,
            "training_schema": "reim-multitask-mlp-training-v1",
            "policy_type": "MLP_BC",
            "state_dim": 49,
            "action_dim": 4,
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
            "benchmark": "MT10",
            "task_vocabulary": TASKS,
            "task_vocabulary_sha256": _vocabulary_hash(),
            "data_manifest_sha256": "manifest-hash",
            "split_sha256": "split-hash",
            "normalization_scope": "raw39_only_task_onehot_unchanged",
            "loss": "mean_squared_error",
            "seed": 42,
        },
        path,
    )

    restored = MLPBCPolicy.from_checkpoint(path)
    np.testing.assert_allclose(restored.act(state), expected, atol=1e-7)
    assert restored.hidden_dims == (16, 8)
    assert restored.provenance["benchmark"] == "MT10"
    assert restored.provenance["data_manifest_sha256"] == "manifest-hash"
    assert restored.provenance["split_sha256"] == "split-hash"


def test_mlp_checkpoint_rejects_false_architecture_metadata(tmp_path: Path) -> None:
    model = MLPBCPolicy(49, 4, hidden_dims=(8, 8))
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "policy_type": "MLP_BC",
            "state_dim": 49,
            "action_dim": 4,
            "model_config": {"hidden_dims": [16, 8]},
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="hidden_dims"):
        MLPBCPolicy.from_checkpoint(path)


def test_manifest_whitelist_hash_and_onehot_are_enforced(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)
    dataset = load_multitask_demonstrations(tmp_path, benchmark="MT10")
    assert len(dataset.states) == 20
    assert dataset.states[0].shape == (3, 49)

    stale = tmp_path / "stale.npz"
    np.savez_compressed(stale, states=np.zeros((1, 49), dtype=np.float32))
    with pytest.raises(ValueError, match="stale"):
        load_multitask_demonstrations(tmp_path, benchmark="MT10")
    stale.unlink()

    first = manifest["trajectories"][0]  # type: ignore[index]
    path = tmp_path / str(first["file"])
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["states"][:, 39:] = 0.0
    np.savez_compressed(path, **payload)
    first["sha256"] = file_sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="one-hot"):
        load_multitask_demonstrations(tmp_path, benchmark="MT10")


def test_trajectory_split_is_deterministic_and_task_stratified() -> None:
    task_ids = np.repeat(np.arange(10), 5)
    first = stratified_trajectory_split(task_ids, 0.2, seed=91)
    second = stratified_trajectory_split(task_ids, 0.2, seed=91)
    assert first == second
    train_groups, validation_groups = first
    assert set(task_ids[train_groups]) == set(range(10))
    assert set(task_ids[validation_groups]) == set(range(10))
    assert not set(train_groups).intersection(validation_groups)


def test_training_and_resume_preserve_auditable_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)
    output = tmp_path / "mlp.pt"
    latest = tmp_path / "mlp_latest.pt"
    config = {
        "benchmark": "MT10",
        "seed": 17,
        "device": "cpu",
        "data_dir": str(data_dir),
        "checkpoint": str(output),
        "latest_checkpoint": str(latest),
        "curve_path": str(tmp_path / "curve.png"),
        "history_path": str(tmp_path / "history.csv"),
        "summary_path": str(tmp_path / "summary.json"),
        "log_path": str(tmp_path / "training.log"),
        "model": {"policy_type": "MLP_BC", "hidden_dims": [8, 8]},
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "validation_fraction": 0.5,
            "num_workers": 0,
            "patience": 0,
            "checkpoint_every": 0,
            "grad_clip_norm": 5.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    parser = build_parser()
    summary = train(parser.parse_args(["--config", str(config_path)]))
    assert summary["epochs_completed"] == 1
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["loss"] == "mean_squared_error"
    assert checkpoint["normalization_scope"] == "raw39_only_task_onehot_unchanged"
    assert len(checkpoint["train_trajectories"]) == 10
    assert len(checkpoint["validation_trajectories"]) == 10

    resumed = train(
        parser.parse_args(
            ["--config", str(config_path), "--epochs", "2", "--resume"]
        )
    )
    assert resumed["epochs_completed"] == 2
    restored = MLPBCPolicy.from_checkpoint(output)
    assert restored.provenance["split_sha256"] == checkpoint["split_sha256"]
