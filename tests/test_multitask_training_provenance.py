from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.io import file_sha256
from models.bc_policy import ACTPolicy
from models.failure_detector import FailureDetector
from trainers.train_multitask_recovery import _load_dataset


TASKS = ["task-a-v3", "task-b-v3"]
SCHEMA = "reim-multitask-trigger-aligned-recovery-v1"


def _write_shard(root: Path, task_id: int, index: int) -> dict[str, object]:
    task_dir = root / f"task_{task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"recovery_{index:04d}.npz"
    one_hot = np.zeros(len(TASKS), dtype=np.float32)
    one_hot[task_id] = 1.0
    states = np.concatenate(
        [np.zeros((3, 39), dtype=np.float32), np.broadcast_to(one_hot, (3, 2))],
        axis=1,
    )
    np.savez_compressed(
        path,
        states=states,
        actions=np.zeros((3, 4), dtype=np.float32),
        success=np.asarray(True),
        task_id=np.asarray(task_id),
        task_name=np.asarray(TASKS[task_id]),
        episode_seed=np.asarray(1000 * task_id + index),
        schema_version=np.asarray(SCHEMA),
    )
    return {
        "file": path.relative_to(root).as_posix(),
        "task_id": task_id,
        "task_name": TASKS[task_id],
        "length": 3,
        "sha256": file_sha256(path),
    }


def _manifest(root: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "files": [_write_shard(root, 0, 0), _write_shard(root, 1, 0)],
    }


def test_recovery_loader_accepts_only_manifest_whitelist(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    states, actions, task_ids, files = _load_dataset(tmp_path, manifest, TASKS)
    assert [array.shape for array in states] == [(3, 41), (3, 41)]
    assert [array.shape for array in actions] == [(3, 4), (3, 4)]
    assert task_ids == [0, 1]
    assert len(files) == 2

    np.savez_compressed(tmp_path / "stale.npz", states=np.zeros((1, 41)))
    with pytest.raises(ValueError, match="stale"):
        _load_dataset(tmp_path, manifest, TASKS)


def test_recovery_loader_rejects_hash_or_onehot_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["files"][0]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_dataset(tmp_path, manifest, TASKS)

    manifest = _manifest(tmp_path)
    path = tmp_path / str(manifest["files"][0]["file"])  # type: ignore[index]
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["states"][:, 39:] = 0.0
    np.savez_compressed(path, **payload)
    manifest["files"][0]["sha256"] = file_sha256(path)  # type: ignore[index]
    with pytest.raises(ValueError, match="one-hot"):
        _load_dataset(tmp_path, manifest, TASKS)


def test_policy_loaders_expose_ordered_task_provenance(tmp_path: Path) -> None:
    vocabulary_hash = "abc123"
    act = ACTPolicy(
        41,
        4,
        chunk_size=2,
        hidden_dim=16,
        latent_dim=4,
        nheads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    act_path = tmp_path / "act.pt"
    torch.save(
        {
            "policy_type": "ACT",
            "model_state_dict": act.state_dict(),
            "state_dim": 41,
            "action_dim": 4,
            "model_config": act.model_config,
            "benchmark": "TEST",
            "task_vocabulary": TASKS,
            "task_vocabulary_sha256": vocabulary_hash,
        },
        act_path,
    )
    loaded_act = ACTPolicy.from_checkpoint(act_path)
    assert loaded_act.provenance["task_vocabulary"] == TASKS
    assert loaded_act.provenance["task_vocabulary_sha256"] == vocabulary_hash

    detector = FailureDetector(41, hidden_dim=8, mlp_hidden=4, dropout=0.0)
    detector_path = tmp_path / "detector.pt"
    torch.save(
        {
            "model_state_dict": detector.state_dict(),
            "state_dim": 41,
            "hidden_dim": 8,
            "mlp_hidden": 4,
            "dropout": 0.0,
            "benchmark": "TEST",
            "task_vocabulary": TASKS,
            "task_vocabulary_sha256": vocabulary_hash,
        },
        detector_path,
    )
    loaded_detector = FailureDetector.from_checkpoint(detector_path)
    assert loaded_detector.provenance["task_vocabulary"] == TASKS
    assert loaded_detector.provenance["task_vocabulary_sha256"] == vocabulary_hash
