from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.io import file_sha256
from models.bc_policy import ACTPolicy
from models.failure_detector import FailureDetector
from models.imitation_recovery_policy import ImitationRecoveryPolicy
import trainers.train_multitask_recovery as recovery_trainer
from trainers.train_multitask_recovery import (
    RECOVERY_DATASET_TYPE,
    _array_content_sha256,
    _canonical_sha256,
    _load_dataset,
)


TASKS = ["task-a-v3", "task-b-v3"]
SCHEMA = "reim-multitask-trigger-aligned-recovery-v2"
ROLLOUT_SEED = 42
PROTOCOL = {
    "detector_threshold": 0.2,
    "max_episode_steps": 500,
    "rollout_seed": ROLLOUT_SEED,
}
PROTOCOL_FINGERPRINT = _canonical_sha256(PROTOCOL)


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
    episode_seed = (
        (ROLLOUT_SEED % 1_000_000_000) * 100_000_000 + task_id * 1_000_000 + index
    )
    task_payload_sha256 = (
        __import__("hashlib").sha256(f"{task_id}:{index}".encode("ascii")).hexdigest()
    )
    arrays = {
        "states": states,
        "raw_observations": np.zeros((3, 39), dtype=np.float32),
        "actions": np.zeros((3, 4), dtype=np.float32),
        "rewards": np.zeros(3, dtype=np.float32),
        "failure_probabilities": np.full(3, 0.9, dtype=np.float32),
        "success": np.asarray(True),
        "task_id": np.asarray(task_id),
        "task_name": np.asarray(TASKS[task_id]),
        "task_variant": np.asarray(index),
        "task_payload_sha256": np.asarray(task_payload_sha256),
        "trigger_step": np.asarray(0),
        "trigger_probability": np.asarray(0.9, dtype=np.float32),
        "episode_seed": np.asarray(episode_seed),
        "attempt_index": np.asarray(index),
        "shard_index": np.asarray(index),
        "protocol_fingerprint_sha256": np.asarray(PROTOCOL_FINGERPRINT),
        "schema_version": np.asarray(SCHEMA),
    }
    np.savez_compressed(path, **arrays)
    return {
        "file": path.relative_to(root).as_posix(),
        "task_id": task_id,
        "task_name": TASKS[task_id],
        "task_variant": index,
        "task_payload_sha256": task_payload_sha256,
        "trigger_step": 0,
        "trigger_probability": float(np.float32(0.9)),
        "episode_seed": episode_seed,
        "attempt_index": index,
        "shard_index": index,
        "length": 3,
        "content_sha256": _array_content_sha256(arrays),
        "sha256": file_sha256(path),
    }


def _manifest(root: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "protocol": PROTOCOL,
        "protocol_fingerprint_sha256": PROTOCOL_FINGERPRINT,
        "files": [_write_shard(root, 0, 0), _write_shard(root, 1, 0)],
    }


def _complete_manifest(root: Path, *, target: int = 2) -> dict[str, object]:
    files = [
        _write_shard(root, task_id, index)
        for task_id in range(len(TASKS))
        for index in range(target)
    ]
    protocol = {
        "schema_version": SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "benchmark": "MT10",
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _canonical_sha256(TASKS),
        "task_bank_sha256": "c" * 64,
        "state_dim": 41,
        "action_dim": 4,
        "target_per_task": target,
        "max_attempts_multiplier": 3,
        "max_attempts_per_task": target * 3,
        "max_episode_steps": 500,
        "detector_threshold": 0.2,
        "rollout_seed": ROLLOUT_SEED,
        "act_checkpoint_sha256": "a" * 64,
        "detector_checkpoint_sha256": "b" * 64,
    }
    fingerprint = _canonical_sha256(protocol)
    # Rebind the synthetic shard fingerprint to this complete protocol.
    for entry in files:
        path = root / str(entry["file"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["protocol_fingerprint_sha256"] = np.asarray(fingerprint)
        np.savez_compressed(path, **arrays)
        entry["content_sha256"] = _array_content_sha256(arrays)
        entry["sha256"] = file_sha256(path)
    per_task: dict[str, dict[str, object]] = {}
    progress: dict[str, dict[str, object]] = {}
    for task_id, task_name in enumerate(TASKS):
        task_entries = [entry for entry in files if entry["task_id"] == task_id]
        rows = sum(int(entry["length"]) for entry in task_entries)
        progress[task_name] = {
            "task_id": task_id,
            "next_attempt_index": target,
            "attempts": target,
            "detector_triggers": target,
            "successful_continuations": target,
        }
        per_task[task_name] = {
            **progress[task_name],
            "rows": rows,
            "attempt_yield": 1.0,
        }
    return {
        "schema_version": SCHEMA,
        "dataset_type": RECOVERY_DATASET_TYPE,
        "benchmark": "MT10",
        "complete": True,
        "task_vocabulary": TASKS,
        "task_vocabulary_sha256": _canonical_sha256(TASKS),
        "task_bank_sha256": "c" * 64,
        "state_dim": 41,
        "action_dim": 4,
        "target_per_task": target,
        "max_episode_steps": 500,
        "act_checkpoint_sha256": "a" * 64,
        "detector_checkpoint_sha256": "b" * 64,
        "protocol": protocol,
        "protocol_fingerprint_sha256": fingerprint,
        "successful_continuations": len(files),
        "rows": sum(int(entry["length"]) for entry in files),
        "attempts": len(files),
        "detector_triggers": len(files),
        "collection_progress": progress,
        "per_task": per_task,
        "files": files,
    }


def _training_args(
    root: Path, output_name: str, *, epochs: int, resume=None
) -> Namespace:
    return Namespace(
        benchmark="MT10",
        data_dir=str(root / "data"),
        output=str(root / f"{output_name}.pt"),
        history=str(root / f"{output_name}.csv"),
        curve=str(root / f"{output_name}.png"),
        summary=str(root / f"{output_name}.json"),
        log_file=str(root / f"{output_name}.log"),
        seed=17,
        device="cpu",
        epochs=epochs,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        validation_fraction=0.5,
        hidden_dims=(8, 8),
        state_noise_std=0.005,
        grad_clip=1.0,
        patience=10,
        min_delta=0.0,
        num_workers=0,
        resume=resume,
    )


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


def test_recovery_training_resume_is_exact_and_provenance_is_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(recovery_trainer.SUPPORTED_BENCHMARKS, "MT10", 2)
    monkeypatch.setattr(recovery_trainer, "OFFICIAL_TARGET_PER_TASK", 2)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = _complete_manifest(data_dir)
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    full = recovery_trainer.train(_training_args(tmp_path, "full", epochs=4))
    recovery_trainer.train(_training_args(tmp_path, "resumed", epochs=2))
    resumed = recovery_trainer.train(
        _training_args(tmp_path, "resumed", epochs=4, resume="auto")
    )

    assert full["best_validation_loss"] == resumed["best_validation_loss"]
    assert resumed["sampling_task_mass"] == [1.0, 1.0]
    full_history = (tmp_path / "full.csv").read_text(encoding="utf-8")
    resumed_history = (tmp_path / "resumed.csv").read_text(encoding="utf-8")
    assert full_history == resumed_history
    full_policy = ImitationRecoveryPolicy.load(tmp_path / "full.pt")
    resumed_policy = ImitationRecoveryPolicy.load(tmp_path / "resumed.pt")
    for key, value in full_policy.state_dict().items():
        assert torch.equal(value, resumed_policy.state_dict()[key])
    provenance = resumed_policy.provenance
    assert provenance["schema_version"] == recovery_trainer.SCHEMA_VERSION
    assert provenance["data_schema_version"] == SCHEMA
    assert provenance["dataset_complete"] is True
    assert provenance["target_per_task"] == 2
    assert provenance["act_checkpoint_sha256"] == "a" * 64
    assert provenance["detector_checkpoint_sha256"] == "b" * 64
    assert provenance["source_training"] == {
        "algorithm": "task_balanced_smooth_l1",
        "num_timesteps": 0,
        "supervised_update_only": True,
    }
    assert len(provenance["split_sha256"]) == 64
    assert len(provenance["training_config_sha256"]) == 64
    assert set(provenance["split"]["train_files"]).isdisjoint(
        provenance["split"]["validation_files"]
    )

    incompatible = _training_args(tmp_path, "resumed", epochs=5, resume="auto")
    incompatible.learning_rate = 2e-3
    with pytest.raises(ValueError, match="training configuration"):
        recovery_trainer.train(incompatible)


def test_recovery_training_rejects_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(recovery_trainer.SUPPORTED_BENCHMARKS, "MT10", 2)
    monkeypatch.setattr(recovery_trainer, "OFFICIAL_TARGET_PER_TASK", 2)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = _complete_manifest(data_dir)
    manifest["complete"] = False
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete"):
        recovery_trainer.train(_training_args(tmp_path, "partial", epochs=1))


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
