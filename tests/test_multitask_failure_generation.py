"""Strict resume and provenance tests for multi-task failure generation."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict, namedtuple
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from data.io import file_sha256
from scripts.generate_multitask_failures import (
    DATASET_TYPE,
    SCHEMA_VERSION,
    FailureGenerationComponents,
    generate,
)


FakeTask = namedtuple("FakeTask", ["env_name", "data"])


class _FakeACT:
    state_dim = 49
    action_dim = 4

    def reset(self) -> None:
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        assert state.shape == (self.state_dim,)
        return np.zeros(self.action_dim, dtype=np.float32)


class _FakeExpert:
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == (39,)
        return np.asarray([0.8, -0.8, 0.4, 1.0], dtype=np.float32)


def _environment_type(task_id: int, task_name: str) -> type:
    class FakeEnvironment:
        def __init__(self, render_mode: str | None = None) -> None:
            assert render_mode is None
            self.task: FakeTask | None = None
            self.closed = False

        def set_task(self, task: FakeTask) -> None:
            assert task.env_name == task_name
            self.task = task

        def reset(self, seed: int):
            assert self.task is not None
            raw = np.full(39, task_id / 10.0, dtype=np.float32)
            raw[-1] = seed % 97
            return raw, {}

        def step(self, action: np.ndarray):
            assert action.shape == (4,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
            raw = np.full(39, task_id / 10.0 + 0.01, dtype=np.float32)
            success = task_id % 2 == 0
            return raw, float(success), False, True, {"success": success}

        def close(self) -> None:
            self.closed = True

    FakeEnvironment.__name__ = f"FakeFailureEnvironment{task_id}"
    return FakeEnvironment


class _FakeBenchmark:
    def __init__(
        self,
        seed: int,
        *,
        bank_token: str,
        vocabulary_prefix: str,
    ) -> None:
        names = [f"{vocabulary_prefix}-{index}-v3" for index in range(10)]
        self.train_classes = OrderedDict(
            (name, _environment_type(index, name))
            for index, name in enumerate(names)
        )
        self.train_tasks = [
            FakeTask(
                name,
                f"seed={seed}|bank={bank_token}|task={name}|variant={variant}".encode(),
            )
            for name in names
            for variant in range(50)
        ]


def _components(
    *,
    bank_token: str = "bank-a",
    vocabulary_prefix: str = "task",
) -> FailureGenerationComponents:
    names = [f"{vocabulary_prefix}-{index}-v3" for index in range(10)]

    def benchmark_factory(name: str, seed: int) -> _FakeBenchmark:
        assert name == "MT10"
        return _FakeBenchmark(
            seed,
            bank_token=bank_token,
            vocabulary_prefix=vocabulary_prefix,
        )

    return FailureGenerationComponents(
        benchmark_factory=benchmark_factory,
        policy_loader=lambda _path, _device: _FakeACT(),
        expert_policy_map={name: _FakeExpert for name in names},
        metaworld_version="3.test",
    )


def _args(
    output: Path,
    checkpoint: Path,
    *,
    rollouts: int = 1,
    resume: bool = False,
    overwrite: bool = False,
) -> Namespace:
    return Namespace(
        benchmark="MT10",
        benchmark_seed=31,
        act_checkpoint=str(checkpoint),
        output_dir=str(output),
        rollouts_per_task=rollouts,
        max_steps=500,
        noise_level=0.2,
        action_std_scale=0.40,
        observation_std_scale=0.025,
        disagreement_threshold=0.35,
        prediction_horizon=10,
        terminal_positive_horizon=25,
        seed=17,
        device="cpu",
        log_file=str(output.parent / "failure-generation.log"),
        resume=resume,
        overwrite=overwrite,
    )


def _changed(args: Namespace, **updates: object) -> Namespace:
    values = vars(args).copy()
    values.update(updates)
    return Namespace(**values)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "act.pt"
    path.write_bytes(b"fake-act-checkpoint-v1")
    return path


def test_manifest_fingerprint_inventory_and_resume_paths_are_canonical(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    output = tmp_path / "failures"
    manifest = generate(_args(output, checkpoint), components=_components())

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["dataset_type"] == DATASET_TYPE
    assert len(manifest["provenance_fingerprint_sha256"]) == 64
    assert len(manifest["benchmark_task_bank_sha256"]) == 64
    assert manifest["act_checkpoint_sha256"] == file_sha256(checkpoint)
    assert manifest["episodes"] == 10
    assert len(manifest["files"]) == 10
    assert all(
        item["file"] == f"task_{item['task_id']:02d}/failure_0000.npz"
        for item in manifest["files"]
    )
    original_hashes = {
        item["file"]: item["sha256"] for item in manifest["files"]
    }

    resumed = generate(
        _args(output, checkpoint, resume=True), components=_components()
    )
    assert {
        item["file"]: item["sha256"] for item in resumed["files"]
    } == original_hashes
    # Regression guard: the old implementation accidentally prefixed the
    # output directory name while summarizing resumed shards.
    assert all(
        not item["file"].startswith(f"{output.name}/")
        for item in resumed["files"]
    )
    assert json.loads((output / "manifest.json").read_text()) == resumed


def test_resume_can_only_expand_and_preserves_existing_shards(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    output = tmp_path / "failures"
    first = generate(_args(output, checkpoint), components=_components())
    original_hashes = {
        item["file"]: file_sha256(output / item["file"])
        for item in first["files"]
    }

    resumed = generate(
        _args(output, checkpoint, rollouts=2, resume=True),
        components=_components(),
    )
    assert resumed["episodes"] == 20
    assert all(
        (output / f"task_{task_id:02d}" / "failure_0001.npz").is_file()
        for task_id in range(10)
    )
    for relative, digest in original_hashes.items():
        assert file_sha256(output / relative) == digest

    with pytest.raises(ValueError, match="Cannot decrease rollouts_per_task"):
        generate(
            _args(output, checkpoint, rollouts=1, resume=True),
            components=_components(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_seed", 32),
        ("noise_level", 0.3),
        ("action_std_scale", 0.5),
        ("observation_std_scale", 0.05),
        ("disagreement_threshold", 0.2),
        ("prediction_horizon", 8),
        ("terminal_positive_horizon", 9),
    ],
)
def test_resume_rejects_protocol_fingerprint_changes(
    tmp_path: Path,
    checkpoint: Path,
    field: str,
    value: object,
) -> None:
    output = tmp_path / "failures"
    args = _args(output, checkpoint)
    generate(args, components=_components())

    with pytest.raises(ValueError, match="incompatible provenance"):
        generate(
            _changed(args, resume=True, **{field: value}),
            components=_components(),
        )


def test_resume_rejects_model_bank_and_vocabulary_changes(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    model_output = tmp_path / "model-change"
    generate(_args(model_output, checkpoint), components=_components())
    checkpoint.write_bytes(b"fake-act-checkpoint-v2")
    with pytest.raises(ValueError, match="act_checkpoint_sha256"):
        generate(
            _args(model_output, checkpoint, resume=True),
            components=_components(),
        )

    checkpoint.write_bytes(b"fake-act-checkpoint-v1")
    bank_output = tmp_path / "bank-change"
    generate(_args(bank_output, checkpoint), components=_components())
    with pytest.raises(ValueError, match="benchmark_task_bank_sha256"):
        generate(
            _args(bank_output, checkpoint, resume=True),
            components=_components(bank_token="bank-b"),
        )

    vocabulary_output = tmp_path / "vocab-change"
    generate(_args(vocabulary_output, checkpoint), components=_components())
    with pytest.raises(ValueError, match="task_vocabulary"):
        generate(
            _args(vocabulary_output, checkpoint, resume=True),
            components=_components(vocabulary_prefix="changed-task"),
        )


def test_resume_rejects_tampered_manifest_fingerprint(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    output = tmp_path / "failures"
    generate(_args(output, checkpoint), components=_components())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["benchmark"] = "MT50"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        generate(
            _args(output, checkpoint, resume=True), components=_components()
        )


@pytest.mark.parametrize("damage", ["hash", "missing", "stale"])
def test_resume_verifies_complete_file_inventory_and_hashes(
    tmp_path: Path,
    checkpoint: Path,
    damage: str,
) -> None:
    output = tmp_path / "failures"
    generate(_args(output, checkpoint), components=_components())
    first = output / "task_00" / "failure_0000.npz"
    if damage == "hash":
        first.write_bytes(first.read_bytes() + b"tamper")
        expected = "SHA256 mismatch"
    elif damage == "missing":
        first.unlink()
        expected = "missing shards"
    else:
        shutil.copyfile(first, output / "task_00" / "failure_9999.npz")
        expected = "stale shards"

    with pytest.raises(ValueError, match=expected):
        generate(
            _args(output, checkpoint, resume=True), components=_components()
        )


def test_overwrite_clears_only_owned_shards_and_leaves_no_stale_shard(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    output = tmp_path / "failures"
    generate(
        _args(output, checkpoint, rollouts=2), components=_components()
    )
    stale = output / "task_00" / "failure_9999.npz"
    shutil.copyfile(output / "task_00" / "failure_0000.npz", stale)
    unrelated_root = output / "notes.txt"
    unrelated_task = output / "task_00" / "keep.npz"
    unrelated_root.write_text("keep me", encoding="utf-8")
    unrelated_task.write_bytes(b"not a failure shard")

    replaced = generate(
        _args(output, checkpoint, rollouts=1, overwrite=True),
        components=_components(),
    )
    assert replaced["episodes"] == 10
    assert len(list(output.glob("task_*/failure_*.npz"))) == 10
    assert not stale.exists()
    assert not any(output.glob("task_*/failure_0001.npz"))
    assert unrelated_root.read_text(encoding="utf-8") == "keep me"
    assert unrelated_task.read_bytes() == b"not a failure shard"


def test_overwrite_refuses_foreign_or_unproven_dataset(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    foreign_manifest = foreign / "manifest.json"
    foreign_manifest.write_text(
        json.dumps({"dataset_type": "expert_demonstrations", "files": []}),
        encoding="utf-8",
    )
    foreign_shard = foreign / "task_00" / "failure_0000.npz"
    foreign_shard.parent.mkdir()
    foreign_shard.write_bytes(b"foreign")
    with pytest.raises(FileExistsError, match="Refusing --overwrite"):
        generate(
            _args(foreign, checkpoint, overwrite=True),
            components=_components(),
        )
    assert foreign_manifest.exists() and foreign_shard.exists()

    unproven = tmp_path / "unproven"
    unproven_shard = unproven / "task_00" / "failure_0000.npz"
    unproven_shard.parent.mkdir(parents=True)
    unproven_shard.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="without a manifest"):
        generate(
            _args(unproven, checkpoint, overwrite=True),
            components=_components(),
        )
    assert unproven_shard.exists()


def test_resume_and_overwrite_are_mutually_exclusive(
    tmp_path: Path,
    checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate(
            _args(
                tmp_path / "failures",
                checkpoint,
                resume=True,
                overwrite=True,
            ),
            components=_components(),
        )
