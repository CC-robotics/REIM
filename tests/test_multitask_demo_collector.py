"""Protocol tests for the direct Meta-World MT10/MT50 demo collector."""

from __future__ import annotations

import json
from argparse import Namespace
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import pytest

from data.io import file_sha256
from scripts.collect_multitask_demonstrations import (
    ACTION_DIM,
    DATASET_TYPE,
    RAW_OBSERVATION_DIM,
    MetaWorldComponents,
    build_parser,
    collect,
    task_vocabulary_sha256,
)


FakeTask = namedtuple("FakeTask", ["env_name", "data"])


class _ClippedExpert:
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == (RAW_OBSERVATION_DIM,)
        return np.asarray([2.0, -3.0, 0.25, 5.0], dtype=np.float32)


def _environment_type(task_id: int, task_name: str) -> type:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.task = None
            self.episode_number = 0
            self.step_number = 0
            self.closed = False

        def set_task(self, task: FakeTask) -> None:
            assert task.env_name == task_name
            self.task = task

        def seed(self, seed: int) -> list[int]:
            self.last_seed = seed
            return [seed]

        def reset(self, seed: int | None = None):
            assert seed == self.last_seed
            assert self.task is not None
            self.episode_number += 1
            self.step_number = 0
            raw = np.full(RAW_OBSERVATION_DIM, task_id, dtype=np.float32)
            raw[-1] = int(self.task.data)
            return raw, {}

        def step(self, action: np.ndarray):
            self.step_number += 1
            assert action.shape == (ACTION_DIM,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
            raw = np.full(
                RAW_OBSERVATION_DIM,
                task_id + self.step_number / 100.0,
                dtype=np.float32,
            )
            # Force one discarded rollout for task 0, proving collect-until-success.
            success = task_id != 0 or self.episode_number >= 2
            return raw, float(success), False, True, {"success": success}

        def close(self) -> None:
            self.closed = True

    FakeEnvironment.__name__ = f"Fake{task_id}Environment"
    return FakeEnvironment


class _FakeMT10:
    def __init__(self, seed: int) -> None:
        del seed
        names = [f"task-{index}-v3" for index in range(10)]
        self.train_classes = OrderedDict(
            (name, _environment_type(index, name))
            for index, name in enumerate(names)
        )
        self.train_tasks = [
            FakeTask(name, variant)
            for name in names
            for variant in range(50)
        ]


class _FakeMetaWorld:
    MT10 = _FakeMT10


def _components() -> MetaWorldComponents:
    names = [f"task-{index}-v3" for index in range(10)]
    return MetaWorldComponents(
        module=_FakeMetaWorld,
        policy_map={name: _ClippedExpert for name in names},
        version="3.test",
    )


def _args(output: Path, *, episodes: int = 1, resume: bool = False) -> Namespace:
    return Namespace(
        benchmark="MT10",
        episodes_per_task=episodes,
        output=str(output),
        seed=17,
        max_attempts=3,
        max_steps=500,
        resume=resume,
    )


def test_balanced_collect_until_success_schema_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "mt10"
    manifest = collect(_args(output), components=_components())

    files = sorted(output.glob("mt10_task*_trajectory_*.npz"))
    assert len(files) == 10
    assert manifest["dataset_type"] == DATASET_TYPE
    assert manifest["complete"] is True
    assert manifest["metaworld_version"] == "3.test"
    assert manifest["task_vocabulary"] == [
        f"task-{index}-v3" for index in range(10)
    ]
    assert manifest["task_vocabulary_sha256"] == task_vocabulary_sha256(
        manifest["task_vocabulary"]
    )
    assert manifest["statistics"]["successful_trajectories"] == 10
    assert manifest["statistics"]["discarded_failed_attempts"] == 1
    assert manifest["per_task_yield"]["task-0-v3"]["attempts"] == 2
    assert (
        manifest["per_task_yield"]["task-0-v3"]["successful_trajectories"]
        == 1
    )
    assert all(
        item["successful_trajectories"] == 1
        for item in manifest["per_task_yield"].values()
    )

    with np.load(files[0], allow_pickle=False) as trajectory:
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
        }
        assert required.issubset(trajectory.files)
        states = np.asarray(trajectory["states"])
        raw = np.asarray(trajectory["raw_observations"])
        actions = np.asarray(trajectory["actions"])
        assert states.shape == (1, RAW_OBSERVATION_DIM + 10)
        assert raw.shape == (1, RAW_OBSERVATION_DIM)
        assert actions.shape == (1, ACTION_DIM)
        np.testing.assert_array_equal(states[:, :RAW_OBSERVATION_DIM], raw)
        np.testing.assert_array_equal(states[0, RAW_OBSERVATION_DIM:], np.eye(10)[0])
        np.testing.assert_array_equal(actions[0], [1.0, -1.0, 0.25, 1.0])
        assert bool(trajectory["success"])
        assert int(trajectory["task_id"]) == 0
        assert str(trajectory["task_name"]) == "task-0-v3"
        assert int(trajectory["task_variant"]) == 1
        assert int(trajectory["seed"]) >= 0

    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_resume_extends_without_rewriting_existing_files(tmp_path: Path) -> None:
    output = tmp_path / "mt10"
    first = collect(_args(output, episodes=1), components=_components())
    original_hashes = {
        item["file"]: file_sha256(output / item["file"])
        for item in first["trajectories"]
    }

    with pytest.raises(FileExistsError, match="--resume"):
        collect(_args(output, episodes=2), components=_components())

    resumed = collect(
        _args(output, episodes=2, resume=True),
        components=_components(),
    )
    assert resumed["complete"] is True
    assert resumed["statistics"]["successful_trajectories"] == 20
    assert all(
        item["successful_trajectories"] == 2
        for item in resumed["per_task_yield"].values()
    )
    for filename, digest in original_hashes.items():
        assert file_sha256(output / filename) == digest


def test_refuses_to_overwrite_legacy_pickplace_collection(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    output.mkdir()
    legacy = output / "trajectory_00000.npz"
    np.savez_compressed(legacy, states=np.zeros((1, 21), dtype=np.float32))
    original = legacy.read_bytes()

    with pytest.raises(FileExistsError, match="legacy PickPlace"):
        collect(_args(output), components=_components())
    assert legacy.read_bytes() == original
    assert not (output / "manifest.json").exists()


def test_cli_exposes_required_multitask_arguments() -> None:
    parsed = build_parser().parse_args(
        [
            "--benchmark",
            "MT50",
            "--episodes-per-task",
            "7",
            "--output",
            "somewhere",
            "--seed",
            "9",
            "--max-attempts",
            "4",
        ]
    )
    assert parsed.benchmark == "MT50"
    assert parsed.episodes_per_task == 7
    assert parsed.output == "somewhere"
    assert parsed.seed == 9
    assert parsed.max_attempts == 4


@pytest.mark.parametrize("episodes", [0, 501])
def test_episodes_per_task_is_bounded(episodes: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="episodes-per-task"):
        collect(_args(tmp_path / "invalid", episodes=episodes), components=_components())
