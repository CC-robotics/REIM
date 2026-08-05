"""Deterministic resume/provenance tests for multi-task recovery collection."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict, namedtuple
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from data.io import file_sha256
import scripts.collect_multitask_recovery as collector


FakeTask = namedtuple("FakeTask", ["env_name", "data", "variant"])


class _Probability:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray([0.9], dtype=np.float32)


class _FakeACT:
    state_dim = 49
    action_dim = 4

    def __init__(self) -> None:
        vocabulary = [f"task-{index}-v3" for index in range(10)]
        self.provenance = {
            "benchmark": "MT10",
            "task_vocabulary": vocabulary,
            "task_vocabulary_sha256": collector._task_vocabulary_sha256(vocabulary),
        }

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray) -> np.ndarray:
        assert state.shape == (49,)
        return np.zeros(4, dtype=np.float32)


class _FakeDetector:
    state_dim = 49
    sequence_length = 2

    def __init__(self) -> None:
        vocabulary = [f"task-{index}-v3" for index in range(10)]
        self.provenance = {
            "benchmark": "MT10",
            "task_vocabulary": vocabulary,
            "task_vocabulary_sha256": collector._task_vocabulary_sha256(vocabulary),
        }

    def predict_proba(self, windows: np.ndarray, lengths: np.ndarray) -> _Probability:
        assert windows.shape == (1, 2, 49)
        assert lengths.shape == (1,)
        return _Probability()


class _FakeExpert:
    def get_action(self, raw: np.ndarray) -> np.ndarray:
        assert raw.shape == (39,)
        return np.asarray([2.0, -2.0, 0.25, 3.0], dtype=np.float32)


def _environment_type(task_id: int, task_name: str) -> type:
    class FakeEnvironment:
        def __init__(self, render_mode=None) -> None:
            assert render_mode is None
            self.task = None
            self.last_seed = None

        def set_task(self, task: FakeTask) -> None:
            assert task.env_name == task_name
            self.task = task

        def seed(self, seed: int) -> list[int]:
            self.last_seed = seed
            return [seed]

        def reset(self, seed: int | None = None):
            assert seed == self.last_seed
            assert self.task is not None
            raw = np.full(39, task_id, dtype=np.float32)
            raw[-1] = self.task.variant
            return raw, {}

        def step(self, action: np.ndarray):
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
            assert self.task is not None
            # Even variants fail; odd variants succeed. Every requested shard
            # therefore consumes exactly two deterministic attempts.
            success = bool(self.task.variant % 2)
            raw = np.full(39, task_id + 0.1, dtype=np.float32)
            return raw, float(success), False, True, {"success": success}

        def close(self) -> None:
            return None

    FakeEnvironment.__name__ = f"FakeRecoveryEnvironment{task_id}"
    return FakeEnvironment


class _FakeBenchmark:
    def __init__(self, seed: int, *, reverse: bool = False) -> None:
        names = [f"task-{index}-v3" for index in range(10)]
        if reverse:
            names.reverse()
        self.train_classes = OrderedDict(
            (name, _environment_type(index, name)) for index, name in enumerate(names)
        )
        self.train_tasks = [
            FakeTask(
                name,
                f"bank={seed};task={name};variant={variant}".encode("ascii"),
                variant,
            )
            for name in names
            for variant in range(50)
        ]


def _components(*, reverse: bool = False) -> collector.MetaWorldComponents:
    module = SimpleNamespace(MT10=lambda seed: _FakeBenchmark(seed, reverse=reverse))
    names = [f"task-{index}-v3" for index in range(10)]
    return collector.MetaWorldComponents(
        module=module,
        policy_map={name: _FakeExpert for name in names},
        version="3.test",
    )


def _args(
    root: Path,
    *,
    target: int = 1,
    resume: bool = False,
    threshold: float = 0.2,
    benchmark_seed: int = 111,
    noise_level: float = 0.2,
) -> Namespace:
    return Namespace(
        benchmark="MT10",
        benchmark_seed=benchmark_seed,
        act_checkpoint=str(root / "act.pt"),
        detector_checkpoint=str(root / "detector.pt"),
        output_dir=str(root / "recovery"),
        target_per_task=target,
        max_attempts_multiplier=3,
        max_steps=1,
        threshold=threshold,
        noise_level=noise_level,
        action_std_scale=0.4,
        observation_std_scale=0.025,
        seed=42,
        device="cpu",
        log_file=str(root / "collector.log"),
        resume=resume,
        overwrite=False,
    )


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        collector.ACTPolicy,
        "from_checkpoint",
        staticmethod(lambda *_args, **_kwargs: _FakeACT()),
    )
    monkeypatch.setattr(
        collector.FailureDetector,
        "from_checkpoint",
        staticmethod(lambda *_args, **_kwargs: _FakeDetector()),
    )


def _prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "act.pt").write_bytes(b"frozen-act")
    (root / "detector.pt").write_bytes(b"frozen-detector")


def _semantic_entries(manifest: dict) -> list[dict]:
    return [
        {key: value for key, value in entry.items() if key != "sha256"}
        for entry in manifest["files"]
    ]


def test_shards_bind_contiguous_cursor_seed_hash_and_onehot(
    tmp_path: Path,
    fake_models,
) -> None:
    _prepare(tmp_path)
    args = _args(tmp_path)
    manifest = collector.collect(args, components=_components())

    assert manifest["complete"] is True
    assert manifest["successful_continuations"] == 10
    assert manifest["attempts"] == 20
    assert manifest["detector_triggers"] == 20
    assert manifest["protocol"]["task_bank_sha256"] == manifest["task_bank_sha256"]
    assert manifest["protocol"]["act_checkpoint_sha256"] == file_sha256(
        tmp_path / "act.pt"
    )
    assert manifest["protocol"]["detector_checkpoint_sha256"] == file_sha256(
        tmp_path / "detector.pt"
    )
    assert len(manifest["protocol_fingerprint_sha256"]) == 64

    for task_id, task_name in enumerate(manifest["task_vocabulary"]):
        progress = manifest["collection_progress"][task_name]
        assert progress == {
            "task_id": task_id,
            "next_attempt_index": 2,
            "attempts": 2,
            "detector_triggers": 2,
            "successful_continuations": 1,
        }
        path = tmp_path / "recovery" / f"task_{task_id:02d}" / "recovery_0000.npz"
        entry = next(item for item in manifest["files"] if item["task_id"] == task_id)
        assert entry["file"] == f"task_{task_id:02d}/recovery_0000.npz"
        assert entry["shard_index"] == 0
        assert entry["attempt_index"] == 1
        assert entry["task_variant"] == 1
        assert entry["sha256"] == file_sha256(path)
        assert len(entry["content_sha256"]) == 64
        with np.load(path, allow_pickle=False) as shard:
            assert int(shard["attempt_index"]) == 1
            assert int(shard["shard_index"]) == 0
            assert int(shard["episode_seed"]) == collector._episode_seed(42, task_id, 1)
            assert (
                str(shard["protocol_fingerprint_sha256"])
                == manifest["protocol_fingerprint_sha256"]
            )
            states = np.asarray(shard["states"])
            assert states.shape == (1, 49)
            expected = np.zeros(10, dtype=np.float32)
            expected[task_id] = 1.0
            np.testing.assert_array_equal(states[0, 39:], expected)
            np.testing.assert_array_equal(shard["actions"][0], [1.0, -1.0, 0.25, 1.0])

    completed_resume = collector.collect(
        _args(tmp_path, resume=True), components=_components()
    )
    assert completed_resume == manifest


def test_resume_matches_uninterrupted_and_preserves_committed_shards(
    tmp_path: Path,
    fake_models,
) -> None:
    resumed_root = tmp_path / "resumed"
    full_root = tmp_path / "full"
    _prepare(resumed_root)
    _prepare(full_root)
    interrupted_args = _args(resumed_root, target=2)

    class StopCollection(RuntimeError):
        pass

    calls = 0

    def stop_after_seven(_event) -> None:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise StopCollection("simulated process interruption")

    with pytest.raises(StopCollection):
        collector.collect(
            interrupted_args,
            components=_components(),
            attempt_hook=stop_after_seven,
        )
    partial_manifest = __import__("json").loads(
        (resumed_root / "recovery" / "manifest.json").read_text(encoding="utf-8")
    )
    committed_hashes = {
        entry["file"]: file_sha256(resumed_root / "recovery" / entry["file"])
        for entry in partial_manifest["files"]
    }

    resumed_args = _args(resumed_root, target=2, resume=True)
    resumed = collector.collect(resumed_args, components=_components())
    uninterrupted = collector.collect(
        _args(full_root, target=2), components=_components()
    )

    assert (
        resumed["protocol_fingerprint_sha256"]
        == uninterrupted["protocol_fingerprint_sha256"]
    )
    assert resumed["collection_progress"] == uninterrupted["collection_progress"]
    assert resumed["per_task"] == uninterrupted["per_task"]
    assert _semantic_entries(resumed) == _semantic_entries(uninterrupted)
    for relative, digest in committed_hashes.items():
        assert file_sha256(resumed_root / "recovery" / relative) == digest
    for left, right in zip(resumed["files"], uninterrupted["files"]):
        assert left["content_sha256"] == right["content_sha256"]
        with (
            np.load(resumed_root / "recovery" / left["file"]) as a,
            np.load(full_root / "recovery" / right["file"]) as b,
        ):
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_resume_rejects_shrink_and_any_fingerprinted_parameter_change(
    tmp_path: Path,
    fake_models,
) -> None:
    root = tmp_path / "strict"
    _prepare(root)
    collector.collect(_args(root, target=2), components=_components())

    with pytest.raises(ValueError, match="shrink"):
        collector.collect(_args(root, target=1, resume=True), components=_components())
    with pytest.raises(ValueError, match="target_per_task"):
        collector.collect(_args(root, target=3, resume=True), components=_components())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.collect(
            _args(root, target=2, resume=True, threshold=0.25),
            components=_components(),
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.collect(
            _args(root, target=2, resume=True, noise_level=0.3),
            components=_components(),
        )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.collect(
            _args(root, target=2, resume=True, benchmark_seed=222),
            components=_components(),
        )
    with pytest.raises(ValueError, match="task vocabulary is incompatible"):
        collector.collect(
            _args(root, target=2, resume=True),
            components=_components(reverse=True),
        )


def test_resume_rejects_changed_checkpoint_and_stale_or_tampered_shard(
    tmp_path: Path,
    fake_models,
) -> None:
    changed_checkpoint_root = tmp_path / "checkpoint"
    _prepare(changed_checkpoint_root)
    collector.collect(_args(changed_checkpoint_root), components=_components())
    (changed_checkpoint_root / "act.pt").write_bytes(b"changed-act")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.collect(
            _args(changed_checkpoint_root, resume=True), components=_components()
        )
    (changed_checkpoint_root / "act.pt").write_bytes(b"frozen-act")
    (changed_checkpoint_root / "detector.pt").write_bytes(b"changed-detector")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.collect(
            _args(changed_checkpoint_root, resume=True), components=_components()
        )

    stale_root = tmp_path / "stale"
    _prepare(stale_root)
    collector.collect(_args(stale_root), components=_components())
    source = stale_root / "recovery" / "task_00" / "recovery_0000.npz"
    shutil.copyfile(source, source.with_name("recovery_9999.npz"))
    with pytest.raises(ValueError, match="Stale/missing"):
        collector.collect(_args(stale_root, resume=True), components=_components())

    tampered_root = tmp_path / "tampered"
    _prepare(tampered_root)
    collector.collect(_args(tampered_root), components=_components())
    tampered = tampered_root / "recovery" / "task_00" / "recovery_0000.npz"
    with tampered.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="file hash mismatch"):
        collector.collect(_args(tampered_root, resume=True), components=_components())


@pytest.mark.parametrize("failing_manifest_commit", [2, 3])
def test_resume_recovers_atomic_attempt_transaction(
    tmp_path: Path,
    fake_models,
    monkeypatch: pytest.MonkeyPatch,
    failing_manifest_commit: int,
) -> None:
    """Resume covers crashes both before and after the successful shard write."""

    resumed_root = tmp_path / f"resumed-{failing_manifest_commit}"
    full_root = tmp_path / f"full-{failing_manifest_commit}"
    _prepare(resumed_root)
    _prepare(full_root)
    real_write = collector.atomic_write_json
    manifest_commits = 0

    def interrupted_write(path: Path, payload):
        nonlocal manifest_commits
        if Path(path).name == "manifest.json":
            manifest_commits += 1
            if manifest_commits == failing_manifest_commit:
                raise RuntimeError("simulated crash between transaction and manifest")
        return real_write(path, payload)

    monkeypatch.setattr(collector, "atomic_write_json", interrupted_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        collector.collect(_args(resumed_root), components=_components())
    assert (resumed_root / "recovery" / collector.PENDING_ATTEMPT_FILENAME).is_file()

    monkeypatch.setattr(collector, "atomic_write_json", real_write)
    resumed = collector.collect(
        _args(resumed_root, resume=True), components=_components()
    )
    uninterrupted = collector.collect(_args(full_root), components=_components())
    assert resumed["collection_progress"] == uninterrupted["collection_progress"]
    assert resumed["per_task"] == uninterrupted["per_task"]
    assert _semantic_entries(resumed) == _semantic_entries(uninterrupted)
    assert not (resumed_root / "recovery" / collector.PENDING_ATTEMPT_FILENAME).exists()
    for left, right in zip(resumed["files"], uninterrupted["files"]):
        assert left["content_sha256"] == right["content_sha256"]


def test_trigger_aligned_targets_use_the_same_pre_action_state() -> None:
    class SequenceProbability(_Probability):
        def __init__(self, value: float) -> None:
            self.value = value

        def numpy(self) -> np.ndarray:
            return np.asarray([self.value], dtype=np.float32)

    class SequenceDetector(_FakeDetector):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def predict_proba(self, windows, lengths):
            value = 0.1 if self.calls == 0 else 0.9
            self.calls += 1
            return SequenceProbability(value)

    class EncodingExpert:
        def get_action(self, raw: np.ndarray) -> np.ndarray:
            return np.asarray([raw[0] / 10.0, 0.0, 0.0, 1.0], dtype=np.float32)

    class TemporalEnvironment:
        def set_task(self, task) -> None:
            self.task = task

        def seed(self, seed: int) -> None:
            self.last_seed = seed

        def reset(self, seed=None):
            self.step_index = 0
            return np.zeros(39, dtype=np.float32), {}

        def step(self, action: np.ndarray):
            self.step_index += 1
            raw = np.zeros(39, dtype=np.float32)
            raw[0] = self.step_index
            success = self.step_index == 3
            return raw, float(action[0]), False, False, {"success": success}

    result = collector._run_attempt(
        env=TemporalEnvironment(),
        task=object(),
        expert=EncodingExpert(),
        act=_FakeACT(),
        detector=SequenceDetector(),
        task_id=2,
        task_count=10,
        episode_seed=123,
        max_steps=3,
        threshold=0.5,
        action_std=0.0,
        observation_std=0.0,
    )
    assert result["triggered"] is True
    assert result["trigger_step"] == 1
    payload = result["payload"]
    assert payload is not None
    np.testing.assert_array_equal(payload["raw_observations"][:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(payload["states"][:, 0], [1.0, 2.0])
    np.testing.assert_allclose(payload["actions"][:, 0], [0.1, 0.2])
    np.testing.assert_allclose(payload["rewards"], [0.1, 0.2])


def test_resume_finalizes_journal_left_after_manifest_commit(
    tmp_path: Path, fake_models
) -> None:
    _prepare(tmp_path)
    manifest = collector.collect(_args(tmp_path), components=_components())
    task_id = 9
    task_name = manifest["task_vocabulary"][task_id]
    entry = next(item for item in manifest["files"] if int(item["task_id"]) == task_id)
    pending = {
        "schema_version": collector.PENDING_ATTEMPT_SCHEMA_VERSION,
        "protocol_fingerprint_sha256": manifest["protocol_fingerprint_sha256"],
        "task_id": task_id,
        "task_name": task_name,
        "task_variant": int(entry["task_variant"]),
        "episode_seed": int(entry["episode_seed"]),
        "attempt_index": int(entry["attempt_index"]),
        "shard_index": int(entry["shard_index"]),
        "previous_progress": {
            "task_id": task_id,
            "next_attempt_index": 1,
            "attempts": 1,
            "detector_triggers": 1,
            "successful_continuations": 0,
        },
        "triggered": True,
        "saved": True,
        "file": entry["file"],
        "content_sha256": entry["content_sha256"],
    }
    pending_path = tmp_path / "recovery" / collector.PENDING_ATTEMPT_FILENAME
    collector.atomic_write_json(pending_path, pending)

    resumed = collector.collect(_args(tmp_path, resume=True), components=_components())
    assert resumed == manifest
    assert not pending_path.exists()
