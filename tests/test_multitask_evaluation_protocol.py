"""Protocol-integrity tests for the MT10/MT50 evaluation entrypoint."""

from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import evaluate_multitask as evaluation


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "benchmark": "MT10",
        "condition": "clean",
        "benchmark_seed": 20265010,
        "act_checkpoint": "act.pt",
        "mlp_checkpoint": "mlp.pt",
        "detector_checkpoint": "detector.pt",
        "recovery_checkpoint": "recovery.pt",
        "output_csv": "episodes.csv",
        "output_summary": "summary.json",
        "methods": ("mlp_bc", "act", "heuristic_recovery", "reim"),
        "episodes_per_task": 50,
        "max_steps": 500,
        "noise_level": 0.0,
        "action_std_scale": 0.4,
        "observation_std_scale": 0.025,
        "threshold": 0.3,
        "recovery_budget": 250,
        "heuristic_min_steps": 30,
        "heuristic_window": 20,
        "heuristic_tolerance": 0.01,
        "bootstrap_samples": 100,
        "task_ids": None,
        "seed": 42,
        "device": "cpu",
        "log_file": None,
        "resume": False,
        "allow_partial": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("condition", "   ", "condition"),
        ("episodes_per_task", 0, "episodes-per-task"),
        ("episodes_per_task", 51, "episodes-per-task"),
        ("max_steps", 0, "max-steps"),
        ("max_steps", 501, "max-steps"),
        ("benchmark_seed", -1, "benchmark-seed"),
        ("seed", -1, "seed"),
        ("noise_level", -0.1, "noise-level"),
        ("noise_level", float("nan"), "noise-level"),
        ("action_std_scale", -0.1, "action-std-scale"),
        ("observation_std_scale", -0.1, "observation-std-scale"),
        ("threshold", 1.01, "threshold"),
        ("recovery_budget", 0, "recovery-budget"),
        ("heuristic_min_steps", 500, "heuristic-min-steps"),
        ("heuristic_window", 501, "heuristic-window"),
        ("heuristic_tolerance", -0.01, "heuristic-tolerance"),
        ("bootstrap_samples", 0, "bootstrap-samples"),
        ("task_ids", (0, 0), "duplicates"),
        ("task_ids", (10,), "task-ids"),
        ("methods", ("act", "act"), "duplicates"),
        ("methods", ("unknown",), "Unknown methods"),
    ],
)
def test_protocol_argument_ranges_fail_closed(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation._validate_evaluation_arguments(
            _args(**{field: value}), task_count=10
        )


def test_zero_noise_clean_run_rejects_retry() -> None:
    assert evaluation.build_parser().get_default("methods") == (
        "mlp_bc",
        "act",
        "heuristic_recovery",
        "reim",
    )
    with pytest.raises(ValueError, match="act_retry.*not permitted"):
        evaluation._validate_evaluation_arguments(
            _args(methods=("act", "act_retry")), task_count=10
        )

    condition, methods, task_ids = evaluation._validate_evaluation_arguments(
        _args(
            condition=" noisy ",
            methods=("act_retry", "act"),
            noise_level=0.2,
            task_ids=(7, 2),
        ),
        task_count=10,
    )
    assert condition == "noisy"
    assert methods == ("act_retry", "act")
    assert task_ids == (2, 7)


def test_mlp_checkpoint_is_required_only_when_mlp_is_selected() -> None:
    condition, methods, _ = evaluation._validate_evaluation_arguments(
        _args(methods=("act",), mlp_checkpoint=None), task_count=10
    )
    assert condition == "clean"
    assert methods == ("act",)

    with pytest.raises(ValueError, match="--mlp-checkpoint.*mlp_bc"):
        evaluation._validate_evaluation_arguments(
            _args(methods=("mlp_bc",), mlp_checkpoint=None), task_count=10
        )


BASE_PROTOCOL = {
    "evaluation_schema_version": evaluation.SCHEMA_VERSION,
    "benchmark": "MT10",
    "condition": "clean",
    "noise_level": 0.0,
    "detector_threshold": 0.3,
    "max_episode_steps": 500,
    "task_ids": list(range(10)),
    "methods": ["act", "reim"],
    "checkpoint_sha256": {
        "act": "a" * 64,
        "detector": "b" * 64,
        "recovery": "c" * 64,
    },
    "task_vocabulary": [f"task-{index}" for index in range(10)],
    "episode_seed_base": 42,
    "benchmark_seed": 20265010,
    "task_bank_sha256": "d" * 64,
}


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("condition", "robust"),
        ("noise_level", 0.2),
        ("detector_threshold", 0.4),
        ("max_episode_steps", 499),
        ("task_ids", [0, 1]),
        ("methods", ["act"]),
        (
            "checkpoint_sha256",
            {
                "act": "e" * 64,
                "detector": "b" * 64,
                "recovery": "c" * 64,
            },
        ),
        ("task_vocabulary", ["permuted"] + BASE_PROTOCOL["task_vocabulary"][1:]),
        ("episode_seed_base", 43),
        ("benchmark_seed", 20265011),
        ("task_bank_sha256", "f" * 64),
    ],
)
def test_resume_sidecar_rejects_every_protocol_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    output_csv = tmp_path / "episodes.csv"
    protocol = copy.deepcopy(BASE_PROTOCOL)
    fingerprint, sidecar, rows = evaluation._prepare_run_artifacts(
        output_csv, protocol, resume=False
    )
    assert rows == []
    assert (
        json.loads(sidecar.read_text(encoding="utf-8"))["run_fingerprint"]
        == fingerprint
    )

    same_fingerprint, same_sidecar, same_rows = evaluation._prepare_run_artifacts(
        output_csv, protocol, resume=True
    )
    assert (same_fingerprint, same_sidecar, same_rows) == (
        fingerprint,
        sidecar,
        [],
    )

    changed = copy.deepcopy(protocol)
    changed[field] = replacement
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluation._prepare_run_artifacts(output_csv, changed, resume=True)


def test_nonresume_refuses_existing_artifacts_and_csv_requires_sidecar(
    tmp_path: Path,
) -> None:
    output_csv = tmp_path / "episodes.csv"
    evaluation._prepare_run_artifacts(output_csv, BASE_PROTOCOL, resume=False)
    with pytest.raises(FileExistsError, match="already exist"):
        evaluation._prepare_run_artifacts(output_csv, BASE_PROTOCOL, resume=False)

    orphan_csv = tmp_path / "orphan.csv"
    orphan_csv.write_text("method\nMT-ACT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without immutable run sidecar"):
        evaluation._prepare_run_artifacts(orphan_csv, BASE_PROTOCOL, resume=True)


class _Task:
    def __init__(self, payload: bytes) -> None:
        self.data = payload


def _row_from_specification(specification: dict[str, object]) -> dict[str, object]:
    return {
        "run_fingerprint": specification["run_fingerprint"],
        "benchmark": specification["benchmark"],
        "condition": specification["condition"],
        "task_name": specification["task_name"],
        "task_id": specification["task_id"],
        "task_variant": specification["task_variant"],
        "method": specification["method"],
        "success": 1,
        "intervention_count": 0,
        "recovery_success": 0,
        "steps": 20,
        "paired_episode_id": specification["paired_episode_id"],
        "episode_seed": specification["episode_seed"],
        "task_payload_sha256": specification["task_payload_sha256"],
        "max_failure_probability": 0.0,
        "trigger_step": -1,
        "attempt_count": 1,
    }


def test_resume_rows_are_exactly_scoped_to_current_fingerprint() -> None:
    fingerprint = "1" * 64
    vocabulary = ("task-a", "task-b")
    tasks = {
        "task-a": (_Task(b"a0"), _Task(b"a1")),
        "task-b": (_Task(b"b0"), _Task(b"b1")),
    }
    expected = evaluation._expected_row_specifications(
        benchmark="MT10",
        condition="unit",
        methods=("act", "reim"),
        task_ids=(0,),
        task_vocabulary=vocabulary,
        tasks_by_name=tasks,
        episodes_per_task=1,
        seed=42,
        run_fingerprint=fingerprint,
    )
    rows = [
        _row_from_specification(dict(specification))
        for specification in expected.values()
    ]
    completed = evaluation._validate_protocol_rows(
        rows, expected, run_fingerprint=fingerprint, max_steps=500
    )
    assert {key[0] for key in completed} == {"act", "reim"}

    foreign = copy.deepcopy(rows)
    foreign[0]["run_fingerprint"] = "2" * 64
    with pytest.raises(ValueError, match="another run fingerprint"):
        evaluation._validate_protocol_rows(
            foreign, expected, run_fingerprint=fingerprint, max_steps=500
        )

    changed_condition = copy.deepcopy(rows)
    changed_condition[0]["condition"] = "other"
    with pytest.raises(ValueError, match="condition does not match"):
        evaluation._validate_protocol_rows(
            changed_condition, expected, run_fingerprint=fingerprint, max_steps=500
        )

    with pytest.raises(ValueError, match="duplicate"):
        evaluation._validate_protocol_rows(
            [rows[0], rows[0]],
            expected,
            run_fingerprint=fingerprint,
            max_steps=500,
        )


def _completed(
    methods: tuple[str, ...], task_count: int, episodes_per_task: int
) -> set[tuple[str, int, str]]:
    return {
        (method, task_id, f"{task_id}-{episode}")
        for method in methods
        for task_id in range(task_count)
        for episode in range(episodes_per_task)
    }


def test_official_clean_eligibility_is_per_method_and_fail_closed() -> None:
    methods = ("mlp_bc", "act", "heuristic_recovery", "reim")
    complete = _completed(methods, task_count=10, episodes_per_task=2)
    eligible = evaluation._official_clean_eligibility(
        noise_level=0.0,
        max_steps=500,
        task_ids=tuple(range(10)),
        task_count=10,
        methods=methods,
        episodes_per_task=2,
        completed=complete,
    )
    assert all(item["eligible"] for item in eligible.values())

    partial = evaluation._official_clean_eligibility(
        noise_level=0.0,
        max_steps=500,
        task_ids=(0, 1),
        task_count=10,
        methods=("act",),
        episodes_per_task=2,
        completed={("act", 0, "0-0"), ("act", 0, "0-1")},
    )
    assert not partial["act"]["eligible"]
    assert "partial_task_suite" in partial["act"]["reasons"]
    assert "incomplete_rows" in partial["act"]["reasons"]

    noisy = evaluation._official_clean_eligibility(
        noise_level=0.2,
        max_steps=500,
        task_ids=tuple(range(10)),
        task_count=10,
        methods=("act", "act_retry"),
        episodes_per_task=1,
        completed=_completed(("act", "act_retry"), 10, 1),
    )
    assert not noisy["act"]["eligible"]
    assert not noisy["act_retry"]["eligible"]
    assert "nonzero_noise" in noisy["act"]["reasons"]
    assert "run_contains_retry" in noisy["act"]["reasons"]


class _RolloutEnv:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.steps = 0

    def set_task(self, task: object) -> None:
        del task

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        del seed
        self.steps = 0
        return np.zeros(39, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.steps += 1
        return (
            np.zeros(39, dtype=np.float32),
            0.0,
            False,
            self.steps >= self.horizon,
            {"success": False},
        )


class _Act:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        pass

    def act(self, state: np.ndarray) -> np.ndarray:
        del state
        self.calls += 1
        return np.zeros(4, dtype=np.float32)


class _Recovery:
    def act(self, state: np.ndarray) -> np.ndarray:
        del state
        return np.zeros(4, dtype=np.float32)


class _MLP:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, state: np.ndarray) -> np.ndarray:
        self.states.append(np.asarray(state).copy())
        return np.full(4, 0.25, dtype=np.float32)


def test_mlp_rollout_uses_current_conditioned_state_without_act() -> None:
    max_steps = 3
    act = _Act()
    mlp = _MLP()
    result = evaluation._rollout(
        env=_RolloutEnv(max_steps),
        task=object(),
        task_id=3,
        task_count=10,
        method="mlp_bc",
        episode_seed=42,
        max_steps=max_steps,
        action_noise=np.zeros((max_steps, 4), dtype=np.float32),
        observation_noise=np.zeros((max_steps, 39), dtype=np.float32),
        act=act,
        detector=SimpleNamespace(sequence_length=1, state_dim=49),
        recovery=_Recovery(),
        threshold=0.5,
        recovery_budget=max_steps,
        heuristic_min_steps=1,
        heuristic_window=1,
        heuristic_tolerance=0.0,
        mlp=mlp,  # type: ignore[arg-type]
    )
    assert result["steps"] == max_steps
    assert mlp.reset_calls == 1
    assert len(mlp.states) == max_steps
    assert act.calls == 0
    for state in mlp.states:
        assert state.shape == (49,)
        np.testing.assert_array_equal(state[39:], np.eye(10, dtype=np.float32)[3])


def _valid_mlp_provenance() -> dict[str, object]:
    vocabulary = [f"task-{index}" for index in range(10)]
    return {
        "training_schema": "reim-multitask-mlp-training-v1",
        "benchmark": "MT10",
        "task_vocabulary": vocabulary,
        "task_vocabulary_sha256": evaluation._canonical_sha256(vocabulary),
        "data_manifest_sha256": "a" * 64,
        "data_schema_version": "reim-multitask-demonstrations-v1",
        "dataset_type": "balanced_multitask_scripted_expert_demonstrations",
        "split_sha256": "b" * 64,
    }


def test_mlp_provenance_requires_benchmark_vocab_hash_and_manifest() -> None:
    provenance = _valid_mlp_provenance()
    vocabulary = provenance["task_vocabulary"]
    assert isinstance(vocabulary, list)
    evaluation._validate_multitask_provenance(
        "MLP-BC",
        provenance,
        benchmark="MT10",
        task_vocabulary=vocabulary,
        require_mlp_manifest=True,
    )

    corruptions = {
        "benchmark": "MT50",
        "task_vocabulary": list(reversed(vocabulary)),
        "task_vocabulary_sha256": "c" * 64,
        "data_manifest_sha256": None,
        "data_schema_version": "unknown",
        "dataset_type": "unknown",
        "split_sha256": "short",
    }
    for field, value in corruptions.items():
        invalid = dict(provenance)
        invalid[field] = value
        with pytest.raises(ValueError):
            evaluation._validate_multitask_provenance(
                "MLP-BC",
                invalid,
                benchmark="MT10",
                task_vocabulary=vocabulary,
                require_mlp_manifest=True,
            )


def test_run_protocol_fingerprints_only_supplied_checkpoint_set() -> None:
    vocabulary = [f"task-{index}" for index in range(10)]
    common = {
        "args": _args(methods=("mlp_bc", "act")),
        "condition": "clean",
        "methods": ("mlp_bc", "act"),
        "task_ids": tuple(range(10)),
        "task_vocabulary": vocabulary,
        "task_bank_sha256": "d" * 64,
        "metaworld_version": "3.0.0",
        "execution_device": "cpu",
    }
    with_mlp = evaluation._build_run_protocol(
        **common,
        checkpoint_sha256={"act": "a" * 64, "mlp_bc": "b" * 64},
    )
    assert with_mlp["checkpoint_sha256"] == {
        "act": "a" * 64,
        "mlp_bc": "b" * 64,
    }
    changed = copy.deepcopy(with_mlp)
    changed["checkpoint_sha256"]["mlp_bc"] = "c" * 64
    assert evaluation._canonical_sha256(with_mlp) != evaluation._canonical_sha256(
        changed
    )


def test_rollout_preserves_first_trigger_across_repeated_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluation, "_risk", lambda detector, history: 1.0)
    max_steps = 4
    result = evaluation._rollout(
        env=_RolloutEnv(max_steps),
        task=object(),
        task_id=0,
        task_count=10,
        method="reim",
        episode_seed=42,
        max_steps=max_steps,
        action_noise=np.zeros((max_steps, 4), dtype=np.float32),
        observation_noise=np.zeros((max_steps, 39), dtype=np.float32),
        act=_Act(),
        detector=SimpleNamespace(sequence_length=1, state_dim=49),
        recovery=_Recovery(),
        threshold=0.5,
        recovery_budget=1,
        heuristic_min_steps=1,
        heuristic_window=1,
        heuristic_tolerance=0.0,
    )
    assert result["intervention_count"] == max_steps
    assert result["trigger_step"] == 0


def test_reim_hysteresis_releases_recovery_back_to_act(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = iter((1.0, 0.1, 0.1, 0.1))
    monkeypatch.setattr(
        evaluation, "_risk", lambda detector, history: next(probabilities)
    )
    max_steps = 4
    result = evaluation._rollout(
        env=_RolloutEnv(max_steps),
        task=object(),
        task_id=0,
        task_count=10,
        method="reim",
        episode_seed=42,
        max_steps=max_steps,
        action_noise=np.zeros((max_steps, 4), dtype=np.float32),
        observation_noise=np.zeros((max_steps, 39), dtype=np.float32),
        act=_Act(),
        detector=SimpleNamespace(sequence_length=1, state_dim=49),
        recovery=_Recovery(),
        threshold=0.5,
        release_threshold=0.2,
        release_patience=2,
        min_recovery_steps=1,
        intervention_cooldown=10,
        recovery_budget=4,
        heuristic_min_steps=1,
        heuristic_window=1,
        heuristic_tolerance=0.0,
    )
    assert result["intervention_count"] == 1
    assert result["recovery_success"] == 1
    assert result["trigger_step"] == 0
