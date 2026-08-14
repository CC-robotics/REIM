"""Tests for the explicit deterministic toy MT10/MT50 CI backend."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from env import toy_multitask as toy
from env.metaworld_multitask import REIMMetaWorldMultiTaskEnv


def _rollout(env, expert, task, *, seed: int, max_steps: int = 500):
    env.set_task(task)
    raw, info = env.reset(seed=seed)
    states = [raw.copy()]
    successes = []
    steps = 0
    for steps in range(1, max_steps + 1):
        action = expert.get_action(raw)
        raw, reward, terminated, truncated, info = env.step(action)
        states.append(raw.copy())
        successes.append(bool(info["success"]))
        if info["success"] or terminated or truncated:
            break
    return np.stack(states), any(successes), steps


def test_toy_benchmark_banks_match_official_shape() -> None:
    for name, expected in (("MT10", 10), ("MT50", 50)):
        benchmark = getattr(toy, name)(seed=0)
        assert len(benchmark.train_classes) == expected
        assert len(benchmark.train_tasks) == expected * 50
        assert not benchmark.test_tasks
        assert not benchmark.test_classes
        per_name: dict[str, int] = {}
        for task in benchmark.train_tasks:
            assert isinstance(bytes(task.data), bytes)
            per_name[task.env_name] = per_name.get(task.env_name, 0) + 1
        assert set(per_name) == set(benchmark.train_classes)
        assert all(count == 50 for count in per_name.values())
        payloads = {bytes(task.data) for task in benchmark.train_tasks}
        assert len(payloads) == expected * 50


def test_toy_task_payloads_track_the_benchmark_seed() -> None:
    first = toy.MT10(seed=0).train_tasks
    same = toy.MT10(seed=0).train_tasks
    other = toy.MT10(seed=99).train_tasks
    assert [bytes(task.data) for task in first] == [
        bytes(task.data) for task in same
    ]
    # Different registered bank seeds must expose disjoint payload identities,
    # mirroring Meta-World's per-seed goal sampling so the bank-separation
    # audit has genuine separation to verify.
    first_payloads = {bytes(task.data) for task in first}
    other_payloads = {bytes(task.data) for task in other}
    assert first_payloads.isdisjoint(other_payloads)
    # Task names and dynamics anchors stay seed-independent.
    assert [task.env_name for task in first] == [task.env_name for task in other]


@pytest.mark.parametrize("task_name", toy.MT10_TASKS)
def test_toy_expert_succeeds_on_every_mt10_task(task_name: str) -> None:
    benchmark = toy.MT10(seed=0)
    env = benchmark.train_classes[task_name](render_mode=None)
    expert = toy.ENV_POLICY_MAP[task_name]()
    variants = [task for task in benchmark.train_tasks if task.env_name == task_name]
    try:
        for variant in range(3):
            _, success, steps = _rollout(
                env, expert, variants[variant], seed=1000 + variant
            )
            assert success, f"{task_name} variant {variant} did not succeed"
            assert steps < 500
    finally:
        env.close()


def test_toy_rollout_is_deterministic() -> None:
    benchmark = toy.MT10(seed=0)
    task = next(
        task for task in benchmark.train_tasks if task.env_name == "pick-place-v3"
    )
    env = benchmark.train_classes["pick-place-v3"]()
    expert = toy.ENV_POLICY_MAP["pick-place-v3"]()
    first, success_a, _ = _rollout(env, expert, task, seed=7)
    second, success_b, _ = _rollout(env, expert, task, seed=7)
    assert success_a and success_b
    assert np.array_equal(first, second)
    third, _, _ = _rollout(env, expert, task, seed=8)
    assert not np.array_equal(first, third)


def test_toy_env_contract_and_failure_modes() -> None:
    benchmark = toy.MT10(seed=0)
    env = benchmark.train_classes["reach-v3"]()
    assert env.observation_space.shape == (39,)
    assert env.action_space.shape == (4,)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))
    task = next(task for task in benchmark.train_tasks if task.env_name == "reach-v3")
    env.set_task(task)
    raw, info = env.reset(seed=5)
    assert raw.shape == (39,) and np.all(np.isfinite(raw))
    assert info["success"] is False
    with pytest.raises(ValueError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step(np.array([0.0, 0.0, np.nan, 0.0]))
    other = next(
        task for task in benchmark.train_tasks if task.env_name == "push-v3"
    )
    with pytest.raises(ValueError):
        env.set_task(other)
    env.close()


def test_wrapper_toy_backend_preserves_one_hot_semantics() -> None:
    env = REIMMetaWorldMultiTaskEnv(
        "MT10", backend="toy", task_id="pick-place-v3", seed=3
    )
    assert env.state_dim == 49
    obs, info = env.reset(seed=11)
    assert obs.shape == (49,)
    assert obs[39:].sum() == 1.0
    assert obs[39 + 2] == 1.0
    metadata = info["task_metadata"]
    assert metadata["backend"] == "toy"
    assert metadata["metaworld_version"] == toy.TOY_VERSION
    assert hashlib.sha256(env.current_task.data).hexdigest() == metadata["task_sha256"]
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(env.get_expert_action())
        if info["success"] or terminated or truncated:
            break
    assert info["success"]
    env.select_task("reach-v3", 5)
    obs, info = env.reset()
    assert obs[39] == 1.0
    assert info["task_name"] == "reach-v3"
    env.close()


def test_wrapper_requires_explicit_toy_backend() -> None:
    with pytest.raises(ValueError):
        REIMMetaWorldMultiTaskEnv("MT10", backend="auto")
    pytest.importorskip("gymnasium")
    try:
        import metaworld  # noqa: F401
    except ImportError:
        # Without Meta-World installed, the default backend must fail loudly
        # rather than silently falling back to the toy backend.
        with pytest.raises(RuntimeError, match="Meta-World 3.1.1 is required"):
            REIMMetaWorldMultiTaskEnv("MT10")
