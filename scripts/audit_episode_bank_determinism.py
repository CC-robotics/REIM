#!/usr/bin/env python3
"""Audit exact CRN replay across constructors and nominal shard boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.metaworld_pickplace import REIMPickPlaceEnv  # noqa: E402
from evaluation.episode_bank import (  # noqa: E402
    bank_file_sha256,
    load_episode_bank,
    runtime_episode_specifications,
)
from utils.common import atomic_json_dump  # noqa: E402


def _array_sha256(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype=np.float32)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _default_indices(episodes: int) -> list[int]:
    candidates = {
        0,
        episodes - 1,
        episodes // 4 - 1,
        episodes // 4,
        episodes // 2 - 1,
        episodes // 2,
        3 * episodes // 4 - 1,
        3 * episodes // 4,
    }
    return sorted(index for index in candidates if 0 <= index < episodes)


def _rollout(
    *,
    bank: dict[str, Any],
    specification: dict[str, Any],
    constructor_seed: int,
    env_config: Path,
    state_mode: str,
    steps: int,
) -> dict[str, Any]:
    disturbance = bank["disturbance"]
    env = REIMPickPlaceEnv(
        config=env_config,
        backend=str(bank["backend"]),
        seed=constructor_seed,
        state_mode=state_mode,
        max_episode_steps=int(bank["max_steps"]),
        **disturbance,
    )
    observations: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    clean_states: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    goal_positions: list[np.ndarray] = []
    disturbance_deltas: list[np.ndarray] = []
    try:
        observation, info = env.reset(
            seed=int(specification["episode_seed"]),
            options={"reim_episode_spec": specification},
        )
        observations.append(observation.copy())
        clean_states.append(env.get_state(noisy=False))
        object_positions.append(np.asarray(info["object_position"], dtype=np.float32))
        goal_positions.append(np.asarray(info["goal_position"], dtype=np.float32))
        for step in range(steps):
            command = np.asarray(
                [
                    0.12 * np.sin(0.31 * step),
                    -0.08 * np.cos(0.23 * step),
                    0.04,
                    1.0 if step % 3 else -1.0,
                ],
                dtype=np.float32,
            )
            observation, _, terminated, truncated, info = env.step(command)
            observations.append(observation.copy())
            clean_states.append(env.get_state(noisy=False))
            executed_actions.append(
                np.asarray(info["executed_action"], dtype=np.float32)
            )
            object_positions.append(
                np.asarray(info["object_position"], dtype=np.float32)
            )
            goal_positions.append(
                np.asarray(info["goal_position"], dtype=np.float32)
            )
            disturbance_deltas.append(
                np.asarray(
                    info.get(
                        "object_disturbance_delta",
                        np.zeros(3, dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "constructor_seed": int(constructor_seed),
        "steps_executed": len(executed_actions),
        "observation_sha256": _array_sha256(observations),
        "clean_state_sha256": _array_sha256(clean_states),
        "executed_action_sha256": _array_sha256(executed_actions),
        "object_position_sha256": _array_sha256(object_positions),
        "goal_position_sha256": _array_sha256(goal_positions),
        "disturbance_delta_sha256": _array_sha256(disturbance_deltas),
        "_arrays": {
            "observations": observations,
            "clean_states": clean_states,
            "executed_actions": executed_actions,
            "object_positions": object_positions,
            "goal_positions": goal_positions,
            "disturbance_deltas": disturbance_deltas,
        },
    }


def _exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["steps_executed"] != right["steps_executed"]:
        return False
    for key in left["_arrays"]:
        left_values = left["_arrays"][key]
        right_values = right["_arrays"][key]
        if len(left_values) != len(right_values):
            return False
        if any(
            not np.array_equal(a, b)
            for a, b in zip(left_values, right_values)
        ):
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-bank", type=Path, required=True)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=ROOT / "configs" / "environment.yaml",
    )
    parser.add_argument("--state-mode", choices=("semantic", "raw"), default="semantic")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--constructor-seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "tables"
        / "episode_bank_determinism_audit.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    bank = load_episode_bank(args.episode_bank)
    episodes = int(bank["episodes"])
    indices = (
        _default_indices(episodes)
        if args.episode_indices is None
        else sorted(set(args.episode_indices))
    )
    if not indices or min(indices) < 0 or max(indices) >= episodes:
        raise ValueError("episode indices are outside the bank")
    constructor_seeds = (
        [
            int(bank["task_bank_seed"]),
            int(bank["task_bank_seed"]) + max(1, episodes // 4),
        ]
        if args.constructor_seeds is None
        else args.constructor_seeds
    )
    if len(set(constructor_seeds)) < 2:
        raise ValueError("at least two distinct constructor seeds are required")

    episode_results: list[dict[str, Any]] = []
    all_exact = True
    for index in indices:
        specification = runtime_episode_specifications(
            bank,
            offset=index,
            count=1,
        )[0]
        constructor_results = [
            _rollout(
                bank=bank,
                specification=specification,
                constructor_seed=constructor_seed,
                env_config=args.env_config,
                state_mode=args.state_mode,
                steps=args.steps,
            )
            for constructor_seed in constructor_seeds
        ]
        reference = constructor_results[0]
        exact = all(
            _exact(reference, candidate)
            for candidate in constructor_results[1:]
        )
        all_exact = all_exact and exact
        for result in constructor_results:
            result.pop("_arrays")
        episode_results.append(
            {
                "episode_index": int(specification["episode_index"]),
                "episode_seed": int(specification["episode_seed"]),
                "episode_specification_sha256": specification[
                    "specification_sha256"
                ],
                "task_sha256": specification["task_sha256"],
                "scheduled_object_displacement_step": specification[
                    "object_disturbance_step"
                ],
                "scheduled_object_displacement_xyz_m": specification[
                    "object_disturbance_delta"
                ],
                "constructors_exact": exact,
                "constructors": constructor_results,
            }
        )

    payload = {
        "audit": "episode_bank_cross_constructor_and_shard_determinism",
        "exact_equality_required": True,
        "all_checks_passed": all_exact,
        "episode_bank": str(args.episode_bank.resolve()),
        "episode_bank_file_sha256": bank_file_sha256(args.episode_bank),
        "episode_bank_sha256": bank["bank_sha256"],
        "backend": bank["backend"],
        "env_name": bank["env_name"],
        "state_mode": args.state_mode,
        "constructor_seeds": constructor_seeds,
        "nominal_shard_boundary_indices": indices,
        "steps_per_episode": args.steps,
        "episodes": episode_results,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))
    if not all_exact:
        raise RuntimeError("episode-bank exact replay audit failed")
    return payload


if __name__ == "__main__":
    main()
