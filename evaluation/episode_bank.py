"""Persistent common-random-number episode specifications for REIM evaluation.

Meta-World's ``RandomTaskSelectWrapper`` samples a task *before* forwarding
``reset(seed=...)`` to the underlying task.  Consequently, an episode seed
alone does not identify the initial object/goal configuration when evaluation
is split across independently constructed shards.  This module makes every
source of episode randomness explicit:

* one serialized Meta-World task payload from a fixed task bank;
* the underlying environment reset seed;
* stateless per-step action- and observation-noise stream seeds; and
* an explicit object-disturbance step and XYZ delta.

The resulting JSON bank is portable across processes and shard boundaries.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
BANK_TYPE = "reim_pickplace_common_random_numbers"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA256 of a JSON-compatible mapping."""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _derive_seed(master_seed: int, episode_seed: int, namespace: str) -> int:
    material = f"REIM-CRN-v1|{master_seed}|{episode_seed}|{namespace}".encode(
        "ascii"
    )
    # NumPy accepts arbitrary Python integers, but a signed 63-bit value also
    # round-trips safely through common JSON and dataframe tooling.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _serialize_metaworld_tasks(
    *,
    env_name: str,
    task_bank_seed: int,
) -> list[dict[str, Any]]:
    try:
        import metaworld
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Meta-World is required to generate a task bank") from exc

    benchmark = metaworld.MT1(env_name, seed=task_bank_seed)
    records: list[dict[str, Any]] = []
    for task_index, task in enumerate(benchmark.train_tasks):
        data = bytes(task.data)
        records.append(
            {
                "task_index": task_index,
                "env_name": str(task.env_name),
                "data_base64": base64.b64encode(data).decode("ascii"),
                "data_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError("Meta-World returned an empty MT1 task bank")
    return records


def _object_schedule(
    *,
    seed: int,
    max_steps: int,
    probability: float,
    std: float,
    magnitude: float,
) -> tuple[int | None, list[float]]:
    if probability <= 0.0 or (std <= 0.0 and magnitude <= 0.0):
        return None, [0.0, 0.0, 0.0]
    rng = np.random.default_rng(seed)
    disturbance_step: int | None = None
    # REIMPickPlaceEnv checks the schedule before executing the action whose
    # zero-based index equals ``step`` and allows object-position displacements
    # from step 2 onward.
    for step in range(2, max_steps):
        if float(rng.random()) < probability:
            disturbance_step = step
            break
    if disturbance_step is None:
        return None, [0.0, 0.0, 0.0]
    if std > 0.0:
        delta = rng.normal(0.0, std, size=3)
    else:
        delta = np.asarray(
            [
                rng.uniform(-magnitude, magnitude),
                rng.uniform(-magnitude, magnitude),
                rng.uniform(-0.2 * magnitude, 0.2 * magnitude),
            ],
            dtype=np.float64,
        )
    return disturbance_step, [float(value) for value in delta]


def _build_specification(
    *,
    task_bank_seed: int,
    episode_index: int,
    episode_seed: int,
    tasks: Sequence[Mapping[str, Any]],
    max_steps: int,
    object_noise_probability: float,
    object_noise_std: float,
    object_noise_magnitude: float,
    namespace: str,
) -> dict[str, Any]:
    """Build one self-contained deterministic reset/disturbance specification."""

    task_count = len(tasks) if tasks else 1
    task_rng = np.random.default_rng(
        _derive_seed(task_bank_seed, episode_seed, f"{namespace}:task")
    )
    task_index = int(task_rng.integers(0, task_count))
    task_sha256 = str(tasks[task_index]["data_sha256"]) if tasks else "toy"
    object_seed = _derive_seed(
        task_bank_seed,
        episode_seed,
        f"{namespace}:object-disturbance",
    )
    disturbance_step, disturbance_delta = _object_schedule(
        seed=object_seed,
        max_steps=max_steps,
        probability=object_noise_probability,
        std=object_noise_std,
        magnitude=object_noise_magnitude,
    )
    core = {
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "reset_seed": int(episode_seed),
        "task_index": task_index,
        "task_sha256": task_sha256,
        "action_noise_seed": _derive_seed(
            task_bank_seed,
            episode_seed,
            f"{namespace}:action-noise",
        ),
        "observation_noise_seed": _derive_seed(
            task_bank_seed,
            episode_seed,
            f"{namespace}:observation-noise",
        ),
        "object_disturbance_seed": object_seed,
        "object_disturbance_step": disturbance_step,
        "object_disturbance_delta": disturbance_delta,
    }
    return {**core, "specification_sha256": payload_sha256(core)}


def build_episode_bank(
    *,
    backend: str,
    env_name: str,
    task_bank_seed: int,
    episode_seed_start: int,
    episodes: int,
    max_steps: int,
    action_noise_std: float,
    observation_noise_std: float,
    object_noise_probability: float,
    object_noise_std: float,
    object_noise_magnitude: float,
    retries_per_episode: int = 1,
) -> dict[str, Any]:
    """Construct and validate a portable episode-bank payload."""

    backend = str(backend).lower()
    if backend not in {"metaworld", "toy"}:
        raise ValueError("backend must be metaworld or toy")
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if retries_per_episode < 0:
        raise ValueError("retries_per_episode must be non-negative")
    for name, value in (
        ("action_noise_std", action_noise_std),
        ("observation_noise_std", observation_noise_std),
        ("object_noise_std", object_noise_std),
        ("object_noise_magnitude", object_noise_magnitude),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 <= object_noise_probability <= 1.0:
        raise ValueError("object_noise_probability must lie in [0, 1]")

    tasks = (
        _serialize_metaworld_tasks(
            env_name=env_name,
            task_bank_seed=task_bank_seed,
        )
        if backend == "metaworld"
        else []
    )
    specifications: list[dict[str, Any]] = []
    retry_specifications: list[list[dict[str, Any]]] = []
    for episode_index in range(episodes):
        episode_seed = int(episode_seed_start + episode_index)
        specifications.append(
            _build_specification(
                task_bank_seed=task_bank_seed,
                episode_index=episode_index,
                episode_seed=episode_seed,
                tasks=tasks,
                max_steps=max_steps,
                object_noise_probability=object_noise_probability,
                object_noise_std=object_noise_std,
                object_noise_magnitude=object_noise_magnitude,
                namespace="primary",
            )
        )
        retries: list[dict[str, Any]] = []
        for retry_index in range(retries_per_episode):
            retry_seed = _derive_seed(
                task_bank_seed,
                episode_seed,
                f"retry:{retry_index}:reset",
            )
            retries.append(
                _build_specification(
                    task_bank_seed=task_bank_seed,
                    episode_index=episode_index,
                    episode_seed=retry_seed,
                    tasks=tasks,
                    max_steps=max_steps,
                    object_noise_probability=object_noise_probability,
                    object_noise_std=object_noise_std,
                    object_noise_magnitude=object_noise_magnitude,
                    namespace=f"retry:{retry_index}",
                )
            )
        retry_specifications.append(retries)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bank_type": BANK_TYPE,
        "backend": backend,
        "env_name": env_name,
        "task_bank_seed": int(task_bank_seed),
        "episode_seed_start": int(episode_seed_start),
        "episode_seed_end": int(episode_seed_start + episodes - 1),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "retries_per_episode": int(retries_per_episode),
        "disturbance": {
            "action_noise_std": float(action_noise_std),
            "observation_noise_std": float(observation_noise_std),
            "object_noise_probability": float(object_noise_probability),
            "object_noise_std": float(object_noise_std),
            "object_noise_magnitude": float(object_noise_magnitude),
        },
        "tasks": tasks,
        "episode_specifications": specifications,
        "retry_specifications": retry_specifications,
    }
    payload["bank_sha256"] = payload_sha256(payload)
    validate_episode_bank(payload)
    return payload


def validate_episode_bank(payload: Mapping[str, Any]) -> None:
    """Fail closed when any task, specification, or manifest hash differs."""

    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported episode-bank schema")
    if payload.get("bank_type") != BANK_TYPE:
        raise ValueError("unexpected episode-bank type")
    recorded_bank_sha256 = str(payload.get("bank_sha256", ""))
    core = {key: value for key, value in payload.items() if key != "bank_sha256"}
    if payload_sha256(core) != recorded_bank_sha256:
        raise ValueError("episode-bank SHA256 mismatch")

    backend = str(payload.get("backend", "")).lower()
    tasks = payload.get("tasks")
    specifications = payload.get("episode_specifications")
    retry_specifications = payload.get("retry_specifications")
    if (
        not isinstance(tasks, list)
        or not isinstance(specifications, list)
        or not isinstance(retry_specifications, list)
    ):
        raise ValueError(
            "episode bank must contain task/primary/retry specification lists"
        )
    if backend == "metaworld" and not tasks:
        raise ValueError("Meta-World episode bank has no serialized tasks")
    if backend == "toy" and tasks:
        raise ValueError("toy episode bank must not contain Meta-World tasks")
    if len(specifications) != int(payload.get("episodes", -1)):
        raise ValueError("episode specification count mismatch")
    retries_per_episode = int(payload.get("retries_per_episode", -1))
    if retries_per_episode < 0:
        raise ValueError("invalid retries_per_episode")
    if len(retry_specifications) != len(specifications):
        raise ValueError("retry specification episode count mismatch")

    for expected_index, task in enumerate(tasks):
        if int(task.get("task_index", -1)) != expected_index:
            raise ValueError("task indices are not contiguous")
        try:
            data = base64.b64decode(task["data_base64"], validate=True)
        except Exception as exc:
            raise ValueError("invalid task base64 payload") from exc
        if hashlib.sha256(data).hexdigest() != task.get("data_sha256"):
            raise ValueError("serialized Meta-World task SHA256 mismatch")

    seed_start = int(payload.get("episode_seed_start", -1))
    seen_spec_hashes: set[str] = set()

    def validate_specification(
        specification: Mapping[str, Any],
        *,
        expected_index: int,
        expected_seed: int | None,
    ) -> None:
        spec = dict(specification)
        recorded = str(spec.pop("specification_sha256", ""))
        if payload_sha256(spec) != recorded:
            raise ValueError("episode specification SHA256 mismatch")
        if recorded in seen_spec_hashes:
            raise ValueError("duplicate episode specification SHA256")
        seen_spec_hashes.add(recorded)
        if int(spec.get("episode_index", -1)) != expected_index:
            raise ValueError("episode indices are not contiguous")
        episode_seed = int(spec.get("episode_seed", -1))
        if expected_seed is not None and episode_seed != expected_seed:
            raise ValueError("episode seeds are not contiguous")
        if int(spec.get("reset_seed", -1)) != episode_seed:
            raise ValueError("episode and reset seeds differ")
        task_index = int(spec.get("task_index", -1))
        if not 0 <= task_index < (len(tasks) if tasks else 1):
            raise ValueError("episode specification has invalid task index")
        expected_task_sha = (
            tasks[task_index]["data_sha256"] if tasks else "toy"
        )
        if spec.get("task_sha256") != expected_task_sha:
            raise ValueError("episode specification task SHA256 mismatch")

    for expected_index, specification in enumerate(specifications):
        validate_specification(
            specification,
            expected_index=expected_index,
            expected_seed=seed_start + expected_index,
        )
        retries = retry_specifications[expected_index]
        if not isinstance(retries, list) or len(retries) != retries_per_episode:
            raise ValueError("retry specification count mismatch")
        for retry in retries:
            validate_specification(
                retry,
                expected_index=expected_index,
                expected_seed=None,
            )


def save_episode_bank(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically persist a validated bank."""

    validate_episode_bank(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_episode_bank(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("episode bank must be a JSON object")
    validate_episode_bank(payload)
    return payload


def runtime_episode_specifications(
    payload: Mapping[str, Any],
    *,
    offset: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    """Return validated runtime specs, including the referenced task payload."""

    validate_episode_bank(payload)
    specifications = payload["episode_specifications"]
    if offset < 0 or offset > len(specifications):
        raise ValueError("episode-bank offset is out of range")
    stop = len(specifications) if count is None else offset + int(count)
    if stop < offset or stop > len(specifications):
        raise ValueError("requested episode-bank slice is out of range")
    tasks = payload["tasks"]
    disturbance = payload["disturbance"]
    retry_specifications = payload["retry_specifications"]

    def as_runtime(specification: Mapping[str, Any]) -> dict[str, Any]:
        task_index = int(specification["task_index"])
        return {
            **copy.deepcopy(specification),
            "schema_version": SCHEMA_VERSION,
            "bank_sha256": payload["bank_sha256"],
            "bank_type": BANK_TYPE,
            "backend": payload["backend"],
            "env_name": payload["env_name"],
            "max_steps": payload["max_steps"],
            "disturbance": copy.deepcopy(disturbance),
            "task": copy.deepcopy(tasks[task_index]) if tasks else None,
        }

    result: list[dict[str, Any]] = []
    for absolute_index in range(offset, stop):
        runtime = as_runtime(specifications[absolute_index])
        runtime["retry_specifications"] = [
            as_runtime(retry)
            for retry in retry_specifications[absolute_index]
        ]
        result.append(runtime)
    return result


def bank_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_spec_map(
    specifications: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for specification in specifications:
        seed = int(specification["episode_seed"])
        if seed in result:
            raise ValueError(f"duplicate episode seed in specification set: {seed}")
        result[seed] = specification
    return result
