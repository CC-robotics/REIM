#!/usr/bin/env python3
"""Audit strict ACT/REIM equivalence when REIM never intervenes.

This is a fail-closed common-random-number (CRN) audit.  It deliberately
constructs the ACT and REIM Meta-World environments with *different*
constructor seeds, then resets both from the same persistent episode-bank
specification.  REIM receives a detector that always returns zero and a
recovery policy that raises if called.  Therefore, every successful audit
episode establishes all of the following:

* the serialized task/reset/disturbance specification overrides constructor
  and shard history;
* detector evaluation alone does not change the ACT trajectory; and
* with no intervention, REIM is bitwise identical to ACT in commanded actions,
  observed states, success, and executed step count.

The complete arrays are compared in memory as contiguous float32 byte strings.
The JSON artifact stores their shapes and SHA256 digests rather than duplicating
potentially hundreds of megabytes of trajectory data.  ``--include-arrays`` is
available for small forensic audits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.metaworld_pickplace import REIMPickPlaceEnv  # noqa: E402
from evaluation.episode_bank import (  # noqa: E402
    bank_file_sha256,
    load_episode_bank,
    runtime_episode_specifications,
)
from evaluation.evaluate_reim import (  # noqa: E402
    ControllerConfig,
    REIMController,
    load_bc_policy,
    run_episode,
    seed_everything,
)


LOGGER = logging.getLogger("reim.audit.no_intervention")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    PROJECT_ROOT
    / "datasets/evaluation/pickplace_crn_seed6000042_n1000_noise020.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/audits/no_intervention_equivalence.json"
)
AUDIT_SCHEMA_VERSION = 1


class ZeroFailureDetector:
    """Detector test double with a visible call count and exactly zero risk."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        state_sequence: np.ndarray,
        valid_length: int | None = None,
    ) -> float:
        del state_sequence, valid_length
        self.calls += 1
        return 0.0


class ForbiddenRecoveryPolicy:
    """Recovery test double that turns any intervention into a hard failure."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        self.calls += 1
        raise AssertionError(
            "recovery policy was called during a no-intervention equivalence audit"
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_array(value: Any, *, width: int, name: str) -> np.ndarray:
    """Return a finite, C-contiguous ``[T, width]`` float32 trajectory."""

    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        array = np.empty((0, width), dtype=np.float32)
    elif array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [T, {width}], received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and exact contiguous bytes of an ndarray."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"REIM-array-v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            list(contiguous.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(adapter: Any) -> str:
    """Hash the two independently loaded ACT parameter/buffer dictionaries."""

    policy = getattr(adapter, "policy", adapter)
    if not hasattr(policy, "state_dict"):
        raise TypeError("loaded ACT policy does not expose state_dict()")
    digest = hashlib.sha256()
    digest.update(b"REIM-ACT-state-dict-v1\0")
    state_dict = policy.state_dict()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _first_bitwise_mismatch(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any] | None:
    if left.shape != right.shape:
        return {
            "kind": "shape",
            "act_shape": list(left.shape),
            "reim_shape": list(right.shape),
        }
    left_bytes = left.view(np.uint8).reshape(left.shape + (left.itemsize,))
    right_bytes = right.view(np.uint8).reshape(right.shape + (right.itemsize,))
    different_elements = np.any(left_bytes != right_bytes, axis=-1)
    positions = np.argwhere(different_elements)
    if positions.size == 0:
        return None
    index = tuple(int(value) for value in positions[0])
    return {
        "kind": "value",
        "index": list(index),
        "act_value": float(left[index]),
        "reim_value": float(right[index]),
        "act_float32_bits": f"0x{int(left[index].view(np.uint32)):08x}",
        "reim_float32_bits": f"0x{int(right[index].view(np.uint32)):08x}",
    }


def _trajectory_record(
    *,
    trace: Mapping[str, Any],
    state_dim: int,
    action_dim: int,
    include_arrays: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    states = _float32_array(trace["states"], width=state_dim, name="states")
    actions = _float32_array(
        trace["actions"],
        width=action_dim,
        name="commanded actions",
    )
    sources = [str(value) for value in trace["sources"]]
    if len(states) != len(actions) + 1:
        raise ValueError(
            "trajectory must contain one initial state plus one state per action"
        )
    if len(sources) != len(actions):
        raise ValueError("control-source count differs from action count")
    source_bytes = json.dumps(
        sources,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    record: dict[str, Any] = {
        "success": bool(trace["success"]),
        "steps": int(len(actions)),
        "state_shape": list(states.shape),
        "commanded_action_shape": list(actions.shape),
        "state_sha256": array_sha256(states),
        "commanded_action_sha256": array_sha256(actions),
        "control_sources_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "control_sources_unique": sorted(set(sources)),
        "only_act_control": all(source == "bc" for source in sources),
    }
    if include_arrays:
        record["states_float32"] = states.tolist()
        record["commanded_actions_float32"] = actions.tolist()
        record["control_sources"] = sources
    return record, states, actions


def compare_episode_traces(
    *,
    act_trace: Mapping[str, Any],
    reim_trace: Mapping[str, Any],
    state_dim: int,
    action_dim: int,
    include_arrays: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Bitwise-compare complete trajectories and return JSON-safe evidence."""

    act_record, act_states, act_actions = _trajectory_record(
        trace=act_trace,
        state_dim=state_dim,
        action_dim=action_dim,
        include_arrays=include_arrays,
    )
    reim_record, reim_states, reim_actions = _trajectory_record(
        trace=reim_trace,
        state_dim=state_dim,
        action_dim=action_dim,
        include_arrays=include_arrays,
    )
    state_mismatch = _first_bitwise_mismatch(act_states, reim_states)
    action_mismatch = _first_bitwise_mismatch(act_actions, reim_actions)
    success_equal = act_record["success"] == reim_record["success"]
    steps_equal = act_record["steps"] == reim_record["steps"]
    states_equal = state_mismatch is None
    actions_equal = action_mismatch is None
    sources_equal = (
        act_record["control_sources_sha256"]
        == reim_record["control_sources_sha256"]
    )
    only_act_control = bool(
        act_record["only_act_control"] and reim_record["only_act_control"]
    )
    checks = {
        "success_equal": success_equal,
        "steps_equal": steps_equal,
        "states_bitwise_equal": states_equal,
        "commanded_actions_bitwise_equal": actions_equal,
        "control_sources_equal": sources_equal,
        "only_act_control": only_act_control,
    }
    evidence: dict[str, Any] = {
        "act": act_record,
        "reim_no_intervention": reim_record,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if state_mismatch is not None:
        evidence["first_state_mismatch"] = state_mismatch
    if action_mismatch is not None:
        evidence["first_commanded_action_mismatch"] = action_mismatch
    return evidence, bool(evidence["passed"])


def _make_bank_env(
    *,
    bank: Mapping[str, Any],
    constructor_seed: int,
    env_config: str | Path | None,
) -> REIMPickPlaceEnv:
    disturbance = bank["disturbance"]
    return REIMPickPlaceEnv(
        config=env_config,
        seed=int(constructor_seed),
        backend=str(bank["backend"]),
        env_name=str(bank["env_name"]),
        max_episode_steps=int(bank["max_steps"]),
        action_noise_std=float(disturbance["action_noise_std"]),
        observation_noise_std=float(disturbance["observation_noise_std"]),
        object_noise_probability=float(
            disturbance["object_noise_probability"]
        ),
        object_noise_std=float(disturbance["object_noise_std"]),
        object_noise_magnitude=float(
            disturbance["object_noise_magnitude"]
        ),
    )


def _atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def run_audit(
    *,
    episode_bank: str | Path,
    act_checkpoint: str | Path,
    output: str | Path,
    offset: int,
    count: int | None,
    constructor_seed_act: int,
    constructor_seed_reim: int,
    device: str,
    env_config: str | Path | None,
    include_arrays: bool = False,
    torch_threads: int = 1,
) -> dict[str, Any]:
    """Run and persist a strict CRN/no-intervention equivalence audit."""

    bank_path = Path(episode_bank).expanduser().resolve()
    checkpoint_path = Path(act_checkpoint).expanduser().resolve()
    bank = load_episode_bank(bank_path)
    if count is None:
        count = int(bank["episodes"]) - int(offset)
    if count <= 0:
        raise ValueError("count must be positive")
    specifications = runtime_episode_specifications(
        bank,
        offset=int(offset),
        count=int(count),
    )
    if constructor_seed_act == constructor_seed_reim:
        raise ValueError(
            "constructor seeds must differ; the audit is intended to prove "
            "episode specifications override constructor state"
        )

    # CPU is the publication/default path.  When a caller explicitly selects a
    # GPU, request deterministic kernels and still fail on any bit difference.
    seed_everything(int(bank["task_bank_seed"]))
    try:
        import torch

        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(torch_threads)
        torch.use_deterministic_algorithms(True)
    except ImportError:  # load_bc_policy will provide the actionable error.
        pass

    act_env: REIMPickPlaceEnv | None = None
    reim_env: REIMPickPlaceEnv | None = None
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_type": "strict_no_intervention_equivalence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "passed": False,
        "configuration": {
            "episode_bank": str(bank_path),
            "episode_bank_file_sha256": bank_file_sha256(bank_path),
            "episode_bank_sha256": str(bank["bank_sha256"]),
            "backend": str(bank["backend"]),
            "env_name": str(bank["env_name"]),
            "episode_offset": int(offset),
            "episode_count": int(count),
            "max_steps": int(bank["max_steps"]),
            "constructor_seed_act": int(constructor_seed_act),
            "constructor_seed_reim": int(constructor_seed_reim),
            "constructor_seeds_deliberately_different": True,
            "act_checkpoint": str(checkpoint_path),
            "act_checkpoint_sha256": _file_sha256(checkpoint_path),
            "device": str(device),
            "torch_threads": int(torch_threads),
            "comparison": (
                "complete finite C-contiguous float32 trajectories compared "
                "byte-for-byte; success and executed steps compared exactly"
            ),
            "arrays_embedded_in_json": bool(include_arrays),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "per_episode": [],
    }
    try:
        act_env = _make_bank_env(
            bank=bank,
            constructor_seed=constructor_seed_act,
            env_config=env_config,
        )
        reim_env = _make_bank_env(
            bank=bank,
            constructor_seed=constructor_seed_reim,
            env_config=env_config,
        )
        state_dim = int(np.prod(act_env.observation_space.shape))
        action_dim = int(np.prod(act_env.action_space.shape))
        if tuple(act_env.observation_space.shape) != tuple(
            reim_env.observation_space.shape
        ):
            raise ValueError("ACT and REIM environment state dimensions differ")
        if tuple(act_env.action_space.shape) != tuple(
            reim_env.action_space.shape
        ):
            raise ValueError("ACT and REIM environment action dimensions differ")

        # This must remain two separate deserializations.  Sharing one ACT
        # object would allow its temporal ensemble to leak between controllers.
        act_policy = load_bc_policy(
            checkpoint_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        reim_act_policy = load_bc_policy(
            checkpoint_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        act_model_sha256 = _state_dict_sha256(act_policy)
        reim_model_sha256 = _state_dict_sha256(reim_act_policy)
        report["configuration"].update(
            {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "act_loaded_model_sha256": act_model_sha256,
                "reim_loaded_model_sha256": reim_model_sha256,
                "independent_model_loads_equal": (
                    act_model_sha256 == reim_model_sha256
                ),
            }
        )
        if act_model_sha256 != reim_model_sha256:
            raise AssertionError("independently loaded ACT models differ")

        zero_detector = ZeroFailureDetector()
        forbidden_recovery = ForbiddenRecoveryPolicy()
        controller_config = ControllerConfig(
            failure_threshold=0.5,
            recovery_exit_threshold=0.0,
            sequence_length=10,
        )
        counters = {
            "success_equal": 0,
            "steps_equal": 0,
            "states_bitwise_equal": 0,
            "commanded_actions_bitwise_equal": 0,
            "control_sources_equal": 0,
            "only_act_control": 0,
            "passed": 0,
        }

        for specification in specifications:
            episode_index = int(specification["episode_index"])
            episode_seed = int(specification["episode_seed"])
            act_controller = REIMController("bc", act_policy)
            reim_controller = REIMController(
                "reim",
                reim_act_policy,
                detector=zero_detector,
                recovery_policy=forbidden_recovery,
                config=controller_config,
            )
            act_metrics, act_trace = run_episode(
                act_env,
                act_controller,
                episode=episode_index,
                seed=episode_seed,
                max_steps=int(bank["max_steps"]),
                capture_trace=True,
                episode_specification=specification,
            )
            reim_metrics, reim_trace = run_episode(
                reim_env,
                reim_controller,
                episode=episode_index,
                seed=episode_seed,
                max_steps=int(bank["max_steps"]),
                capture_trace=True,
                episode_specification=specification,
            )
            if act_trace is None or reim_trace is None:  # Defensive invariant.
                raise RuntimeError("run_episode did not return requested traces")
            evidence, passed = compare_episode_traces(
                act_trace=act_trace,
                reim_trace=reim_trace,
                state_dim=state_dim,
                action_dim=action_dim,
                include_arrays=include_arrays,
            )
            # Metrics are independently checked so a future trace regression
            # cannot silently weaken the audit.
            metrics_checks = {
                "metrics_success_equal": (
                    bool(act_metrics.success) == bool(reim_metrics.success)
                ),
                "metrics_steps_equal": (
                    int(act_metrics.steps) == int(reim_metrics.steps)
                ),
                "reim_recovery_attempts_zero": (
                    int(reim_metrics.recovery_attempts) == 0
                ),
                "reim_detector_triggers_zero": (
                    int(reim_metrics.detector_triggers) == 0
                ),
                "specification_hash_equal": (
                    act_metrics.episode_specification_sha256
                    == reim_metrics.episode_specification_sha256
                    == str(specification["specification_sha256"])
                ),
                "task_hash_equal": (
                    act_metrics.metaworld_task_sha256
                    == reim_metrics.metaworld_task_sha256
                    == str(specification["task_sha256"])
                ),
            }
            passed = bool(passed and all(metrics_checks.values()))
            evidence.update(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "episode_specification_sha256": str(
                        specification["specification_sha256"]
                    ),
                    "metaworld_task_sha256": str(
                        specification["task_sha256"]
                    ),
                    "metrics_checks": metrics_checks,
                    "passed": passed,
                }
            )
            report["per_episode"].append(evidence)
            for name, value in evidence["checks"].items():
                counters[name] += int(bool(value))
            counters["passed"] += int(passed)

        aggregate_passed = (
            counters["passed"] == len(specifications)
            and forbidden_recovery.calls == 0
            and all(
                len(entry["reim_no_intervention"]["control_sources_unique"])
                <= 1
                and entry["reim_no_intervention"]["only_act_control"]
                for entry in report["per_episode"]
            )
        )
        report["aggregate"] = {
            "episodes_expected": len(specifications),
            "episodes_audited": len(report["per_episode"]),
            **{f"{name}_episodes": value for name, value in counters.items()},
            "zero_detector_calls": zero_detector.calls,
            "forbidden_recovery_calls": forbidden_recovery.calls,
            "all_episode_specification_hashes_unique": (
                len(
                    {
                        entry["episode_specification_sha256"]
                        for entry in report["per_episode"]
                    }
                )
                == len(specifications)
            ),
        }
        report["passed"] = bool(aggregate_passed)
        report["status"] = "passed" if aggregate_passed else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["passed"] = False
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _atomic_write_json(report, output)
        raise
    finally:
        if act_env is not None:
            act_env.close()
        if reim_env is not None:
            reim_env.close()

    _atomic_write_json(report, output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--act-checkpoint",
        "--checkpoint",
        dest="act_checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/bc_policy.pt",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of bank episodes to audit; use 0 for the complete bank.",
    )
    parser.add_argument(
        "--constructor-seed-act",
        type=int,
        default=6_000_042,
    )
    parser.add_argument(
        "--constructor-seed-reim",
        type=int,
        default=9_000_091,
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="CPU is the publication audit default for bitwise reproducibility.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="PyTorch intra-op CPU threads.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs/environment.yaml",
    )
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="Embed full float32 arrays in JSON (intended only for small audits).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    count = None if args.count == 0 else args.count
    if count is not None and count <= 0:
        raise ValueError("count must be positive or zero for the complete bank")
    try:
        report = run_audit(
            episode_bank=args.episode_bank,
            act_checkpoint=args.act_checkpoint,
            output=args.output,
            offset=args.offset,
            count=count,
            constructor_seed_act=args.constructor_seed_act,
            constructor_seed_reim=args.constructor_seed_reim,
            device=args.device,
            env_config=args.env_config,
            include_arrays=args.include_arrays,
            torch_threads=args.torch_threads,
        )
    except Exception:
        LOGGER.exception(
            "No-intervention equivalence audit errored; evidence was written to %s",
            args.output,
        )
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": report["status"],
                "passed": report["passed"],
                "episode_bank_sha256": report["configuration"][
                    "episode_bank_sha256"
                ],
                "episodes_audited": report["aggregate"]["episodes_audited"],
                "state_equal_episodes": report["aggregate"][
                    "states_bitwise_equal_episodes"
                ],
                "commanded_action_equal_episodes": report["aggregate"][
                    "commanded_actions_bitwise_equal_episodes"
                ],
                "forbidden_recovery_calls": report["aggregate"][
                    "forbidden_recovery_calls"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
