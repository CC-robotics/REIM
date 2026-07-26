#!/usr/bin/env python3
"""Export and audit the zero-PPO-step recovery actor as standalone PyTorch.

The command refuses checkpoints with any PPO interaction/update history. It
copies only observation preprocessing, the policy MLP, the action head, and
action bounds, then compares the reloaded artifact against the SB3 source on
every train/validation demonstration state and on a deterministic random bank.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.imitation_recovery_policy import (  # noqa: E402
    ImitationRecoveryPolicy,
)
from utils.common import (  # noqa: E402
    atomic_json_dump,
    configure_logging,
    load_yaml,
    resolve_path,
)


LOGGER = configure_logging("export_imitation_recovery")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _verified_path(section: Mapping[str, Any], key: str) -> Path:
    path = resolve_path(section[key])
    if not path.is_file():
        raise FileNotFoundError(f"Required export input is missing: {path}")
    expected = str(section[f"{key}_sha256"])
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _load_demo_states(path: Path, expected_dim: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
        if schema != "reim-recovery-starts-v1":
            raise ValueError(f"Unsupported recovery-start schema {schema!r}.")
        states = np.asarray(archive["demo_states"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != expected_dim:
        raise ValueError(
            f"{path}: expected demonstration states (*, {expected_dim}), "
            f"got {states.shape}."
        )
    if not np.isfinite(states).all():
        raise ValueError(f"{path}: demonstration states contain NaN or Inf.")
    return states


def _tensor_hash(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        tensor = values[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _extract_policy(
    source: Any,
    *,
    expected_model: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[ImitationRecoveryPolicy, dict[str, Any]]:
    policy = source.policy
    if bool(policy.squash_output):
        raise ValueError("Squashed SB3 actions are not supported by this exporter.")
    if policy.features_extractor.__class__.__name__ != "FlattenExtractor":
        raise ValueError("Only identity FlattenExtractor observations are supported.")
    if getattr(source, "_vec_normalize_env", None) is not None:
        raise ValueError("A source with VecNormalize state cannot be exported.")
    observation_shape = tuple(int(value) for value in policy.observation_space.shape)
    action_shape = tuple(int(value) for value in policy.action_space.shape)
    if len(observation_shape) != 1 or len(action_shape) != 1:
        raise ValueError("Only flat vector observation/action spaces are supported.")

    state_dim = observation_shape[0]
    action_dim = action_shape[0]
    hidden_dims = tuple(
        int(module.out_features)
        for module in policy.mlp_extractor.policy_net
        if isinstance(module, nn.Linear)
    )
    activation_modules = [
        module
        for module in policy.mlp_extractor.policy_net
        if not isinstance(module, nn.Linear)
    ]
    if not activation_modules or not all(
        isinstance(module, nn.Tanh) for module in activation_modules
    ):
        raise ValueError("Source actor must use Tanh hidden activations.")
    expected_hidden = tuple(int(value) for value in expected_model["hidden_dims"])
    invariants = {
        "state_dim": state_dim == int(expected_model["state_dim"]),
        "action_dim": action_dim == int(expected_model["action_dim"]),
        "hidden_dims": hidden_dims == expected_hidden,
        "activation": str(expected_model["activation"]).lower() == "tanh",
        "observation_normalization": str(
            expected_model["observation_normalization"]
        ).lower()
        == "identity",
        "action_clipping": bool(expected_model["action_clipping"]),
    }
    if not all(invariants.values()):
        raise ValueError(f"Source/model config mismatch: {invariants}")

    action_low = np.asarray(policy.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(policy.action_space.high, dtype=np.float32).reshape(-1)
    standalone = ImitationRecoveryPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation="tanh",
        observation_mean=np.zeros(state_dim, dtype=np.float32),
        observation_std=np.ones(state_dim, dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
        provenance=provenance,
    )
    standalone.policy_net.load_state_dict(
        policy.mlp_extractor.policy_net.state_dict(), strict=True
    )
    standalone.action_net.load_state_dict(policy.action_net.state_dict(), strict=True)
    standalone.eval()

    actor_tensors = {
        f"policy_net.{key}": value
        for key, value in policy.mlp_extractor.policy_net.state_dict().items()
    }
    actor_tensors.update(
        {
            f"action_net.{key}": value
            for key, value in policy.action_net.state_dict().items()
        }
    )
    details = {
        "source_framework": "stable-baselines3",
        "source_policy_class": policy.__class__.__name__,
        "source_features_extractor": policy.features_extractor.__class__.__name__,
        "source_squash_output": bool(policy.squash_output),
        "source_vec_normalize_present": bool(
            getattr(source, "_vec_normalize_env", None) is not None
        ),
        "observation_preprocessing": "float32_identity_flatten",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dims": list(hidden_dims),
        "activation": "tanh",
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
        "actor_parameter_count": int(
            sum(parameter.numel() for parameter in standalone.parameters())
        ),
        "source_actor_tensor_sha256": _tensor_hash(actor_tensors),
        "standalone_state_tensor_sha256": _tensor_hash(standalone.state_dict()),
        "excluded_source_state": [
            "log_std",
            "mlp_extractor.value_net",
            "value_net",
            "policy.optimizer",
        ],
    }
    return standalone, details


def _random_states(
    train_states: np.ndarray,
    validation_states: np.ndarray,
    *,
    samples: int,
    seed: int,
    support_expansion: float,
) -> np.ndarray:
    combined = np.concatenate((train_states, validation_states), axis=0)
    lower = np.min(combined, axis=0)
    upper = np.max(combined, axis=0)
    span = np.maximum(upper - lower, np.float32(1e-3))
    lower = lower - np.float32(support_expansion) * span
    upper = upper + np.float32(support_expansion) * span
    generator = np.random.default_rng(seed)
    return generator.uniform(lower, upper, size=(samples, combined.shape[1])).astype(
        np.float32
    )


def _equivalence(
    source: Any,
    standalone: ImitationRecoveryPolicy,
    states: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    source_means: list[np.ndarray] = []
    source_actions: list[np.ndarray] = []
    standalone_means: list[np.ndarray] = []
    standalone_actions: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = states[start : start + batch_size]
        with torch.inference_mode():
            source_tensor = torch.as_tensor(
                batch, dtype=torch.float32, device=source.device
            )
            source_mean = (
                source.policy.get_distribution(source_tensor)
                .distribution.mean.detach()
                .cpu()
                .numpy()
            )
        source_action, _ = source.predict(batch, deterministic=True)
        standalone_mean = standalone.predict_mean(batch)
        standalone_action = standalone.predict(batch, deterministic=True)
        if isinstance(standalone_action, tuple):  # Defensive typing guard.
            standalone_action = standalone_action[0]
        source_means.append(np.asarray(source_mean, dtype=np.float32))
        source_actions.append(np.asarray(source_action, dtype=np.float32))
        standalone_means.append(np.asarray(standalone_mean, dtype=np.float32))
        standalone_actions.append(np.asarray(standalone_action, dtype=np.float32))

    source_mean_all = np.concatenate(source_means)
    source_action_all = np.concatenate(source_actions)
    standalone_mean_all = np.concatenate(standalone_means)
    standalone_action_all = np.concatenate(standalone_actions)
    mean_error = np.abs(source_mean_all - standalone_mean_all)
    action_error = np.abs(source_action_all - standalone_action_all)
    return {
        "samples": int(len(states)),
        "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
        "source_mean_actions_sha256": hashlib.sha256(
            source_mean_all.tobytes()
        ).hexdigest(),
        "standalone_mean_actions_sha256": hashlib.sha256(
            standalone_mean_all.tobytes()
        ).hexdigest(),
        "source_clipped_actions_sha256": hashlib.sha256(
            source_action_all.tobytes()
        ).hexdigest(),
        "standalone_clipped_actions_sha256": hashlib.sha256(
            standalone_action_all.tobytes()
        ).hexdigest(),
        "mean_actions_array_equal": bool(
            np.array_equal(source_mean_all, standalone_mean_all)
        ),
        "clipped_actions_array_equal": bool(
            np.array_equal(source_action_all, standalone_action_all)
        ),
        "mean_action_max_abs_error": float(np.max(mean_error, initial=0.0)),
        "mean_action_mean_abs_error": float(np.mean(mean_error)),
        "clipped_action_max_abs_error": float(np.max(action_error, initial=0.0)),
        "clipped_action_mean_abs_error": float(np.mean(action_error)),
    }


def export(config_path: str | Path) -> dict[str, Any]:
    config_file = resolve_path(config_path)
    config = load_yaml(config_file)
    source_config = dict(config["source"])
    data_config = dict(config["data"])
    model_config = dict(config["model"])
    export_config = dict(config["export"])
    equivalence_config = dict(config["equivalence"])

    source_checkpoint = _verified_path(source_config, "checkpoint")
    source_metrics_path = _verified_path(source_config, "metrics")
    source_training_config = _verified_path(source_config, "training_config")
    verified_data_paths = {
        key: _verified_path(data_config, key)
        for key in (
            "train_npz",
            "train_manifest",
            "validation_npz",
            "validation_manifest",
            "bank_audit",
            "act_checkpoint",
            "detector_checkpoint",
        )
    }

    try:
        from stable_baselines3 import PPO
        import stable_baselines3
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required only for export.") from error

    source = PPO.load(str(source_checkpoint), device="cpu")
    num_timesteps = int(source.num_timesteps)
    n_updates = int(getattr(source, "_n_updates", 0))
    optimizer_state_entries = int(len(source.policy.optimizer.state))
    source_invariants = {
        "num_timesteps": num_timesteps
        == int(source_config["require_num_timesteps"]),
        "n_updates": n_updates == int(source_config["require_n_updates"]),
        "optimizer_empty": (
            optimizer_state_entries == 0
            if bool(source_config["require_empty_optimizer"])
            else True
        ),
    }
    if not all(source_invariants.values()):
        raise ValueError(
            "Source is not a zero-PPO-step actor-only checkpoint: "
            f"{source_invariants}"
        )

    metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    metrics_invariants = {
        "selected_model_type": metrics.get("selected_model_type")
        == "expert_actor_only",
        "trained_timesteps": int(metrics.get("trained_timesteps", -1)) == 0,
        "selected_model_timesteps": int(
            metrics.get("selected_model_timesteps", -1)
        )
        == 0,
        "additional_timesteps": int(metrics.get("additional_timesteps", -1))
        == 0,
    }
    if not all(metrics_invariants.values()):
        raise ValueError(f"Source metrics fail actor-only audit: {metrics_invariants}")

    input_hashes = {
        "source_actor_only_checkpoint": sha256(source_checkpoint),
        "source_actor_only_metrics": sha256(source_metrics_path),
        "source_training_config": sha256(source_training_config),
        **{
            key: sha256(path)
            for key, path in verified_data_paths.items()
        },
    }
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    provenance = {
        "created_utc": created_utc,
        "method": "trigger_aligned_imitation_recovery",
        "source_checkpoint": _relative(source_checkpoint),
        "source_training": {
            "algorithm_container": "stable_baselines3.PPO",
            "num_timesteps": num_timesteps,
            "n_updates": n_updates,
            "optimizer_state_entries": optimizer_state_entries,
            "proof": "zero environment interactions and zero PPO updates",
        },
        "supervision": {
            key: metrics["expert_pretraining"][key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "observation_noise_std",
                "train_samples",
                "validation_samples",
                "best_validation_loss",
            )
        },
        "input_paths": {
            "source_actor_only_metrics": _relative(source_metrics_path),
            "source_training_config": _relative(source_training_config),
            **{
                key: _relative(path)
                for key, path in verified_data_paths.items()
            },
        },
        "input_sha256": input_hashes,
    }
    standalone, extraction = _extract_policy(
        source, expected_model=model_config, provenance=provenance
    )
    standalone.provenance["extraction"] = extraction

    output_path = resolve_path(export_config["checkpoint"])
    audit_path = resolve_path(export_config["audit"])
    pending_path = output_path.with_suffix(output_path.suffix + ".pending")
    standalone.save(pending_path)
    reloaded = ImitationRecoveryPolicy.load(pending_path, device="cpu")

    train_states = _load_demo_states(
        verified_data_paths["train_npz"], reloaded.state_dim
    )
    validation_states = _load_demo_states(
        verified_data_paths["validation_npz"], reloaded.state_dim
    )
    random_states = _random_states(
        train_states,
        validation_states,
        samples=int(equivalence_config["random_samples"]),
        seed=int(equivalence_config["random_seed"]),
        support_expansion=float(
            equivalence_config["random_support_expansion"]
        ),
    )
    batch_size = int(equivalence_config["batch_size"])
    split_results = {
        "train_demo_states": _equivalence(
            source, reloaded, train_states, batch_size=batch_size
        ),
        "validation_demo_states": _equivalence(
            source, reloaded, validation_states, batch_size=batch_size
        ),
        "random_support_states": _equivalence(
            source, reloaded, random_states, batch_size=batch_size
        ),
    }
    tolerance = float(equivalence_config["tolerance"])
    maximum_error = max(
        max(
            result["mean_action_max_abs_error"],
            result["clipped_action_max_abs_error"],
        )
        for result in split_results.values()
    )
    if maximum_error > tolerance:
        pending_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Standalone action equivalence failed: {maximum_error} > {tolerance}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.replace(output_path)
    payload = {
        "schema_version": 1,
        "created_utc": created_utc,
        "audit_result": "pass",
        "artifact": {
            "path": _relative(output_path),
            "sha256": sha256(output_path),
            "bytes": output_path.stat().st_size,
            "policy_type": reloaded.policy_type,
            "schema_version": reloaded.schema_version,
        },
        "source": {
            "checkpoint": _relative(source_checkpoint),
            "checkpoint_sha256": sha256(source_checkpoint),
            "metrics": _relative(source_metrics_path),
            "metrics_sha256": sha256(source_metrics_path),
            "stable_baselines3_version": stable_baselines3.__version__,
            "num_timesteps": num_timesteps,
            "n_updates": n_updates,
            "optimizer_state_entries": optimizer_state_entries,
            "invariants": source_invariants,
            "metrics_invariants": metrics_invariants,
        },
        "architecture": extraction,
        "provenance": provenance,
        "equivalence": {
            "tolerance": tolerance,
            "total_states": int(
                sum(result["samples"] for result in split_results.values())
            ),
            "maximum_abs_error": float(maximum_error),
            "all_splits_within_tolerance": True,
            "random_seed": int(equivalence_config["random_seed"]),
            "random_support_expansion": float(
                equivalence_config["random_support_expansion"]
            ),
            "splits": split_results,
        },
        "config": {
            "path": _relative(config_file),
            "sha256": sha256(config_file),
        },
    }
    atomic_json_dump(payload, audit_path)
    LOGGER.info(
        "Exported %s (%d parameters); checked %d states, max error %.3g; audit %s",
        output_path,
        extraction["actor_parameter_count"],
        payload["equivalence"]["total_states"],
        maximum_error,
        audit_path,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/recovery_imitation.yaml",
        help="Standalone recovery export configuration.",
    )
    args = parser.parse_args(argv)
    payload = export(args.config)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
