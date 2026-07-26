#!/usr/bin/env python3
"""Select the deployed recovery option on a disjoint CRN development bank.

The rule is deliberately simple and fixed in code:

1. maximize end-to-end task successes;
2. maximize completions reached while recovery is active;
3. minimize recovery-policy environment interaction steps;
4. minimize mean episode length.

The formal 1,000-episode test bank is not an input to this command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation file: {path}")
    return rows


def _truth(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _condition(
    *,
    label: str,
    raw_path: Path,
    checkpoint: Path,
    metrics_path: Path,
) -> tuple[dict[str, Any], list[bool]]:
    rows = _read_csv(raw_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics must be a JSON object: {metrics_path}")
    required = {
        "seed",
        "success",
        "steps",
        "recovery_attempts",
        "recovery_successes",
        "episode_specification_sha256",
        "episode_bank_sha256",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{raw_path} is missing columns: {sorted(missing)}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(rows):
        raise ValueError(f"duplicate evaluation seeds in {raw_path}")
    specification_hashes = [
        str(row["episode_specification_sha256"]) for row in rows
    ]
    if any(not value for value in specification_hashes):
        raise ValueError(f"unverified episode specification in {raw_path}")
    bank_hashes = {str(row["episode_bank_sha256"]) for row in rows}
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError(f"expected one verified episode bank in {raw_path}")
    successes = [_truth(row["success"]) for row in rows]
    recovery_successes = sum(int(row["recovery_successes"]) for row in rows)
    attempts = sum(int(row["recovery_attempts"]) for row in rows)
    trained_steps = int(metrics.get("trained_timesteps", -1))
    if trained_steps < 0:
        raise ValueError(f"invalid trained_timesteps in {metrics_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    record = {
        "label": label,
        "episodes": len(rows),
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "ordered_seeds": seeds,
        "ordered_specification_sha256": specification_hashes,
        "episode_bank_sha256": next(iter(bank_hashes)),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(rows),
        "recovery_attempts": attempts,
        "recovery_successes": recovery_successes,
        "recovery_rate": recovery_successes / attempts if attempts else 0.0,
        "average_steps": sum(int(row["steps"]) for row in rows) / len(rows),
        "recovery_environment_training_steps": trained_steps,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "raw_episodes": str(raw_path.resolve()),
        "raw_episodes_sha256": _sha256(raw_path),
    }
    return record, successes


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-only-raw",
        type=Path,
        default=ROOT
        / "results/tables/model_selection_actor_only_seed7800042_episodes.csv",
    )
    parser.add_argument(
        "--ppo-raw",
        type=Path,
        default=ROOT / "results/tables/model_selection_ppo_seed7800042_episodes.csv",
    )
    parser.add_argument(
        "--actor-only-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/recovery_ablation_warmstart_only.zip",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/recovery_trigger_seed42.zip",
    )
    parser.add_argument(
        "--actor-only-metrics",
        type=Path,
        default=ROOT
        / "results/tables/recovery_ablation_warmstart_only_metrics.json",
    )
    parser.add_argument(
        "--ppo-metrics",
        type=Path,
        default=ROOT / "results/tables/recovery_trigger_seed42_500k_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tables/recovery_option_selection.json",
    )
    args = parser.parse_args()

    actor, actor_success = _condition(
        label="expert_actor_only",
        raw_path=args.actor_only_raw,
        checkpoint=args.actor_only_checkpoint,
        metrics_path=args.actor_only_metrics,
    )
    ppo, ppo_success = _condition(
        label="expert_actor_plus_ppo",
        raw_path=args.ppo_raw,
        checkpoint=args.ppo_checkpoint,
        metrics_path=args.ppo_metrics,
    )
    for field in (
        "episodes",
        "seed_start",
        "seed_end",
        "ordered_seeds",
        "ordered_specification_sha256",
        "episode_bank_sha256",
    ):
        if actor[field] != ppo[field]:
            raise ValueError(f"development evaluations differ in {field}")

    candidates = [actor, ppo]
    selected = max(
        candidates,
        key=lambda item: (
            int(item["successes"]),
            int(item["recovery_successes"]),
            -int(item["recovery_environment_training_steps"]),
            -float(item["average_steps"]),
        ),
    )
    actor_rescues = sum(a and not p for a, p in zip(actor_success, ppo_success))
    ppo_rescues = sum(p and not a for a, p in zip(actor_success, ppo_success))
    payload = {
        "schema_version": "reim-recovery-option-selection-v1",
        "scope": "development_only_not_formal_test",
        "selection_rule": [
            "maximize_task_successes",
            "maximize_completions_while_recovery_active",
            "minimize_recovery_environment_training_steps",
            "minimize_average_episode_steps",
        ],
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key
                not in {"ordered_seeds", "ordered_specification_sha256"}
            }
            for candidate in candidates
        ],
        "paired_discordance": {
            "actor_only_success_ppo_failure": actor_rescues,
            "ppo_success_actor_only_failure": ppo_rescues,
        },
        "selected_label": selected["label"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "formal_test_bank_was_not_read": True,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
