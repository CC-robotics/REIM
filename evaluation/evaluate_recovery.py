#!/usr/bin/env python3
"""Evaluate a PPO checkpoint on independently seeded recovery initial states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.train_recovery import (  # noqa: E402
    _wrapper_kwargs,
    evaluate_recovery,
    make_environment_factory,
)
from utils.common import (  # noqa: E402
    atomic_json_dump,
    load_yaml,
    resolve_path,
    seed_everything,
    select_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/recovery_trigger_seed42.zip",
    )
    parser.add_argument("--config", default="configs/ppo_trigger.yaml")
    parser.add_argument("--env-config", default="configs/environment.yaml")
    parser.add_argument("--backend", choices=("metaworld", "toy"), default="metaworld")
    parser.add_argument("--state-mode", choices=("semantic", "raw"), default="semantic")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/tables/recovery_evaluation.json",
    )
    return parser


def main() -> dict[str, object]:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    from stable_baselines3 import PPO

    seed_everything(args.seed)
    device = select_device(args.device)
    recovery_config = load_yaml(args.config)
    environment_config = load_yaml(args.env_config)
    wrapper = _wrapper_kwargs(recovery_config, environment_config)
    checkpoint = resolve_path(args.checkpoint)
    model = PPO.load(str(checkpoint), device=device)
    factory = make_environment_factory(
        environment_config=environment_config,
        wrapper_kwargs=wrapper,
        seed=args.seed,
        backend=args.backend,
        state_mode=args.state_mode,
        monitor_file=None,
    )
    metrics = evaluate_recovery(model, factory, args.episodes)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "policy_timesteps": int(model.num_timesteps),
            "backend": args.backend,
            "state_mode": args.state_mode,
            "episodes": args.episodes,
            "seed": args.seed,
            "device": device,
            "reward": wrapper,
            "config": str(resolve_path(args.config)),
        }
    )
    atomic_json_dump(metrics, args.output)
    return metrics


if __name__ == "__main__":
    main()
