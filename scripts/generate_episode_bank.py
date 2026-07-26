#!/usr/bin/env python3
"""Generate a persistent REIM common-random-number evaluation bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.episode_bank import (  # noqa: E402
    bank_file_sha256,
    build_episode_bank,
    save_episode_bank,
)
from evaluation.evaluate_reim import (  # noqa: E402
    ROBUSTNESS_ACTION_STD_SCALE,
    ROBUSTNESS_OBJECT_STD_SCALE,
    ROBUSTNESS_OBSERVATION_STD_SCALE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metaworld", "toy"), default="metaworld")
    parser.add_argument("--env-name", default="pick-place-v3")
    parser.add_argument("--task-bank-seed", type=int, default=6_000_042)
    parser.add_argument("--episode-seed-start", type=int, default=6_000_042)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--retries-per-episode",
        type=int,
        default=1,
        help="Deterministic independent reset specs reserved for Random Reset.",
    )
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "datasets/evaluation/"
        "pickplace_crn_seed6000042_n1000_noise020.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.noise_level < 0.0:
        raise ValueError("noise-level must be non-negative")
    object_noise_std = float(args.noise_level) * ROBUSTNESS_OBJECT_STD_SCALE
    payload = build_episode_bank(
        backend=args.backend,
        env_name=args.env_name,
        task_bank_seed=args.task_bank_seed,
        episode_seed_start=args.episode_seed_start,
        episodes=args.episodes,
        max_steps=args.max_steps,
        action_noise_std=(
            float(args.noise_level) * ROBUSTNESS_ACTION_STD_SCALE
        ),
        observation_noise_std=(
            float(args.noise_level) * ROBUSTNESS_OBSERVATION_STD_SCALE
        ),
        object_noise_probability=1.0 if object_noise_std > 0.0 else 0.0,
        object_noise_std=object_noise_std,
        object_noise_magnitude=0.03,
        retries_per_episode=args.retries_per_episode,
    )
    output = save_episode_bank(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "file_sha256": bank_file_sha256(output),
                "bank_sha256": payload["bank_sha256"],
                "backend": payload["backend"],
                "episodes": payload["episodes"],
                "episode_seed_start": payload["episode_seed_start"],
                "episode_seed_end": payload["episode_seed_end"],
                "task_count": len(payload["tasks"]),
                "retries_per_episode": payload["retries_per_episode"],
            },
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    main()
