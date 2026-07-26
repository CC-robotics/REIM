"""Paired comparison with bootstrap gains relative to the ACT baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.baseline import DEFAULT_METHODS, evaluate_baselines
from evaluation.evaluate_reim import (
    ControllerConfig,
    CLI_METHOD_CHOICES,
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_RECOVERY_BUDGET,
    DEFAULT_RECOVERY_CHECKPOINT,
    DEFAULT_RECOVERY_CLEAR_STEPS,
    DEFAULT_RECOVERY_EXIT_THRESHOLD,
    DEFAULT_RECOVERY_MIN_STEPS,
    METHOD_LABELS,
    PROJECT_ROOT,
    _atomic_write_csv,
    effective_profile,
    seed_everything,
)
from evaluation.metrics import EpisodeMetrics, bootstrap_ci

LOGGER = logging.getLogger("reim.comparison")


def _paired_success_gain(
    baseline: Sequence[EpisodeMetrics],
    contender: Sequence[EpisodeMetrics],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float, float]:
    baseline_by_seed = {record.seed: record for record in baseline}
    contender_by_seed = {record.seed: record for record in contender}
    shared = sorted(set(baseline_by_seed) & set(contender_by_seed))
    if not shared:
        raise ValueError("no paired episode seeds between baseline and contender")
    differences = np.asarray(
        [
            float(contender_by_seed[item].success)
            - float(baseline_by_seed[item].success)
            for item in shared
        ],
        dtype=np.float64,
    )
    lower, upper = bootstrap_ci(
        differences, n_bootstrap=bootstrap_samples, seed=seed
    )
    return float(differences.mean()), lower, upper


def run_comparison(
    *,
    methods: Sequence[str],
    episodes: int,
    seed: int,
    backend: str,
    env_config: str | Path | None,
    bc_checkpoint: str | Path,
    detector_checkpoint: str | Path,
    recovery_checkpoint: str | Path,
    device: str,
    max_steps: int,
    controller_config: ControllerConfig,
    bootstrap_samples: int,
    noise_level: float,
) -> tuple[list[dict[str, Any]], list[EpisodeMetrics]]:
    summaries, records, _ = evaluate_baselines(
        methods=methods,
        episodes=episodes,
        seed=seed,
        backend=backend,
        env_config=env_config,
        bc_checkpoint=bc_checkpoint,
        detector_checkpoint=detector_checkpoint,
        recovery_checkpoint=recovery_checkpoint,
        device=device,
        max_steps=max_steps,
        controller_config=controller_config,
        bootstrap_samples=bootstrap_samples,
        noise_level=noise_level,
        capture_traces=0,
    )
    by_method: dict[str, list[EpisodeMetrics]] = {}
    for record in records:
        by_method.setdefault(record.method, []).append(record)
    baseline_label = METHOD_LABELS["bc"]
    if baseline_label not in by_method:
        raise ValueError("comparison requires the ACT baseline method")

    for index, summary in enumerate(summaries):
        label = str(summary["Method"])
        gain, lower, upper = _paired_success_gain(
            by_method[baseline_label],
            by_method[label],
            bootstrap_samples=bootstrap_samples,
            seed=seed + 17 * index,
        )
        summary["Success Gain vs ACT"] = gain
        summary["Gain CI Lower"] = lower
        summary["Gain CI Upper"] = upper
        summary["Noise Level"] = float(noise_level)
    ranked = sorted(summaries, key=lambda row: float(row["Success Rate"]), reverse=True)
    rank_by_method = {str(row["Method"]): rank + 1 for rank, row in enumerate(ranked)}
    for row in summaries:
        row["Rank"] = rank_by_method[str(row["Method"])]
    summaries.sort(key=lambda row: int(row["Rank"]))
    return summaries, records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=CLI_METHOD_CHOICES,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=("metaworld", "auto", "toy"),
        default="metaworld",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "environment.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        "--act-checkpoint",
        "--bc-checkpoint",
        dest="bc_checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "bc_policy.pt",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "failure_detector.pt",
    )
    parser.add_argument(
        "--recovery-checkpoint",
        type=Path,
        default=DEFAULT_RECOVERY_CHECKPOINT,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD
    )
    parser.add_argument(
        "--recovery-exit-threshold",
        type=float,
        default=DEFAULT_RECOVERY_EXIT_THRESHOLD,
    )
    parser.add_argument(
        "--recovery-budget", type=int, default=DEFAULT_RECOVERY_BUDGET
    )
    parser.add_argument(
        "--recovery-min-steps", type=int, default=DEFAULT_RECOVERY_MIN_STEPS
    )
    parser.add_argument(
        "--recovery-clear-steps", type=int, default=DEFAULT_RECOVERY_CLEAR_STEPS
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Dimensionless calibrated disturbance fraction (main protocol: 0.2).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "comparison.csv",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    episodes = args.episodes
    if episodes is None:
        episodes = 1_000 if args.profile == "full" else 5
    profile = effective_profile(
        args.profile, episodes, full_episodes=1_000, smoke_episodes=5
    )
    seed_everything(args.seed)
    controller_config = ControllerConfig(
        failure_threshold=args.failure_threshold,
        recovery_exit_threshold=args.recovery_exit_threshold,
        recovery_budget=args.recovery_budget,
        recovery_min_steps=args.recovery_min_steps,
        recovery_clear_steps=args.recovery_clear_steps,
    )
    summaries, records = run_comparison(
        methods=args.methods,
        episodes=episodes,
        seed=args.seed,
        backend=args.backend,
        env_config=args.env_config,
        bc_checkpoint=args.bc_checkpoint,
        detector_checkpoint=args.detector_checkpoint,
        recovery_checkpoint=args.recovery_checkpoint,
        device=args.device,
        max_steps=args.max_steps,
        controller_config=controller_config,
        bootstrap_samples=args.bootstrap_samples,
        noise_level=args.noise_level,
    )
    for row in summaries:
        row["Profile"] = profile
        row["Benchmark Eligible"] = bool(
            profile == "full" and row.get("Backend") == "metaworld"
        )
    _atomic_write_csv(args.output, summaries)
    raw_path = args.output.with_name(f"{args.output.stem}_episodes.csv")
    _atomic_write_csv(
        raw_path,
        [{**record.to_dict(), "Profile": profile} for record in records],
    )
    LOGGER.info("Saved comparison results to %s", args.output)
    return summaries


if __name__ == "__main__":
    main()
