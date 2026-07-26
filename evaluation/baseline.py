"""Evaluate the four primary REIM baselines on paired episode seeds."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    ROBUSTNESS_ACTION_STD_SCALE,
    ROBUSTNESS_OBJECT_STD_SCALE,
    ROBUSTNESS_OBSERVATION_STD_SCALE,
    REIMController,
    _atomic_write_csv,
    _space_dim,
    canonical_method,
    evaluate_controller,
    effective_profile,
    load_bc_policy,
    load_failure_detector,
    load_recovery_policy,
    make_env,
    save_traces,
    seed_everything,
)
from evaluation.metrics import EpisodeMetrics, aggregate_episode_metrics

LOGGER = logging.getLogger("reim.baselines")
DEFAULT_METHODS = ("bc", "bc_random_reset", "bc_rl_recovery", "reim")


def evaluate_baselines(
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
    noise_level: float = 0.0,
    capture_traces: int = 4,
    episode_specifications: Sequence[dict[str, Any]] | None = None,
    environment_constructor_seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[EpisodeMetrics], list[dict[str, Any]]]:
    """Run methods with paired seeds and return summaries plus raw episodes."""

    canonical_methods = [canonical_method(method) for method in methods]
    if len(set(canonical_methods)) != len(canonical_methods):
        raise ValueError("methods contain aliases resolving to a duplicate controller")

    constructor_seed = (
        seed
        if environment_constructor_seed is None
        else int(environment_constructor_seed)
    )
    reference_env = make_env(
        backend=backend,
        seed=constructor_seed,
        env_config=env_config,
        noise_level=noise_level,
    )
    try:
        state_dim = _space_dim(reference_env.observation_space, "observation")
        action_dim = _space_dim(reference_env.action_space, "action")
        bc = load_bc_policy(
            bc_checkpoint,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        detector = (
            load_failure_detector(
                detector_checkpoint, state_dim=state_dim, device=device
            )
            if any(method in {"reim", "bc_detector"} for method in canonical_methods)
            else None
        )
        recovery = (
            load_recovery_policy(recovery_checkpoint, env=None, device=device)
            if any(
                method in {"reim", "bc_rl_recovery"}
                for method in canonical_methods
            )
            else None
        )
    finally:
        if hasattr(reference_env, "close"):
            reference_env.close()

    summaries: list[dict[str, Any]] = []
    all_records: list[EpisodeMetrics] = []
    all_traces: list[dict[str, Any]] = []
    for method_index, method in enumerate(canonical_methods):
        LOGGER.info(
            "Evaluating %s (%d episodes, backend=%s)",
            METHOD_LABELS[method],
            episodes,
            backend,
        )
        env = make_env(
            backend=backend,
            seed=constructor_seed,
            env_config=env_config,
            noise_level=noise_level,
        )

        def factory(method_name: str = method) -> REIMController:
            return REIMController(
                method_name,
                bc,
                detector=detector if method_name in {"reim", "bc_detector"} else None,
                recovery_policy=(
                    recovery
                    if method_name in {"reim", "bc_rl_recovery"}
                    else None
                ),
                config=controller_config,
            )

        try:
            records, traces = evaluate_controller(
                env,
                factory,
                episodes=episodes,
                seed=seed,
                max_steps=max_steps,
                capture_traces=capture_traces if method == "reim" else 0,
                episode_specifications=episode_specifications,
            )
        finally:
            if hasattr(env, "close"):
                env.close()
        summary = aggregate_episode_metrics(
            records,
            method=METHOD_LABELS[method],
            n_bootstrap=bootstrap_samples,
            seed=seed + 101 * method_index,
        )
        summaries.append(summary)
        all_records.extend(records)
        all_traces.extend(traces)
    return summaries, all_records, all_traces


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=CLI_METHOD_CHOICES,
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
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="PyTorch intra-op CPU threads (1 is fastest for small ACT batches).",
    )
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
    parser.add_argument("--max-random-resets", type=int, default=1)
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Dimensionless calibrated disturbance fraction (main protocol: 0.2).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--episode-bank",
        type=Path,
        default=None,
        help=(
            "Persistent common-random-number episode bank. Required for new "
            "benchmark-eligible Meta-World results."
        ),
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Zero-based slice offset when evaluating an episode-bank shard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "baseline.csv",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "recovery_traces.json",
    )
    parser.add_argument("--capture-traces", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    try:
        import torch

        torch.set_num_threads(args.torch_threads)
    except ImportError:
        pass
    episodes = args.episodes
    if episodes is None:
        episodes = 1_000 if args.profile == "full" else 5
    profile = effective_profile(
        args.profile, episodes, full_episodes=1_000, smoke_episodes=5
    )
    seed_everything(args.seed)
    episode_bank = None
    episode_specifications = None
    episode_bank_file_sha256 = ""
    environment_constructor_seed = args.seed
    if args.episode_bank is not None:
        from evaluation.episode_bank import (
            bank_file_sha256,
            load_episode_bank,
            runtime_episode_specifications,
        )

        episode_bank = load_episode_bank(args.episode_bank)
        expected_backend = (
            "metaworld" if args.backend == "auto" else str(args.backend).lower()
        )
        if str(episode_bank["backend"]).lower() != expected_backend:
            raise ValueError("episode bank backend differs from --backend")
        if int(episode_bank["max_steps"]) != int(args.max_steps):
            raise ValueError("episode bank max_steps differs from --max-steps")
        expected_disturbance = {
            "action_noise_std": float(args.noise_level)
            * ROBUSTNESS_ACTION_STD_SCALE,
            "observation_noise_std": float(args.noise_level)
            * ROBUSTNESS_OBSERVATION_STD_SCALE,
            "object_noise_probability": (
                1.0
                if float(args.noise_level) * ROBUSTNESS_OBJECT_STD_SCALE > 0.0
                else 0.0
            ),
            "object_noise_std": float(args.noise_level)
            * ROBUSTNESS_OBJECT_STD_SCALE,
            "object_noise_magnitude": 0.03,
        }
        for field, expected in expected_disturbance.items():
            actual = float(episode_bank["disturbance"][field])
            if abs(actual - expected) > 1e-12:
                raise ValueError(
                    f"episode bank {field}={actual} differs from the "
                    f"--noise-level protocol value {expected}"
                )
        if (
            "bc_random_reset"
            in {canonical_method(method) for method in args.methods}
            and int(episode_bank["retries_per_episode"])
            < int(args.max_random_resets)
        ):
            raise ValueError(
                "episode bank has fewer deterministic retries than "
                "--max-random-resets"
            )
        episode_specifications = runtime_episode_specifications(
            episode_bank,
            offset=args.episode_offset,
            count=episodes,
        )
        episode_bank_file_sha256 = bank_file_sha256(args.episode_bank)
        environment_constructor_seed = int(episode_bank["task_bank_seed"])
    controller_config = ControllerConfig(
        failure_threshold=args.failure_threshold,
        recovery_exit_threshold=args.recovery_exit_threshold,
        recovery_budget=args.recovery_budget,
        recovery_min_steps=args.recovery_min_steps,
        recovery_clear_steps=args.recovery_clear_steps,
        max_random_resets=args.max_random_resets,
    )
    summaries, records, traces = evaluate_baselines(
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
        capture_traces=args.capture_traces,
        episode_specifications=episode_specifications,
        environment_constructor_seed=environment_constructor_seed,
    )
    full_bank_coverage = bool(
        episode_bank is not None
        and args.episode_offset == 0
        and episodes == 1_000
        and int(episode_bank["episodes"]) >= 1_000
    )
    for summary in summaries:
        summary["Profile"] = profile
        summary["Noise Level"] = float(args.noise_level)
        summary["Action Noise Std"] = (
            float(args.noise_level) * ROBUSTNESS_ACTION_STD_SCALE
        )
        summary["Observation Noise Std"] = (
            float(args.noise_level) * ROBUSTNESS_OBSERVATION_STD_SCALE
        )
        summary["Object Impulse Std (m)"] = (
            float(args.noise_level) * ROBUSTNESS_OBJECT_STD_SCALE
        )
        summary["Benchmark Eligible"] = bool(
            profile == "full"
            and summary.get("Backend") == "metaworld"
            and full_bank_coverage
        )
        summary["Episode Bank SHA256"] = (
            str(episode_bank["bank_sha256"]) if episode_bank is not None else ""
        )
        summary["Episode Bank File SHA256"] = episode_bank_file_sha256
        summary["Episode Offset"] = int(args.episode_offset)
        summary["CRN Episode Specifications Verified"] = bool(
            episode_bank is not None
        )
    _atomic_write_csv(args.output, summaries)
    raw_output = args.output.with_name(f"{args.output.stem}_episodes.csv")
    _atomic_write_csv(
        raw_output,
        [
            {
                **record.to_dict(),
                "Profile": profile,
                "noise_level": float(args.noise_level),
                "Benchmark Eligible": bool(
                    profile == "full"
                    and record.backend == "metaworld"
                    and full_bank_coverage
                ),
            }
            for record in records
        ],
    )
    for trace in traces:
        trace["profile"] = profile
        trace["evaluation_episodes"] = args.episodes
        trace["benchmark_eligible"] = bool(
            profile == "full"
            and args.backend == "metaworld"
            and full_bank_coverage
        )
        trace["episode_bank_sha256"] = (
            str(episode_bank["bank_sha256"]) if episode_bank is not None else ""
        )
    save_traces(args.trace_output, traces)
    LOGGER.info("Baseline results:\n%s", json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    main()
