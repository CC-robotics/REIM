"""Evaluate ACT and REIM under 0--40% action/object/observation noise."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.baseline import evaluate_baselines
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
    _atomic_write_csv,
    canonical_method,
    effective_profile,
    save_traces,
    seed_everything,
)

LOGGER = logging.getLogger("reim.robustness")
DEFAULT_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)


def run_robustness(
    *,
    methods: Sequence[str],
    noise_levels: Sequence[float],
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
    capture_traces: int = 4,
    episode_specifications_by_condition: (
        Sequence[Sequence[dict[str, Any]] | None] | None
    ) = None,
    environment_constructor_seeds: Sequence[int | None] | None = None,
    episode_bank_metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if any(not 0.0 <= float(level) <= 1.0 for level in noise_levels):
        raise ValueError("noise levels must be fractions in [0, 1]")
    for name, values in (
        ("episode specifications", episode_specifications_by_condition),
        ("constructor seeds", environment_constructor_seeds),
        ("episode bank metadata", episode_bank_metadata),
    ):
        if values is not None and len(values) != len(noise_levels):
            raise ValueError(f"{name} must align one-to-one with noise levels")

    summaries: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    for level_index, level in enumerate(noise_levels):
        level = float(level)
        LOGGER.info("Robustness condition: %.0f%% noise", level * 100.0)
        condition_summaries, records, traces = evaluate_baselines(
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
            noise_level=level,
            capture_traces=max(0, capture_traces - len(captured)),
            episode_specifications=(
                None
                if episode_specifications_by_condition is None
                else episode_specifications_by_condition[level_index]
            ),
            environment_constructor_seed=(
                None
                if environment_constructor_seeds is None
                else environment_constructor_seeds[level_index]
            ),
        )
        bank_metadata = (
            None
            if episode_bank_metadata is None
            else episode_bank_metadata[level_index]
        )
        for row in condition_summaries:
            row["Noise Level"] = level
            row["Noise (%)"] = 100.0 * level
            row["Action Noise Std"] = level * ROBUSTNESS_ACTION_STD_SCALE
            row["Observation Noise Std"] = (
                level * ROBUSTNESS_OBSERVATION_STD_SCALE
            )
            row["Object Impulse Std (m)"] = (
                level * ROBUSTNESS_OBJECT_STD_SCALE
            )
            row["Condition Seed"] = seed
            row["Benchmark Eligible"] = False
            row["Episode Bank SHA256"] = (
                "" if bank_metadata is None else bank_metadata["bank_sha256"]
            )
            row["Episode Bank File SHA256"] = (
                ""
                if bank_metadata is None
                else bank_metadata["file_sha256"]
            )
            row["Episode Offset"] = (
                0 if bank_metadata is None else bank_metadata["offset"]
            )
            row["CRN Episode Specifications Verified"] = bool(
                bank_metadata is not None
            )
            summaries.append(row)
        for record in records:
            row = record.to_dict()
            row["noise_level"] = level
            row["noise_percent"] = 100.0 * level
            row["action_noise_std"] = level * ROBUSTNESS_ACTION_STD_SCALE
            row["observation_noise_std"] = (
                level * ROBUSTNESS_OBSERVATION_STD_SCALE
            )
            row["object_impulse_std_m"] = level * ROBUSTNESS_OBJECT_STD_SCALE
            row["condition_index"] = level_index
            raw_rows.append(row)
        for trace in traces:
            trace["noise_level"] = level
            captured.append(trace)
    return summaries, raw_rows, captured[:capture_traces]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=CLI_METHOD_CHOICES,
        default=["bc", "reim"],
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_LEVELS),
        metavar="FRACTION",
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
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--episode-bank",
        dest="episode_banks",
        action="append",
        type=Path,
        default=None,
        help=(
            "Persistent CRN bank for one noise level. Repeat in the same order "
            "as --noise-levels (five times for the formal robustness sweep)."
        ),
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Zero-based common slice offset for bank-backed shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "robustness.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "robustness.png",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "recovery_traces.json",
    )
    parser.add_argument("--capture-traces", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
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
        episodes = 200 if args.profile == "full" else 3
    profile = effective_profile(
        args.profile, episodes, full_episodes=200, smoke_episodes=3
    )
    seed_everything(args.seed)
    episode_specifications_by_condition = None
    environment_constructor_seeds = None
    episode_bank_metadata = None
    if args.episode_banks:
        if len(args.episode_banks) != len(args.noise_levels):
            raise ValueError(
                "repeat --episode-bank exactly once per --noise-levels entry"
            )
        from evaluation.episode_bank import (
            bank_file_sha256,
            load_episode_bank,
            runtime_episode_specifications,
        )

        episode_specifications_by_condition = []
        environment_constructor_seeds = []
        episode_bank_metadata = []
        expected_backend = (
            "metaworld" if args.backend == "auto" else args.backend
        )
        for level, path in zip(args.noise_levels, args.episode_banks):
            bank = load_episode_bank(path)
            if str(bank["backend"]).lower() != expected_backend:
                raise ValueError(f"{path}: episode bank backend mismatch")
            if int(bank["max_steps"]) != int(args.max_steps):
                raise ValueError(f"{path}: episode bank max_steps mismatch")
            expected_disturbance = {
                "action_noise_std": float(level)
                * ROBUSTNESS_ACTION_STD_SCALE,
                "observation_noise_std": float(level)
                * ROBUSTNESS_OBSERVATION_STD_SCALE,
                "object_noise_probability": (
                    1.0
                    if float(level) * ROBUSTNESS_OBJECT_STD_SCALE > 0.0
                    else 0.0
                ),
                "object_noise_std": float(level)
                * ROBUSTNESS_OBJECT_STD_SCALE,
                "object_noise_magnitude": 0.03,
            }
            for field, expected in expected_disturbance.items():
                actual = float(bank["disturbance"][field])
                if abs(actual - expected) > 1e-12:
                    raise ValueError(
                        f"{path}: {field}={actual} does not match "
                        f"noise level {level} ({expected})"
                    )
            episode_specifications_by_condition.append(
                runtime_episode_specifications(
                    bank,
                    offset=args.episode_offset,
                    count=episodes,
                )
            )
            environment_constructor_seeds.append(int(bank["task_bank_seed"]))
            episode_bank_metadata.append(
                {
                    "bank_sha256": str(bank["bank_sha256"]),
                    "file_sha256": bank_file_sha256(path),
                    "offset": int(args.episode_offset),
                    "episodes": int(bank["episodes"]),
                }
            )
    controller_config = ControllerConfig(
        failure_threshold=args.failure_threshold,
        recovery_exit_threshold=args.recovery_exit_threshold,
        recovery_budget=args.recovery_budget,
        recovery_min_steps=args.recovery_min_steps,
        recovery_clear_steps=args.recovery_clear_steps,
    )
    summaries, raw_rows, traces = run_robustness(
        methods=[canonical_method(method) for method in args.methods],
        noise_levels=args.noise_levels,
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
        capture_traces=args.capture_traces,
        episode_specifications_by_condition=(
            episode_specifications_by_condition
        ),
        environment_constructor_seeds=environment_constructor_seeds,
        episode_bank_metadata=episode_bank_metadata,
    )
    full_bank_coverage = bool(
        episode_bank_metadata is not None
        and args.episode_offset == 0
        and episodes == 200
        and all(
            int(metadata["episodes"]) >= 200
            for metadata in episode_bank_metadata
        )
    )
    for row in summaries:
        row["Profile"] = profile
        row["Benchmark Eligible"] = bool(
            profile == "full"
            and row.get("Backend") == "metaworld"
            and full_bank_coverage
        )
    for row in raw_rows:
        row["Profile"] = profile
        row["Benchmark Eligible"] = bool(
            profile == "full"
            and str(row.get("backend", "")).lower() == "metaworld"
            and full_bank_coverage
        )
    for trace in traces:
        trace["profile"] = profile
        trace["evaluation_episodes"] = args.episodes
        trace["benchmark_eligible"] = bool(
            profile == "full"
            and args.backend == "metaworld"
            and full_bank_coverage
        )
    _atomic_write_csv(args.output, summaries)
    raw_path = args.output.with_name(f"{args.output.stem}_episodes.csv")
    _atomic_write_csv(raw_path, raw_rows)
    save_traces(args.trace_output, traces)
    LOGGER.info("Saved robustness results to %s", args.output)
    if not args.no_plot:
        from visualization.plot_results import plot_robustness

        plot_robustness(args.output, args.figure)
    return summaries


if __name__ == "__main__":
    main()
