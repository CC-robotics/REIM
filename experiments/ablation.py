"""Four-way ablation: ACT, detector only, recovery only, and full REIM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.baseline import evaluate_baselines
from evaluation.evaluate_reim import (
    ControllerConfig,
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_RECOVERY_BUDGET,
    DEFAULT_RECOVERY_CHECKPOINT,
    DEFAULT_RECOVERY_CLEAR_STEPS,
    DEFAULT_RECOVERY_EXIT_THRESHOLD,
    DEFAULT_RECOVERY_MIN_STEPS,
    PROJECT_ROOT,
    _atomic_write_csv,
    effective_profile,
    seed_everything,
)

LOGGER = logging.getLogger("reim.ablation")
ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "bc", "ACT"),
    ("B", "bc_detector", "ACT + Detector"),
    ("C", "bc_rl_recovery", "ACT + Recovery"),
    ("D", "reim", "REIM"),
)


def run_ablation(
    *,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, records, _ = evaluate_baselines(
        methods=[method for _, method, _ in ABLATIONS],
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
    label_by_controller = {
        "ACT": ("A", "ACT"),
        "ACT + Detector": ("B", "ACT + Detector"),
        "ACT + Heuristic Recovery": ("C", "ACT + Recovery"),
        "REIM (ACT + Detector + Recovery)": ("D", "REIM"),
    }
    output: list[dict[str, Any]] = []
    for summary in summaries:
        controller_name = str(summary["Method"])
        variant, public_name = label_by_controller[controller_name]
        row = {
            "Variant": variant,
            "Method": public_name,
            **{key: value for key, value in summary.items() if key != "Method"},
            "Controller": controller_name,
            "Noise Level": float(noise_level),
        }
        output.append(row)

    raw_rows: list[dict[str, Any]] = []
    for record in records:
        variant, public_name = label_by_controller[record.method]
        row = record.to_dict()
        row["variant"] = variant
        row["method"] = public_name
        row["controller"] = record.method
        row["noise_level"] = float(noise_level)
        raw_rows.append(row)
    return output, raw_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "ablation.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "ablation.png",
    )
    parser.add_argument("--no-plot", action="store_true")
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
    rows, raw_rows = run_ablation(
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
    for row in rows:
        row["Profile"] = profile
        row["Benchmark Eligible"] = bool(
            profile == "full" and row.get("Backend") == "metaworld"
        )
    for row in raw_rows:
        row["Profile"] = profile
    _atomic_write_csv(args.output, rows)
    raw_path = args.output.with_name(f"{args.output.stem}_episodes.csv")
    _atomic_write_csv(raw_path, raw_rows)
    LOGGER.info("Saved ablation results to %s", args.output)
    if not args.no_plot:
        from visualization.plot_results import plot_ablation

        plot_ablation(args.output, args.figure)
    return rows


if __name__ == "__main__":
    main()
