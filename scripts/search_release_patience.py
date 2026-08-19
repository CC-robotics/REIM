#!/usr/bin/env python3
"""Validation-bank grid search over REIM release parameters.

Addresses the pre-submission P0 concern about "premature release": REIM fell
below the Heuristic-Gated baseline at low disturbance (noise 0.1) and
over-intervened on clean conditions.  This tool sweeps
``release_threshold x release_patience`` for the REIM gated recovery on a
*validation* bank (benchmark seed distinct from the final evaluation bank),
holding the tuned trigger ``--threshold`` fixed.

Methodology notes
-----------------
- Only the ``reim`` method depends on the swept parameters.  The ``act`` and
  ``heuristic_recovery`` references are rolled out once per noise condition and
  reused across every grid cell (CRN-paired episode seeds, as in the official
  protocol).
- The search consumes only the validation bank seed, so the official
  final-evaluation bank remains untouched until parameters are locked.
- Each rollout records the per-episode ``recovery_steps_total`` field, so this
  run also produces the recovery-occupancy metric that the published CSVs
  could not (the "known gap" from the intervention-burden review).

Outputs (under ``results/diagnostics/release_patience_search/``, never touched
by the official pipeline):
- ``<benchmark>_reim_grid.csv``   one row per (noise, release, patience) cell
- ``<benchmark>_references.csv``  one row per (noise, method) act/heuristic
- ``<benchmark>_search.json``     protocol fingerprint + nested results
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.io import atomic_write_json, file_sha256
from evaluation import evaluate_multitask as evaluation
from models.bc_policy import ACTPolicy
from models.failure_detector import FailureDetector
from models.imitation_recovery_policy import ImitationRecoveryPolicy
from utils.common import configure_logging, seed_everything, select_device

SCHEMA_VERSION = "reim-release-patience-search-v1"

DEFAULT_VALIDATION_SEEDS = {"MT10": 20264010, "MT50": 20264050}
DEFAULT_TRIGGER_THRESHOLDS = {"MT10": 0.65, "MT50": 0.64}

GRID_FIELDS = (
    "condition",
    "noise_level",
    "release_threshold",
    "release_patience",
    "episodes",
    "micro_success",
    "task_macro_success",
    "intervened_episode_rate",
    "interventions_per_episode_mean",
    "interventions_per_episode_median",
    "release_rate",
    "median_trigger_step",
    "recovery_occupancy_mean",
    "mean_steps",
)

REFERENCE_FIELDS = (
    "condition",
    "noise_level",
    "method",
    "episodes",
    "micro_success",
    "task_macro_success",
    "mean_steps",
)


def _aggregate_reim(rows: list[dict[str, Any]], task_count: int) -> dict[str, Any]:
    """Aggregate a list of per-episode reim rollout records into one cell row."""
    total = len(rows)
    if total == 0:
        raise RuntimeError("Cannot aggregate an empty grid cell")
    successes = [int(row["success"]) for row in rows]
    micro = sum(successes) / total
    per_task: dict[int, list[int]] = {}
    for row in rows:
        per_task.setdefault(int(row["task_id"]), []).append(int(row["success"]))
    macro = statistics.mean(
        [statistics.mean(per_task[tid]) for tid in range(task_count) if tid in per_task]
    )
    intervened = [row for row in rows if int(row["intervention_count"]) > 0]
    intervened_rate = len(intervened) / total if total else 0.0
    interventions = [int(row["intervention_count"]) for row in rows]
    released = [
        row for row in intervened if int(row["recovery_success"]) >= 1
    ]
    release_rate = len(released) / len(intervened) if intervened else 0.0
    triggers = [
        int(row["trigger_step"])
        for row in rows
        if int(row["trigger_step"]) >= 0
    ]
    occupancy = [
        int(row["recovery_steps_total"]) / max(1, int(row["steps"])) for row in rows
    ]
    return {
        "episodes": total,
        "micro_success": round(micro, 6),
        "task_macro_success": round(float(macro), 6),
        "intervened_episode_rate": round(intervened_rate, 6),
        "interventions_per_episode_mean": round(
            float(statistics.mean(interventions)), 6
        ),
        "interventions_per_episode_median": round(
            float(statistics.median(interventions)), 6
        ),
        "release_rate": round(release_rate, 6),
        "median_trigger_step": (
            round(float(statistics.median(triggers)), 3) if triggers else -1
        ),
        "recovery_occupancy_mean": round(float(statistics.mean(occupancy)), 6),
        "mean_steps": round(float(statistics.mean(int(row["steps"]) for row in rows)), 3),
    }


def _aggregate_reference(
    rows: list[dict[str, Any]], task_count: int
) -> dict[str, Any]:
    total = len(rows)
    successes = [int(row["success"]) for row in rows]
    micro = sum(successes) / total
    per_task: dict[int, list[int]] = {}
    for row in rows:
        per_task.setdefault(int(row["task_id"]), []).append(int(row["success"]))
    macro = statistics.mean(
        [statistics.mean(per_task[tid]) for tid in range(task_count) if tid in per_task]
    )
    return {
        "episodes": total,
        "micro_success": round(micro, 6),
        "task_macro_success": round(float(macro), 6),
        "mean_steps": round(
            float(statistics.mean(int(row["steps"]) for row in rows)), 3
        ),
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_models(
    args: argparse.Namespace,
    task_count: int,
    task_vocabulary: list[str],
    device: str,
) -> tuple[ACTPolicy, FailureDetector, ImitationRecoveryPolicy]:
    act_path = Path(args.act_checkpoint).expanduser().resolve()
    detector_path = Path(args.detector_checkpoint).expanduser().resolve()
    recovery_path = Path(args.recovery_checkpoint).expanduser().resolve()
    act = ACTPolicy.from_checkpoint(act_path, map_location=device)
    detector = FailureDetector.from_checkpoint(
        detector_path, map_location=device, state_dim=39 + task_count
    )
    recovery = ImitationRecoveryPolicy.load(recovery_path, device=device)
    if {act.state_dim, detector.state_dim, recovery.state_dim} != {39 + task_count}:
        raise ValueError("Checkpoint observation dimensions do not match benchmark")
    if act.action_dim != 4 or recovery.action_dim != 4:
        raise ValueError("Checkpoint action dimensions do not match Meta-World")
    if not args.skip_provenance:
        sha256 = {
            "act": file_sha256(act_path),
            "detector": file_sha256(detector_path),
            "recovery": file_sha256(recovery_path),
        }
        for name, path, kind in (
            ("ACT", act_path, "act"),
            ("detector", detector_path, "detector"),
        ):
            evaluation._validate_multitask_provenance(
                name,
                evaluation._checkpoint_metadata(path),
                benchmark=args.benchmark,
                task_vocabulary=task_vocabulary,
                checkpoint_kind=kind,
            )
        evaluation._validate_multitask_provenance(
            "recovery",
            getattr(recovery, "provenance", None),
            benchmark=args.benchmark,
            task_vocabulary=task_vocabulary,
            checkpoint_kind="recovery",
            linked_checkpoint_sha256={
                "act_checkpoint_sha256": sha256["act"],
                "detector_checkpoint_sha256": sha256["detector"],
            },
        )
    return act, detector, recovery


def _run_cell(
    *,
    benchmark: Any,
    tasks_by_name: dict[str, list[Any]],
    task_vocabulary: list[str],
    task_count: int,
    method: str,
    episodes_per_task: int,
    episode_seed_base: int,
    max_steps: int,
    action_std: float,
    observation_std: float,
    act: ACTPolicy,
    detector: FailureDetector,
    recovery: ImitationRecoveryPolicy,
    threshold: float,
    release_threshold: float,
    release_patience: int,
    min_recovery_steps: int,
    intervention_cooldown: int,
    recovery_budget: int,
    heuristic_min_steps: int,
    heuristic_window: int,
    heuristic_tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_ids = list(range(task_count))
    for task_id in task_ids:
        task_name = task_vocabulary[task_id]
        env = benchmark.train_classes[task_name](render_mode=None)
        try:
            for episode_index in range(episodes_per_task):
                task = tasks_by_name[task_name][episode_index]
                episode_seed = episode_seed_base + task_id * 100_000 + episode_index
                action_noise, observation_noise = evaluation._noise_arrays(
                    episode_seed, max_steps, action_std, observation_std
                )
                result = evaluation._rollout(
                    env=env,
                    task=task,
                    task_id=task_id,
                    task_count=task_count,
                    method=method,
                    episode_seed=episode_seed,
                    max_steps=max_steps,
                    action_noise=action_noise,
                    observation_noise=observation_noise,
                    act=act,
                    detector=detector,
                    recovery=recovery,
                    threshold=threshold,
                    release_threshold=release_threshold,
                    release_patience=release_patience,
                    min_recovery_steps=min_recovery_steps,
                    intervention_cooldown=intervention_cooldown,
                    recovery_budget=recovery_budget,
                    heuristic_min_steps=heuristic_min_steps,
                    heuristic_window=heuristic_window,
                    heuristic_tolerance=heuristic_tolerance,
                )
                result["task_id"] = task_id
                rows.append(result)
        finally:
            env.close()
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("MT10", "MT50"))
    parser.add_argument("--benchmark-seed", type=int)
    parser.add_argument("--backend", default="metaworld", choices=("metaworld", "toy"))
    parser.add_argument("--act-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--release-thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    parser.add_argument(
        "--release-patiences", type=int, nargs="+", default=[1, 3, 5, 10]
    )
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-std-scale", type=float, default=0.40)
    parser.add_argument("--observation-std-scale", type=float, default=0.025)
    parser.add_argument("--min-recovery-steps", type=int, default=5)
    parser.add_argument("--intervention-cooldown", type=int, default=10)
    parser.add_argument("--recovery-budget", type=int, default=250)
    parser.add_argument("--heuristic-min-steps", type=int, default=30)
    parser.add_argument("--heuristic-window", type=int, default=20)
    parser.add_argument("--heuristic-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-provenance", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.benchmark_seed is None:
        args.benchmark_seed = DEFAULT_VALIDATION_SEEDS[args.benchmark]
    if args.threshold is None:
        args.threshold = DEFAULT_TRIGGER_THRESHOLDS[args.benchmark]
    if not args.release_thresholds or not args.release_patiences:
        parser.error("--release-thresholds and --release-patiences must be non-empty")
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")

    backend_module, backend_version = evaluation._backend_components(args.backend)
    benchmark = getattr(backend_module, args.benchmark)(seed=args.benchmark_seed)
    task_vocabulary = list(benchmark.train_classes.keys())
    task_count = len(task_vocabulary)
    expected = 10 if args.benchmark == "MT10" else 50
    if task_count != expected:
        raise RuntimeError(f"Expected {expected} tasks, got {task_count}")
    args.benchmark_module = backend_module

    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir or (PROJECT_ROOT / "results" / "diagnostics" / "release_patience_search"))
    grid_path = output_dir / f"{args.benchmark.lower()}_reim_grid.csv"
    reference_path = output_dir / f"{args.benchmark.lower()}_references.csv"
    json_path = output_dir / f"{args.benchmark.lower()}_search.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        "search_release_patience",
        output_dir / f"{args.benchmark.lower()}_search.log",
    )

    tasks_by_name = {
        name: [task for task in benchmark.train_tasks if task.env_name == name]
        for name in task_vocabulary
    }
    for name, variants in tasks_by_name.items():
        if len(variants) < args.episodes_per_task:
            raise RuntimeError(
                f"{args.benchmark} bank has {len(variants)} variants for {name}, "
                f"fewer than requested {args.episodes_per_task}"
            )

    act, detector, recovery = _load_models(args, task_count, task_vocabulary, device)
    checkpoint_sha256 = {
        "act": file_sha256(Path(args.act_checkpoint).expanduser().resolve()),
        "detector": file_sha256(Path(args.detector_checkpoint).expanduser().resolve()),
        "recovery": file_sha256(Path(args.recovery_checkpoint).expanduser().resolve()),
    }

    grid_rows = _read_csv(grid_path) if args.resume else []
    reference_rows = _read_csv(reference_path) if args.resume else []
    completed_grid = {
        (
            row["condition"],
            float(row["release_threshold"]),
            int(float(row["release_patience"])),
        )
        for row in grid_rows
    }
    completed_reference = {(row["condition"], row["method"]) for row in reference_rows}

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": args.benchmark,
        "benchmark_seed": args.benchmark_seed,
        "backend": args.backend,
        "backend_version": backend_version,
        "checkpoint_sha256": checkpoint_sha256,
        "trigger_threshold": args.threshold,
        "grid": {
            "release_thresholds": args.release_thresholds,
            "release_patiences": args.release_patiences,
            "noise_levels": args.noise_levels,
            "episodes_per_task": args.episodes_per_task,
            "episode_seed_base": args.seed,
        },
        "fixed_recovery_params": {
            "min_recovery_steps": args.min_recovery_steps,
            "intervention_cooldown": args.intervention_cooldown,
            "recovery_budget": args.recovery_budget,
        },
    }

    noise_levels = sorted(set(args.noise_levels))
    results: dict[str, Any] = {}
    for noise_level in noise_levels:
        condition = (
            "official_clean" if noise_level == 0.0 else f"robustness_noise_{int(noise_level * 100):02d}"
        )
        condition_results: dict[str, Any] = {
            "noise_level": noise_level,
            "references": {},
            "grid": {},
        }
        action_std = args.action_std_scale * noise_level
        observation_std = args.observation_std_scale * noise_level

        for method in ("act", "heuristic_recovery"):
            if (condition, method) in completed_reference:
                existing = next(
                    row
                    for row in reference_rows
                    if row["condition"] == condition and row["method"] == method
                )
                condition_results["references"][method] = existing
                logger.info("reference %s %s reused from resume", condition, method)
                continue
            logger.info(
                "reference %s %s (%d episodes/task)",
                condition,
                method,
                args.episodes_per_task,
            )
            reference_episodes = _run_cell(
                benchmark=benchmark,
                tasks_by_name=tasks_by_name,
                task_vocabulary=task_vocabulary,
                task_count=task_count,
                method=method,
                episodes_per_task=args.episodes_per_task,
                episode_seed_base=args.seed,
                max_steps=args.max_steps,
                action_std=action_std,
                observation_std=observation_std,
                act=act,
                detector=detector,
                recovery=recovery,
                threshold=args.threshold,
                release_threshold=args.release_thresholds[0],
                release_patience=args.release_patiences[0],
                min_recovery_steps=args.min_recovery_steps,
                intervention_cooldown=args.intervention_cooldown,
                recovery_budget=args.recovery_budget,
                heuristic_min_steps=args.heuristic_min_steps,
                heuristic_window=args.heuristic_window,
                heuristic_tolerance=args.heuristic_tolerance,
            )
            aggregated = _aggregate_reference(reference_episodes, task_count)
            row = {
                "condition": condition,
                "noise_level": noise_level,
                "method": method,
                **aggregated,
            }
            reference_rows.append(row)
            _write_csv(reference_path, REFERENCE_FIELDS, reference_rows)
            condition_results["references"][method] = row
            logger.info(
                "reference %s %s success %.3f", condition, method, row["micro_success"]
            )

        for release_threshold in args.release_thresholds:
            for release_patience in args.release_patiences:
                key = (condition, release_threshold, release_patience)
                if key in completed_grid:
                    existing = next(
                        row
                        for row in grid_rows
                        if row["condition"] == condition
                        and float(row["release_threshold"]) == release_threshold
                        and int(float(row["release_patience"])) == release_patience
                    )
                    condition_results["grid"][f"{release_threshold:g}/{release_patience}"] = existing
                    continue
                logger.info(
                    "%s release=%.2f patience=%d (%d episodes/task)",
                    condition,
                    release_threshold,
                    release_patience,
                    args.episodes_per_task,
                )
                episodes = _run_cell(
                    benchmark=benchmark,
                    tasks_by_name=tasks_by_name,
                    task_vocabulary=task_vocabulary,
                    task_count=task_count,
                    method="reim",
                    episodes_per_task=args.episodes_per_task,
                    episode_seed_base=args.seed,
                    max_steps=args.max_steps,
                    action_std=action_std,
                    observation_std=observation_std,
                    act=act,
                    detector=detector,
                    recovery=recovery,
                    threshold=args.threshold,
                    release_threshold=release_threshold,
                    release_patience=release_patience,
                    min_recovery_steps=args.min_recovery_steps,
                    intervention_cooldown=args.intervention_cooldown,
                    recovery_budget=args.recovery_budget,
                    heuristic_min_steps=args.heuristic_min_steps,
                    heuristic_window=args.heuristic_window,
                    heuristic_tolerance=args.heuristic_tolerance,
                )
                aggregated = _aggregate_reim(episodes, task_count)
                row = {
                    "condition": condition,
                    "noise_level": noise_level,
                    "release_threshold": release_threshold,
                    "release_patience": release_patience,
                    **aggregated,
                }
                grid_rows.append(row)
                _write_csv(grid_path, GRID_FIELDS, grid_rows)
                condition_results["grid"][f"{release_threshold:g}/{release_patience}"] = row
                logger.info(
                    "%s release=%.2f patience=%d macro=%.4f intervened=%.3f occ=%.3f",
                    condition,
                    release_threshold,
                    release_patience,
                    row["task_macro_success"],
                    row["intervened_episode_rate"],
                    row["recovery_occupancy_mean"],
                )
        results[condition] = condition_results

    atomic_write_json(json_path, {"protocol": protocol, "results": results})
    logger.info("search complete -> %s", output_dir)
    print(f"Search complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
