#!/usr/bin/env python3
"""Strict paired audit of heuristic and learned gates on one frozen CRN bank.

The script is intentionally analysis-only: it never runs a controller or
modifies any input result.  It validates every episode against the serialized
gate bank before atomically writing one deterministic JSON artifact.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.episode_bank import (  # noqa: E402
    bank_file_sha256,
    load_episode_bank,
    runtime_episode_specifications,
)
from utils.common import atomic_json_dump  # noqa: E402


SCHEMA_VERSION = 1
HEURISTIC_METHOD = "ACT + Heuristic Recovery"
REIM_METHOD = "REIM (ACT + Detector + Recovery)"
REQUIRED_FIELDS = {
    "method",
    "episode",
    "seed",
    "backend",
    "success",
    "recovery_attempts",
    "recovery_successes",
    "detector_triggers",
    "episode_specification_sha256",
    "episode_bank_sha256",
    "metaworld_task_sha256",
    "noise_level",
}


@dataclass(frozen=True)
class EpisodeSeries:
    """Validated, episode-sorted binary outcomes and intervention counts."""

    analysis_name: str
    path: Path
    method: str
    episodes: np.ndarray
    seeds: np.ndarray
    successes: np.ndarray
    recovery_attempts: np.ndarray
    recovery_successes: np.ndarray
    detector_triggers: np.ndarray
    specification_sha256: tuple[str, ...]
    bank_sha256: tuple[str, ...]
    task_sha256: tuple[str, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_bool(value: Any, *, path: Path, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{path}: row {row_number} has invalid boolean {value!r}")


def _parse_nonnegative_int(
    value: Any,
    *,
    field: str,
    path: Path,
    row_number: int,
) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: row {row_number} has invalid integer {field}={value!r}"
        ) from exc
    if result < 0:
        raise ValueError(
            f"{path}: row {row_number} has negative {field}={result}"
        )
    return result


def _validate_sha256(
    value: Any,
    *,
    field: str,
    path: Path,
    row_number: int,
) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(
            f"{path}: row {row_number} has invalid {field} SHA256"
        )
    return result


def read_episode_series(
    path: str | Path,
    *,
    analysis_name: str,
    expected_method: str,
    expected_backend: str,
    expected_noise_level: float,
    expected_episodes: int,
    expected_seed_start: int,
) -> EpisodeSeries:
    """Read and fail closed on malformed or protocol-incompatible raw rows."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source}: missing CSV header")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{source}: missing fields {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_episodes:
        raise ValueError(
            f"{source}: expected {expected_episodes} episodes, found {len(rows)}"
        )

    parsed: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        method = str(row["method"]).strip()
        if method != expected_method:
            raise ValueError(
                f"{source}: row {row_number} method {method!r} differs from "
                f"expected {expected_method!r}"
            )
        backend = str(row["backend"]).strip().lower()
        if backend != expected_backend:
            raise ValueError(
                f"{source}: row {row_number} backend {backend!r} differs from "
                f"expected {expected_backend!r}"
            )
        try:
            noise_level = float(row["noise_level"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: row {row_number} has invalid noise level"
            ) from exc
        if not math.isfinite(noise_level) or not np.isclose(
            noise_level,
            expected_noise_level,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{source}: row {row_number} noise={noise_level} differs "
                f"from expected {expected_noise_level}"
            )
        episode = _parse_nonnegative_int(
            row["episode"],
            field="episode",
            path=source,
            row_number=row_number,
        )
        seed = _parse_nonnegative_int(
            row["seed"],
            field="seed",
            path=source,
            row_number=row_number,
        )
        attempts = _parse_nonnegative_int(
            row["recovery_attempts"],
            field="recovery_attempts",
            path=source,
            row_number=row_number,
        )
        recovery_successes = _parse_nonnegative_int(
            row["recovery_successes"],
            field="recovery_successes",
            path=source,
            row_number=row_number,
        )
        detector_triggers = _parse_nonnegative_int(
            row["detector_triggers"],
            field="detector_triggers",
            path=source,
            row_number=row_number,
        )
        if recovery_successes > attempts:
            raise ValueError(
                f"{source}: row {row_number} has more recovery successes "
                "than attempts"
            )
        parsed.append(
            {
                "method": method,
                "episode": episode,
                "seed": seed,
                "success": _parse_bool(
                    row["success"],
                    path=source,
                    row_number=row_number,
                ),
                "recovery_attempts": attempts,
                "recovery_successes": recovery_successes,
                "detector_triggers": detector_triggers,
                "specification_sha256": _validate_sha256(
                    row["episode_specification_sha256"],
                    field="episode specification",
                    path=source,
                    row_number=row_number,
                ),
                "bank_sha256": _validate_sha256(
                    row["episode_bank_sha256"],
                    field="episode bank",
                    path=source,
                    row_number=row_number,
                ),
                "task_sha256": _validate_sha256(
                    row["metaworld_task_sha256"],
                    field="Meta-World task",
                    path=source,
                    row_number=row_number,
                ),
            }
        )

    parsed.sort(key=lambda row: row["episode"])
    episodes = np.asarray([row["episode"] for row in parsed], dtype=np.int64)
    seeds = np.asarray([row["seed"] for row in parsed], dtype=np.int64)
    expected_episode_ids = np.arange(expected_episodes, dtype=np.int64)
    expected_seeds = expected_seed_start + expected_episode_ids
    if not np.array_equal(episodes, expected_episode_ids):
        raise ValueError(f"{source}: episode IDs are not exactly 0..N-1")
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError(
            f"{source}: seeds are not exactly "
            f"{expected_seed_start}..{expected_seed_start + expected_episodes - 1}"
        )

    return EpisodeSeries(
        analysis_name=analysis_name,
        path=source,
        method=expected_method,
        episodes=episodes,
        seeds=seeds,
        successes=np.asarray(
            [row["success"] for row in parsed],
            dtype=np.bool_,
        ),
        recovery_attempts=np.asarray(
            [row["recovery_attempts"] for row in parsed],
            dtype=np.int64,
        ),
        recovery_successes=np.asarray(
            [row["recovery_successes"] for row in parsed],
            dtype=np.int64,
        ),
        detector_triggers=np.asarray(
            [row["detector_triggers"] for row in parsed],
            dtype=np.int64,
        ),
        specification_sha256=tuple(
            row["specification_sha256"] for row in parsed
        ),
        bank_sha256=tuple(row["bank_sha256"] for row in parsed),
        task_sha256=tuple(row["task_sha256"] for row in parsed),
    )


def exact_two_sided_mcnemar_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided sign/binomial test over discordant paired outcomes."""

    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    one_sided = sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2.0**discordant)
    return float(min(1.0, 2.0 * one_sided))


def paired_bootstrap_delta_ci(
    reference: Sequence[bool] | np.ndarray,
    candidate: Sequence[bool] | np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI for the paired candidate-minus-reference rate."""

    reference_array = np.asarray(reference, dtype=np.bool_).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.bool_).reshape(-1)
    if reference_array.shape != candidate_array.shape or reference_array.size == 0:
        raise ValueError("paired outcomes must have the same non-empty shape")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    differences = (
        candidate_array.astype(np.float64)
        - reference_array.astype(np.float64)
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = min(samples, 2_048)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        estimates[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def paired_binary_comparison(
    reference: Sequence[bool] | np.ndarray,
    candidate: Sequence[bool] | np.ndarray,
    *,
    reference_name: str,
    candidate_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return paired delta, discordances, bootstrap CI, and exact p-value."""

    reference_array = np.asarray(reference, dtype=np.bool_).reshape(-1)
    candidate_array = np.asarray(candidate, dtype=np.bool_).reshape(-1)
    if reference_array.shape != candidate_array.shape or reference_array.size == 0:
        raise ValueError("paired outcomes must have the same non-empty shape")
    wins = int(np.count_nonzero(~reference_array & candidate_array))
    losses = int(np.count_nonzero(reference_array & ~candidate_array))
    both_success = int(np.count_nonzero(reference_array & candidate_array))
    both_failure = int(np.count_nonzero(~reference_array & ~candidate_array))
    delta = float(candidate_array.mean() - reference_array.mean())
    lower, upper = paired_bootstrap_delta_ci(
        reference_array,
        candidate_array,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    clean = lambda value: float(round(float(value), 12))
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "episodes": int(reference_array.size),
        "candidate_rate": clean(candidate_array.mean()),
        "reference_rate": clean(reference_array.mean()),
        "paired_delta": clean(delta),
        "paired_delta_percentage_points": clean(100.0 * delta),
        "paired_delta_bootstrap_95_ci": [clean(lower), clean(upper)],
        "paired_delta_bootstrap_95_ci_percentage_points": [
            clean(100.0 * lower),
            clean(100.0 * upper),
        ],
        "wins": wins,
        "losses": losses,
        "discordant_pairs": wins + losses,
        "both_success": both_success,
        "both_failure": both_failure,
        "exact_two_sided_mcnemar_binomial_p": (
            exact_two_sided_mcnemar_binomial_p(wins, losses)
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def summarize_method(series: EpisodeSeries) -> dict[str, Any]:
    episodes = int(series.successes.size)
    attempts = int(series.recovery_attempts.sum())
    recovery_successes = int(series.recovery_successes.sum())
    intervened = series.recovery_attempts > 0
    intervened_episodes = int(intervened.sum())
    post_intervention_successes = int(
        np.count_nonzero(intervened & series.successes)
    )
    recovery_rate = (
        float(recovery_successes / attempts) if attempts else 0.0
    )
    return {
        "analysis_name": series.analysis_name,
        "method": series.method,
        "episodes": episodes,
        "successes": int(series.successes.sum()),
        "success_rate": float(series.successes.mean()),
        "intervention_attempts": attempts,
        "intervened_episodes": intervened_episodes,
        "intervention_rate": float(intervened_episodes / episodes),
        "intervention_rate_definition": (
            "episodes_with_at_least_one_recovery_attempt / episodes"
        ),
        "mean_interventions_per_episode": float(attempts / episodes),
        "detector_triggers": int(series.detector_triggers.sum()),
        "recovery_successes": recovery_successes,
        "recovery_rate": recovery_rate,
        "recovery_rate_per_attempt": recovery_rate,
        "recovery_rate_definition": (
            "sum(recovery_successes) / sum(recovery_attempts)"
        ),
        "post_intervention_episode_successes": post_intervention_successes,
        "post_intervention_episode_success_rate": (
            float(post_intervention_successes / intervened_episodes)
            if intervened_episodes
            else 0.0
        ),
    }


def _validate_bank_and_pairing(
    *,
    bank: Mapping[str, Any],
    series: Sequence[EpisodeSeries],
    expected_task_bank_seed: int,
    expected_seed_start: int,
    expected_episodes: int,
    expected_noise_level: float,
) -> None:
    if str(bank["backend"]).lower() != "metaworld":
        raise ValueError("gate bank backend must be Meta-World")
    if int(bank["task_bank_seed"]) != expected_task_bank_seed:
        raise ValueError("gate bank task seed mismatch")
    if int(bank["episode_seed_start"]) != expected_seed_start:
        raise ValueError("gate bank evaluation seed start mismatch")
    if int(bank["episodes"]) != expected_episodes:
        raise ValueError("gate bank episode count mismatch")
    expected_noise = {
        "action_noise_std": 0.40 * expected_noise_level,
        "observation_noise_std": 0.025 * expected_noise_level,
        "object_noise_probability": 1.0 if expected_noise_level > 0.0 else 0.0,
        "object_noise_std": 0.10 * expected_noise_level,
    }
    for field, expected in expected_noise.items():
        if not np.isclose(
            float(bank["disturbance"][field]),
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"gate bank {field} differs from protocol")

    runtime_specs = runtime_episode_specifications(
        bank,
        offset=0,
        count=expected_episodes,
    )
    expected_spec_sha = tuple(
        str(spec["specification_sha256"]) for spec in runtime_specs
    )
    expected_task_sha = tuple(str(spec["task_sha256"]) for spec in runtime_specs)
    expected_bank_sha = tuple(
        str(bank["bank_sha256"]) for _ in range(expected_episodes)
    )
    reference = series[0]
    for candidate in series:
        if not np.array_equal(candidate.episodes, reference.episodes):
            raise ValueError("episode IDs differ across matched-gate inputs")
        if not np.array_equal(candidate.seeds, reference.seeds):
            raise ValueError("episode seeds differ across matched-gate inputs")
        if candidate.specification_sha256 != reference.specification_sha256:
            raise ValueError("episode specification hashes differ across inputs")
        if candidate.bank_sha256 != reference.bank_sha256:
            raise ValueError("episode bank hashes differ across inputs")
        if candidate.task_sha256 != reference.task_sha256:
            raise ValueError("Meta-World task hashes differ across inputs")
        if candidate.specification_sha256 != expected_spec_sha:
            raise ValueError("raw specification hashes differ from gate bank")
        if candidate.bank_sha256 != expected_bank_sha:
            raise ValueError("raw episode-bank hashes differ from gate bank")
        if candidate.task_sha256 != expected_task_sha:
            raise ValueError("raw task hashes differ from gate bank")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    bank_path = args.episode_bank.expanduser().resolve()
    bank = load_episode_bank(bank_path)
    heuristic = read_episode_series(
        args.heuristic,
        analysis_name="heuristic_gate",
        expected_method=HEURISTIC_METHOD,
        expected_backend="metaworld",
        expected_noise_level=args.noise_level,
        expected_episodes=args.episodes,
        expected_seed_start=args.seed_start,
    )
    tau_0175 = read_episode_series(
        args.tau_0175,
        analysis_name="reim_tau_0.175",
        expected_method=REIM_METHOD,
        expected_backend="metaworld",
        expected_noise_level=args.noise_level,
        expected_episodes=args.episodes,
        expected_seed_start=args.seed_start,
    )
    tau_020 = read_episode_series(
        args.tau_020,
        analysis_name="reim_tau_0.20",
        expected_method=REIM_METHOD,
        expected_backend="metaworld",
        expected_noise_level=args.noise_level,
        expected_episodes=args.episodes,
        expected_seed_start=args.seed_start,
    )
    all_series = (heuristic, tau_0175, tau_020)
    _validate_bank_and_pairing(
        bank=bank,
        series=all_series,
        expected_task_bank_seed=args.task_bank_seed,
        expected_seed_start=args.seed_start,
        expected_episodes=args.episodes,
        expected_noise_level=args.noise_level,
    )

    tau_0175_success = paired_binary_comparison(
        heuristic.successes,
        tau_0175.successes,
        reference_name=heuristic.analysis_name,
        candidate_name=tau_0175.analysis_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    tau_020_success = paired_binary_comparison(
        heuristic.successes,
        tau_020.successes,
        reference_name=heuristic.analysis_name,
        candidate_name=tau_020.analysis_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1,
    )
    threshold_success = paired_binary_comparison(
        tau_020.successes,
        tau_0175.successes,
        reference_name=tau_020.analysis_name,
        candidate_name=tau_0175.analysis_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 2,
    )
    intervention_comparison = paired_binary_comparison(
        heuristic.recovery_attempts > 0,
        tau_0175.recovery_attempts > 0,
        reference_name=heuristic.analysis_name,
        candidate_name=tau_0175.analysis_name,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 3,
    )

    inputs = {
        series_item.analysis_name: {
            "path": str(series_item.path),
            "sha256": file_sha256(series_item.path),
            "method": series_item.method,
            "rows": int(series_item.episodes.size),
        }
        for series_item in all_series
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_type": "strict_matched_gate_paired_comparison",
        "analysis_only": True,
        "protocol": {
            "backend": "metaworld",
            "noise_level": float(args.noise_level),
            "task_bank_seed": int(args.task_bank_seed),
            "episode_seed_start": int(args.seed_start),
            "episode_seed_end": int(args.seed_start + args.episodes - 1),
            "episodes": int(args.episodes),
            "episode_bank": str(bank_path),
            "episode_bank_file_sha256": bank_file_sha256(bank_path),
            "episode_bank_sha256": str(bank["bank_sha256"]),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "inputs": inputs,
        "validation": {
            "all_checks_passed": True,
            "expected_method_per_input_verified": True,
            "reim_method_equal_across_thresholds": (
                tau_0175.method == tau_020.method == REIM_METHOD
            ),
            "heuristic_method_distinct_by_design": (
                heuristic.method == HEURISTIC_METHOD
            ),
            "backend_equal": True,
            "noise_level_equal": True,
            "episode_ids_equal": True,
            "episode_seeds_equal": True,
            "episode_specification_sha256_equal": True,
            "episode_bank_sha256_equal": True,
            "metaworld_task_sha256_equal": True,
            "raw_rows_match_serialized_bank": True,
        },
        "methods": {
            item.analysis_name: summarize_method(item) for item in all_series
        },
        "paired_success_comparisons": {
            "reim_tau_0.175_minus_heuristic": tau_0175_success,
            "reim_tau_0.20_minus_heuristic": tau_020_success,
            "reim_tau_0.175_minus_reim_tau_0.20": threshold_success,
        },
        "tau_0.175_minus_heuristic_intervention_rate": intervention_comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heuristic",
        type=Path,
        default=ROOT
        / "results/tables/formal_crn/gate_matched_heuristic_episodes.csv",
    )
    parser.add_argument(
        "--tau-0175",
        dest="tau_0175",
        type=Path,
        default=ROOT / "results/tables/gate_calibration_tau0175_episodes.csv",
    )
    parser.add_argument(
        "--tau-020",
        dest="tau_020",
        type=Path,
        default=ROOT / "results/tables/gate_calibration_tau020_episodes.csv",
    )
    parser.add_argument(
        "--episode-bank",
        type=Path,
        default=ROOT
        / "datasets/evaluation/"
        "pickplace_gate_task20260726_ep8200042_n200_noise020.json",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--task-bank-seed", type=int, default=20_260_726)
    parser.add_argument("--seed-start", type=int, default=8_200_042)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=8_200_042)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/tables/gate_matched_comparison.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    report = build_report(args)
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validation_passed": report["validation"]["all_checks_passed"],
                "methods": report["methods"],
                "paired_success_comparisons": report[
                    "paired_success_comparisons"
                ],
                "tau_0.175_minus_heuristic_intervention_rate": report[
                    "tau_0.175_minus_heuristic_intervention_rate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


if __name__ == "__main__":
    main()
