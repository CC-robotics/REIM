#!/usr/bin/env python3
"""Reuse paired full-protocol episodes for ablation and comparison tables.

The primary baseline already evaluates ACT, ACT+Recovery, and REIM with the
same seeds and disturbance condition required by the ablation/comparison.
This utility combines those measured episodes with one ACT+Detector run,
avoiding scientifically redundant simulator rollouts while preserving 1,000
episodes per condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_reim import _atomic_write_csv  # noqa: E402
from evaluation.metrics import bootstrap_ci  # noqa: E402
from utils.common import atomic_json_dump  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty measured result: {path}")
    return rows


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    selected = [row for row in rows if row.get("Method") == method]
    if len(selected) != 1:
        raise ValueError(f"expected one summary for {method!r}, found {len(selected)}")
    return selected[0]


def _validate_protocol(rows: list[dict[str, str]], *, episodes: int) -> None:
    for row in rows:
        if row.get("Backend") != "metaworld" or row.get("Profile") != "full":
            raise ValueError("cached scientific results must be full Meta-World runs")
        if int(float(row.get("Episodes", 0))) != episodes:
            raise ValueError(f"cached result does not contain {episodes} episodes")


def build(args: argparse.Namespace) -> None:
    baseline_rows = _read(args.baseline)
    detector_rows = _read(args.detector_only)
    baseline_episodes = _read(args.baseline_episodes)
    detector_episodes = _read(args.detector_only_episodes)
    _validate_protocol(baseline_rows, episodes=args.episodes)
    _validate_protocol(detector_rows, episodes=args.episodes)

    expected = {
        "ACT",
        "ACT + Random Reset",
        "ACT + Heuristic Recovery",
        "REIM (ACT + Detector + Recovery)",
    }
    if {row.get("Method") for row in baseline_rows} != expected:
        raise ValueError("baseline cache does not contain the canonical four methods")
    detector_summary = _one(detector_rows, "ACT + Detector")
    all_summaries = baseline_rows + detector_rows
    protocol_signatures = {
        (
            float(row.get("Noise Level", "nan")),
            float(row.get("Action Noise Std", "nan")),
            float(row.get("Observation Noise Std", "nan")),
            float(row.get("Object Impulse Std (m)", "nan")),
        )
        for row in all_summaries
    }
    if len(protocol_signatures) != 1:
        raise ValueError(
            "cached baseline and detector-only runs use different disturbances"
        )
    episode_sources = baseline_episodes + detector_episodes
    expected_episode_methods = expected | {"ACT + Detector"}
    seeds_by_method: dict[str, set[int]] = {}
    for row in episode_sources:
        method = str(row.get("method", ""))
        if method not in expected_episode_methods:
            continue
        if (
            str(row.get("backend", "")).lower() != "metaworld"
            or str(row.get("Profile", row.get("profile", ""))).lower() != "full"
        ):
            raise ValueError("cached episode provenance is not full Meta-World")
        if not np.isclose(float(row.get("noise_level", "nan")), 0.2):
            raise ValueError("cached episode does not use the 20% main condition")
        seeds_by_method.setdefault(method, set()).add(int(row["seed"]))
    if set(seeds_by_method) != expected_episode_methods:
        raise ValueError("cached episodes do not contain all five controllers")
    if any(len(seeds) != args.episodes for seeds in seeds_by_method.values()):
        raise ValueError("cached controller episode counts or seeds are invalid")
    reference_seeds = next(iter(seeds_by_method.values()))
    if any(seeds != reference_seeds for seeds in seeds_by_method.values()):
        raise ValueError("cached controllers are not evaluated on paired seeds")

    sources = {
        "A": _one(baseline_rows, "ACT"),
        "B": detector_summary,
        "C": _one(baseline_rows, "ACT + Heuristic Recovery"),
        "D": _one(baseline_rows, "REIM (ACT + Detector + Recovery)"),
    }
    public_names = {
        "A": "ACT",
        "B": "ACT + Detector",
        "C": "ACT + Recovery",
        "D": "REIM",
    }
    ablation_rows: list[dict[str, Any]] = []
    for variant in ("A", "B", "C", "D"):
        source = dict(sources[variant])
        controller = str(source.pop("Method"))
        ablation_rows.append(
            {
                "Variant": variant,
                "Method": public_names[variant],
                **source,
                "Controller": controller,
            }
        )
    _atomic_write_csv(args.ablation, ablation_rows)

    controller_to_variant = {
        str(row["Controller"]): (str(row["Variant"]), str(row["Method"]))
        for row in ablation_rows
    }
    combined_episodes = baseline_episodes + detector_episodes
    ablation_episode_rows: list[dict[str, Any]] = []
    for row in combined_episodes:
        controller = str(row.get("method"))
        if controller not in controller_to_variant:
            continue
        variant, public_name = controller_to_variant[controller]
        converted: dict[str, Any] = dict(row)
        converted["variant"] = variant
        converted["method"] = public_name
        converted["controller"] = controller
        ablation_episode_rows.append(converted)
    counts = {
        variant: sum(row["variant"] == variant for row in ablation_episode_rows)
        for variant in ("A", "B", "C", "D")
    }
    if set(counts.values()) != {args.episodes}:
        raise ValueError(f"ablation episode counts are invalid: {counts}")
    _atomic_write_csv(
        args.ablation.with_name(f"{args.ablation.stem}_episodes.csv"),
        ablation_episode_rows,
    )

    by_method: dict[str, dict[int, bool]] = {}
    for row in baseline_episodes:
        by_method.setdefault(str(row["method"]), {})[int(row["seed"])] = _truth(
            row["success"]
        )
    act = by_method["ACT"]
    comparison_rows: list[dict[str, Any]] = []
    for index, source in enumerate(baseline_rows):
        row: dict[str, Any] = dict(source)
        label = str(source["Method"])
        shared = sorted(set(act) & set(by_method[label]))
        if len(shared) != args.episodes:
            raise ValueError(f"{label} does not have {args.episodes} paired seeds")
        differences = np.asarray(
            [float(by_method[label][seed]) - float(act[seed]) for seed in shared]
        )
        lower, upper = bootstrap_ci(
            differences,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + 17 * index,
        )
        row["Success Gain vs ACT"] = float(differences.mean())
        row["Gain CI Lower"] = lower
        row["Gain CI Upper"] = upper
        comparison_rows.append(row)
    ranked = sorted(
        comparison_rows, key=lambda row: float(row["Success Rate"]), reverse=True
    )
    distinct_rates = sorted(
        {float(row["Success Rate"]) for row in ranked}, reverse=True
    )
    dense_rank = {rate: index + 1 for index, rate in enumerate(distinct_rates)}
    for row in ranked:
        row["Rank"] = dense_rank[float(row["Success Rate"])]
    _atomic_write_csv(args.comparison, ranked)
    _atomic_write_csv(
        args.comparison.with_name(f"{args.comparison.stem}_episodes.csv"),
        baseline_episodes,
    )

    inputs = (
        args.baseline,
        args.baseline_episodes,
        args.detector_only,
        args.detector_only_episodes,
    )
    atomic_json_dump(
        {
            "strategy": "paired_episode_reuse",
            "episodes_per_condition": args.episodes,
            "seed": args.seed,
            "evaluation_seed_start": min(reference_seeds),
            "evaluation_seed_end": max(reference_seeds),
            "bootstrap_samples": args.bootstrap_samples,
            "inputs": {str(path): _sha256(path) for path in inputs},
            "outputs": {
                "ablation": str(args.ablation),
                "comparison": str(args.comparison),
            },
        },
        args.manifest,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--baseline-episodes", type=Path, required=True)
    result.add_argument("--detector-only", type=Path, required=True)
    result.add_argument("--detector-only-episodes", type=Path, required=True)
    result.add_argument(
        "--ablation",
        type=Path,
        default=PROJECT_ROOT / "results/tables/ablation.csv",
    )
    result.add_argument(
        "--comparison",
        type=Path,
        default=PROJECT_ROOT / "results/tables/comparison.csv",
    )
    result.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results/tables/evaluation_reuse_manifest.json",
    )
    result.add_argument("--episodes", type=int, default=1000)
    result.add_argument("--bootstrap-samples", type=int, default=2000)
    result.add_argument("--seed", type=int, default=42)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
