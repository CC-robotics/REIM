#!/usr/bin/env python3
"""Merge independently evaluated result shards with protocol validation.

The full experiment can evaluate controllers in parallel.  This utility
combines their summary and episode CSVs only after checking that every summary
comes from the requested backend/profile and that each condition contains the
declared number of episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_reim import _atomic_write_csv  # noqa: E402
from evaluation.metrics import aggregate_episode_metrics  # noqa: E402
from utils.common import atomic_json_dump  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"result shard is empty: {path}")
    return rows


def _episode_path(summary_path: Path) -> Path:
    return summary_path.with_name(f"{summary_path.stem}_episodes.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_key(
    row: dict[str, str],
    fields: Sequence[str],
) -> tuple[str, ...]:
    try:
        return tuple(str(row[field]) for field in fields)
    except KeyError as exc:
        raise ValueError(f"missing condition key field: {exc.args[0]}") from exc


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def merge(args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    seen: set[tuple[str, ...]] = set()
    paired_seed_sets: list[set[int]] = []
    crn_maps: list[
        tuple[
            tuple[str, ...],
            dict[int, tuple[int, str, str]],
        ]
    ] = []
    bank_hashes: set[str] = set()

    for summary_path in args.summaries:
        shard_summaries = _read(summary_path)
        episode_path = _episode_path(summary_path)
        shard_episodes = _read(episode_path)
        inputs.extend((summary_path, episode_path))
        for row in shard_summaries:
            if str(row.get("Backend", "")).lower() != args.backend:
                raise ValueError(
                    f"{summary_path} backend is not {args.backend!r}: "
                    f"{row.get('Backend')!r}"
                )
            if str(row.get("Profile", "")).lower() != args.profile:
                raise ValueError(
                    f"{summary_path} profile is not {args.profile!r}: "
                    f"{row.get('Profile')!r}"
                )
            if int(float(row.get("Episodes", 0))) != args.episodes_per_condition:
                raise ValueError(
                    f"{summary_path} does not contain "
                    f"{args.episodes_per_condition} episodes per condition"
                )
            key = _condition_key(row, args.key_fields)
            if key in seen:
                raise ValueError(f"duplicate condition across shards: {key}")
            seen.add(key)
            method = str(row.get("Method", ""))
            matches = [
                episode
                for episode in shard_episodes
                if str(episode.get("method", "")) == method
                and (
                    "Noise Level" not in row
                    or np.isclose(
                        float(episode.get("noise_level", "nan")),
                        float(row["Noise Level"]),
                    )
                )
            ]
            if len(matches) != args.episodes_per_condition:
                raise ValueError(
                    f"{summary_path} raw rows for {key} number {len(matches)}, "
                    f"expected {args.episodes_per_condition}"
                )
            episode_ids = [int(float(item["episode"])) for item in matches]
            seeds = [int(float(item["seed"])) for item in matches]
            if len(set(episode_ids)) != len(episode_ids):
                raise ValueError(f"duplicate episode indices for condition {key}")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"duplicate evaluation seeds for condition {key}")
            for episode in matches:
                if str(episode.get("backend", "")).lower() != args.backend:
                    raise ValueError(f"raw episode backend mismatch for {key}")
                episode_profile = str(
                    episode.get("Profile", episode.get("profile", ""))
                ).lower()
                if episode_profile != args.profile:
                    raise ValueError(f"raw episode profile mismatch for {key}")
            require_crn = args.require_episode_bank or _truth(
                row.get("Benchmark Eligible", False)
            )
            crn_map: dict[int, tuple[int, str, str]] = {}
            for episode in matches:
                bank_sha = str(episode.get("episode_bank_sha256", "")).strip()
                spec_sha = str(
                    episode.get("episode_specification_sha256", "")
                ).strip()
                task_sha = str(
                    episode.get("metaworld_task_sha256", "")
                ).strip()
                if require_crn and not all((bank_sha, spec_sha, task_sha)):
                    raise ValueError(
                        f"raw episode is missing bank/spec/task SHA256 for {key}"
                    )
                if bank_sha:
                    bank_hashes.add(bank_sha)
                    if (
                        args.episode_bank_sha256
                        and bank_sha not in args.episode_bank_sha256
                    ):
                        raise ValueError(
                            f"unexpected episode-bank SHA256 for {key}: {bank_sha}"
                        )
                    crn_map[int(float(episode["seed"]))] = (
                        int(float(episode["episode"])),
                        spec_sha,
                        task_sha,
                    )
            protocol = (
                str(row.get("Noise Level", "")),
                str(row.get("Action Noise Std", "")),
                str(row.get("Observation Noise Std", "")),
                str(row.get("Object Impulse Std (m)", "")),
            )
            if crn_map:
                if len(crn_map) != args.episodes_per_condition:
                    raise ValueError(f"incomplete CRN map for condition {key}")
                crn_maps.append((protocol, crn_map))
            paired_seed_sets.append(set(seeds))
            recomputed = aggregate_episode_metrics(
                matches,
                method=method,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed + len(summary_rows) * 17,
            )
            recomputed["Benchmark Eligible"] = (
                args.backend == "metaworld" and args.profile == "full"
            )
            recomputed["Evaluation Seed Start"] = min(seeds)
            recomputed["Evaluation Seed End"] = max(seeds)
            summary_rows.append({**row, **recomputed})
        expected_episode_rows = (
            len(shard_summaries) * args.episodes_per_condition
        )
        if len(shard_episodes) != expected_episode_rows:
            raise ValueError(
                f"{episode_path} has {len(shard_episodes)} rows, expected "
                f"{expected_episode_rows}"
            )
        episode_rows.extend(shard_episodes)

    if len(summary_rows) != args.expected_conditions:
        raise ValueError(
            f"found {len(summary_rows)} conditions, expected "
            f"{args.expected_conditions}"
        )
    expected_total = args.expected_conditions * args.episodes_per_condition
    if len(episode_rows) != expected_total:
        raise ValueError(
            f"found {len(episode_rows)} episode rows, expected {expected_total}"
        )
    if paired_seed_sets and any(
        seeds != paired_seed_sets[0] for seeds in paired_seed_sets[1:]
    ):
        raise ValueError("conditions do not use the same paired evaluation seeds")

    crn_verified = bool(crn_maps)
    maps_by_protocol: dict[
        tuple[str, ...],
        list[dict[int, tuple[int, str, str]]],
    ] = {}
    for protocol, mapping in crn_maps:
        maps_by_protocol.setdefault(protocol, []).append(mapping)
    for protocol, mappings in maps_by_protocol.items():
        reference = mappings[0]
        if any(mapping != reference for mapping in mappings[1:]):
            raise ValueError(
                "conditions with the same disturbance protocol do not use "
                f"identical episode/spec/task mappings: {protocol}"
            )
    if args.require_episode_bank and len(crn_maps) != len(summary_rows):
        raise ValueError("not every merged condition has complete CRN provenance")

    # Across robustness levels the disturbance spec hash legitimately changes,
    # but task identity and episode index must remain paired.
    if crn_maps:
        reference_tasks = {
            seed: (episode_index, task_sha)
            for seed, (episode_index, _, task_sha) in crn_maps[0][1].items()
        }
        for _, mapping in crn_maps[1:]:
            candidate_tasks = {
                seed: (episode_index, task_sha)
                for seed, (episode_index, _, task_sha) in mapping.items()
            }
            if candidate_tasks != reference_tasks:
                raise ValueError(
                    "conditions do not share identical episode/task mappings"
                )

    protocol_signatures = {
        (
            row.get("Noise Level", ""),
            row.get("Action Noise Std", ""),
            row.get("Observation Noise Std", ""),
            row.get("Object Impulse Std (m)", ""),
        )
        for row in summary_rows
    }
    noise_levels = {signature[0] for signature in protocol_signatures}
    if len(protocol_signatures) != len(noise_levels):
        raise ValueError("noise levels map to inconsistent physical disturbance scales")

    for row in summary_rows:
        row["CRN Episode Specifications Verified"] = crn_verified
        row["Benchmark Eligible"] = bool(
            args.backend == "metaworld"
            and args.profile == "full"
            and crn_verified
        )

    _atomic_write_csv(args.output, summary_rows)
    episode_output = _episode_path(args.output)
    _atomic_write_csv(episode_output, episode_rows)
    manifest_path = args.output.with_name(f"{args.output.stem}_merge_manifest.json")
    atomic_json_dump(
        {
            "strategy": "validated_parallel_shard_merge",
            "backend": args.backend,
            "profile": args.profile,
            "condition_key_fields": list(args.key_fields),
            "conditions": [list(key) for key in seen],
            "episodes_per_condition": args.episodes_per_condition,
            "evaluation_seed_start": min(paired_seed_sets[0]),
            "evaluation_seed_end": max(paired_seed_sets[0]),
            "summary_recomputed_from_raw_episodes": True,
            "crn_episode_specifications_verified": crn_verified,
            "episode_bank_sha256s": sorted(bank_hashes),
            "summary_rows": len(summary_rows),
            "episode_rows": len(episode_rows),
            "inputs": {str(path): _sha256(path) for path in inputs},
            "outputs": {
                "summary": str(args.output),
                "episodes": str(episode_output),
            },
        },
        manifest_path,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "conditions": len(summary_rows),
                "episodes": len(episode_rows),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="metaworld")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--episodes-per-condition", type=int, required=True)
    parser.add_argument("--expected-conditions", type=int, required=True)
    parser.add_argument("--key-fields", nargs="+", default=["Method"])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--require-episode-bank",
        action="store_true",
        help="Fail unless every raw row has validated bank/spec/task hashes.",
    )
    parser.add_argument(
        "--episode-bank-sha256",
        nargs="*",
        default=[],
        help="Optional allow-list of canonical episode-bank SHA256 values.",
    )
    return parser


if __name__ == "__main__":
    merge(build_parser().parse_args())
