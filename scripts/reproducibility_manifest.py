#!/usr/bin/env python3
"""Write an auditable manifest for a completed REIM run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from utils.common import PROJECT_ROOT, atomic_json_dump


TRACKED_PACKAGES = (
    "torch",
    "gymnasium",
    "metaworld",
    "mujoco",
    "stable-baselines3",
    "numpy",
    "scikit-learn",
    "pandas",
)

EPISODE_BANK_TYPE = "reim_pickplace_common_random_numbers"
GATE_AUDIT_SEED_OFFSET = 200_000
FORMAL_BANK_SLOTS = (
    ("baseline_and_robustness_noise_20", "crn", 0, 1000, "020"),
    ("robustness_noise_00", "crn", 0, 200, "000"),
    ("robustness_noise_10", "crn", 0, 200, "010"),
    ("robustness_noise_30", "crn", 0, 200, "030"),
    ("robustness_noise_40", "crn", 0, 200, "040"),
    (
        "matched_gate_noise_20",
        "gate",
        GATE_AUDIT_SEED_OFFSET,
        200,
        "020",
    ),
)
FORMAL_PAPER_FIGURE_STEMS = (
    "Figure1_final_framework",
    "Figure2_final_results",
    "Figure3_detector",
    "Figure4_gate_sensitivity",
    "Figure5_operation_sequence",
    "Figure3_final_ablation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any] | None:
    """Return one project-relative content-addressed artifact record."""

    if not path.is_file():
        return None
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_metadata() -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", str(PROJECT_ROOT)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"root": None, "commit": None, "dirty": None}


def hardware_metadata() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
    }
    try:
        import torch

        payload["torch_cuda_available"] = torch.cuda.is_available()
        payload["torch_cuda_version"] = torch.version.cuda
        payload["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        payload["torch_cuda_available"] = False
    return payload


def collect_artifacts() -> list[dict[str, Any]]:
    roots = (
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "datasets" / "evaluation",
        PROJECT_ROOT / "results" / "tables",
        PROJECT_ROOT / "results" / "figures",
        PROJECT_ROOT / "results" / "traces",
        PROJECT_ROOT / "paper_assets",
    )
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name == ".gitkeep":
                continue
            record = artifact_record(path)
            if record is not None:
                records.append(record)
    return records


def collect_source_hashes() -> dict[str, str]:
    """Hash the executable source snapshot when no Git commit is available."""

    roots = (
        "data",
        "env",
        "evaluation",
        "experiments",
        "models",
        "scripts",
        "trainers",
        "utils",
        "visualization",
        "tests",
    )
    files = [
        PROJECT_ROOT / name
        for name in (
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "setup.sh",
            "run_all.sh",
            "compile_paper.sh",
        )
    ]
    for root_name in roots:
        root = PROJECT_ROOT / root_name
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256(path)
        for path in sorted(set(files))
        if path.is_file()
    }


def collect_dataset_provenance() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for dataset_name in ("demonstrations", "failures"):
        root = PROJECT_ROOT / "datasets" / dataset_name
        files: dict[str, str] = {}
        for filename in ("manifest.json", "statistics.json"):
            path = root / filename
            if path.is_file():
                files[filename] = sha256(path)
        records[dataset_name] = files
    recovery_root = PROJECT_ROOT / "datasets" / "recovery_starts"
    recovery_files: dict[str, str] = {}
    for filename in (
        "train.json",
        "train.npz",
        "validation.json",
        "validation.npz",
    ):
        path = recovery_root / filename
        if path.is_file():
            recovery_files[filename] = sha256(path)
    records["recovery_starts"] = recovery_files
    evaluation_root = PROJECT_ROOT / "datasets" / "evaluation"
    records["evaluation_episode_banks"] = {
        str(path.relative_to(evaluation_root)): sha256(path)
        for path in sorted(evaluation_root.glob("*.json"))
        if path.is_file()
    }
    return records


def _episode_bank_record(
    path: Path,
    *,
    expected_seed: int,
    expected_episodes: int,
    expected_noise: str,
) -> dict[str, Any]:
    """Record and independently verify an episode-bank JSON envelope."""

    record = artifact_record(path)
    if record is None:  # pragma: no cover - callers pass existing glob matches
        raise FileNotFoundError(path)
    metadata_valid = False
    embedded_sha256_valid = False
    payload: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("episode-bank root is not an object")
        payload = loaded
        embedded_sha256 = str(payload.get("bank_sha256", ""))
        core = {key: value for key, value in payload.items() if key != "bank_sha256"}
        canonical = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        embedded_sha256_valid = (
            bool(re.fullmatch(r"[0-9a-f]{64}", embedded_sha256))
            and hashlib.sha256(canonical).hexdigest() == embedded_sha256
        )
        filename_match = re.fullmatch(
            r"pickplace_(?:crn|gate)_task(?P<task>\d+)_ep(?P<seed>\d+)"
            r"_n(?P<episodes>\d+)_noise(?P<noise>\d{3})\.json",
            path.name,
        )
        metadata_valid = bool(
            filename_match
            and payload.get("bank_type") == EPISODE_BANK_TYPE
            and str(payload.get("backend", "")).lower() == "metaworld"
            and int(payload.get("task_bank_seed", -1))
            == int(filename_match.group("task"))
            and int(payload.get("episode_seed_start", -1)) == expected_seed
            and int(filename_match.group("seed")) == expected_seed
            and int(payload.get("episodes", -1)) == expected_episodes
            and int(filename_match.group("episodes")) == expected_episodes
            and filename_match.group("noise") == expected_noise
            and len(payload.get("episode_specifications", []))
            == expected_episodes
            and embedded_sha256_valid
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    record.update(
        {
            "task_bank_seed": payload.get("task_bank_seed"),
            "episode_seed_start": payload.get("episode_seed_start"),
            "episodes": payload.get("episodes"),
            "embedded_bank_sha256": payload.get("bank_sha256"),
            "embedded_bank_sha256_valid": embedded_sha256_valid,
            "metadata_valid": metadata_valid,
        }
    )
    return record


def _operation_trace_evidence() -> tuple[list[dict[str, Any]], bool]:
    """Resolve and verify every file referenced by the operation-sequence manifest."""

    manifest_path = (
        PROJECT_ROOT / "results" / "figures" / "recovery_operation_sequence.json"
    )
    manifest_record = artifact_record(manifest_path)
    if manifest_record is None:
        return [], False
    records = [manifest_record]
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        embedded_artifacts = manifest.get("artifacts", {})
        if not isinstance(embedded_artifacts, dict) or not embedded_artifacts:
            return records, False
        all_valid = True
        required_paths = {
            "results/figures/recovery_operation_sequence_act_failure_trace.json",
            "results/figures/recovery_operation_sequence_reim_success_trace.json",
            "results/figures/recovery_operation_sequence.png",
            "results/figures/recovery_operation_sequence.pdf",
            "paper_assets/Figure5_operation_sequence.png",
            "paper_assets/Figure5_operation_sequence.pdf",
        }
        for relative_path, expected in sorted(embedded_artifacts.items()):
            path = PROJECT_ROOT / relative_path
            record = artifact_record(path)
            if record is None or not isinstance(expected, dict):
                all_valid = False
                continue
            record["embedded_sha256_valid"] = (
                record["sha256"] == expected.get("sha256")
            )
            record["embedded_size_valid"] = record["bytes"] == expected.get("bytes")
            all_valid = (
                all_valid
                and bool(record["embedded_sha256_valid"])
                and bool(record["embedded_size_valid"])
            )
            records.append(record)
        referenced_paths = set(embedded_artifacts)
        keyframes = manifest.get("keyframes", [])
        keyframe_paths = {
            str(item.get("raw_frame"))
            for item in keyframes
            if isinstance(item, dict) and item.get("raw_frame")
        }
        all_valid = (
            all_valid
            and required_paths.issubset(referenced_paths)
            and len(keyframe_paths) >= 8
            and keyframe_paths.issubset(referenced_paths)
        )
        return records, all_valid
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return records, False


def collect_formal_evidence(
    *,
    profile: str,
    backend: str,
    evaluation_seed: int | None,
) -> dict[str, Any]:
    """Collect the publication-critical evidence explicitly and fail closed."""

    formal_required = profile == "full" and backend == "metaworld"
    bank_root = PROJECT_ROOT / "datasets" / "evaluation"
    bank_slots: list[dict[str, Any]] = []
    task_bank_seeds: set[int] = set()
    banks_complete = evaluation_seed is not None
    if evaluation_seed is not None:
        for label, prefix, seed_offset, episodes, noise in FORMAL_BANK_SLOTS:
            expected_seed = evaluation_seed + seed_offset
            pattern = (
                f"pickplace_{prefix}_task*_ep{expected_seed}"
                f"_n{episodes}_noise{noise}.json"
            )
            matches = sorted(bank_root.glob(pattern))
            records = [
                _episode_bank_record(
                    path,
                    expected_seed=expected_seed,
                    expected_episodes=episodes,
                    expected_noise=noise,
                )
                for path in matches
            ]
            slot_complete = (
                len(records) == 1 and bool(records[0].get("metadata_valid"))
            )
            banks_complete = banks_complete and slot_complete
            for record in records:
                seed = record.get("task_bank_seed")
                if isinstance(seed, int):
                    task_bank_seeds.add(seed)
            bank_slots.append(
                {
                    "label": label,
                    "expected_episode_seed_start": expected_seed,
                    "expected_episodes": episodes,
                    "expected_noise_code": noise,
                    "matches": records,
                    "complete": slot_complete,
                }
            )
    banks_complete = banks_complete and len(task_bank_seeds) == 1

    gate_path = PROJECT_ROOT / "results" / "tables" / "gate_matched_comparison.json"
    gate_record = artifact_record(gate_path)
    gate_valid = False
    if gate_record is not None and evaluation_seed is not None:
        try:
            with gate_path.open("r", encoding="utf-8") as handle:
                gate_payload = json.load(handle)
            protocol = gate_payload.get("protocol", {})
            validation = gate_payload.get("validation", {})
            gate_valid = bool(
                gate_payload.get("audit_type")
                == "strict_matched_gate_paired_comparison"
                and validation.get("all_checks_passed") is True
                and int(protocol.get("episode_seed_start", -1))
                == evaluation_seed + GATE_AUDIT_SEED_OFFSET
                and int(protocol.get("episodes", -1)) == 200
                and int(protocol.get("task_bank_seed", -1))
                in task_bank_seeds
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            gate_valid = False

    operation_records, operation_valid = _operation_trace_evidence()

    final_pdf = artifact_record(PROJECT_ROOT / "paper_assets" / "reim_results.pdf")
    figure_records: list[dict[str, Any]] = []
    figures_complete = True
    for stem in FORMAL_PAPER_FIGURE_STEMS:
        for suffix in (".png", ".pdf"):
            record = artifact_record(PROJECT_ROOT / "paper_assets" / f"{stem}{suffix}")
            if record is None:
                figures_complete = False
            else:
                figure_records.append(record)

    checks = {
        "required_for_this_run": formal_required,
        "crn_episode_banks_complete_and_valid": banks_complete,
        "matched_gate_audit_present_and_valid": gate_valid,
        "operation_trace_bundle_complete_and_valid": operation_valid,
        "final_compiled_pdf_present": final_pdf is not None,
        "publication_figures_complete": figures_complete,
    }
    checks["all_required_present_and_valid"] = all(
        value
        for key, value in checks.items()
        if key != "required_for_this_run"
    )
    return {
        "checks": checks,
        "task_bank_seeds": sorted(task_bank_seeds),
        "crn_episode_bank_slots": bank_slots,
        "gate_matched_comparison": gate_record,
        "operation_trace_bundle": operation_records,
        "final_compiled_pdf": final_pdf,
        "publication_figures": figure_records,
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _dataset_episode_seeds() -> set[int]:
    seeds: set[int] = set()
    for dataset_name in ("demonstrations", "failures"):
        path = PROJECT_ROOT / "datasets" / dataset_name / "manifest.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for item in manifest.get("trajectories", []):
            if isinstance(item, dict) and "episode_seed" in item:
                seeds.add(int(item["episode_seed"]))
    recovery_root = PROJECT_ROOT / "datasets" / "recovery_starts"
    for filename in ("train.json", "validation.json"):
        path = recovery_root / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        seed_min = manifest.get("seed_min")
        seed_max = manifest.get("seed_max")
        if seed_min is None or seed_max is None:
            continue
        lower, upper = int(seed_min), int(seed_max)
        if upper < lower:
            raise ValueError(f"invalid seed range in {path}: {lower}..{upper}")
        seeds.update(range(lower, upper + 1))
    return seeds


def benchmark_provenance_checks(
    *,
    profile: str,
    backend: str,
    train_seed: int | None,
    evaluation_seed: int | None,
    controller_protocol: dict[str, float | int],
) -> tuple[dict[str, Any], set[int]]:
    """Validate actual result contents rather than trusting CLI declarations."""

    tables = PROJECT_ROOT / "results" / "tables"
    baseline = _csv_rows(tables / "baseline.csv")
    baseline_episodes = _csv_rows(tables / "baseline_episodes.csv")
    robustness = _csv_rows(tables / "robustness.csv")
    robustness_episodes = _csv_rows(tables / "robustness_episodes.csv")
    ablation = _csv_rows(tables / "ablation.csv")
    ablation_episodes = _csv_rows(tables / "ablation_episodes.csv")

    expected_baseline = {
        "ACT",
        "ACT + Random Reset",
        "ACT + Heuristic Recovery",
        "REIM (ACT + Detector + Recovery)",
    }
    expected_ablation = {"ACT", "ACT + Detector", "ACT + Recovery", "REIM"}
    expected_noise = {0.0, 0.1, 0.2, 0.3, 0.4}

    def full_rows(rows: list[dict[str, str]]) -> bool:
        return bool(rows) and all(
            str(row.get("Backend", row.get("backend", ""))).lower() == "metaworld"
            and str(row.get("Profile", row.get("profile", ""))).lower() == "full"
            for row in rows
        )

    baseline_counts: dict[str, int] = {}
    baseline_seed_sets: dict[str, set[int]] = {}
    for row in baseline_episodes:
        method = str(row.get("method", ""))
        baseline_counts[method] = baseline_counts.get(method, 0) + 1
        baseline_seed_sets.setdefault(method, set()).add(int(row["seed"]))
    baseline_paired = (
        set(baseline_seed_sets) == expected_baseline
        and len({frozenset(value) for value in baseline_seed_sets.values()}) == 1
    )

    robustness_counts: dict[tuple[str, float], int] = {}
    robustness_seed_sets: dict[tuple[str, float], set[int]] = {}
    for row in robustness_episodes:
        key = (str(row.get("method", "")), float(row.get("noise_level", "nan")))
        robustness_counts[key] = robustness_counts.get(key, 0) + 1
        robustness_seed_sets.setdefault(key, set()).add(int(row["seed"]))
    expected_robustness_keys = {
        (method, level)
        for method in ("ACT", "REIM (ACT + Detector + Recovery)")
        for level in expected_noise
    }
    robustness_paired = (
        set(robustness_seed_sets) == expected_robustness_keys
        and len({frozenset(value) for value in robustness_seed_sets.values()}) == 1
    )

    ablation_counts: dict[str, int] = {}
    for row in ablation_episodes:
        method = str(row.get("method", ""))
        ablation_counts[method] = ablation_counts.get(method, 0) + 1

    evaluation_seeds = {
        int(row["seed"]) for row in baseline_episodes if row.get("seed") not in (None, "")
    }
    expected_seed_set = (
        set(range(evaluation_seed, evaluation_seed + 1000))
        if evaluation_seed is not None
        else set()
    )
    training_seeds = _dataset_episode_seeds()
    seed_overlap = sorted(training_seeds & evaluation_seeds)

    checks: dict[str, Any] = {
        "requested_full_metaworld": profile == "full" and backend == "metaworld",
        "canonical_trigger_protocol": (
            float(controller_protocol["recovery_start_threshold"]) == 0.1
            and float(controller_protocol["failure_threshold"]) == 0.2
            and float(controller_protocol["recovery_exit_threshold"]) == 0.0
            and int(controller_protocol["recovery_budget"]) == 150
            and int(controller_protocol["recovery_min_steps"]) == 150
            and int(controller_protocol["recovery_clear_steps"]) == 200
        ),
        "checkpoints_present": (
            train_seed is not None
            and all(
                (PROJECT_ROOT / path).is_file()
                for path in (
                    "checkpoints/bc_policy.pt",
                    "checkpoints/failure_detector.pt",
                    f"checkpoints/recovery_trigger_seed{train_seed}.zip",
                )
            )
        ),
        "recovery_start_datasets_present": all(
            (PROJECT_ROOT / "datasets" / "recovery_starts" / filename).is_file()
            for filename in (
                "train.json",
                "train.npz",
                "validation.json",
                "validation.npz",
            )
        ),
        "baseline_summary": (
            len(baseline) == 4
            and {row.get("Method") for row in baseline} == expected_baseline
            and all(int(float(row.get("Episodes", 0))) == 1000 for row in baseline)
            and full_rows(baseline)
        ),
        "baseline_raw": (
            len(baseline_episodes) == 4000
            and set(baseline_counts) == expected_baseline
            and set(baseline_counts.values()) == {1000}
            and full_rows(baseline_episodes)
            and baseline_paired
        ),
        "robustness_summary": (
            len(robustness) == 10
            and {float(row.get("Noise Level", "nan")) for row in robustness}
            == expected_noise
            and all(int(float(row.get("Episodes", 0))) == 200 for row in robustness)
            and full_rows(robustness)
        ),
        "robustness_raw": (
            len(robustness_episodes) == 2000
            and set(robustness_counts) == expected_robustness_keys
            and set(robustness_counts.values()) == {200}
            and full_rows(robustness_episodes)
            and robustness_paired
        ),
        "ablation_summary": (
            len(ablation) == 4
            and {row.get("Method") for row in ablation} == expected_ablation
            and all(int(float(row.get("Episodes", 0))) == 1000 for row in ablation)
            and full_rows(ablation)
        ),
        "ablation_raw": (
            len(ablation_episodes) == 4000
            and set(ablation_counts) == expected_ablation
            and set(ablation_counts.values()) == {1000}
            and full_rows(ablation_episodes)
        ),
        "evaluation_seed_range": (
            bool(expected_seed_set) and evaluation_seeds == expected_seed_set
        ),
        "training_evaluation_seed_disjoint": not seed_overlap,
        "seed_overlap_count": len(seed_overlap),
    }
    return checks, evaluation_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--backend", choices=("toy", "metaworld"), required=True)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--recovery-start-threshold", type=float, default=0.1)
    parser.add_argument("--failure-threshold", type=float, default=0.2)
    parser.add_argument("--recovery-exit-threshold", type=float, default=0.0)
    parser.add_argument("--recovery-budget", type=int, default=150)
    parser.add_argument("--recovery-min-steps", type=int, default=150)
    parser.add_argument("--recovery-clear-steps", type=int, default=200)
    parser.add_argument(
        "--output", default="results/run_manifest.json", help="Project-relative path"
    )
    args = parser.parse_args()

    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    configs = {}
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        configs[str(path.relative_to(PROJECT_ROOT))] = sha256(path)

    controller_protocol: dict[str, float | int] = {
        "recovery_start_threshold": args.recovery_start_threshold,
        "failure_threshold": args.failure_threshold,
        "recovery_exit_threshold": args.recovery_exit_threshold,
        "recovery_budget": args.recovery_budget,
        "recovery_min_steps": args.recovery_min_steps,
        "recovery_clear_steps": args.recovery_clear_steps,
    }
    provenance_checks, actual_evaluation_seeds = benchmark_provenance_checks(
        profile=args.profile,
        backend=args.backend,
        train_seed=args.train_seed,
        evaluation_seed=args.evaluation_seed,
        controller_protocol=controller_protocol,
    )
    formal_evidence = collect_formal_evidence(
        profile=args.profile,
        backend=args.backend,
        evaluation_seed=args.evaluation_seed,
    )
    boolean_checks = [
        value for value in provenance_checks.values() if isinstance(value, bool)
    ]
    formal_evidence_ok = (
        formal_evidence["checks"]["all_required_present_and_valid"]
        if formal_evidence["checks"]["required_for_this_run"]
        else True
    )
    is_benchmark_result = (
        bool(boolean_checks) and all(boolean_checks) and formal_evidence_ok
    )

    payload = {
        "schema_version": 4,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "backend": args.backend,
        "controller_protocol": controller_protocol,
        "is_benchmark_result": is_benchmark_result,
        "benchmark_provenance_checks": provenance_checks,
        "versions": versions,
        "hardware": hardware_metadata(),
        "git": git_metadata(),
        "source_sha256": collect_source_hashes(),
        "config_sha256": configs,
        "dataset_provenance_sha256": collect_dataset_provenance(),
        "formal_evidence": formal_evidence,
        "seeds": {
            "training_data": args.train_seed,
            "evaluation": args.evaluation_seed,
            "actual_evaluation_min": (
                min(actual_evaluation_seeds) if actual_evaluation_seeds else None
            ),
            "actual_evaluation_max": (
                max(actual_evaluation_seeds) if actual_evaluation_seeds else None
            ),
            "verified_disjoint": provenance_checks[
                "training_evaluation_seed_disjoint"
            ],
        },
        "artifacts": collect_artifacts(),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
