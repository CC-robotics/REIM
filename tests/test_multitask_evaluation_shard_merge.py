from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.io import atomic_write_json, file_sha256
from evaluation import evaluate_multitask as evaluation
from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    paired_task_stratified_bootstrap_delta,
)
from scripts import merge_multitask_evaluation_shards as merger


METHODS = tuple(evaluation.DEFAULT_OFFICIAL_METHODS)
TASK_COUNT = 10
EPISODES_PER_TASK = 50


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=evaluation.CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_records(root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    hashes: dict[str, str] = {}
    records: dict[str, dict[str, str]] = {}
    for name in ("mlp_bc", "act", "detector", "recovery"):
        path = root / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + "-immutable-test-checkpoint").encode("ascii"))
        digest = file_sha256(path)
        hashes[name] = digest
        records[name] = {"path": str(path.resolve()), "sha256": digest}
    return hashes, records


def _protocol(
    *,
    root: Path,
    task_ids: list[int],
    checkpoint_hashes: dict[str, str],
    episode_seed_base: int = 42,
    detector_threshold: float = 0.69,
) -> dict[str, object]:
    vocabulary = [f"task-{task_id:02d}-v3" for task_id in range(TASK_COUNT)]
    return {
        "evaluation_schema_version": evaluation.SCHEMA_VERSION,
        "benchmark": "MT10",
        "condition": "official_clean",
        "metaworld_version": "3.0.0-test",
        "execution_device": "cuda",
        "benchmark_seed": 20265010,
        "episode_seed_base": episode_seed_base,
        "task_bank_sha256": _sha("task-bank"),
        "task_vocabulary": vocabulary,
        "task_vocabulary_sha256": merger._canonical_sha256(vocabulary),
        "task_ids": task_ids,
        "methods": list(METHODS),
        "episodes_per_task": EPISODES_PER_TASK,
        "max_episode_steps": 500,
        "noise_level": 0.0,
        "action_std_scale": 0.40,
        "observation_std_scale": 0.025,
        "object_position_noise": False,
        "detector_threshold": detector_threshold,
        "release_threshold": 0.15,
        "release_patience": 5,
        "min_recovery_steps": 5,
        "intervention_cooldown": 10,
        "recovery_budget": 250,
        "heuristic_min_steps": 30,
        "heuristic_window": 20,
        "heuristic_tolerance": 0.01,
        "bootstrap_samples": 7,
        "checkpoint_sha256": checkpoint_hashes,
    }


def _rows(protocol: dict[str, object], fingerprint: str) -> list[dict[str, object]]:
    task_ids = list(protocol["task_ids"])
    vocabulary = list(protocol["task_vocabulary"])
    seed = int(protocol["episode_seed_base"])
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        for variant in range(EPISODES_PER_TASK):
            payload = _sha(f"payload:{task_id}:{variant}")
            paired = f"{task_id:02d}-{variant:04d}-{payload[:12]}"
            episode_seed = seed + task_id * 100_000 + variant
            for method_index, method in enumerate(METHODS):
                recovery_method = method in {"heuristic_recovery", "reim"}
                intervened = int(recovery_method and variant % 7 == 0)
                success = int((task_id + variant + method_index) % 5 != 0)
                rows.append(
                    {
                        "run_fingerprint": fingerprint,
                        "benchmark": "MT10",
                        "condition": "official_clean",
                        "task_name": vocabulary[task_id],
                        "task_id": task_id,
                        "task_variant": variant,
                        "method": evaluation.METHOD_LABELS[method],
                        "success": success,
                        "intervention_count": intervened,
                        "recovery_success": int(intervened and success),
                        "steps": 100 + variant,
                        "paired_episode_id": paired,
                        "episode_seed": episode_seed,
                        "task_payload_sha256": payload,
                        "max_failure_probability": 0.8 if intervened else 0.1,
                        "trigger_step": 50 if intervened else -1,
                        "attempt_count": 1,
                    }
                )
    return rows


def _statistics(
    rows: list[dict[str, object]], protocol: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    selected = {
        method: [
            row
            for row in rows
            if row["method"] == evaluation.METHOD_LABELS[method]
        ]
        for method in METHODS
    }
    aggregates = {
        method: aggregate_multitask_metrics(selected[method]) for method in METHODS
    }
    paired = {
        method: paired_task_stratified_bootstrap_delta(
            selected["act"],
            selected[method],
            metric="success",
            n_bootstrap=int(protocol["bootstrap_samples"]),
            seed=int(protocol["episode_seed_base"]) + 2026,
        )
        for method in METHODS
        if method != "act"
    }
    return aggregates, paired


def _make_shard(
    root: Path,
    name: str,
    task_ids: list[int],
    *,
    checkpoint_hashes: dict[str, str],
    checkpoint_records: dict[str, dict[str, str]],
    episode_seed_base: int = 42,
    detector_threshold: float = 0.69,
) -> tuple[Path, Path]:
    protocol = _protocol(
        root=root,
        task_ids=task_ids,
        checkpoint_hashes=checkpoint_hashes,
        episode_seed_base=episode_seed_base,
        detector_threshold=detector_threshold,
    )
    fingerprint = merger._canonical_sha256(protocol)
    csv_path = root / f"{name}_episodes.csv"
    summary_path = root / f"{name}_summary.json"
    sidecar_path = evaluation._run_sidecar_path(csv_path)
    rows = _rows(protocol, fingerprint)
    _write_csv(csv_path, rows)
    atomic_write_json(
        sidecar_path,
        {
            "schema_version": evaluation.RUN_SIDECAR_SCHEMA_VERSION,
            "run_fingerprint": fingerprint,
            "protocol": protocol,
        },
    )
    completed = {
        (
            method,
            task_id,
            f"{task_id:02d}-{variant:04d}-{_sha(f'payload:{task_id}:{variant}')[:12]}",
        )
        for method in METHODS
        for task_id in task_ids
        for variant in range(EPISODES_PER_TASK)
    }
    eligibility = evaluation._official_clean_eligibility(
        condition="official_clean",
        noise_level=0.0,
        max_steps=500,
        task_ids=task_ids,
        task_count=TASK_COUNT,
        methods=METHODS,
        episodes_per_task=EPISODES_PER_TASK,
        completed=completed,
    )
    official = all(value["eligible"] for value in eligibility.values())
    readiness = evaluation._publication_readiness(
        benchmark="MT10", official_clean_protocol=official
    )
    aggregates, paired = _statistics(rows, protocol)
    summary = {
        "schema_version": evaluation.SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "run_sidecar": str(sidecar_path.resolve()),
        "run_sidecar_sha256": file_sha256(sidecar_path),
        "benchmark": "MT10",
        "condition": "official_clean",
        "official_clean_protocol": official,
        "official_clean_protocol_scope": "rollout_protocol_only",
        "official_clean_eligibility_by_method": eligibility,
        "publication_eligible": readiness["eligible"],
        "publication_audit_required": readiness["audit_required"],
        "publication_readiness": readiness,
        "robustness_extension": False,
        "metaworld_version": protocol["metaworld_version"],
        "benchmark_seed": protocol["benchmark_seed"],
        "seed": protocol["episode_seed_base"],
        "task_bank_sha256": protocol["task_bank_sha256"],
        "observation_schema": "raw39_plus_official_task_one_hot",
        "task_vocabulary": protocol["task_vocabulary"],
        "task_ids": task_ids,
        "max_episode_steps": 500,
        "episodes_per_task": 50,
        "noise_level": 0.0,
        "action_noise_std": 0.0,
        "observation_noise_std": 0.0,
        "object_position_noise": False,
        "detector_threshold": protocol["detector_threshold"],
        "release_threshold": protocol["release_threshold"],
        "release_patience": protocol["release_patience"],
        "min_recovery_steps": protocol["min_recovery_steps"],
        "intervention_cooldown": protocol["intervention_cooldown"],
        "recovery_budget": protocol["recovery_budget"],
        "methods": list(METHODS),
        "episode_csv": str(csv_path.resolve()),
        "episode_csv_sha256": file_sha256(csv_path),
        "checkpoints": checkpoint_records,
        "aggregates": aggregates,
        "paired_vs_act": paired,
    }
    atomic_write_json(summary_path, summary)
    return summary_path, csv_path


@pytest.fixture
def shard_factory(tmp_path: Path):
    hashes, records = _checkpoint_records(tmp_path)

    def build(
        name: str,
        task_ids: list[int],
        *,
        episode_seed_base: int = 42,
        detector_threshold: float = 0.69,
    ) -> tuple[Path, Path]:
        return _make_shard(
            tmp_path,
            name,
            task_ids,
            checkpoint_hashes=hashes,
            checkpoint_records=records,
            episode_seed_base=episode_seed_base,
            detector_threshold=detector_threshold,
        )

    return build


def _merge(tmp_path: Path, shards: list[tuple[Path, Path]]):
    output_csv = tmp_path / "merged_episodes.csv"
    output_summary = tmp_path / "merged_summary.json"
    summary = merger.merge_evaluation_shards(
        shards,
        output_csv=output_csv,
        output_summary=output_summary,
    )
    return summary, output_summary, output_csv


def test_merge_full_disjoint_suite_recomputes_publishable_protocol(
    tmp_path: Path, shard_factory
) -> None:
    left = shard_factory("left", list(range(5)))
    right = shard_factory("right", list(range(5, 10)))

    summary, summary_path, csv_path = _merge(tmp_path, [right, left])

    assert summary["task_ids"] == list(range(10))
    assert summary["official_clean_protocol"] is True
    assert summary["publication_eligible"] is False
    assert summary["publication_audit_required"] is True
    assert summary["episode_csv_sha256"] == file_sha256(csv_path)
    assert summary["merge_provenance"]["source_shard_count"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    sidecar = json.loads(
        evaluation._run_sidecar_path(csv_path).read_text(encoding="utf-8")
    )
    assert sidecar["protocol"]["task_ids"] == list(range(10))
    assert sidecar["run_fingerprint"] == summary["run_fingerprint"]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == TASK_COUNT * EPISODES_PER_TASK * len(METHODS)
    assert all(row["run_fingerprint"] == summary["run_fingerprint"] for row in rows)
    ordering = [
        (int(row["task_id"]), int(row["task_variant"]), row["method"])
        for row in rows
    ]
    method_order = {
        evaluation.METHOD_LABELS[method]: index for index, method in enumerate(METHODS)
    }
    assert ordering == sorted(
        ordering, key=lambda value: (value[0], value[1], method_order[value[2]])
    )


def test_tampered_csv_is_rejected_by_summary_digest(
    tmp_path: Path, shard_factory
) -> None:
    left = shard_factory("left", list(range(5)))
    right = shard_factory("right", list(range(5, 10)))
    with right[1].open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")

    with pytest.raises(merger.EvaluationShardMergeError, match="CSV SHA-256"):
        _merge(tmp_path, [left, right])


def test_missing_task_is_rejected(tmp_path: Path, shard_factory) -> None:
    left = shard_factory("left", list(range(5)))
    right = shard_factory("right", list(range(5, 9)))

    with pytest.raises(merger.EvaluationShardMergeError, match="full MT10 task suite"):
        _merge(tmp_path, [left, right])


def test_overlapping_task_is_rejected(tmp_path: Path, shard_factory) -> None:
    left = shard_factory("left", list(range(6)))
    right = shard_factory("right", list(range(5, 10)))

    with pytest.raises(merger.EvaluationShardMergeError, match="task_id 5 overlaps"):
        _merge(tmp_path, [left, right])


def test_episode_seed_protocol_mismatch_is_rejected(
    tmp_path: Path, shard_factory
) -> None:
    left = shard_factory("left", list(range(5)), episode_seed_base=42)
    right = shard_factory("right", list(range(5, 10)), episode_seed_base=43)

    with pytest.raises(
        merger.EvaluationShardMergeError, match="episode_seed_base"
    ):
        _merge(tmp_path, [left, right])


def test_threshold_protocol_mismatch_is_rejected(
    tmp_path: Path, shard_factory
) -> None:
    left = shard_factory("left", list(range(5)), detector_threshold=0.69)
    right = shard_factory("right", list(range(5, 10)), detector_threshold=0.70)

    with pytest.raises(
        merger.EvaluationShardMergeError, match="detector_threshold"
    ):
        _merge(tmp_path, [left, right])
