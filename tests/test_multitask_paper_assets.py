"""Publication-gate tests for MT10/MT50 tables, figures, and TeX."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.multitask_metrics import (
    aggregate_multitask_metrics,
    paired_task_stratified_bootstrap_delta,
)
import visualization.generate_multitask_paper_assets as paper


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows(
    *,
    benchmark: str,
    condition: str,
    fingerprint: str,
    noise: float,
) -> list[dict[str, object]]:
    del noise
    rows: list[dict[str, object]] = []
    labels = paper.METHOD_LABELS
    for task_id in range(2):
        for variant in range(2):
            success_by_method = {
                "mlp_bc": variant == 0,
                "act": task_id == 0 or variant == 0,
                "heuristic_recovery": not (task_id == 1 and variant == 1),
                "reim": True,
            }
            payload = hashlib.sha256(
                f"{benchmark}-{task_id}-{variant}".encode()
            ).hexdigest()
            paired_id = f"{task_id:02d}-{variant:04d}-{payload[:12]}"
            for method in paper.CANONICAL_METHODS:
                intervention = int(method in {"heuristic_recovery", "reim"})
                rows.append(
                    {
                        "run_fingerprint": fingerprint,
                        "benchmark": benchmark,
                        "condition": condition,
                        "task_name": f"task-{task_id}",
                        "task_id": task_id,
                        "task_variant": variant,
                        "method": labels[method],
                        "success": success_by_method[method],
                        "intervention_count": intervention,
                        "recovery_success": int(
                            intervention and success_by_method[method]
                        ),
                        "steps": 50 + variant,
                        "paired_episode_id": paired_id,
                        "episode_seed": 10_000 + task_id * 100 + variant,
                        "task_payload_sha256": payload,
                    }
                )
    return rows


def _condition(
    root: Path,
    *,
    benchmark: str,
    stem: str,
    condition: str,
    noise: float,
    checkpoints: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    slug = benchmark.lower()
    episode_path = root / "tables" / f"{slug}_{stem}_episodes.csv"
    summary_path = root / "tables" / f"{slug}_{stem}_summary.json"
    sidecar_path = episode_path.with_suffix(".csv.run.json")
    protocol = {
        "benchmark": benchmark,
        "condition": condition,
        "benchmark_seed": 900 + (10 if benchmark == "MT10" else 50),
        "task_bank_sha256": hashlib.sha256(
            f"{benchmark}-final-bank".encode()
        ).hexdigest(),
        "episodes_per_task": 2,
        "max_episode_steps": 500,
        "noise_level": noise,
        "methods": list(paper.CANONICAL_METHODS),
    }
    fingerprint = _canonical_sha256(protocol)
    rows = _rows(
        benchmark=benchmark,
        condition=condition,
        fingerprint=fingerprint,
        noise=noise,
    )
    _write_csv(episode_path, rows)
    sidecar = {
        "schema_version": paper.RUN_SIDECAR_SCHEMA,
        "run_fingerprint": fingerprint,
        "protocol": protocol,
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    grouped: dict[str, list[dict[str, object]]] = {}
    for method, label in paper.METHOD_LABELS.items():
        grouped[method] = [row for row in rows if row["method"] == label]
    aggregates = {
        method: aggregate_multitask_metrics(values)
        for method, values in grouped.items()
    }
    paired = {
        method: paired_task_stratified_bootstrap_delta(
            grouped["act"], values, n_bootstrap=10, seed=17
        )
        for method, values in grouped.items()
        if method != "act"
    }
    official = condition == "official_clean"
    expected_per_method = 4
    summary = {
        "schema_version": paper.EVALUATION_SCHEMA,
        "run_fingerprint": fingerprint,
        "run_sidecar": str(sidecar_path.resolve()),
        "run_sidecar_sha256": _sha256(sidecar_path),
        "benchmark": benchmark,
        "condition": condition,
        "official_clean_protocol": official,
        "official_clean_protocol_scope": "rollout_protocol_only",
        "official_clean_eligibility_by_method": {
            method: {
                "label": paper.METHOD_LABELS[method],
                "eligible": official,
                "reasons": [] if official else ["non_official_condition"],
                "completed_rows": expected_per_method,
                "expected_rows": expected_per_method,
            }
            for method in paper.CANONICAL_METHODS
        },
        "publication_eligible": False,
        "publication_audit_required": True,
        "publication_readiness": {
            "eligible": False,
            "rollout_protocol_eligible": official,
            "external_audit_consumed": False,
        },
        "robustness_extension": noise != 0.0,
        "metaworld_version": "test",
        "benchmark_seed": protocol["benchmark_seed"],
        "seed": 42,
        "task_bank_sha256": protocol["task_bank_sha256"],
        "observation_schema": "raw39_plus_official_task_one_hot",
        "task_vocabulary": ["task-0", "task-1"],
        "task_ids": [0, 1],
        "max_episode_steps": 500,
        "episodes_per_task": 2,
        "noise_level": noise,
        "action_noise_std": noise * 0.4,
        "observation_noise_std": noise * 0.025,
        "object_position_noise": False,
        "detector_threshold": 0.69,
        "release_threshold": 0.15,
        "release_patience": 5,
        "min_recovery_steps": 5,
        "intervention_cooldown": 10,
        "recovery_budget": 250,
        "methods": list(paper.CANONICAL_METHODS),
        "episode_csv": str(episode_path.resolve()),
        "episode_csv_sha256": _sha256(episode_path),
        "checkpoints": checkpoints,
        "aggregates": aggregates,
        "paired_vs_act": paired,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, episode_path


def _audit(root: Path, *, benchmark: str, clean_csv: Path) -> Path:
    stages: dict[str, dict[str, object]] = {}
    for index, stage in enumerate(paper.STAGES):
        record: dict[str, object] = {
            "source_paths": [],
            "source_sha256": [],
            "benchmark_seed": index + (10 if benchmark == "MT10" else 100),
            "embedded_fingerprint_sha256": hashlib.sha256(
                f"{benchmark}-{stage}-embedded".encode()
            ).hexdigest(),
            "audit_descriptor_sha256": hashlib.sha256(
                f"{benchmark}-{stage}-descriptor".encode()
            ).hexdigest(),
            "task_bank_content_sha256": hashlib.sha256(
                f"{benchmark}-{stage}-bank".encode()
            ).hexdigest(),
            "reserved_task_payloads": 4,
            "materialized_task_payloads": 1,
            "materialized_records": 1,
            "unique_episode_seeds": 1,
        }
        if stage == "final_evaluation":
            sidecar = clean_csv.with_suffix(".csv.run.json")
            record["source_paths"] = [str(sidecar.resolve()), str(clean_csv.resolve())]
            record["source_sha256"] = [_sha256(sidecar), _sha256(clean_csv)]
            record["materialized_task_payloads"] = 4
            record["materialized_records"] = 4
            record["unique_episode_seeds"] = 4
        stages[stage] = record
    pairwise = []
    for left_index, left in enumerate(paper.STAGES):
        for right in paper.STAGES[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "benchmark_seed_equal": False,
                    "task_bank_content_fingerprint_equal": False,
                    "reserved_task_payload_overlap_count": 0,
                    "materialized_task_payload_overlap_count": 0,
                    "sampling_unit_identity_overlap_count": 0,
                    "raw_episode_seed_overlap_count": 0,
                }
            )
    payload = {
        "schema_version": paper.AUDIT_SCHEMA,
        "read_only": True,
        "passed": True,
        "benchmark": benchmark,
        "checks": {name: True for name in paper.AUDIT_CHECKS},
        "stages": stages,
        "pairwise": pairwise,
    }
    path = root / "audits" / f"{benchmark.lower()}_bank_separation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setitem(paper.SUITE_TASK_COUNT, "MT10", 2)
    monkeypatch.setitem(paper.SUITE_TASK_COUNT, "MT50", 2)
    monkeypatch.setattr(paper, "OFFICIAL_EPISODES_PER_TASK", 2)
    monkeypatch.setattr(paper, "ROBUSTNESS_LEVELS", (0.0, 0.2))
    monkeypatch.setattr(paper, "MIN_BOOTSTRAP_SAMPLES", 1)
    tables = tmp_path / "tables"
    audits = tmp_path / "audits"
    assets = tmp_path / "assets"
    for benchmark in ("MT10", "MT50"):
        checkpoint_dir = tmp_path / "checkpoints" / benchmark.lower()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoints: dict[str, dict[str, str]] = {}
        for name in ("mlp_bc", "act", "detector", "recovery"):
            path = checkpoint_dir / f"{name}.pt"
            path.write_bytes(f"{benchmark}:{name}".encode())
            checkpoints[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        _, clean_csv = _condition(
            tmp_path,
            benchmark=benchmark,
            stem="clean",
            condition="official_clean",
            noise=0.0,
            checkpoints=checkpoints,
        )
        for noise in paper.ROBUSTNESS_LEVELS:
            tag = paper._noise_tag(noise)
            _condition(
                tmp_path,
                benchmark=benchmark,
                stem=f"disturbed_noise_{tag}",
                condition=f"robustness_noise_{tag}",
                noise=noise,
                checkpoints=checkpoints,
            )
        _audit(tmp_path, benchmark=benchmark, clean_csv=clean_csv)
    return tables, audits, assets


def test_missing_results_leave_gate_closed(tmp_path: Path) -> None:
    with pytest.raises(paper.PublicationGateError):
        paper.generate_assets(
            tables_dir=tmp_path / "missing",
            audits_dir=tmp_path / "missing",
            assets_dir=tmp_path / "assets",
            n_bootstrap=10,
        )
    assert not (tmp_path / "assets" / "multitask_results.tex").exists()


def test_complete_study_generates_gate_statistics_figure_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables, audits, assets = _complete_study(tmp_path, monkeypatch)
    manifest = paper.generate_assets(
        tables_dir=tables,
        audits_dir=audits,
        assets_dir=assets,
        n_bootstrap=25,
        bootstrap_seed=7,
    )
    gate = (assets / "multitask_results.tex").read_text(encoding="utf-8")
    assert r"\REIMMultiTaskResultstrue" in gate
    assert r"\renewcommand{\MTTenREIMCleanSuccess}" in gate
    assert r"\renewcommand{\MTFiftyREIMDeltaVsACTCI}" in gate
    assert (assets / "Figure_multitask_robustness.png").stat().st_size > 1_000
    assert (assets / "Figure_multitask_robustness.pdf").stat().st_size > 1_000
    assert manifest["publication_gate"]["eligible"] is True
    assert manifest["publication_gate"]["robustness_scope"] == (
        "non_official_reim_extension"
    )
    with (assets / "multitask_clean_statistics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["success_ci_lower"] and row["success_ci_upper"] for row in rows)
    reim = next(
        row
        for row in rows
        if row["benchmark"] == "MT10" and row["method"] == "MT-REIM"
    )
    assert reim["delta_ci_lower"] and reim["delta_ci_upper"]


def test_tampered_summary_or_audit_never_creates_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables, audits, assets = _complete_study(tmp_path, monkeypatch)
    summary_path = tables / "mt50_disturbed_noise_20_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["aggregates"]["reim"]["summary"]["success_rate_task_macro"] = 0.123
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(paper.PublicationGateError, match="recomputation"):
        paper.generate_assets(
            tables_dir=tables,
            audits_dir=audits,
            assets_dir=assets,
            n_bootstrap=10,
        )
    assert not assets.exists()

    # Restore inputs, then bind the MT10 audit to the wrong clean CSV digest.
    tables, audits, assets = _complete_study(tmp_path, monkeypatch)
    audit_path = audits / "mt10_bank_separation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["stages"]["final_evaluation"]["source_sha256"][-1] = "f" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(paper.PublicationGateError, match="does not bind"):
        paper.generate_assets(
            tables_dir=tables,
            audits_dir=audits,
            assets_dir=assets,
            n_bootstrap=10,
        )
    assert not assets.exists()


def test_paper_source_is_closed_by_default_and_labels_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    macros = (root / "paper_assets" / "reim_macros.tex").read_text(
        encoding="utf-8"
    )
    manuscript = (root / "paper_assets" / "reim_results.tex").read_text(
        encoding="utf-8"
    )
    table = (root / "paper_assets" / "Table_multitask_clean.tex").read_text(
        encoding="utf-8"
    )
    assert r"\REIMMultiTaskResultsfalse" in macros
    assert r"\InputIfFileExists{multitask_results.tex}" in macros
    assert "Non-official REIM robustness extension" in manuscript
    assert "paired, task-stratified" in table
    assert "95\\% within-task" in table
