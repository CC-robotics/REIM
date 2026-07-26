"""Targeted tests for publication-critical reproducibility evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.reproducibility_manifest as manifest


def _write(path: Path, content: bytes = b"evidence") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_bank(
    root: Path,
    *,
    prefix: str,
    task_seed: int,
    episode_seed: int,
    episodes: int,
    noise: str,
) -> None:
    specifications = [{"episode_seed": episode_seed + index} for index in range(episodes)]
    payload = {
        "schema_version": 2,
        "bank_type": manifest.EPISODE_BANK_TYPE,
        "backend": "metaworld",
        "task_bank_seed": task_seed,
        "episode_seed_start": episode_seed,
        "episodes": episodes,
        "episode_specifications": specifications,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    payload["bank_sha256"] = hashlib.sha256(canonical).hexdigest()
    filename = (
        f"pickplace_{prefix}_task{task_seed}_ep{episode_seed}"
        f"_n{episodes}_noise{noise}.json"
    )
    path = root / "datasets" / "evaluation" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_operation_bundle(root: Path) -> None:
    artifact_paths = [
        "results/figures/recovery_operation_sequence_act_failure_trace.json",
        "results/figures/recovery_operation_sequence_reim_success_trace.json",
        "results/figures/recovery_operation_sequence.png",
        "results/figures/recovery_operation_sequence.pdf",
        "paper_assets/Figure5_operation_sequence.png",
        "paper_assets/Figure5_operation_sequence.pdf",
    ]
    artifact_paths.extend(
        f"results/figures/recovery_operation_sequence_frames/{index:02d}.png"
        for index in range(1, 9)
    )
    embedded: dict[str, dict[str, int | str]] = {}
    for relative_path in artifact_paths:
        path = _write(root / relative_path, relative_path.encode("utf-8"))
        embedded[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": manifest.sha256(path),
        }
    operation_manifest = {
        "keyframes": [
            {
                "raw_frame": (
                    "results/figures/recovery_operation_sequence_frames/"
                    f"{index:02d}.png"
                )
            }
            for index in range(1, 9)
        ],
        "artifacts": embedded,
    }
    path = root / "results" / "figures" / "recovery_operation_sequence.json"
    path.write_text(json.dumps(operation_manifest), encoding="utf-8")


def _build_formal_evidence_tree(root: Path, evaluation_seed: int) -> None:
    task_seed = 20260726
    for _, prefix, seed_offset, episodes, noise in manifest.FORMAL_BANK_SLOTS:
        _write_bank(
            root,
            prefix=prefix,
            task_seed=task_seed,
            episode_seed=evaluation_seed + seed_offset,
            episodes=episodes,
            noise=noise,
        )
    gate_payload = {
        "audit_type": "strict_matched_gate_paired_comparison",
        "protocol": {
            "episode_seed_start": evaluation_seed + manifest.GATE_AUDIT_SEED_OFFSET,
            "episodes": 200,
            "task_bank_seed": task_seed,
        },
        "validation": {"all_checks_passed": True},
    }
    gate_path = root / "results" / "tables" / "gate_matched_comparison.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate_payload), encoding="utf-8")
    _write(root / "paper_assets" / "reim_results.pdf", b"%PDF-1.4")
    for stem in manifest.FORMAL_PAPER_FIGURE_STEMS:
        _write(root / "paper_assets" / f"{stem}.png")
        _write(root / "paper_assets" / f"{stem}.pdf", b"%PDF-1.4")
    # The operation manifest embeds the final Figure 5 hashes, so build it last.
    _write_operation_bundle(root)


def test_collect_artifacts_hashes_episode_banks_and_runtime_traces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    bank = _write(tmp_path / "datasets" / "evaluation" / "bank.json")
    trace = _write(tmp_path / "results" / "traces" / "formal" / "trace.json")

    records = {record["path"]: record for record in manifest.collect_artifacts()}

    assert records["datasets/evaluation/bank.json"]["sha256"] == manifest.sha256(bank)
    assert (
        records["results/traces/formal/trace.json"]["sha256"]
        == manifest.sha256(trace)
    )


def test_collect_formal_evidence_accepts_complete_content_addressed_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    evaluation_seed = 8_000_042
    _build_formal_evidence_tree(tmp_path, evaluation_seed)

    evidence = manifest.collect_formal_evidence(
        profile="full",
        backend="metaworld",
        evaluation_seed=evaluation_seed,
    )

    assert evidence["checks"]["all_required_present_and_valid"] is True
    assert evidence["task_bank_seeds"] == [20260726]
    assert len(evidence["crn_episode_bank_slots"]) == 6
    assert all(slot["complete"] for slot in evidence["crn_episode_bank_slots"])
    assert len(evidence["operation_trace_bundle"]) == 15


def test_collect_formal_evidence_fails_closed_when_a_bank_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    evaluation_seed = 8_000_042
    _build_formal_evidence_tree(tmp_path, evaluation_seed)
    missing = (
        tmp_path
        / "datasets"
        / "evaluation"
        / "pickplace_crn_task20260726_ep8000042_n200_noise040.json"
    )
    missing.unlink()

    evidence = manifest.collect_formal_evidence(
        profile="full",
        backend="metaworld",
        evaluation_seed=evaluation_seed,
    )

    assert evidence["checks"]["crn_episode_banks_complete_and_valid"] is False
    assert evidence["checks"]["all_required_present_and_valid"] is False
