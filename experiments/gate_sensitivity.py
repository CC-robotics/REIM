#!/usr/bin/env python3
"""Audit and plot the post-freeze REIM gate sensitivity diagnostic.

The frozen primary benchmark uses tau_on=0.20.  This script does not select or
change that threshold.  It validates a separate paired 200-seed sweep and
exports a success-versus-intervention Pareto diagnostic with full provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
METHOD = "REIM (ACT + Detector + Recovery)"
THRESHOLD_FILES = {
    0.100: "gate_calibration_tau010.csv",
    0.125: "gate_calibration_tau0125.csv",
    0.150: "gate_calibration_tau015.csv",
    0.175: "gate_calibration_tau0175.csv",
    0.200: "gate_calibration_tau020.csv",
}


class SensitivityError(ValueError):
    """Raised when a sweep artifact fails the diagnostic protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SensitivityError(f"missing sweep artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SensitivityError(f"empty sweep artifact: {path}")
    return rows


def _integer(row: dict[str, str], key: str) -> int:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SensitivityError(f"invalid {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise SensitivityError(f"{key!r} must be a finite integer")
    return int(value)


def _boolean(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise SensitivityError(f"invalid Boolean value: {value!r}")


def _digest(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SensitivityError(f"{field} is not a SHA256 digest")
    return normalized


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _audit_condition(
    threshold: float,
    summary_path: Path,
    *,
    episodes: int,
    seed_start: int,
) -> tuple[dict[str, Any], dict[str, str], dict[int, tuple[str, str, str]]]:
    summaries = _read(summary_path)
    if len(summaries) != 1:
        raise SensitivityError(f"{summary_path} must contain one summary row")
    summary = summaries[0]
    raw_path = summary_path.with_name(f"{summary_path.stem}_episodes.csv")
    raw = _read(raw_path)
    if len(raw) != episodes:
        raise SensitivityError(
            f"tau={threshold:g}: expected {episodes} raw episodes, got {len(raw)}"
        )
    expected_seeds = set(range(seed_start, seed_start + episodes))
    seeds = {_integer(row, "seed") for row in raw}
    indices = {_integer(row, "episode") for row in raw}
    if seeds != expected_seeds or indices != set(range(episodes)):
        raise SensitivityError(f"tau={threshold:g}: paired seed/index set mismatch")
    bank_sha = _digest(
        summary.get("Episode Bank SHA256", ""), field="summary episode bank"
    )
    bank_file_sha = _digest(
        summary.get("Episode Bank File SHA256", ""),
        field="summary episode-bank file",
    )
    if not _boolean(summary.get("CRN Episode Specifications Verified", "")):
        raise SensitivityError(f"tau={threshold:g}: CRN verification is false")
    crn_by_seed: dict[int, tuple[str, str, str]] = {}
    for row in raw:
        if row.get("method") != METHOD:
            raise SensitivityError(f"tau={threshold:g}: method mismatch")
        if row.get("backend", "").lower() != "metaworld":
            raise SensitivityError(f"tau={threshold:g}: backend is not Meta-World")
        if not math.isclose(float(row["noise_level"]), 0.2, abs_tol=1e-12):
            raise SensitivityError(f"tau={threshold:g}: noise level drift")
        seed = _integer(row, "seed")
        raw_bank = _digest(
            row.get("episode_bank_sha256", ""),
            field=f"tau={threshold:g} seed={seed} episode bank",
        )
        if raw_bank != bank_sha:
            raise SensitivityError(
                f"tau={threshold:g} seed={seed}: raw/summary bank mismatch"
            )
        crn_by_seed[seed] = (
            _digest(
                row.get("episode_specification_sha256", ""),
                field=f"tau={threshold:g} seed={seed} episode specification",
            ),
            raw_bank,
            _digest(
                row.get("metaworld_task_sha256", ""),
                field=f"tau={threshold:g} seed={seed} task",
            ),
        )

    successes = sum(_boolean(row["success"]) for row in raw)
    attempts = sum(_integer(row, "recovery_attempts") for row in raw)
    recovery_successes = sum(_integer(row, "recovery_successes") for row in raw)
    intervened = sum(_integer(row, "recovery_attempts") > 0 for row in raw)
    total_steps = sum(_integer(row, "steps") for row in raw)
    if _integer(summary, "Episodes") != episodes:
        raise SensitivityError(f"tau={threshold:g}: summary episode count drift")
    for key, actual in (
        ("Successes", successes),
        ("Recovery Attempts", attempts),
        ("Recovery Successes", recovery_successes),
        ("Intervened Episodes", intervened),
    ):
        if _integer(summary, key) != actual:
            raise SensitivityError(f"tau={threshold:g}: {key} disagrees with raw")
    success_rate = successes / episodes
    intervention_rate = intervened / episodes
    recovery_rate = recovery_successes / attempts if attempts else 0.0
    for key, expected in (
        ("Success Rate", success_rate),
        ("Recovery Rate", recovery_rate),
        ("Average Steps", total_steps / episodes),
    ):
        if not math.isclose(float(summary[key]), expected, abs_tol=1e-10):
            raise SensitivityError(f"tau={threshold:g}: {key} disagrees with raw")
    intervention_low, intervention_high = _wilson(intervened, episodes)
    result = {
        "Threshold": threshold,
        "Success Rate": success_rate,
        "Success CI Lower": float(summary["Success CI Lower"]),
        "Success CI Upper": float(summary["Success CI Upper"]),
        "Intervention Rate": intervention_rate,
        "Intervention CI Lower": intervention_low,
        "Intervention CI Upper": intervention_high,
        "Intervention Success": recovery_rate,
        "Average Steps": total_steps / episodes,
        "Episodes": episodes,
        "Seed Start": seed_start,
        "Seed End": seed_start + episodes - 1,
        "Episode Bank SHA256": bank_sha,
        "Episode Bank File SHA256": bank_file_sha,
        "CRN Verified": True,
        "Scope": "post_freeze_sensitivity_not_primary_benchmark",
    }
    return (
        result,
        {
            _project_key(summary_path): _sha256(summary_path),
            _project_key(raw_path): _sha256(raw_path),
        },
        crn_by_seed,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_matched_gate_audit(
    path: Path,
    *,
    expected_bank_sha256: str,
    expected_episodes: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise SensitivityError(f"missing matched-gate audit: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validation = payload.get("validation", {})
    if not isinstance(validation, dict) or not validation.get("all_checks_passed"):
        raise SensitivityError("matched-gate audit did not pass all checks")
    protocol = payload.get("protocol", {})
    if (
        protocol.get("episode_bank_sha256") != expected_bank_sha256
        or int(protocol.get("episodes", 0)) != expected_episodes
    ):
        raise SensitivityError("matched-gate audit uses a different protocol")
    methods = payload.get("methods", {})
    comparison = payload.get("paired_success_comparisons", {}).get(
        "reim_tau_0.175_minus_heuristic", {}
    )
    heuristic = methods.get("heuristic_gate", {})
    if (
        heuristic.get("method") != "ACT + Heuristic Recovery"
        or comparison.get("candidate") != "reim_tau_0.175"
        or comparison.get("reference") != "heuristic_gate"
    ):
        raise SensitivityError("matched-gate audit has unexpected method labels")
    return payload


def _plot(
    path: Path,
    rows: list[dict[str, Any]],
    matched_gate_audit: dict[str, Any],
) -> None:
    # Flat-vector style aligned with the Figure 1/2 paper design language:
    # white page, slate text, light horizontal grid only, generous type sizes.
    C_INK = "#0F172A"
    C_GRAY = "#64748B"
    C_GRID = "#E9EEF3"
    C_ORANGE = "#D97706"
    C_BLUE = "#2563EB"
    C_GREEN = "#059669"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": C_INK,
            "text.color": C_INK,
            "xtick.color": C_GRAY,
            "ytick.color": C_GRAY,
        }
    )
    success = 100.0 * np.asarray([row["Success Rate"] for row in rows])
    burden = 100.0 * np.asarray([row["Intervention Rate"] for row in rows])
    thresholds = [float(row["Threshold"]) for row in rows]
    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    axis.plot(
        burden,
        success,
        color=C_ORANGE,
        linewidth=2.4,
        marker="D",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=3,
    )
    annotation_layout = {
        0.100: ((-4, 11), "right"),
        0.125: ((8, -20), "left"),
        0.150: ((10, -20), "left"),
        0.175: ((-10, -22), "right"),
        0.200: ((10, -22), "left"),
    }
    for x_value, y_value, threshold in zip(
        burden, success, thresholds, strict=True
    ):
        offset, alignment = annotation_layout[round(threshold, 3)]
        axis.annotate(
            rf"$\tau={threshold:g}$",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=C_GRAY,
            ha=alignment,
        )
    axis.scatter(
        [burden[-1]],
        [success[-1]],
        s=140,
        facecolors="none",
        edgecolors=C_BLUE,
        linewidths=2.2,
        label=r"Frozen primary gate ($\tau=0.20$)",
        zorder=4,
    )
    heuristic = matched_gate_audit["methods"]["heuristic_gate"]
    heuristic_x = 100.0 * float(heuristic["intervention_rate"])
    heuristic_y = 100.0 * float(heuristic["success_rate"])
    axis.scatter(
        [heuristic_x],
        [heuristic_y],
        s=110,
        marker="X",
        color=C_GREEN,
        edgecolors="white",
        linewidths=1.0,
        label="Heuristic gate",
        zorder=5,
    )
    matched = matched_gate_audit["paired_success_comparisons"][
        "reim_tau_0.175_minus_heuristic"
    ]
    matched_x = 100.0 * float(
        matched_gate_audit["methods"]["reim_tau_0.175"]["intervention_rate"]
    )
    matched_y = 100.0 * float(
        matched_gate_audit["methods"]["reim_tau_0.175"]["success_rate"]
    )
    axis.annotate(
        "",
        xy=(matched_x, matched_y - 0.25),
        xytext=(heuristic_x, heuristic_y + 0.25),
        arrowprops={
            "arrowstyle": "->",
            "color": C_GREEN,
            "linewidth": 1.6,
            "linestyle": "--",
        },
        zorder=3,
    )
    axis.text(
        matched_x + 1.2,
        0.5 * (matched_y + heuristic_y),
        (
            f"+{float(matched['paired_delta_percentage_points']):.1f} pp\n"
            f"$p={float(matched['exact_two_sided_mcnemar_binomial_p']):.3f}$"
        ),
        color=C_GREEN,
        fontsize=9.5,
        fontweight="bold",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ECFDF5",
            "edgecolor": C_GREEN,
            "linewidth": 0.9,
        },
    )
    axis.set(
        xlabel="Episodes with recovery intervention (%)",
        ylabel="Task success (%)",
        xlim=(59, 101.5),
        ylim=(83.5, 98.9),
    )
    axis.set_title(
        "Post-freeze gate sensitivity",
        fontsize=13,
        fontweight="bold",
        color=C_INK,
        pad=10,
    )
    axis.tick_params(labelsize=10)
    for label in (axis.xaxis.label, axis.yaxis.label):
        label.set_fontsize(11)
    axis.grid(axis="y", color=C_GRID, linewidth=1.0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    figure.text(
        0.5,
        0.005,
        "Separate paired Meta-World diagnostic • n=200 • not used for selection",
        ha="center",
        fontsize=8.5,
        color=C_GRAY,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results" / "tables"
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=8_200_042)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "tables" / "gate_sensitivity.csv"
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results" / "figures" / "gate_sensitivity.png",
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=ROOT / "paper_assets" / "Figure4_gate_sensitivity.png",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "tables" / "gate_sensitivity_manifest.json",
    )
    parser.add_argument(
        "--matched-gate-audit",
        type=Path,
        default=ROOT / "results" / "tables" / "gate_matched_comparison.json",
    )
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    reference_crn: dict[int, tuple[str, str, str]] | None = None
    for threshold, filename in THRESHOLD_FILES.items():
        row, condition_hashes, condition_crn = _audit_condition(
            threshold,
            args.input_dir / filename,
            episodes=args.episodes,
            seed_start=args.seed_start,
        )
        if reference_crn is None:
            reference_crn = condition_crn
        elif condition_crn != reference_crn:
            raise SensitivityError(
                f"tau={threshold:g}: CRN episode/task specifications differ"
            )
        rows.append(row)
        hashes.update(condition_hashes)
    _write_csv(args.output, rows)
    matched_gate_audit = _load_matched_gate_audit(
        args.matched_gate_audit,
        expected_bank_sha256=str(rows[0]["Episode Bank SHA256"]),
        expected_episodes=args.episodes,
    )
    _plot(args.figure, rows, matched_gate_audit)
    args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
    args.paper_figure.write_bytes(args.figure.read_bytes())
    args.paper_figure.with_suffix(".pdf").write_bytes(
        args.figure.with_suffix(".pdf").read_bytes()
    )
    manifest = {
        "artifact": "post_freeze_gate_sensitivity",
        "selection_rule": "none; frozen primary tau=0.20 is not changed",
        "paired_seed_start": args.seed_start,
        "paired_seed_end": args.seed_start + args.episodes - 1,
        "episodes_per_threshold": args.episodes,
        "thresholds": list(THRESHOLD_FILES),
        "crn_episode_specifications_identical": True,
        "episode_bank_sha256": rows[0]["Episode Bank SHA256"],
        "episode_bank_file_sha256": rows[0]["Episode Bank File SHA256"],
        "inputs": hashes,
        "matched_gate_audit": {
            "path": _project_key(args.matched_gate_audit),
            "sha256": _sha256(args.matched_gate_audit),
        },
        "outputs": {
            _project_key(args.output): _sha256(args.output),
            _project_key(args.figure): _sha256(args.figure),
            _project_key(args.figure.with_suffix(".pdf")): _sha256(
                args.figure.with_suffix(".pdf")
            ),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
