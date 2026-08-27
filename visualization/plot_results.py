"""Generate REIM figures and LaTeX tables strictly from measured results.

All numerical figures validate their source files before drawing.  Toy-backend
outputs are visibly stamped as integration tests and are never captioned as
Meta-World benchmark results.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_reim import PROJECT_ROOT

LOGGER = logging.getLogger("reim.visualization")

# Restrained, color-blind-friendly palette: two functional colors + one accent.
BLUE = "#356A9A"
TEAL = "#4B8F8C"
ORANGE = "#D97935"
INK = "#25313C"
MID_GRAY = "#75808A"
LIGHT_GRAY = "#E9EDF0"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F3F1"
PALE_ORANGE = "#FAEEE5"
WHITE = "#FFFFFF"

CURRICULUM_TRIGGER_THRESHOLD = 0.10
FINAL_TRIGGER_THRESHOLD = 0.20
FINAL_RECOVERY_BUDGET = 150
FINAL_RECOVERY_DEFINITION = (
    "task_success_while_recovery_active_per_intervention"
)


class ResultValidationError(ValueError):
    """Raised when a measured result file is absent or malformed."""


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        # A headless backend is required on training servers.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate figures") from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": MID_GRAY,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return matplotlib, plt


def _read_csv(path: str | Path, required: Sequence[str]) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"required measured result is missing: {source}. Run its experiment first."
        )
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise ResultValidationError(
                f"{source} is missing required columns: {', '.join(missing)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ResultValidationError(f"{source} contains no result rows")
    return rows


def _number(
    row: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultValidationError(f"invalid numeric value in column {key!r}") from exc
    if not math.isfinite(value):
        raise ResultValidationError(f"non-finite numeric value in column {key!r}")
    if minimum is not None and value < minimum:
        raise ResultValidationError(f"{key!r}={value} is below {minimum}")
    if maximum is not None and value > maximum:
        raise ResultValidationError(f"{key!r}={value} is above {maximum}")
    return value


def _validate_rates(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> None:
    for row in rows:
        for key in keys:
            _number(row, key, minimum=0.0, maximum=1.0)


def _backend(rows: Sequence[Mapping[str, Any]]) -> str:
    values = {
        str(row.get("Backend", row.get("backend", ""))).strip().lower()
        for row in rows
    }
    values.discard("")
    if not values:
        raise ResultValidationError(
            "result file has no Backend metadata; refusing to present provenance-free metrics"
        )
    return values.pop() if len(values) == 1 else "mixed"


def _profile(rows: Sequence[Mapping[str, Any]]) -> str:
    values = {
        str(row.get("Profile", row.get("profile", ""))).strip().lower()
        for row in rows
    }
    values.discard("")
    if not values:
        return "unspecified"
    return values.pop() if len(values) == 1 else "mixed"


def _provenance(rows: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    backend = _backend(rows)
    profile = _profile(rows)
    episode_values: list[int] = []
    for row in rows:
        raw = row.get("Episodes", row.get("episodes"))
        if raw not in (None, ""):
            try:
                episode_values.append(int(float(raw)))
            except (TypeError, ValueError):
                pass
    n_text = (
        f" • n={min(episode_values)} episodes/condition"
        if episode_values
        else ""
    )
    eligibility_values = {
        str(row.get("Benchmark Eligible", "")).strip().lower() for row in rows
    }
    explicitly_eligible = eligibility_values <= {"true", "1", "yes"} and bool(
        eligibility_values
    )
    enough_episodes = bool(episode_values) and min(episode_values) >= 200
    if (
        backend == "metaworld"
        and profile == "full"
        and explicitly_eligible
        and enough_episodes
    ):
        return f"Meta-World PickPlace • full protocol{n_text}", True
    if backend == "metaworld":
        return (
            f"Meta-World {profile.upper()} EVALUATION — NOT THE FULL BENCHMARK{n_text}",
            False,
        )
    if backend == "toy":
        profile_label = "SMOKE " if profile == "smoke" else ""
        return (
            f"TOY {profile_label}INTEGRATION TEST — NOT A BENCHMARK RESULT{n_text}",
            False,
        )
    return f"Mixed/unknown evaluation backends — not a benchmark{n_text}", False


def _footer(fig: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    label, eligible = _provenance(rows)
    fig.text(
        0.5,
        0.005,
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=MID_GRAY if eligible else ORANGE,
        weight="normal" if eligible else "bold",
    )


def _save(fig: Any, path: str | Path, *, title: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.tmp{output.suffix}"
    )
    raster_metadata = {
        "Title": title,
        "Author": "REIM experiment pipeline",
        "Software": "Matplotlib",
    }
    fig.savefig(temporary_output, dpi=320, metadata=raster_metadata)
    temporary_output.replace(output)
    # A vector companion is useful for paper editing without changing the
    # user-requested PNG path.
    vector = output.with_suffix(".pdf")
    temporary_vector = vector.with_name(f".{vector.stem}.tmp{vector.suffix}")
    vector_metadata = {
        "Title": title,
        "Author": "REIM experiment pipeline",
        "Creator": "Matplotlib",
    }
    fig.savefig(temporary_vector, metadata=vector_metadata)
    temporary_vector.replace(vector)
    LOGGER.info("Saved %s (and vector companion %s)", output, vector)
    return output


def _copy_figure_pair(source_png: str | Path, destination_png: str | Path) -> Path:
    """Atomically copy both raster and vector companions of one figure."""

    source = Path(source_png)
    destination = Path(destination_png)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source_item, destination_item in (
        (source, destination),
        (source.with_suffix(".pdf"), destination.with_suffix(".pdf")),
    ):
        if not source_item.is_file():
            raise FileNotFoundError(f"missing figure companion: {source_item}")
        temporary = destination_item.with_name(
            f".{destination_item.stem}.tmp{destination_item.suffix}"
        )
        shutil.copyfile(source_item, temporary)
        temporary.replace(destination_item)
    return destination


def _despine(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _ci_errors(
    rows: Sequence[Mapping[str, Any]], value_key: str, lower_key: str, upper_key: str
) -> np.ndarray | None:
    if not all(lower_key in row and upper_key in row for row in rows):
        return None
    values = np.asarray([_number(row, value_key) for row in rows])
    lower = np.asarray([_number(row, lower_key) for row in rows])
    upper = np.asarray([_number(row, upper_key) for row in rows])
    if np.any(lower > values) or np.any(values > upper):
        raise ResultValidationError(
            f"invalid confidence interval ordering for {value_key!r}"
        )
    return np.vstack((values - lower, upper - values))


def _short_method(name: str) -> str:
    normalized = name.replace("BC", "ACT")
    if normalized.startswith("REIM"):
        return "REIM"
    if normalized == "ACT + Detector":
        return "LSTM 0.2× Hold"
    return normalized


def _draw_success(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    _validate_rates(rows, ("Success Rate", "Recovery Rate"))
    labels = [_short_method(str(row["Method"])) for row in rows]
    success = np.asarray([_number(row, "Success Rate") for row in rows])
    x = np.arange(len(rows), dtype=np.float64)
    success_errors = _ci_errors(
        rows, "Success Rate", "Success CI Lower", "Success CI Upper"
    )
    ax.bar(
        x,
        success * 100.0,
        0.62,
        yerr=None if success_errors is None else success_errors * 100.0,
        color=BLUE,
        edgecolor=WHITE,
        linewidth=0.5,
        capsize=2.5,
        label="Task success (Wilson 95% CI)",
        zorder=3,
    )
    if "REIM" in labels:
        idx = labels.index("REIM")
        ax.scatter(
            [x[idx]],
            [success[idx] * 100.0 + 2.5],
            marker="D",
            s=18,
            color=ORANGE,
            zorder=5,
            label="Full REIM",
        )
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Closed-loop task performance")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _despine(ax)


def _intervention_fraction(row: Mapping[str, Any]) -> float:
    episodes = _number(row, "Episodes", minimum=1.0)
    intervened = _number(
        row, "Intervened Episodes", minimum=0.0, maximum=episodes
    )
    return intervened / episodes


def _draw_intervention_burden(
    ax: Any, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Show the runtime cost hidden by similar task-success bars."""

    labels = [_short_method(str(row["Method"])) for row in rows]
    fractions = np.asarray([_intervention_fraction(row) for row in rows])
    x = np.arange(len(rows), dtype=np.float64)
    colors = [ORANGE if label == "REIM" else TEAL for label in labels]
    bars = ax.bar(
        x,
        fractions * 100.0,
        width=0.62,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.5,
        zorder=3,
    )
    for bar, value, label in zip(bars, fractions * 100.0, labels, strict=True):
        text = "--" if label == "ACT" and value == 0.0 else f"{value:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            text,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    rl_label = next(
        (label for label in labels if label == "ACT + Heuristic Recovery"), None
    )
    if rl_label is not None and "REIM" in labels:
        rl_index = labels.index(rl_label)
        reim_index = labels.index("REIM")
        rl_rate = fractions[rl_index]
        reim_rate = fractions[reim_index]
        if rl_rate > 0.0 and reim_rate <= rl_rate:
            reduction = 100.0 * (rl_rate - reim_rate) / rl_rate
            line_y = min(96.0, max(rl_rate, reim_rate) * 100.0 + 7.0)
            ax.plot(
                [rl_index, rl_index, reim_index, reim_index],
                [line_y - 1.5, line_y, line_y, line_y - 1.5],
                color=INK,
                linewidth=0.8,
                zorder=4,
            )
            ax.text(
                (rl_index + reim_index) / 2,
                line_y + 1.2,
                f"{reduction:.1f}% fewer",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK,
            )
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel("Episodes intervened (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Intervention burden")
    _despine(ax)


def plot_success_comparison(
    baseline_csv: str | Path,
    output_path: str | Path,
) -> Path:
    rows = _read_csv(
        baseline_csv, ("Method", "Success Rate", "Recovery Rate", "Average Steps")
    )
    _, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    _draw_success(ax, rows)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _footer(fig, rows)
    result = _save(fig, output_path, title="REIM success and recovery comparison")
    plt.close(fig)
    return result


def _noise_fraction(row: Mapping[str, Any]) -> float:
    if "Noise Level" in row and row["Noise Level"] != "":
        return _number(row, "Noise Level", minimum=0.0, maximum=1.0)
    if "Noise (%)" in row and row["Noise (%)"] != "":
        return _number(row, "Noise (%)", minimum=0.0, maximum=100.0) / 100.0
    raise ResultValidationError("robustness CSV needs Noise Level or Noise (%)")


def _draw_robustness(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    _validate_rates(rows, ("Success Rate",))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_short_method(str(row["Method"])), []).append(row)
    styles = [
        (BLUE, "--", "o"),
        (ORANGE, "-", "D"),
        (TEAL, "-.", "s"),
    ]
    plotted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, (method, method_rows) in enumerate(grouped.items()):
        if index >= len(styles):
            color, line_style, marker = MID_GRAY, (0, (1, 1)), "^"
        else:
            color, line_style, marker = styles[index]
        ordered = sorted(method_rows, key=_noise_fraction)
        noise = np.asarray([_noise_fraction(row) * 100.0 for row in ordered])
        success = np.asarray([_number(row, "Success Rate") * 100.0 for row in ordered])
        plotted[method] = (noise, success)
        ax.plot(
            noise,
            success,
            color=color,
            linestyle=line_style,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=method,
            zorder=4 - min(index, 2),
        )
        errors = _ci_errors(
            ordered, "Success Rate", "Success CI Lower", "Success CI Upper"
        )
        if errors is not None:
            ax.fill_between(
                noise,
                success - errors[0] * 100.0,
                success + errors[1] * 100.0,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
    plotted_values = list(plotted.values())
    if len(plotted_values) >= 2 and all(
        np.array_equal(plotted_values[0][0], item[0])
        and np.allclose(plotted_values[0][1], item[1], atol=1e-12)
        for item in plotted_values[1:]
    ):
        ax.text(
            0.02,
            0.04,
            "Paired task outcomes overlap at every noise level",
            transform=ax.transAxes,
            fontsize=7.5,
            color=MID_GRAY,
        )
    ax.set_xlabel("Injected disturbance level (%)")
    ax.set_ylabel("Task success (%)")
    ax.set_xticks(sorted({_noise_fraction(row) * 100.0 for row in rows}))
    ax.set_ylim(0.0, 103.0)
    ax.set_title("Robustness to execution disturbances")
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)


def plot_robustness(
    robustness_csv: str | Path,
    output_path: str | Path,
) -> Path:
    rows = _read_csv(robustness_csv, ("Method", "Success Rate"))
    # Force validation before constructing a figure.
    for row in rows:
        _noise_fraction(row)
    _, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    _draw_robustness(ax, rows)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _footer(fig, rows)
    result = _save(fig, output_path, title="REIM robustness curve")
    plt.close(fig)
    return result


def _draw_ablation(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    _validate_rates(rows, ("Success Rate",))
    ordered_all = sorted(rows, key=lambda row: str(row.get("Variant", "")))
    # The detector-only safety hold is retained in the raw engineering audit,
    # but it is not a coherent controller baseline: it changes the action
    # magnitude without providing any recovery behavior.  The paper-facing
    # ablation therefore isolates the two meaningful removals from REIM:
    # recovery itself and the learned gate.
    ordered = [
        row for row in ordered_all if str(row.get("Variant", "")) in {"A", "C", "D"}
    ]
    if len(ordered) != 3:
        raise ResultValidationError(
            "paper ablation requires variants A (ACT), C (heuristic gate), "
            "and D (full REIM)"
        )
    labels_by_variant = {
        "A": "ACT\n(no recovery)",
        "C": "Heuristic gate\n+ recovery",
        "D": "REIM\n(LSTM gate)",
    }
    labels = [labels_by_variant[str(row["Variant"])] for row in ordered]
    values = np.asarray([_number(row, "Success Rate") * 100.0 for row in ordered])
    colors = [BLUE, TEAL, ORANGE]
    errors = _ci_errors(
        ordered, "Success Rate", "Success CI Lower", "Success CI Upper"
    )
    bars = ax.bar(
        np.arange(len(ordered)),
        values,
        yerr=None if errors is None else errors * 100.0,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.5,
        capsize=2.5,
        width=0.68,
        zorder=3,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(ordered)), labels)
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0.0, 108.0)
    ax.set_title("Recovery and gate ablation")
    _despine(ax)


def plot_ablation(ablation_csv: str | Path, output_path: str | Path) -> Path:
    rows = _read_csv(ablation_csv, ("Method", "Success Rate"))
    _, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    _draw_ablation(ax, rows)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _footer(fig, rows)
    result = _save(fig, output_path, title="REIM component ablation")
    plt.close(fig)
    return result


def _load_confusion(path: str | Path) -> tuple[np.ndarray, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"detector metrics are missing: {source}. Train the detector first."
        )
    backend = "dataset"
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ResultValidationError(f"{source} must contain a JSON object")
        deployment_metrics = data.get("deployment_threshold_metrics")
        if isinstance(deployment_metrics, Mapping):
            candidate: Any = deployment_metrics.get("confusion_matrix")
            reported_threshold = deployment_metrics.get("threshold")
        else:
            candidate = None
            reported_threshold = None
        if candidate is None:
            candidate = data.get("confusion_matrix")
            reported_threshold = data.get("threshold")
        if candidate is None and isinstance(data.get("metrics"), Mapping):
            candidate = data["metrics"].get("confusion_matrix")
            reported_threshold = data["metrics"].get(
                "threshold", reported_threshold
            )
        if candidate is None and all(key in data for key in ("tn", "fp", "fn", "tp")):
            candidate = [[data["tn"], data["fp"]], [data["fn"], data["tp"]]]
        if candidate is None:
            raise ResultValidationError(
                f"{source} lacks confusion_matrix or tn/fp/fn/tp"
            )
        matrix = np.asarray(candidate, dtype=np.float64)
        backend = str(
            data.get("backend", data.get("dataset", "failure validation split"))
        )
        if reported_threshold is not None:
            backend = f"{backend} (deployment threshold = {float(reported_threshold):.2f})"
    elif source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = "confusion_matrix"
            if key not in data:
                raise ResultValidationError(f"{source} lacks confusion_matrix")
            matrix = np.asarray(data[key], dtype=np.float64)
    elif source.suffix.lower() == ".csv":
        raw = np.loadtxt(source, delimiter=",", dtype=np.float64)
        matrix = np.asarray(raw)
    else:
        raise ResultValidationError(
            f"unsupported confusion source {source.suffix}; use JSON, NPZ, or CSV"
        )
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ResultValidationError("confusion matrix must be a non-negative 2x2 matrix")
    if matrix.sum() <= 0:
        raise ResultValidationError("confusion matrix contains no evaluated samples")
    return matrix.astype(np.int64), backend


def plot_confusion_matrix(
    detector_metrics: str | Path,
    output_path: str | Path,
) -> Path:
    matrix, split_label = _load_confusion(detector_metrics)
    matplotlib, plt = _matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("reim_blues", [WHITE, BLUE])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    image = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    total_by_row = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fractions = matrix / total_by_row
    threshold = float(matrix.max()) * 0.55
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{fractions[row, column] * 100:.1f}%",
                ha="center",
                va="center",
                color=WHITE if matrix[row, column] > threshold else INK,
                fontsize=10,
            )
    ax.set_xticks((0, 1), ("Pred. low risk", "Pred. event risk"))
    ax.set_yticks(
        (0, 1),
        ("No event\nwithin 10 steps", "Event now or\nwithin 10 steps"),
    )
    ax.tick_params(axis="both", labelsize=8.5)
    ax.tick_params(length=0)
    ax.set_title("Causal risk-monitor confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Samples")
    fig.text(
        0.5,
        0.005,
        f"Measured on {split_label}",
        ha="center",
        color=MID_GRAY,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    result = _save(fig, output_path, title="Causal risk-monitor confusion matrix")
    plt.close(fig)
    return result


def _load_traces(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"recovery traces are missing: {source}. Evaluate REIM with trace capture first."
        )
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ResultValidationError(f"{source} contains no recovery traces")
    traces = [item for item in data if isinstance(item, dict)]
    traces = [trace for trace in traces if int(trace.get("recovery_attempts", 0)) > 0]
    if not traces:
        raise ResultValidationError(
            f"{source} has no episodes with an actual recovery intervention"
        )
    traces.sort(
        key=lambda trace: (
            int(trace.get("recovery_successes", 0)) <= 0,
            not bool(trace.get("success", False)),
        )
    )
    for trace in traces:
        risks = np.asarray(trace.get("failure_probabilities", []), dtype=np.float64)
        if risks.ndim != 1 or risks.size == 0 or not np.isfinite(risks).all():
            raise ResultValidationError(
                f"{source} contains a trace with invalid failure probabilities"
            )
        infos = trace.get("info", [])
        if not isinstance(infos, list) or not any(
            isinstance(info, Mapping) and info.get("backend") for info in infos
        ):
            raise ResultValidationError(
                f"{source} trace lacks backend provenance; rerun evaluation"
            )
    return traces


def _position_series(
    infos: Sequence[Mapping[str, Any]], key: str
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    for info in infos:
        if key not in info:
            return None
        value = np.asarray(info[key], dtype=np.float64).reshape(-1)
        if value.size < 2 or not np.isfinite(value[:2]).all():
            return None
        values.append(value)
    return np.stack(values)


def plot_recovery_examples(
    trace_json: str | Path,
    output_path: str | Path,
    *,
    max_examples: int = 3,
    trigger_threshold: float = FINAL_TRIGGER_THRESHOLD,
    provenance_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    traces = _load_traces(trace_json)[:max_examples]
    _, plt = _matplotlib()
    fig, axes = plt.subplots(
        2,
        len(traces),
        figsize=(4.0 * len(traces), 5.7),
        squeeze=False,
        constrained_layout=False,
    )
    for index, trace in enumerate(traces):
        infos = [item for item in trace.get("info", []) if isinstance(item, dict)]
        object_positions = _position_series(infos, "object_position")
        ee_positions = _position_series(infos, "ee_position")
        goal_positions = _position_series(infos, "goal_position")
        top = axes[0, index]
        if object_positions is not None:
            top.plot(
                object_positions[:, 0],
                object_positions[:, 1],
                color=ORANGE,
                linewidth=1.8,
                marker="o",
                markevery=max(1, len(object_positions) // 8),
                markersize=2.8,
                label="Object",
            )
        if ee_positions is not None:
            top.plot(
                ee_positions[:, 0],
                ee_positions[:, 1],
                color=BLUE,
                linewidth=1.2,
                linestyle="--",
                label="End effector",
            )
        if goal_positions is not None:
            goal = goal_positions[-1]
            top.scatter(
                [goal[0]],
                [goal[1]],
                s=65,
                marker="*",
                color=TEAL,
                edgecolor=WHITE,
                linewidth=0.4,
                label="Goal",
                zorder=4,
            )
        if object_positions is None and ee_positions is None:
            actions = np.asarray(trace.get("actions", []), dtype=np.float64)
            if actions.ndim != 2 or not actions.size:
                raise ResultValidationError("trace has neither positions nor valid actions")
            top.plot(
                np.linalg.norm(actions[:, : min(3, actions.shape[1])], axis=1),
                color=BLUE,
                linewidth=1.5,
            )
            top.set_xlabel("Control step")
            top.set_ylabel("Translation action norm")
        else:
            top.set_xlabel("Workspace x")
            top.set_ylabel("Workspace y")
            top.set_aspect("equal", adjustable="datalim")
            top.legend(frameon=False, fontsize=6.8, loc="best")
        controller_status = (
            "recovery completed the task"
            if int(trace.get("recovery_successes", 0)) > 0
            else "recovery exhausted"
        )
        task_status = "task success" if bool(trace.get("success", False)) else "task failed"
        top.set_title(
            f"Episode {trace.get('episode', index)}",
            fontsize=9.5,
            fontweight="bold",
            loc="left",
            pad=10,
        )
        top.text(
            0.0,
            1.01,
            f"{controller_status}; {task_status}",
            transform=top.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            color=MID_GRAY,
        )
        _despine(top)

        bottom = axes[1, index]
        risks = np.asarray(trace.get("failure_probabilities", []), dtype=np.float64)
        sources = list(trace.get("sources", []))
        steps = np.arange(risks.size)
        bottom.plot(steps, risks, color=BLUE, linewidth=1.5, label="Predicted risk")
        bottom.axhline(
            trigger_threshold,
            color=ORANGE,
            linestyle="--",
            linewidth=1.1,
            label=f"Trigger {trigger_threshold:.2f}",
        )
        active = np.asarray([source == "recovery" for source in sources[: risks.size]])
        if active.size < risks.size:
            active = np.pad(active, (0, risks.size - active.size))
        bottom.fill_between(
            steps,
            0.0,
            1.0,
            where=active,
            transform=bottom.get_xaxis_transform(),
            color=TEAL,
            alpha=0.16,
            step="mid",
            label="recovery option",
        )
        bottom.set_ylim(-0.02, 1.03)
        bottom.set_xlabel("Control step")
        bottom.set_ylabel("Failure probability")
        bottom.legend(frameon=False, fontsize=6.8, loc="upper right")
        _despine(bottom)

    first_infos = traces[0].get("info", [])
    backend_rows = (
        list(provenance_rows)
        if provenance_rows is not None
        else [
            {
                "backend": info.get("backend", ""),
                "Profile": traces[0].get("profile", "unspecified"),
                "Episodes": traces[0].get("evaluation_episodes", ""),
                "Benchmark Eligible": traces[0].get(
                    "benchmark_eligible", False
                ),
            }
            for info in first_infos
            if isinstance(info, dict) and info.get("backend")
        ]
    )
    label, eligible = _provenance(backend_rows)
    fig.text(
        0.5,
        0.005,
        label,
        ha="center",
        color=MID_GRAY if eligible else ORANGE,
        fontsize=7.5,
        weight="normal" if eligible else "bold",
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    result = _save(fig, output_path, title="Measured REIM recovery examples")
    plt.close(fig)
    return result


def plot_framework(output_path: str | Path) -> Path:
    """Draw the online trigger-state REIM training and execution architecture."""
    from visualization.generate_paper_figure1 import create_top_tier_framework_figure

    output = Path(output_path)
    vector = output.with_suffix(".pdf")
    create_top_tier_framework_figure(output, vector, dpi=320)
    LOGGER.info("Saved %s (and vector companion %s)", output, vector)
    return output


def plot_result_composite(
    baseline_csv: str | Path,
    robustness_csv: str | Path,
    output_path: str | Path,
) -> Path:
    baseline = _read_csv(
        baseline_csv,
        (
            "Method",
            "Success Rate",
            "Recovery Rate",
            "Average Steps",
            "Episodes",
            "Intervened Episodes",
        ),
    )
    robustness = _read_csv(robustness_csv, ("Method", "Success Rate"))
    if _backend(baseline) != _backend(robustness):
        raise ResultValidationError(
            "baseline and robustness files use different backends; refusing to combine"
        )
    _, plt = _matplotlib()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 4.15),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)},
    )
    _draw_success(axes[0], baseline)
    axes[0].text(
        -0.12,
        1.03,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=10,
        weight="bold",
    )
    _draw_intervention_burden(axes[1], baseline)
    axes[1].text(
        -0.12,
        1.03,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=10,
        weight="bold",
    )
    _draw_robustness(axes[2], robustness)
    axes[2].text(
        -0.12,
        1.03,
        "(c)",
        transform=axes[2].transAxes,
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    _footer(fig, baseline)
    result = _save(fig, output_path, title="REIM quantitative results")
    plt.close(fig)
    return result


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _rate_with_ci(
    row: Mapping[str, Any], key: str, lower_key: str, upper_key: str
) -> str:
    value = 100.0 * _number(row, key, minimum=0.0, maximum=1.0)
    if lower_key in row and upper_key in row:
        lower = 100.0 * _number(row, lower_key, minimum=0.0, maximum=1.0)
        upper = 100.0 * _number(row, upper_key, minimum=0.0, maximum=1.0)
        return f"{value:.1f} [{lower:.1f}, {upper:.1f}]"
    return f"{value:.1f}"


def _latex_caption(rows: Sequence[Mapping[str, Any]], table_name: str) -> str:
    provenance, eligible = _provenance(rows)
    if eligible:
        return f"{table_name} on {_latex_escape(provenance)}. Rates are percentages."
    return (
        f"{table_name}: {_latex_escape(provenance)}. "
        "These values validate software integration only."
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def export_baseline_latex(
    baseline_csv: str | Path, output_path: str | Path
) -> Path:
    rows = _read_csv(
        baseline_csv, ("Method", "Success Rate", "Recovery Rate", "Average Steps")
    )
    _validate_rates(rows, ("Success Rate", "Recovery Rate"))
    body = []
    for row in rows:
        method = _latex_escape(_short_method(str(row["Method"])))
        success = _rate_with_ci(
            row, "Success Rate", "Success CI Lower", "Success CI Upper"
        )
        recovery = (
            "--"
            if int(float(row.get("Recovery Attempts", 0))) == 0
            else _rate_with_ci(
                row, "Recovery Rate", "Recovery CI Lower", "Recovery CI Upper"
            )
        )
        steps = _number(row, "Average Steps", minimum=0.0)
        body.append(f"{method} & {success} & {recovery} & {steps:.2f} \\\\")
    content = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            (
                rf"\caption{{{_latex_caption(rows, 'Policy comparison')} "
                r"Task-success intervals are Wilson 95\% CIs; intervention "
                r"ratios use episode bootstrap 95\% CIs.}"
            ),
            r"\label{tab:baseline}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Method & Success $\uparrow$ & Intervention outcome$^\ast$ & Steps $\downarrow$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\vspace{1mm}\parbox{0.98\linewidth}{\footnotesize "
                r"$^\ast$Random Reset: post-reset task success per reset; "
                r"Heuristic Recovery/REIM: task success reached while the recovery controller "
                r"is active, per intervention. "
                r"These rates have different semantics and are not compared directly.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output = Path(output_path)
    _atomic_text(output, content)
    LOGGER.info("Saved measured baseline LaTeX table to %s", output)
    return output


def export_ablation_latex(
    ablation_csv: str | Path, output_path: str | Path
) -> Path:
    rows = _read_csv(ablation_csv, ("Method", "Success Rate", "Average Steps"))
    _validate_rates(rows, ("Success Rate",))
    body = []
    ordered = [
        row
        for row in sorted(rows, key=lambda item: str(item.get("Variant", "")))
        if str(row.get("Variant", "")) in {"A", "C", "D"}
    ]
    display_names = {
        "A": "ACT (no recovery)",
        "C": "Heuristic gate + recovery",
        "D": "REIM (LSTM gate)",
    }
    for row in ordered:
        variant = str(row["Variant"])
        method = _latex_escape(display_names[variant])
        success = _rate_with_ci(
            row, "Success Rate", "Success CI Lower", "Success CI Upper"
        )
        recovery = (
            _rate_with_ci(
                row, "Recovery Rate", "Recovery CI Lower", "Recovery CI Upper"
            )
            if "Recovery Rate" in row
            and int(float(row.get("Recovery Attempts", 0))) > 0
            else "--"
        )
        body.append(f"{method} & {success} & {recovery} \\\\")
    content = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            (
                rf"\caption{{{_latex_caption(rows, 'REIM component ablation')} "
                r"Task-success intervals are Wilson 95\% CIs.}"
            ),
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Configuration & Success $\uparrow$ & Intervention outcome$^\ast$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\vspace{1mm}\parbox{0.98\linewidth}{\footnotesize "
                r"$^\ast$Heuristic Recovery/REIM values are task completions while the "
                r"recovery controller is active, per intervention; "
                r"methods without an intervention report --.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output = Path(output_path)
    _atomic_text(output, content)
    LOGGER.info("Saved measured ablation LaTeX table to %s", output)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_rel(path: Any) -> str:
    """Return a repo-relative forward-slash path so manifests stay machine-agnostic."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_final_intervention_semantics(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_name: str,
) -> None:
    """Reject legacy risk-clear results before any publication asset is drawn."""

    for row in rows:
        method = str(row.get("Method", ""))
        attempts = int(float(row.get("Recovery Attempts", 0) or 0))
        definition = str(row.get("Recovery Definition", "")).strip()
        if attempts <= 0:
            if definition not in {"", "not_applicable"}:
                raise ResultValidationError(
                    f"{source_name}: {method} has no intervention but reports "
                    f"Recovery Definition={definition!r}"
                )
            continue
        expected = (
            "post_reset_task_success_per_reset"
            if "Random Reset" in method
            else FINAL_RECOVERY_DEFINITION
        )
        if definition != expected:
            raise ResultValidationError(
                f"{source_name}: {method} uses legacy/incompatible recovery "
                f"semantics {definition!r}; expected {expected!r}"
            )


def _validate_publication_gate(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    detector_metrics: str | Path,
) -> None:
    """Require the frozen final protocol before producing paper-named assets."""

    for name, rows in (
        ("baseline.csv", baseline_rows),
        ("robustness.csv", robustness_rows),
        ("ablation.csv", ablation_rows),
    ):
        provenance, eligible = _provenance(rows)
        if not eligible:
            raise ResultValidationError(
                f"{name} is not a full benchmark input ({provenance})"
            )
        if not all(_truthy(row.get("Benchmark Eligible", "")) for row in rows):
            raise ResultValidationError(f"{name} contains an ineligible row")
        _validate_final_intervention_semantics(rows, source_name=name)

    source = Path(detector_metrics)
    with source.open("r", encoding="utf-8") as handle:
        detector = json.load(handle)
    if not isinstance(detector, Mapping):
        raise ResultValidationError(f"{source} must contain a JSON object")
    deployment = detector.get("deployment_threshold_metrics")
    if not isinstance(deployment, Mapping):
        raise ResultValidationError(
            f"{source} lacks deployment_threshold_metrics"
        )
    threshold = _number(deployment, "threshold", minimum=0.0, maximum=1.0)
    if not math.isclose(
        threshold, FINAL_TRIGGER_THRESHOLD, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ResultValidationError(
            f"{source} reports deployment threshold {threshold:.3f}; "
            f"the frozen final gate is {FINAL_TRIGGER_THRESHOLD:.2f}"
        )


def generate_all(
    *,
    baseline_csv: str | Path,
    robustness_csv: str | Path,
    ablation_csv: str | Path,
    detector_metrics: str | Path,
    trace_json: str | Path,
    figures_dir: str | Path,
    paper_assets_dir: str | Path,
) -> dict[str, str]:
    """Validate all measured inputs, then atomically generate the paper assets."""

    # Validate every required source before producing partial output.
    baseline_rows = _read_csv(
        baseline_csv, ("Method", "Success Rate", "Recovery Rate", "Average Steps")
    )
    robustness_rows = _read_csv(robustness_csv, ("Method", "Success Rate"))
    ablation_rows = _read_csv(ablation_csv, ("Method", "Success Rate", "Average Steps"))
    _validate_rates(baseline_rows, ("Success Rate", "Recovery Rate"))
    _validate_rates(robustness_rows, ("Success Rate",))
    _validate_rates(ablation_rows, ("Success Rate",))
    for row in baseline_rows:
        _number(row, "Average Steps", minimum=0.0)
    for row in ablation_rows:
        _number(row, "Average Steps", minimum=0.0)
    for row in robustness_rows:
        _noise_fraction(row)
    _load_confusion(detector_metrics)
    traces = _load_traces(trace_json)
    backends = {_backend(baseline_rows), _backend(robustness_rows), _backend(ablation_rows)}
    profiles = {
        _profile(baseline_rows),
        _profile(robustness_rows),
        _profile(ablation_rows),
    }
    trace_backend = _backend(
        [
            {"backend": info["backend"]}
            for trace in traces
            for info in trace["info"]
            if isinstance(info, Mapping) and info.get("backend")
        ]
    )
    backends.add(trace_backend)
    if len(backends) != 1:
        raise ResultValidationError(
            f"paper inputs mix evaluation backends: {sorted(backends)}"
        )
    trace_profile = str(traces[0].get("profile", "unspecified")).lower()
    profiles.add(trace_profile)
    if len(profiles) != 1:
        raise ResultValidationError(
            f"paper inputs mix evaluation profiles: {sorted(profiles)}"
        )
    backend = next(iter(backends))
    profile = next(iter(profiles))
    if backend == "metaworld" and profile == "full":
        expected_baselines = {
            "ACT",
            "ACT + Random Reset",
            "ACT + Heuristic Recovery",
            "REIM (ACT + Detector + Recovery)",
        }
        if {row["Method"] for row in baseline_rows} != expected_baselines or any(
            int(float(row.get("Episodes", 0))) != 1000 for row in baseline_rows
        ):
            raise ResultValidationError(
                "full baseline assets require four canonical 1,000-episode methods"
            )
        expected_robustness = {
            (method, level)
            for method in ("ACT", "REIM (ACT + Detector + Recovery)")
            for level in (0.0, 0.1, 0.2, 0.3, 0.4)
        }
        actual_robustness = {
            (str(row["Method"]), _noise_fraction(row)) for row in robustness_rows
        }
        if actual_robustness != expected_robustness or any(
            int(float(row.get("Episodes", 0))) != 200 for row in robustness_rows
        ):
            raise ResultValidationError(
                "full robustness assets require two methods at five "
                "200-episode noise levels"
            )
        if len(ablation_rows) != 4 or any(
            int(float(row.get("Episodes", 0))) != 1000 for row in ablation_rows
        ):
            raise ResultValidationError(
                "full ablation assets require four 1,000-episode variants"
            )
    _validate_publication_gate(
        baseline_rows=baseline_rows,
        robustness_rows=robustness_rows,
        ablation_rows=ablation_rows,
        detector_metrics=detector_metrics,
    )

    figures = Path(figures_dir)
    assets = Path(paper_assets_dir)
    figures.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    framework = plot_framework(figures / "framework_architecture.png")
    outputs["framework"] = str(framework)
    paper_framework = _copy_figure_pair(
        framework, assets / "Figure1_final_framework.png"
    )
    legacy_framework = _copy_figure_pair(
        framework, assets / "Figure1_framework.png"
    )
    outputs["paper_figure_1"] = str(paper_framework)
    outputs["legacy_figure_1"] = str(legacy_framework)

    outputs["success"] = str(
        plot_success_comparison(baseline_csv, figures / "success_comparison.png")
    )
    outputs["robustness"] = str(
        plot_robustness(robustness_csv, figures / "robustness.png")
    )
    outputs["ablation"] = str(
        plot_ablation(ablation_csv, figures / "ablation.png")
    )
    outputs["confusion"] = str(
        plot_confusion_matrix(detector_metrics, figures / "confusion_matrix.png")
    )
    outputs["recovery_examples"] = str(
        plot_recovery_examples(
            trace_json,
            figures / "recovery_examples.png",
            provenance_rows=baseline_rows,
        )
    )
    for key, filename in (
        ("confusion", "Figure3_detector.png"),
        ("recovery_examples", "Figure4_recovery.png"),
        ("ablation", "Figure5_ablation.png"),
    ):
        destination = _copy_figure_pair(outputs[key], assets / filename)
        outputs[f"paper_{key}"] = str(destination)
    final_result = plot_result_composite(
        baseline_csv, robustness_csv, assets / "Figure2_final_results.png"
    )
    outputs["paper_figure_2"] = str(final_result)
    outputs["legacy_figure_2"] = str(
        _copy_figure_pair(final_result, assets / "Figure2_result.png")
    )
    outputs["paper_figure_3"] = str(
        plot_ablation(ablation_csv, assets / "Figure3_final_ablation.png")
    )
    outputs["paper_table_1"] = str(
        export_baseline_latex(baseline_csv, assets / "Table1_baseline.tex")
    )
    outputs["paper_table_2"] = str(
        export_ablation_latex(ablation_csv, assets / "Table2_ablation.tex")
    )

    sources = [
        Path(baseline_csv),
        Path(robustness_csv),
        Path(ablation_csv),
        Path(detector_metrics),
        Path(trace_json),
    ]
    benchmark_label, benchmark_eligible = _provenance(baseline_rows)
    del benchmark_label
    output_records = {
        key: {
            "path": _repo_rel(value),
            "sha256": _sha256(Path(value)),
            "bytes": Path(value).stat().st_size,
        }
        for key, value in outputs.items()
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "profile": profile,
        "benchmark_eligible": benchmark_eligible,
        "warning": (
            ""
            if benchmark_eligible
            else "Only full-profile Meta-World outputs are benchmark results."
        ),
        "inputs": {
            _repo_rel(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sources
        },
        "outputs": output_records,
        "figure_design": {
            "paper_type": "embodied robot method paper",
            "structure": "closed-loop feedback",
            "style": "flat vector, white background, functional three-color palette",
        },
    }
    manifest_path = assets / "assets_manifest.json"
    _atomic_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    outputs["manifest"] = str(manifest_path)
    return outputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "baseline.csv",
    )
    parser.add_argument(
        "--robustness",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "robustness.csv",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "ablation.csv",
    )
    parser.add_argument(
        "--detector-metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "detector_metrics.json",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=PROJECT_ROOT / "results" / "recovery_traces.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument(
        "--paper-assets-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_assets",
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "framework",
            "success",
            "robustness",
            "ablation",
            "confusion",
            "recovery_examples",
            "paper",
        ),
        default="all",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, str]:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.paper_assets_dir.mkdir(parents=True, exist_ok=True)
    if args.only == "all":
        outputs = generate_all(
            baseline_csv=args.baseline,
            robustness_csv=args.robustness,
            ablation_csv=args.ablation,
            detector_metrics=args.detector_metrics,
            trace_json=args.traces,
            figures_dir=args.figures_dir,
            paper_assets_dir=args.paper_assets_dir,
        )
    elif args.only == "framework":
        outputs = {
            "framework": str(
                plot_framework(args.figures_dir / "framework_architecture.png")
            )
        }
    elif args.only == "success":
        outputs = {
            "success": str(
                plot_success_comparison(
                    args.baseline, args.figures_dir / "success_comparison.png"
                )
            )
        }
    elif args.only == "robustness":
        outputs = {
            "robustness": str(
                plot_robustness(
                    args.robustness, args.figures_dir / "robustness.png"
                )
            )
        }
    elif args.only == "ablation":
        outputs = {
            "ablation": str(
                plot_ablation(args.ablation, args.figures_dir / "ablation.png")
            )
        }
    elif args.only == "confusion":
        outputs = {
            "confusion": str(
                plot_confusion_matrix(
                    args.detector_metrics, args.figures_dir / "confusion_matrix.png"
                )
            )
        }
    elif args.only == "recovery_examples":
        outputs = {
            "recovery_examples": str(
                plot_recovery_examples(
                    args.traces, args.figures_dir / "recovery_examples.png"
                )
            )
        }
    else:  # paper
        baseline_rows = _read_csv(
            args.baseline,
            ("Method", "Success Rate", "Recovery Rate", "Average Steps"),
        )
        robustness_rows = _read_csv(
            args.robustness, ("Method", "Success Rate")
        )
        ablation_rows = _read_csv(
            args.ablation, ("Method", "Success Rate", "Average Steps")
        )
        _validate_publication_gate(
            baseline_rows=baseline_rows,
            robustness_rows=robustness_rows,
            ablation_rows=ablation_rows,
            detector_metrics=args.detector_metrics,
        )
        figure_1 = plot_framework(
            args.paper_assets_dir / "Figure1_final_framework.png"
        )
        legacy_figure_1 = _copy_figure_pair(
            figure_1, args.paper_assets_dir / "Figure1_framework.png"
        )
        figure_2 = plot_result_composite(
            args.baseline,
            args.robustness,
            args.paper_assets_dir / "Figure2_final_results.png",
        )
        legacy_figure_2 = _copy_figure_pair(
            figure_2, args.paper_assets_dir / "Figure2_result.png"
        )
        figure_3 = plot_ablation(
            args.ablation,
            args.paper_assets_dir / "Figure3_final_ablation.png",
        )
        table_1 = export_baseline_latex(
            args.baseline, args.paper_assets_dir / "Table1_baseline.tex"
        )
        table_2 = export_ablation_latex(
            args.ablation, args.paper_assets_dir / "Table2_ablation.tex"
        )
        outputs = {
            "paper_figure_1": str(figure_1),
            "legacy_figure_1": str(legacy_figure_1),
            "paper_figure_2": str(figure_2),
            "legacy_figure_2": str(legacy_figure_2),
            "paper_figure_3": str(figure_3),
            "paper_table_1": str(table_1),
            "paper_table_2": str(table_2),
        }
    LOGGER.info("Generated assets:\n%s", json.dumps(outputs, indent=2))
    return outputs


if __name__ == "__main__":
    main()
