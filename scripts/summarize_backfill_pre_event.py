"""Build the per-horizon pre-event / lead-time summary for the backfill ablation."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "mt10_backfill_pre_event_summary.csv"

HORIZONS = [
    (0, "mt10_horizon0", "mt10_horizon0"),
    (10, "mt10_horizon10", "mt10_horizon10"),
    (25, "mt10_horizon25", "mt10"),  # canonical threshold file has no suffix
    (50, "mt10_horizon50", "mt10_horizon50"),
]

FIELDS = [
    "terminal_positive_horizon",
    "selected_threshold",
    "val_task_macro_precision",
    "val_task_macro_recall",
    "val_task_macro_f1",
    "val_task_macro_auprc",
    "strict_pre_event_precision",
    "strict_pre_event_recall",
    "strict_pre_event_f1",
    "current_event_window_recall",
    "event_trajectories",
    "strict_pre_event_detection_rate",
    "lead_time_mean_steps",
    "lead_time_median_steps",
    "non_event_trajectory_alert_rate",
]

rows = []
for horizon, audit_stem, threshold_stem in HORIZONS:
    audit = json.loads(
        (ROOT / "results" / "tables" / f"{audit_stem}_pre_event_audit.json").read_text(encoding="utf-8")
    )
    threshold_json = json.loads(
        (ROOT / "results" / "tables" / f"{threshold_stem}_detector_threshold.json").read_text(encoding="utf-8")
    )
    sel = threshold_json["selection"]
    sp = audit["strict_pre_event_window_metrics"]
    tm = audit["trajectory_metrics"]
    lead = tm["latest_pre_event_alert_lead_steps"]
    rows.append({
        "terminal_positive_horizon": horizon,
        "selected_threshold": audit["threshold"],
        "val_task_macro_precision": sel["task_macro_precision"],
        "val_task_macro_recall": sel.get("task_macro_recall"),
        "val_task_macro_f1": sel.get("task_macro_f1"),
        "val_task_macro_auprc": sel.get("task_macro_auprc"),
        "strict_pre_event_precision": sp["precision"],
        "strict_pre_event_recall": sp["recall"],
        "strict_pre_event_f1": sp["f1"],
        "current_event_window_recall": audit["current_event_window_recall"],
        "event_trajectories": tm["event_trajectories"],
        "strict_pre_event_detection_rate": tm["strict_pre_event_detection_rate"],
        "lead_time_mean_steps": lead["mean"],
        "lead_time_median_steps": lead["median"],
        "non_event_trajectory_alert_rate": tm["non_event_trajectory_alert_rate"],
    })

tmp = OUT.with_suffix(".csv.tmp")
with tmp.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
tmp.replace(OUT)
print(f"wrote {OUT} ({len(rows)} horizons)")
