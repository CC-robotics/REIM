"""Cross-check the README MT10/MT50 results section against source files.

Verifies every number in the README '### MT10/MT50 breadth results' section
against paper_assets/multitask_results.tex (clean table, incl. intervention
rates), results/tables/*.json (robustness table, occupancy table),
results/tables/mt10_matched_occupancy_comparison.json +
results/diagnostics/release_patience_search/mt10_references.csv
(matched-occupancy control), and
results/tables/mt10_backfill_pre_event_summary.csv (backfill audit table).
Exits 0 iff all numbers match.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

readme = (ROOT / "README.md").read_text(encoding="utf-8")
start = readme.find("### MT10/MT50 breadth results")
end = readme.find("## Paper assets")
section = readme[start:end]
assert start > 0 and end > start, "README section not found"

tex = (ROOT / "paper_assets" / "multitask_results.tex").read_text(encoding="utf-8")


def macro(name: str) -> float:
    m = re.search(r"\\renewcommand\{\\" + name + r"\}\{([0-9.]+)", tex)
    assert m, f"macro {name} not found"
    return float(m.group(1))


errors = []

# 1) clean table (values + intervention rates come from the tex macros)
for bench, p in [("MT10", "MTTen"), ("MT50", "MTFifty")]:
    row = next(l for l in section.splitlines() if l.startswith("| " + bench))
    for label, mac in [
        ("mlp", "MLPCleanSuccess"), ("act", "ACTCleanSuccess"),
        ("heur", "HeuristicCleanSuccess"), ("reim", "REIMCleanSuccess"),
        ("heur_int", "HeuristicIntervention"), ("reim_int", "REIMIntervention"),
    ]:
        v = macro(p + mac)
        if f"{v:.1f}%" not in row:
            errors.append(f"clean {bench} {label}: tex={v:.1f}% not in README row: {row}")

# 2) robustness table vs summary JSONs
conds = [("0.1", "disturbed_noise_10"), ("0.2", "disturbed_noise_20"),
         ("0.3", "disturbed_noise_30"), ("0.4", "disturbed_noise_40")]
for label, cond in conds:
    row = next(l for l in section.splitlines() if l.startswith("| " + label))
    for bench in ["mt10", "mt50"]:
        data = json.loads(
            (ROOT / "results" / "tables" / f"{bench}_{cond}_summary.json").read_text(encoding="utf-8"))
        for method in ["act", "heuristic_recovery", "reim"]:
            v = data["aggregates"][method]["summary"]["success_rate_task_macro"] * 100
            if f"{v:.1f}%" not in row:
                errors.append(f"noise {label} {bench} {method}: json={v:.1f}% not in README row: {row}")

# 3) occupancy table vs burden summary (first-class occupancy reporting)
burden = json.loads(
    (ROOT / "results" / "tables" / "intervention_burden_summary.json").read_text(encoding="utf-8"))
occ_conds = [("clean", "official_clean"), ("0.1", "robustness_noise_10"),
             ("0.2", "robustness_noise_20"), ("0.3", "robustness_noise_30"),
             ("0.4", "robustness_noise_40")]
for label, cond in occ_conds:
    vals = []
    for bench in ["MT10", "MT50"]:
        b = burden["benchmarks"][bench][cond]["burden"]
        for method in ["reim", "heuristic_recovery"]:
            vals.append(f"{b[method]['recovery_occupancy_mean'] * 100:.1f}%")
    row = next((l for l in section.splitlines()
                if l.startswith(f"| {label} |") and all(v in l for v in vals)), None)
    if row is None:
        errors.append(f"occupancy row {label}: expected values {vals} not found together")

# 3b) legacy occupancy spot-check (MT10 noise 0.4: REIM 69.0%, heuristic 47.9%)
noise40 = burden["benchmarks"]["MT10"]["robustness_noise_40"]["burden"]
for who, expect in [("reim", 69.0), ("heuristic_recovery", 47.9)]:
    v = noise40[who]["recovery_occupancy_mean"] * 100
    if abs(v - expect) > 0.05 or f"{expect:.1f}%" not in section:
        errors.append(f"occupancy {who}: json={v:.2f}% vs README={expect:.1f}%")

# 4) matched-occupancy comparison vs machine-readable record
import csv
comp = json.loads(
    (ROOT / "results" / "tables" / "mt10_matched_occupancy_comparison.json").read_text(encoding="utf-8"))
refs = {(r["condition"], r["method"]): r for r in csv.DictReader(
    open(ROOT / "results" / "diagnostics" / "release_patience_search" / "mt10_references.csv",
         encoding="utf-8"))}
for c in comp["comparisons"]:
    nl = c["noise_level"]
    for v in [c["heuristic"]["recovery_occupancy"] * 100,
              c["heuristic"]["success"] * 100,
              c["nearest_grid_point"]["recovery_occupancy"] * 100,
              c["nearest_grid_point"]["micro_success"] * 100]:
        if f"{v:.1f}%" not in section:
            errors.append(f"matched-occupancy noise {nl}: {v:.1f}% not in README section")
    # cross-check heuristic occupancy against the re-run reference CSV
    ref = refs[(f"robustness_noise_{int(round(nl * 100))}", "heuristic_recovery")]
    if abs(float(ref["recovery_occupancy_mean"]) - c["heuristic"]["recovery_occupancy"]) > 1e-6:
        errors.append(f"matched-occupancy noise {nl}: comparison JSON disagrees with references CSV")

# 5) backfill pre-event table vs summary CSV
backfill = list(csv.DictReader(
    open(ROOT / "results" / "tables" / "mt10_backfill_pre_event_summary.csv", encoding="utf-8")))
for r in backfill:
    h = r["terminal_positive_horizon"]
    row = next((l for l in section.splitlines() if l.startswith(f"| h{h} |")), None)
    if row is None:
        errors.append(f"backfill h{h}: row missing")
        continue
    checks = [f"{float(r['selected_threshold']):.2f}",
              f"{float(r['strict_pre_event_f1']):.3f}",
              f"{float(r['strict_pre_event_detection_rate']) * 100:.1f}%",
              f"| {int(float(r['lead_time_median_steps']))} |"]
    for chk in checks:
        if chk not in row:
            errors.append(f"backfill h{h}: {chk} not in README row: {row}")

if errors:
    print("MISMATCHES:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("ALL README MT10/MT50 NUMBERS VERIFIED against tex + results/tables")
