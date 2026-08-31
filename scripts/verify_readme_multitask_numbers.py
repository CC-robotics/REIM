"""Cross-check the README MT10/MT50 results section against source files.

Verifies every number in the README '### MT10/MT50 breadth results' section
against paper_assets/multitask_results.tex (clean table, incl. intervention
rates), results/tables/confirmation_202660xx/*.json (robustness table,
confirmation library banks 20266010/20266050),
paper_assets/multitask_clean_statistics.csv +
paper_assets/multitask_robustness_statistics.csv (occupancy table,
gate-generated from the confirmation library),
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

# 2) robustness table vs confirmation-library summary JSONs
CONF = ROOT / "results" / "tables" / "confirmation_202660xx"
conds = [("0.0", "robustness_noise_00"), ("0.1", "robustness_noise_10"),
         ("0.2", "robustness_noise_20"), ("0.3", "robustness_noise_30"),
         ("0.4", "robustness_noise_40")]
for label, cond in conds:
    row = next(l for l in section.splitlines() if l.startswith("| " + label))
    for bench in ["mt10", "mt50"]:
        data = json.loads(
            (CONF / f"{bench}_confirm_{cond}_summary.json").read_text(encoding="utf-8"))
        for method in ["act", "heuristic_recovery", "reim"]:
            v = data["aggregates"][method]["summary"]["success_rate_task_macro"] * 100
            if f"{v:.1f}%" not in row:
                errors.append(f"noise {label} {bench} {method}: json={v:.1f}% not in README row: {row}")

# 3) occupancy table vs gate-generated confirmation statistics CSVs
import csv

def _occ(cond_label):
    vals = []
    for bench in ["MT10", "MT50"]:
        if cond_label == "clean":
            src = ROOT / "paper_assets" / "multitask_clean_statistics.csv"
            rows = [r for r in csv.DictReader(open(src, encoding="utf-8"))
                    if r["benchmark"] == bench and r["condition"] == "official_clean"]
        else:
            src = ROOT / "paper_assets" / "multitask_robustness_statistics.csv"
            nl = f"{float(cond_label):.6g}"
            rows = [r for r in csv.DictReader(open(src, encoding="utf-8"))
                    if r["benchmark"] == bench
                    and abs(float(r["noise_level"]) - float(cond_label)) < 1e-9]
        by_method = {r["method"]: r for r in rows}
        for method in ["MT-REIM", "MT-ACT + Heuristic-Gated Learned Recovery"]:
            vals.append(f"{float(by_method[method]['recovery_occupancy_mean']) * 100:.1f}%")
    return vals

for label in ["clean", "0.1", "0.2", "0.3", "0.4"]:
    vals = _occ(label)
    row = next((l for l in section.splitlines()
                if l.startswith(f"| {label} |") and all(v in l for v in vals)), None)
    if row is None:
        errors.append(f"occupancy row {label}: expected values {vals} not found together")

# 3b) occupancy spot-check (confirmation MT10 noise 0.4: REIM 69.2%, heuristic 47.7%)
spot = dict(zip(["reim", "heuristic_recovery"], _occ("0.4")[:2]))
for who, expect in [("reim", 69.2), ("heuristic_recovery", 47.7)]:
    v = float(spot[who].rstrip("%"))
    if abs(v - expect) > 0.05 or f"{expect:.1f}%" not in section:
        errors.append(f"occupancy {who}: csv={v:.2f}% vs README={expect:.1f}%")

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

# 6) tuned backfill closed-loop table vs aggregate CSV
tuned = list(csv.DictReader(
    open(ROOT / "results" / "tables" / "mt10_backfill_tuned_closed_loop_summary.csv",
         encoding="utf-8")))
for h in ("0", "10", "50"):
    sub = {float(r["noise_level"]): r for r in tuned if r["horizon"] == h}
    if set(sub) != {0.0, 0.1, 0.4}:
        errors.append(f"tuned closed loop h{h}: incomplete conditions")
        continue
    row = next((l for l in section.splitlines()
                if l.startswith(f"| h{h} |") and "%" in l and "/" in l), None)
    if row is None:
        errors.append(f"tuned closed loop h{h}: row missing")
        continue
    checks = [f"{float(sub[0.0]['tuned_threshold']):.2f}"]
    checks += [f"{float(sub[nl]['success_tuned_20ep']) * 100:.1f}%" for nl in (0.0, 0.1, 0.4)]
    checks += [f"{float(sub[nl]['recovery_occupancy_tuned']) * 100:.1f}%" for nl in (0.0, 0.1, 0.4)]
    for chk in checks:
        if chk not in row:
            errors.append(f"tuned closed loop h{h}: {chk} not in README row: {row}")

# 7) multi-seed paragraph vs aggregate CSV
multiseed = list(csv.DictReader(
    open(ROOT / "results" / "tables" / "mt10_multiseed_summary.csv", encoding="utf-8")))
for r in multiseed:
    for field in ("mean",):
        chk = f"{float(r[field]) * 100:.1f}%"
        if chk not in section:
            errors.append(f"multi-seed {r['method']} noise {r['noise_level']}: "
                          f"{chk} not in README section")
    if r["method"] == "reim":
        rng = (f"{float(r['min']) * 100:.1f}-{float(r['max']) * 100:.1f}%")
        if rng not in section:
            errors.append(f"multi-seed reim noise {r['noise_level']}: range {rng} missing")

if errors:
    print("MISMATCHES:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("ALL README MT10/MT50 NUMBERS VERIFIED against tex + confirmation library + gate CSVs")
