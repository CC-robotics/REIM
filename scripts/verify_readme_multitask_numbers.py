"""Cross-check the README MT10/MT50 results section against source files.

Verifies every number in the README '### MT10/MT50 breadth results' section
against paper_assets/multitask_results.tex (clean table, incl. intervention
rates) and results/tables/*.json (robustness table, occupancy sentence).
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

# 3) occupancy sentence vs burden summary
burden = json.loads(
    (ROOT / "results" / "tables" / "intervention_burden_summary.json").read_text(encoding="utf-8"))
noise40 = burden["benchmarks"]["MT10"]["robustness_noise_40"]["burden"]
for who, expect in [("reim", 69.0), ("heuristic_recovery", 47.9)]:
    v = noise40[who]["recovery_occupancy_mean"] * 100
    if abs(v - expect) > 0.05 or f"{expect:.1f}%" not in section:
        errors.append(f"occupancy {who}: json={v:.2f}% vs README={expect:.1f}%")

if errors:
    print("MISMATCHES:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("ALL README MT10/MT50 NUMBERS VERIFIED against tex + results/tables")
