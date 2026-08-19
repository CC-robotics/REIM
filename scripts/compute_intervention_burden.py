"""Paired intervention-burden metrics for the MT10/MT50 REIM study.

Computes the metrics requested by the pre-submission review (问题 2, P0):
success alongside intervention burden, including CRN-paired rescued/harmed
counts, switch counts, and trigger timing — from the existing official
evaluation CSVs without re-running any rollout.

Not computable from the current CSV schema (deferred to the seed/ablation
re-runs, which will add a per-episode ``recovery_steps_total`` field):
recovery-controlled steps / occupancy and per-segment lengths.

Outputs:
- results/tables/intervention_burden_paired.csv  (per benchmark x condition)
- results/tables/intervention_burden_summary.json (machine-readable + gaps)
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "results" / "tables"

METHOD_KEY = {
    "MT-MLP BC": "mlp_bc",
    "MT-ACT": "act",
    "MT-ACT + Heuristic-Gated Learned Recovery": "heuristic_recovery",
    "MT-REIM": "reim",
}

CONDITIONS = [
    ("official_clean", "clean"),
    ("robustness_noise_00", "disturbed_noise_00"),
    ("robustness_noise_10", "disturbed_noise_10"),
    ("robustness_noise_20", "disturbed_noise_20"),
    ("robustness_noise_30", "disturbed_noise_30"),
    ("robustness_noise_40", "disturbed_noise_40"),
]

GATED_METHODS = ("heuristic_recovery", "reim")


def _load_episodes(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    """paired_episode_id -> method -> row (parsed)."""
    paired: dict[str, dict[str, dict[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = METHOD_KEY[row["method"]]
            paired.setdefault(row["paired_episode_id"], {})[method] = {
                "success": row["success"] == "1",
                "interventions": int(row["intervention_count"]),
                "trigger_step": int(row["trigger_step"]),
                "steps": int(row["steps"]),
                "recovery_success": int(row["recovery_success"]),
            }
    return paired


def _paired_outcomes(
    episodes: dict[str, dict[str, dict[str, object]]],
    treatment: str,
    control: str,
) -> tuple[int, int, int]:
    rescued = harmed = unchanged = 0
    for methods in episodes.values():
        if treatment not in methods or control not in methods:
            continue
        t = bool(methods[treatment]["success"])
        c = bool(methods[control]["success"])
        if t and not c:
            rescued += 1
        elif c and not t:
            harmed += 1
        else:
            unchanged += 1
    return rescued, harmed, unchanged


def _burden(episodes: dict[str, dict[str, dict[str, object]]], method: str) -> dict[str, object]:
    rows = [methods[method] for methods in episodes.values() if method in methods]
    intervened = [r for r in rows if int(r["interventions"]) > 0]
    triggers = [int(r["trigger_step"]) for r in intervened if int(r["trigger_step"]) >= 0]
    counts = [int(r["interventions"]) for r in rows]
    return {
        "episodes": len(rows),
        "intervened_episode_rate": len(intervened) / max(1, len(rows)),
        "interventions_per_episode_mean": statistics.fmean(counts) if counts else 0.0,
        "interventions_per_episode_median": statistics.median(counts) if counts else 0.0,
        "max_interventions_in_episode": max(counts) if counts else 0,
        "trigger_step_median": statistics.median(triggers) if triggers else None,
        "clean_release_rate": (
            sum(int(r["recovery_success"]) for r in intervened)
            / sum(int(r["interventions"]) for r in intervened)
            if intervened and sum(int(r["interventions"]) for r in intervened) > 0
            else None
        ),
    }


def main() -> None:
    table_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema_version": "reim-intervention-burden-v1",
        "source": "official evaluation CSVs (CRN-paired), no re-run",
        "known_gaps": [
            "recovery-controlled steps / occupancy and per-segment lengths "
            "require a new per-episode recovery_steps_total field; deferred to "
            "the seed-study and terminal-backfill-ablation re-runs"
        ],
        "benchmarks": {},
    }
    for benchmark in ("mt10", "mt50"):
        per_condition: dict[str, object] = {}
        for condition, slug in CONDITIONS:
            path = TABLES / f"{benchmark}_{slug}_episodes.csv"
            episodes = _load_episodes(path)
            n_pairs = len(episodes)
            reim_vs_act = _paired_outcomes(episodes, "reim", "act")
            heur_vs_act = _paired_outcomes(episodes, "heuristic_recovery", "act")
            reim_vs_heur = _paired_outcomes(episodes, "reim", "heuristic_recovery")
            burden = {m: _burden(episodes, m) for m in GATED_METHODS}
            per_condition[condition] = {
                "paired_episodes": n_pairs,
                "rescued_harmed_unchanged": {
                    "reim_vs_act": reim_vs_act,
                    "heuristic_vs_act": heur_vs_act,
                    "reim_vs_heuristic": reim_vs_heur,
                },
                "burden": burden,
            }
            for method in GATED_METHODS:
                b = burden[method]
                table_rows.append(
                    {
                        "benchmark": benchmark.upper(),
                        "condition": condition,
                        "method": method,
                        "paired_episodes": n_pairs,
                        "intervened_episode_rate": round(
                            float(b["intervened_episode_rate"]), 4
                        ),
                        "interventions_per_episode_mean": round(
                            float(b["interventions_per_episode_mean"]), 4
                        ),
                        "interventions_per_episode_median": b[
                            "interventions_per_episode_median"
                        ],
                        "max_interventions_in_episode": b[
                            "max_interventions_in_episode"
                        ],
                        "trigger_step_median": b["trigger_step_median"],
                        "clean_release_rate": (
                            round(float(b["clean_release_rate"]), 4)
                            if b["clean_release_rate"] is not None
                            else ""
                        ),
                        "rescued_vs_act": _paired_outcomes(
                            episodes, method, "act"
                        )[0],
                        "harmed_vs_act": _paired_outcomes(
                            episodes, method, "act"
                        )[1],
                    }
                )
            if condition != "official_clean":
                pass
        summary["benchmarks"][benchmark.upper()] = per_condition

    out_csv = TABLES / "intervention_burden_paired.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    out_json = TABLES / "intervention_burden_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")

    # Console digest: reim vs heuristic paired outcomes per condition.
    for benchmark in ("MT10", "MT50"):
        print(f"\n== {benchmark}: reim vs heuristic (rescued/harmed/unchanged) ==")
        for condition, payload in summary["benchmarks"][benchmark].items():
            r, h, u = payload["rescued_harmed_unchanged"]["reim_vs_heuristic"]
            n = payload["paired_episodes"]
            print(
                f"  {condition:<22} rescued={r:>4} ({r / n:5.1%})  "
                f"harmed={h:>4} ({h / n:5.1%})  unchanged={u:>4}"
            )


if __name__ == "__main__":
    main()
