"""Disagreement-state success stats per horizon + h0/h50 recovery-mismatch check.

Per the review (section 3.3), computed from the floor-0.65 closed-loop
episodes (bank 20265010, 20 ep/task, seed 42), no new simulation:

1. Disagreement-state success split, per horizon {0,10,25,50} x condition
   {clean, 0.1, 0.4}: episodes are split by whether the detector fired
   (``intervention_count > 0`` = detector declared disagreement with the ACT
   action stream) and success is reported per branch.

2. End-to-end recovery-mismatch validation for h0 and h50: each REIM episode
   is paired (``paired_episode_id``) with the same-seed MT-ACT episode from
   the canonical 50 ep/task runs, giving a 2x2 of detector-fired x ACT-alone
   outcome:
   - fired & ACT succeeds  -> unnecessary intervention (over-trigger)
   - fired & ACT fails     -> justified intervention
   - silent & ACT succeeds -> correctly silent
   - silent & ACT fails    -> missed failure (under-trigger)

Output: results/tables/mt10_disagreement_horizon_stats.json (LF).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "mt10_disagreement_horizon_stats.json"

CONDITIONS = {
    "official_clean": "mt10_clean_episodes.csv",
    "robustness_noise_10": "mt10_disturbed_noise_10_episodes.csv",
    "robustness_noise_40": "mt10_disturbed_noise_40_episodes.csv",
}
HORIZONS = [0, 10, 25, 50]
MISMATCH_HORIZONS = [0, 50]


def load_reim(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["method"] == "MT-REIM"]


def load_act_outcomes(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as f:
        return {
            r["paired_episode_id"]: int(r["success"])
            for r in csv.DictReader(f)
            if r["method"] == "MT-ACT"
        }


def branch_stats(rows: list[dict]) -> dict:
    fired = [r for r in rows if int(r["intervention_count"]) > 0]
    silent = [r for r in rows if int(r["intervention_count"]) == 0]

    def sr(rs: list[dict]) -> float | None:
        return round(sum(int(r["success"]) for r in rs) / len(rs), 4) if rs else None

    return {
        "episodes": len(rows),
        "success_overall": sr(rows),
        "detector_fired": {"episodes": len(fired), "success": sr(fired)},
        "detector_silent": {"episodes": len(silent), "success": sr(silent)},
    }


def mismatch(rows: list[dict], act: dict[str, int]) -> dict:
    cells = {
        "fired_act_success": 0,
        "fired_act_fail": 0,
        "silent_act_success": 0,
        "silent_act_fail": 0,
    }
    paired = 0
    for r in rows:
        pid = r["paired_episode_id"]
        if pid not in act:
            continue
        paired += 1
        fired = int(r["intervention_count"]) > 0
        key = ("fired" if fired else "silent") + (
            "_act_success" if act[pid] == 1 else "_act_fail"
        )
        cells[key] += 1
    n = paired or 1
    return {
        "paired_episodes": paired,
        "cells": cells,
        "unnecessary_intervention_rate": round(cells["fired_act_success"] / n, 4),
        "justified_intervention_rate": round(cells["fired_act_fail"] / n, 4),
        "missed_failure_rate": round(cells["silent_act_fail"] / n, 4),
        "correctly_silent_rate": round(cells["silent_act_success"] / n, 4),
    }


def main() -> None:
    out = {
        "schema_version": "reim-disagreement-horizon-stats-v1",
        "definitions": {
            "disagreement_state": (
                "detector fired (intervention_count > 0): the detector declared "
                "the ACT action stream risky and recovery took over"
            ),
            "mismatch_cells": (
                "REIM episode paired with same-seed MT-ACT episode via "
                "paired_episode_id; mismatch = unnecessary intervention "
                "(fired & ACT succeeds) or missed failure (silent & ACT fails)"
            ),
        },
        "protocol": {
            "benchmark": "MT10",
            "benchmark_seed": 20265010,
            "episodes_per_task": 20,
            "seed": 42,
            "precision_floor": 0.65,
            "thresholds": {0: 0.79, 10: 0.78, 25: 0.73, 50: 0.71},
        },
        "disagreement_success_by_horizon": {},
        "recovery_mismatch_h0_h50": {},
    }

    for h in HORIZONS:
        per_cond = {}
        for cond in CONDITIONS:
            rows = load_reim(
                TABLES / f"mt10_horizon{h}_floor065_{cond}_episodes.csv"
            )
            per_cond[cond] = branch_stats(rows)
        out["disagreement_success_by_horizon"][f"horizon_{h}"] = per_cond

    for h in MISMATCH_HORIZONS:
        per_cond = {}
        for cond, act_file in CONDITIONS.items():
            rows = load_reim(
                TABLES / f"mt10_horizon{h}_floor065_{cond}_episodes.csv"
            )
            act = load_act_outcomes(TABLES / act_file)
            per_cond[cond] = mismatch(rows, act)
        out["recovery_mismatch_h0_h50"][f"horizon_{h}"] = per_cond

    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")

    print("\n== disagreement-state success (fired vs silent) ==")
    for h in HORIZONS:
        for cond in CONDITIONS:
            s = out["disagreement_success_by_horizon"][f"horizon_{h}"][cond]
            f_, s_ = s["detector_fired"], s["detector_silent"]
            print(
                f"h{h:>2} {cond:>22}: fired {f_['episodes']:>3} ep "
                f"succ={f_['success']} | silent {s_['episodes']:>3} ep succ={s_['success']}"
            )
    print("\n== h0/h50 recovery mismatch (vs paired ACT) ==")
    for h in MISMATCH_HORIZONS:
        for cond in CONDITIONS:
            m = out["recovery_mismatch_h0_h50"][f"horizon_{h}"][cond]
            print(
                f"h{h:>2} {cond:>22}: unnecessary={m['unnecessary_intervention_rate']:.3f} "
                f"justified={m['justified_intervention_rate']:.3f} "
                f"missed={m['missed_failure_rate']:.3f} "
                f"correct_silent={m['correctly_silent_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
