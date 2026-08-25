"""Paired rescued/harmed stats for the matched-occupancy supplementary runs.

Pairs each matched REIM run against the heuristic reference run on the same
search bank (seed 20264010, 20 ep/task, seed 42) via ``paired_episode_id``:

- noise 0.4: REIM release 0.3/patience 5  vs  heuristic noise40 reference
- noise 0.1: REIM release 0.02/patience 3 vs  heuristic noise10 reference

rescued = REIM success & heuristic failure; harmed = REIM failure & heuristic
success.  Also reports task-macro success and mean per-episode occupancy for
both arms.  Prints a summary; no files are written (callers update the
comparison JSON separately).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "diagnostics" / "release_patience_search"

CELLS = [
    (
        "robustness_noise_40",
        "mt10_matched_reim_rel030_pat5_noise40_episodes.csv",
        "mt10_ref_heuristic_noise40_episodes.csv",
    ),
    (
        "robustness_noise_10",
        "mt10_matched_reim_rel002_pat3_noise10_episodes.csv",
        "mt10_ref_heuristic_noise10_episodes.csv",
    ),
]


def load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["paired_episode_id"]] = {
                "success": int(r["success"]),
                "steps": int(r["steps"]),
                "recovery_steps": int(r["recovery_steps_total"]),
                "task_name": r["task_name"],
            }
    return out


def macro_success(rows: dict[str, dict]) -> float:
    by_task: dict[str, list[int]] = {}
    for v in rows.values():
        by_task.setdefault(v["task_name"], []).append(v["success"])
    return sum(sum(s) / len(s) for s in by_task.values()) / len(by_task)


def mean_occupancy(rows: dict[str, dict]) -> float:
    occ = [v["recovery_steps"] / v["steps"] if v["steps"] else 0.0 for v in rows.values()]
    return sum(occ) / len(occ)


def main() -> None:
    for cond, reim_file, heur_file in CELLS:
        reim = load(DIR / reim_file)
        heur = load(DIR / heur_file)
        common = sorted(set(reim) & set(heur))
        rescued = sum(
            1 for k in common if reim[k]["success"] == 1 and heur[k]["success"] == 0
        )
        harmed = sum(
            1 for k in common if reim[k]["success"] == 0 and heur[k]["success"] == 1
        )
        print(f"== {cond}: paired {len(common)} episodes")
        print(
            f"  REIM     macro_success={macro_success(reim):.4f} "
            f"occupancy={mean_occupancy(reim):.4f}"
        )
        print(
            f"  heuristic macro_success={macro_success(heur):.4f} "
            f"occupancy={mean_occupancy(heur):.4f}"
        )
        print(f"  rescued={rescued} harmed={harmed} (REIM vs heuristic)")


if __name__ == "__main__":
    main()
