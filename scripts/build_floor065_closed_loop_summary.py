"""Summarize the floor-0.65 closed-loop runs alongside the floor-0.60 results.

Reads the 12 floor-0.65 cells (mt10_horizon{0,10,25,50}_floor065_*) and the
existing floor-0.60 tuned closed-loop summary, and writes a combined
dual-caliber table to
``results/tables/mt10_backfill_floor065_closed_loop_summary.csv`` (LF).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "mt10_backfill_floor065_closed_loop_summary.csv"

CONDITIONS = ["official_clean", "robustness_noise_10", "robustness_noise_40"]
HORIZONS = [0, 10, 25, 50]
FLOOR065_THRESHOLDS = {0: 0.79, 10: 0.78, 25: 0.73, 50: 0.71}


def mean_occupancy(csv_path: Path) -> float:
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "MT-REIM"]
    occ = [
        int(r["recovery_steps_total"]) / int(r["steps"]) if int(r["steps"]) else 0.0
        for r in rows
    ]
    return sum(occ) / len(occ)


def main() -> None:
    out_rows = []
    for h in HORIZONS:
        for cond in CONDITIONS:
            tag = f"mt10_horizon{h}_floor065_{cond}"
            summary = json.loads(
                (TABLES / f"{tag}_summary.json").read_text(encoding="utf-8")
            )
            s = summary["aggregates"]["reim"]["summary"]
            out_rows.append(
                {
                    "horizon": h,
                    "precision_floor": 0.65,
                    "threshold": FLOOR065_THRESHOLDS[h],
                    "condition": cond,
                    "success_rate_task_macro": round(s["success_rate_task_macro"], 4),
                    "success_rate_micro": round(s["success_rate_micro"], 4),
                    "recovery_occupancy_mean": round(
                        mean_occupancy(TABLES / f"{tag}_episodes.csv"), 4
                    ),
                    "episodes": s["episode_count"],
                }
            )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {OUT.relative_to(ROOT)}")

    # Dual-caliber console table against the floor-0.60 tuned closed loop.
    noise_of = {"official_clean": "0.0", "robustness_noise_10": "0.1", "robustness_noise_40": "0.4"}
    old: dict[tuple[int, str], float] = {}
    old_csv = TABLES / "mt10_backfill_tuned_closed_loop_summary.csv"
    if old_csv.exists():
        with old_csv.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                v = r.get("success_tuned_20ep")
                if v:
                    old[(int(r["horizon"]), r["noise_level"])] = float(v)
    print("\nhorizon | clean 0.60->0.65 | 0.1 0.60->0.65 | 0.4 0.60->0.65")
    for h in HORIZONS:
        cells = []
        for cond in CONDITIONS:
            new = next(
                r for r in out_rows if r["horizon"] == h and r["condition"] == cond
            )
            prev = old.get((h, noise_of[cond]))
            cells.append(
                f"{new['success_rate_task_macro'] * 100:.1f}%"
                + (f" (was {prev * 100:.1f}%)" if prev is not None else "")
            )
        print(f"h{h}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
