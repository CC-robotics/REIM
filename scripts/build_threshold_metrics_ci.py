"""Clustered (by-task) bootstrap CIs for the threshold-selection metrics.

Per the review's P0 standard ("阈值相关的聚类置信区间"), the tuning metrics
reported for each horizon's selected threshold get percentile bootstrap CIs
that respect the task cluster structure: the 10 tasks are resampled with
replacement (B=10000), the task-macro metric is recomputed on each resample,
and the 2.5%/97.5% percentiles form the 95% CI.

Covers both precision-floor calibers (0.60 tuned selections and the 0.65
probe) for horizons {0, 10, 25, 50}.

Output: results/tables/mt10_threshold_metrics_clustered_bootstrap_ci.json
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "mt10_threshold_metrics_clustered_bootstrap_ci.json"

SOURCES = {
    ("0.60", 0): "mt10_horizon0_detector_threshold.json",
    ("0.60", 10): "mt10_horizon10_detector_threshold.json",
    ("0.60", 25): "mt10_detector_threshold.json",
    ("0.60", 50): "mt10_horizon50_detector_threshold.json",
    ("0.65", 0): "mt10_horizon0_detector_threshold_floor065.json",
    ("0.65", 10): "mt10_horizon10_detector_threshold_floor065.json",
    ("0.65", 25): "mt10_horizon25_detector_threshold_floor065.json",
    ("0.65", 50): "mt10_horizon50_detector_threshold_floor065.json",
}

METRICS = ["precision", "recall", "f1"]
B = 10000
SEED = 20260825


def bootstrap_ci(values: list[float], rng: random.Random) -> dict:
    n = len(values)
    point = sum(values) / n
    boots = []
    for _ in range(B):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return {
        "point_estimate": round(point, 4),
        "ci95_lower": round(boots[int(0.025 * B)], 4),
        "ci95_upper": round(boots[int(0.975 * B) - 1], 4),
    }


def main() -> None:
    rng = random.Random(SEED)
    out = {
        "schema_version": "reim-threshold-metrics-ci-v1",
        "method": (
            "percentile bootstrap clustered by task: the 10 MT10 tasks are "
            "resampled with replacement (B=10000, seed %d); the task-macro "
            "metric is recomputed per resample. Episodes within a task are "
            "treated as a single cluster, matching the review requirement "
            "that intra-task episodes are correlated." % SEED
        ),
        "source": "validation-bank per-task metrics at each selected threshold (no re-run)",
        "results": [],
    }
    for (floor, horizon), fname in sorted(SOURCES.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        d = json.loads((TABLES / fname).read_text(encoding="utf-8"))
        per_task = d["dataset"]["per_task"]
        sel = d["selection"]
        entry = {
            "precision_floor": float(floor),
            "horizon": horizon,
            "threshold": sel["threshold"],
            "task_count": len(per_task),
            "metrics": {
                m: bootstrap_ci([float(t[m]) for t in per_task], rng) for m in METRICS
            },
        }
        out["results"].append(entry)
        p, r, f1 = (entry["metrics"][m] for m in METRICS)
        print(
            f"floor={floor} h{horizon:>2} thr={sel['threshold']:.2f} | "
            f"prec {p['point_estimate']:.4f} [{p['ci95_lower']:.4f}, {p['ci95_upper']:.4f}] | "
            f"recall {r['point_estimate']:.4f} [{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}] | "
            f"F1 {f1['point_estimate']:.4f} [{f1['ci95_lower']:.4f}, {f1['ci95_upper']:.4f}]"
        )

    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
