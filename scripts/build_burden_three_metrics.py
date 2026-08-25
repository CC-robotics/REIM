"""Compute the three intervention-burden metrics required by the review.

Reads the official-bank (seed 20265010) MT10 episode CSVs and computes, per
condition and method:

1. mean per-episode occupancy   = mean(recovery_steps_total / steps)
2. pooled control share         = sum(recovery_steps_total) / sum(steps)
3. absolute burden              = interventions per episode (mean/median) and
   mean segment length (recovery steps / interventions, the on/off cycle size)

Also verifies the "denominator effect" numbers quoted in the review
(REIM ~157 vs heuristic ~167 recovery steps per episode at noise 0.4) as
mean recovery_steps_total per episode.

Output: results/tables/intervention_burden_three_metrics.json (LF).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "intervention_burden_three_metrics.json"

CONDITIONS = {
    "official_clean": "mt10_clean_episodes.csv",
    "robustness_noise_10": "mt10_disturbed_noise_10_episodes.csv",
    "robustness_noise_20": "mt10_disturbed_noise_20_episodes.csv",
    "robustness_noise_30": "mt10_disturbed_noise_30_episodes.csv",
    "robustness_noise_40": "mt10_disturbed_noise_40_episodes.csv",
}

METHODS = {
    "MT-REIM": "reim",
    "MT-ACT + Heuristic-Gated Learned Recovery": "heuristic_recovery",
    "MT-ACT": "act",
}


def burden(rows: list[dict]) -> dict:
    n = len(rows)
    steps = [int(r["steps"]) for r in rows]
    rec = [int(r["recovery_steps_total"]) for r in rows]
    inter = [int(r["intervention_count"]) for r in rows]
    occ = [rv / st if st else 0.0 for rv, st in zip(rec, steps)]
    total_inter = sum(inter)
    return {
        "episodes": n,
        "mean_per_episode_occupancy": sum(occ) / n,
        "pooled_control_share": sum(rec) / sum(steps),
        "absolute_burden": {
            "interventions_per_episode_mean": total_inter / n,
            "interventions_per_episode_median": sorted(inter)[n // 2]
            if n % 2
            else (sorted(inter)[n // 2 - 1] + sorted(inter)[n // 2]) / 2,
            "max_interventions_in_episode": max(inter),
            "mean_segment_length": (sum(rec) / total_inter) if total_inter else 0.0,
            "mean_recovery_steps_per_episode": sum(rec) / n,
            "mean_steps_per_episode": sum(steps) / n,
        },
    }


def main() -> None:
    out = {
        "schema_version": "reim-intervention-burden-three-metrics-v1",
        "source": (
            "official evaluation bank seed 20265010 episode CSVs (seed 42, "
            "500 episodes per method per condition), no re-run"
        ),
        "metric_definitions": {
            "mean_per_episode_occupancy": "mean over episodes of recovery_steps_total / steps",
            "pooled_control_share": "sum(recovery_steps_total) / sum(steps) over all episodes",
            "absolute_burden": (
                "interventions per episode (mean/median/max), mean segment "
                "length = total recovery steps / total interventions (the "
                "on/off cycle size), and mean recovery steps per episode"
            ),
        },
        "conditions": {},
        "denominator_effect_check": {},
    }

    for cond, fname in CONDITIONS.items():
        path = TABLES / fname
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        by_method: dict[str, list[dict]] = {}
        for r in rows:
            label = METHODS.get(r["method"])
            if label:
                by_method.setdefault(label, []).append(r)
        out["conditions"][cond] = {
            label: burden(mrows) for label, mrows in sorted(by_method.items())
        }

    n40 = out["conditions"]["robustness_noise_40"]
    out["denominator_effect_check"] = {
        "noise_0.4_mean_recovery_steps_per_episode": {
            "reim": n40["reim"]["absolute_burden"]["mean_recovery_steps_per_episode"],
            "heuristic_recovery": n40["heuristic_recovery"]["absolute_burden"][
                "mean_recovery_steps_per_episode"
            ],
        },
        "review_quoted_values": {"reim": 157, "heuristic": 167},
    }

    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    for cond, blk in out["conditions"].items():
        line = [cond]
        for label in ("reim", "heuristic_recovery"):
            b = blk[label]
            line.append(
                f"{label}: occ={b['mean_per_episode_occupancy']:.3f} "
                f"pooled={b['pooled_control_share']:.3f} "
                f"int/ep={b['absolute_burden']['interventions_per_episode_mean']:.2f} "
                f"rec_steps/ep={b['absolute_burden']['mean_recovery_steps_per_episode']:.1f}"
            )
        print("\n  ".join(line))
    chk = out["denominator_effect_check"]["noise_0.4_mean_recovery_steps_per_episode"]
    print(
        f"denominator effect: reim={chk['reim']:.1f} (quoted 157), "
        f"heuristic={chk['heuristic_recovery']:.1f} (quoted 167)"
    )


if __name__ == "__main__":
    main()
