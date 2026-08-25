# -*- coding: utf-8 -*-
"""Extend mt10_multiseed_summary.csv with sample standard deviations and
paired rescued/harmed counts (REIM vs ACT, Heuristic vs ACT), per the
2026-08-21 review PDF section 3 reporting requirements.

Reads the per-episode CSVs for seeds 42/43/44 at clean / noise 0.1 / 0.4
(bank 20265010, 50 episodes per task) and rewrites
results/tables/mt10_multiseed_summary.csv.
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK = 20265010
EPISODES_PER_TASK = 50
SEEDS = (42, 43, 44)
NOISES = (0.0, 0.1, 0.4)
METHODS = ("reim", "heuristic_recovery")

METHOD_LABELS = {
    "reim": "MT-REIM",
    "heuristic_recovery": "MT-ACT + Heuristic-Gated Learned Recovery",
    "act": "MT-ACT",
}
LABEL_TO_KEY = {v: k for k, v in METHOD_LABELS.items()}

# File naming differs between the canonical seed-42 runs and seed 43/44 runs.
NAME = {
    42: {0.0: "clean", 0.1: "disturbed_noise_10", 0.4: "disturbed_noise_40"},
    43: {0.0: "official_clean", 0.1: "robustness_noise_10", 0.4: "robustness_noise_40"},
    44: {0.0: "official_clean", 0.1: "robustness_noise_10", 0.4: "robustness_noise_40"},
}


def episodes_path(seed: int, noise: float) -> Path:
    prefix = "mt10_" if seed == 42 else f"mt10_seed{seed}_"
    return ROOT / "results" / "tables" / f"{prefix}{NAME[seed][noise]}_episodes.csv"


def load_pairs(seed: int, noise: float) -> dict:
    """paired_episode_id -> {method_key: success(0/1)} for act/reim/heuristic."""
    pairs: dict = {}
    with open(episodes_path(seed, noise), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = LABEL_TO_KEY.get(r["method"])
            if key is None:
                continue
            pairs.setdefault(r["paired_episode_id"], {})[key] = int(r["success"])
    return pairs


def main() -> None:
    out_rows = []
    for noise in NOISES:
        per_seed = {seed: load_pairs(seed, noise) for seed in SEEDS}
        for method in METHODS:
            succ, rescued, harmed = [], [], []
            for seed in SEEDS:
                pairs = per_seed[seed]
                ids = [k for k, v in pairs.items() if method in v and "act" in v]
                m_succ = [pairs[i][method] for i in ids]
                a_succ = [pairs[i]["act"] for i in ids]
                succ.append(sum(m_succ) / len(m_succ))
                rescued.append(sum(1 for m, a in zip(m_succ, a_succ) if m == 1 and a == 0))
                harmed.append(sum(1 for m, a in zip(m_succ, a_succ) if m == 0 and a == 1))
            out_rows.append({
                "noise_level": noise,
                "method": method,
                "seed42": round(succ[0], 4),
                "seed43": round(succ[1], 4),
                "seed44": round(succ[2], 4),
                "mean": round(statistics.fmean(succ), 4),
                "sample_std": round(statistics.stdev(succ), 4),
                "min": round(min(succ), 4),
                "max": round(max(succ), 4),
                "rescued_per_seed": "/".join(str(x) for x in rescued),
                "rescued_mean": round(statistics.fmean(rescued), 2),
                "rescued_sample_std": round(statistics.stdev(rescued), 2),
                "harmed_per_seed": "/".join(str(x) for x in harmed),
                "harmed_mean": round(statistics.fmean(harmed), 2),
                "harmed_sample_std": round(statistics.stdev(harmed), 2),
                "episodes_per_task": EPISODES_PER_TASK,
                "bank": BANK,
            })
    out = ROOT / "results" / "tables" / "mt10_multiseed_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(out, "written")
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
