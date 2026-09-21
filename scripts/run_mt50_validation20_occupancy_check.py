"""Run the MT50 20-episode/task validation occupancy check in shards."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OUT = ROOT / "results" / "diagnostics" / "occupancy_match_20260921"
LOG = ROOT / "results" / "logs" / "occupancy_match_20260921"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    stem = "mt50_validation20_tol002"
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    workers = 4
    for shard in range(workers):
        task_ids = list(range(shard, 50, workers))
        csv_path = OUT / f"{stem}_shard{shard}_episodes.csv"
        summary_path = OUT / f"{stem}_shard{shard}_summary.json"
        log_path = LOG / f"{stem}_shard{shard}.log"
        argv = [
            str(PYTHON), "evaluation/evaluate_multitask.py",
            "--benchmark", "MT50", "--condition", "robustness_noise_40",
            "--benchmark-seed", "20264050",
            "--act-checkpoint", str(ROOT / "checkpoints/mt50/seed_42/act.pt"),
            "--detector-checkpoint", str(ROOT / "checkpoints/mt50/seed_42/failure_detector.pt"),
            "--recovery-checkpoint", str(ROOT / "checkpoints/mt50/seed_42/recovery.pt"),
            "--mlp-checkpoint", str(ROOT / "checkpoints/mt50/seed_42/mlp_bc.pt"),
            "--output-csv", str(csv_path), "--output-summary", str(summary_path),
            "--methods", "heuristic_recovery", "reim",
            "--episodes-per-task", "20", "--max-steps", "500",
            "--noise-level", "0.4", "--action-std-scale", "0.4",
            "--observation-std-scale", "0.025", "--threshold", "0.64",
            "--release-threshold", "0.05", "--release-patience", "10",
            "--min-recovery-steps", "5", "--intervention-cooldown", "10",
            "--recovery-budget", "250", "--heuristic-min-steps", "30",
            "--heuristic-window", "20", "--heuristic-tolerance", "0.02",
            "--task-ids", *(str(value) for value in task_ids),
            "--seed", "42", "--device", "cuda", "--log-file", str(log_path),
            "--resume", "--allow-partial",
        ]
        console = log_path.with_suffix(".console.log").open("ab")
        process = subprocess.Popen(argv, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        processes.append((shard, process, console))
        print(f"started shard {shard}: tasks={task_ids}")

    failures: list[tuple[int, int]] = []
    for shard, process, console in processes:
        code = process.wait()
        console.close()
        print(f"finished shard {shard}: return_code={code}")
        if code != 0:
            failures.append((shard, code))
    if failures:
        raise SystemExit(f"shards failed: {failures}")

    merged_csv = OUT / f"{stem}_episodes.csv"
    merged_summary = OUT / f"{stem}_summary.json"
    merge = [str(PYTHON), "scripts/merge_multitask_evaluation_shards.py"]
    for shard in range(workers):
        merge.extend(("--shard", str(OUT / f"{stem}_shard{shard}_summary.json"),
                      str(OUT / f"{stem}_shard{shard}_episodes.csv")))
    merge.extend(("--output-csv", str(merged_csv), "--output-summary", str(merged_summary),
                  "--overwrite", "--allow-method-subset"))
    subprocess.run(merge, cwd=ROOT, check=True)
    print(f"merged: {merged_csv}")


if __name__ == "__main__":
    main()
