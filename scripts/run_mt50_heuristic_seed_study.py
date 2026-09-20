#!/usr/bin/env python3
"""Evaluate seed-specific MT50 recovery policies under the heuristic gate.

Seeds 43--45 reuse the completed component-seed recovery checkpoints.  The
heuristic trigger itself has no learned seed; the varying component is the
recovery policy.  ACT, task bank, episode seeds, perturbations, and heuristic
parameters are identical to the four-seed REIM study.  Each condition is
split into four disjoint task shards and merged with fail-closed validation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

SEED42 = ROOT / "checkpoints" / "mt50" / "seed_42"
CONDITIONS = (
    ("official_clean", 0.0),
    ("robustness_noise_10", 0.1),
    ("robustness_noise_40", 0.4),
)
FINAL_BANK_SEED = 20266050


def _run(argv: list[str], *, dry_run: bool) -> None:
    print("\n>>> " + " ".join(argv), flush=True)
    if not dry_run:
        subprocess.run(argv, cwd=ROOT, check=True)


def _evaluate(seed: int, workers: int, device: str, *, dry_run: bool) -> None:
    if not 1 <= workers <= 8:
        raise ValueError("workers must lie in [1, 8]")
    detector = ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "failure_detector.pt"
    recovery = ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "recovery.pt"
    for path in (SEED42 / "act.pt", detector, recovery):
        if not path.is_file():
            raise FileNotFoundError(path)

    shards = [list(range(offset, 50, workers)) for offset in range(workers)]
    for condition, noise in CONDITIONS:
        stem = f"mt50_seed{seed}_heuristic_confirm_{condition}"
        output_csv = ROOT / "results" / "tables" / f"{stem}_episodes.csv"
        output_summary = ROOT / "results" / "tables" / f"{stem}_summary.json"
        if output_summary.is_file():
            print(f"[skip] seed {seed} {condition}: {output_summary.name}")
            continue

        shard_specs: list[tuple[Path, Path]] = []
        running: list[tuple[int, subprocess.Popen[bytes], object]] = []
        for shard_index, task_ids in enumerate(shards):
            shard_stem = f"{stem}_shard{shard_index}"
            shard_csv = output_csv.with_name(f"{shard_stem}_episodes.csv")
            shard_summary = output_summary.with_name(f"{shard_stem}_summary.json")
            shard_specs.append((shard_summary, shard_csv))
            if shard_summary.is_file():
                print(f"[skip] complete shard {shard_index}: {shard_summary.name}")
                continue
            argv = [
                str(PYTHON),
                "evaluation/evaluate_multitask.py",
                "--benchmark", "MT50",
                "--condition", condition,
                "--benchmark-seed", str(FINAL_BANK_SEED),
                "--act-checkpoint", str(SEED42 / "act.pt"),
                "--detector-checkpoint", str(detector),
                "--recovery-checkpoint", str(recovery),
                "--output-csv", str(shard_csv),
                "--output-summary", str(shard_summary),
                "--methods", "heuristic_recovery",
                "--episodes-per-task", "50",
                "--max-steps", "500",
                "--noise-level", str(noise),
                "--action-std-scale", "0.4",
                "--observation-std-scale", "0.025",
                "--threshold", "0.64",
                "--release-threshold", "0.05",
                "--release-patience", "10",
                "--min-recovery-steps", "5",
                "--intervention-cooldown", "10",
                "--recovery-budget", "250",
                "--heuristic-min-steps", "30",
                "--heuristic-window", "20",
                "--heuristic-tolerance", "0.01",
                "--task-ids", *(str(value) for value in task_ids),
                "--seed", "42",
                "--device", device,
                "--log-file",
                f"results/logs/mt50_heuristic_seed_study/eval_seed{seed}_{condition}_shard{shard_index}.log",
                "--resume",
                "--allow-partial",
            ]
            if dry_run:
                print("\n>>> " + " ".join(argv))
                continue
            console_path = (
                ROOT
                / "results"
                / "logs"
                / "mt50_heuristic_seed_study"
                / f"seed{seed}_{condition}_shard{shard_index}.console.log"
            )
            console_path.parent.mkdir(parents=True, exist_ok=True)
            console = console_path.open("ab")
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
            )
            running.append((shard_index, process, console))
            print(f"[start] seed {seed} {condition} shard {shard_index}: {task_ids}")

        failures: list[tuple[int, int]] = []
        for shard_index, process, console in running:
            return_code = process.wait()
            console.close()
            print(f"[done] seed {seed} {condition} shard {shard_index}: {return_code}")
            if return_code != 0:
                failures.append((shard_index, return_code))
        if failures:
            raise RuntimeError(f"heuristic evaluation failures: {failures}")

        merge = [str(PYTHON), "scripts/merge_multitask_evaluation_shards.py"]
        for shard_summary, shard_csv in shard_specs:
            merge.extend(("--shard", str(shard_summary), str(shard_csv)))
        merge.extend(
            (
                "--output-csv", str(output_csv),
                "--output-summary", str(output_summary),
                "--overwrite",
                "--allow-method-subset",
            )
        )
        _run(merge, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(seed == 42 for seed in args.seeds):
        raise ValueError("seed 42 heuristic is already in the canonical confirmation bank")
    for seed in args.seeds:
        _evaluate(seed, args.eval_workers, args.device, dry_run=args.dry_run)
    print("\nMT50 seed-specific heuristic evaluation complete.")


if __name__ == "__main__":
    main()
