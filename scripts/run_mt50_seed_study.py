#!/usr/bin/env python3
"""Run the four-seed MT50 detector/recovery stability study.

Study definition
----------------
* Seed 42 is the frozen canonical stack already used by the manuscript.
* Seeds 43, 44, and 45 retrain only the failure detector and recovery MLP.
* The seed-42 ACT remains fixed for every run, matching the existing MT10
  component-seed study.
* Recovery supervision remains the same frozen seed-42 collection bank.  The
  recovery checkpoint records both collection and deployment detector hashes.
* The final confirmation bank is fixed at benchmark seed 20266050.
* Only REIM is re-evaluated for new seeds because MLP BC, ACT, and the
  heuristic-gated controller are invariant across component seeds.
* Conditions are official clean, robustness noise 0.10, and noise 0.40, with
  50 episodes per task.
* Evaluation is split into disjoint task shards and merged with the repository's
  fail-closed merger. This makes the CPU-bound MuJoCo rollout stage practical
  without changing any task, seed, checkpoint, or perturbation.

Every stage is resumable. Completed checkpoints and summaries are skipped;
partial training/evaluation artifacts resume in place.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

SEED42 = PROJECT_ROOT / "checkpoints" / "mt50" / "seed_42"
CONDITIONS = (
    ("official_clean", 0.0),
    ("robustness_noise_10", 0.1),
    ("robustness_noise_40", 0.4),
)
FINAL_BANK_SEED = 20266050
TRIGGER_THRESHOLD = 0.64
RELEASE_THRESHOLD = 0.05
RELEASE_PATIENCE = 10


def _run(argv: list[str], *, dry_run: bool) -> None:
    print("\n>>> " + " ".join(argv), flush=True)
    if not dry_run:
        subprocess.run(argv, check=True, cwd=PROJECT_ROOT)


def _train_detector(seed: int, device: str, *, dry_run: bool) -> Path:
    config = PROJECT_ROOT / "configs" / "multitask" / f"mt50_detector_seed{seed}.yaml"
    checkpoint = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "failure_detector.pt"
    latest = checkpoint.with_name("failure_detector_latest.pt")
    if not config.is_file():
        raise FileNotFoundError(f"missing detector config: {config}")
    if checkpoint.is_file():
        print(f"[skip] detector seed {seed} already trained: {checkpoint}")
        return checkpoint
    argv = [
        str(PYTHON),
        "trainers/train_detector.py",
        "--config",
        str(config),
        "--device",
        device,
    ]
    if latest.is_file():
        argv.append("--resume")
    _run(argv, dry_run=dry_run)
    return checkpoint


def _train_recovery(seed: int, device: str, *, dry_run: bool) -> Path:
    checkpoint = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "recovery.pt"
    latest = checkpoint.with_name("recovery_latest.pt")
    if checkpoint.is_file():
        print(f"[skip] recovery seed {seed} already trained: {checkpoint}")
        return checkpoint
    argv = [
        str(PYTHON),
        "trainers/train_multitask_recovery.py",
        "--benchmark",
        "MT50",
        "--data-dir",
        "datasets/mt50/recovery",
        "--output",
        str(checkpoint),
        "--history",
        f"results/tables/mt50_seed{seed}_recovery_training.csv",
        "--curve",
        f"results/figures/mt50_seed{seed}_recovery_training.png",
        "--summary",
        f"results/tables/mt50_seed{seed}_recovery_training.json",
        "--log-file",
        f"results/logs/mt50_seed_study/train_recovery_seed{seed}.log",
        "--seed",
        str(seed),
        "--device",
        device,
        "--epochs",
        "60",
        "--batch-size",
        "1024",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "1e-05",
        "--validation-fraction",
        "0.2",
        "--hidden-dims",
        "512",
        "512",
        "256",
        "--state-noise-std",
        "0.005",
        "--grad-clip",
        "1.0",
        "--patience",
        "12",
        "--min-delta",
        "1e-06",
        "--num-workers",
        "0",
        "--expected-target-per-task",
        "50",
    ]
    if latest.is_file():
        argv.append("--resume")
    _run(argv, dry_run=dry_run)
    return checkpoint


def _rebind_recovery_provenance(seed: int, *, dry_run: bool) -> None:
    from data.io import file_sha256
    from models.imitation_recovery_policy import ImitationRecoveryPolicy

    checkpoint = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "recovery.pt"
    detector = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "failure_detector.pt"
    act = SEED42 / "act.pt"
    if dry_run:
        print(f"\n>>> rebind provenance: {checkpoint} -> act(seed42)+detector(seed{seed})")
        return
    if not checkpoint.is_file() or not detector.is_file() or not act.is_file():
        raise FileNotFoundError("required seed-study deployment checkpoint is missing")

    deploy_act = file_sha256(act)
    deploy_detector = file_sha256(detector)
    model = ImitationRecoveryPolicy.load(checkpoint, device="cpu")
    provenance = model.provenance
    old_act = provenance.get("act_checkpoint_sha256")
    old_detector = provenance.get("detector_checkpoint_sha256")
    existing = provenance.get("seed_study") or {}
    if (
        old_act == deploy_act
        and old_detector == deploy_detector
        and existing.get("training_seed") == seed
        and existing.get("benchmark") == "MT50"
    ):
        print(f"[skip] recovery seed {seed} provenance already rebound")
        return

    backup = checkpoint.with_name("recovery_prebind.pt")
    if not backup.exists():
        backup.write_bytes(checkpoint.read_bytes())
    provenance["seed_study"] = {
        "study": "MT50 four-seed component stability study",
        "benchmark": "MT50",
        "training_seed": seed,
        "data_collection_act_checkpoint_sha256": old_act,
        "data_collection_detector_checkpoint_sha256": old_detector,
        "deployment_act_checkpoint_sha256": deploy_act,
        "deployment_detector_checkpoint_sha256": deploy_detector,
        "note": (
            "ACT and the recovery collection bank are fixed at seed 42; "
            "detector and recovery optimization use this component seed."
        ),
    }
    provenance["act_checkpoint_sha256"] = deploy_act
    provenance["detector_checkpoint_sha256"] = deploy_detector
    model.save(checkpoint)
    print(f"[rebind] MT50 seed {seed}: detector -> {deploy_detector[:12]}...")


def _evaluation_argv(
    *,
    seed: int,
    device: str,
    condition: str,
    noise: float,
    task_ids: list[int],
    csv_path: Path,
    summary: Path,
    shard_index: int,
) -> list[str]:
    detector = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "failure_detector.pt"
    recovery = PROJECT_ROOT / "checkpoints" / "mt50" / f"seed_{seed}" / "recovery.pt"
    return [
        str(PYTHON),
        "evaluation/evaluate_multitask.py",
        "--benchmark",
        "MT50",
        "--condition",
        condition,
        "--benchmark-seed",
        str(FINAL_BANK_SEED),
        "--act-checkpoint",
        str(SEED42 / "act.pt"),
        "--detector-checkpoint",
        str(detector),
        "--recovery-checkpoint",
        str(recovery),
        "--output-csv",
        str(csv_path),
        "--output-summary",
        str(summary),
        "--methods",
        "reim",
        "--episodes-per-task",
        "50",
        "--max-steps",
        "500",
        "--noise-level",
        str(noise),
        "--action-std-scale",
        "0.4",
        "--observation-std-scale",
        "0.025",
        "--threshold",
        str(TRIGGER_THRESHOLD),
        "--release-threshold",
        str(RELEASE_THRESHOLD),
        "--release-patience",
        str(RELEASE_PATIENCE),
        "--min-recovery-steps",
        "5",
        "--intervention-cooldown",
        "10",
        "--recovery-budget",
        "250",
        "--task-ids",
        *(str(value) for value in task_ids),
        "--seed",
        "42",
        "--device",
        device,
        "--log-file",
        f"results/logs/mt50_seed_study/eval_seed{seed}_{condition}_shard{shard_index}.log",
        "--resume",
        "--allow-partial",
    ]


def _evaluate(
    seed: int,
    device: str,
    *,
    workers: int,
    dry_run: bool,
) -> None:
    if workers <= 0 or workers > 8:
        raise ValueError("evaluation workers must lie in [1, 8]")
    task_shards = [list(range(offset, 50, workers)) for offset in range(workers)]
    for condition, noise in CONDITIONS:
        stem = f"mt50_seed{seed}_confirm_{condition}"
        summary = PROJECT_ROOT / "results" / "tables" / f"{stem}_summary.json"
        csv_path = summary.with_name(f"{stem}_episodes.csv")
        if summary.is_file():
            print(f"[skip] MT50 seed {seed} {condition}: {summary.name}")
            continue
        shard_specs: list[tuple[Path, Path]] = []
        running: list[tuple[int, subprocess.Popen[bytes], object]] = []
        for shard_index, task_ids in enumerate(task_shards):
            shard_stem = f"{stem}_shard{shard_index}"
            shard_summary = summary.with_name(f"{shard_stem}_summary.json")
            shard_csv = summary.with_name(f"{shard_stem}_episodes.csv")
            shard_specs.append((shard_summary, shard_csv))
            if shard_summary.is_file():
                print(f"[skip] complete shard {shard_index}: {shard_summary.name}")
                continue
            argv = _evaluation_argv(
                seed=seed,
                device=device,
                condition=condition,
                noise=noise,
                task_ids=task_ids,
                csv_path=shard_csv,
                summary=shard_summary,
                shard_index=shard_index,
            )
            if dry_run:
                print("\n>>> " + " ".join(argv))
                continue
            console_path = (
                PROJECT_ROOT
                / "results"
                / "logs"
                / "mt50_seed_study"
                / f"seed{seed}_{condition}_shard{shard_index}.console.log"
            )
            console_path.parent.mkdir(parents=True, exist_ok=True)
            console = console_path.open("ab")
            process = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
            )
            running.append((shard_index, process, console))
            print(
                f"[start] seed {seed} {condition} shard {shard_index}: "
                f"tasks={task_ids}"
            )
        failures: list[tuple[int, int]] = []
        for shard_index, process, console in running:
            return_code = process.wait()
            console.close()
            print(
                f"[done] seed {seed} {condition} shard {shard_index}: "
                f"return_code={return_code}"
            )
            if return_code != 0:
                failures.append((shard_index, return_code))
        if failures:
            raise RuntimeError(
                f"MT50 evaluation shard failures for seed {seed} {condition}: {failures}"
            )
        merge_argv = [
            str(PYTHON),
            "scripts/merge_multitask_evaluation_shards.py",
        ]
        for shard_summary, shard_csv in shard_specs:
            merge_argv.extend(("--shard", str(shard_summary), str(shard_csv)))
        merge_argv.extend(
            (
                "--output-csv",
                str(csv_path),
                "--output-summary",
                str(summary),
                "--overwrite",
                "--allow-method-subset",
            )
        )
        _run(merge_argv, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if any(seed == 42 for seed in args.seeds):
        raise ValueError("seed 42 is canonical and must not be retrained by this script")

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    (PROJECT_ROOT / "results" / "logs" / "mt50_seed_study").mkdir(
        parents=True, exist_ok=True
    )

    print("MT50 component seeds:", args.seeds)
    print("Fixed ACT=42, trigger=0.64, release=0.05/10, bank=20266050")
    print("Conditions:", ", ".join(f"{name}({noise})" for name, noise in CONDITIONS))
    for seed in args.seeds:
        if not args.skip_training:
            _train_detector(seed, args.device, dry_run=args.dry_run)
            _train_recovery(seed, args.device, dry_run=args.dry_run)
            _rebind_recovery_provenance(seed, dry_run=args.dry_run)
        if not args.skip_evaluation:
            _evaluate(
                seed,
                args.device,
                workers=args.eval_workers,
                dry_run=args.dry_run,
            )

    print("\nMT50 seed study stages complete.")


if __name__ == "__main__":
    main()
