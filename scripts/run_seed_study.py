#!/usr/bin/env python3
"""Component-seed study runner — pre-submission P0 item 4 (training seeds).

Design (per the pre-submission review's minimum-acceptable version):
- ACT stays fixed at seed 42; the failure detector and the recovery policy
  are retrained with seeds 43/44 ("component-seed study").
- Evaluation uses the same disturbance CRN bank as the official runs
  (final evaluation bank seed 20265010), conditions official_clean /
  robustness_noise_10 / robustness_noise_40, 50 episodes per task.
- The canonical trigger threshold (0.65) and the frozen release parameters
  (release 0.05 / patience 10) are held fixed across seeds, so the study
  isolates training-seed variance rather than re-tuning per seed.

Provenance note (fail-closed evaluation requires this):
  The recovery dataset was collected with the seed-42 stack, so
  train_multitask_recovery.py bakes the seed-42 act/detector SHA256 into the
  checkpoint provenance.  After training, this runner rebinds the two hash
  fields to the deployment checkpoints (seed-42 ACT + seed-43/44 detector)
  and records both pairs under provenance["seed_study"]; the pre-bind file
  is kept next to the checkpoint as recovery_prebind.pt for audit.

Everything is resumable: completed steps are skipped on re-run, partially
completed trainings/evaluations resume from their latest state.

Usage:
  .venv\\Scripts\\python.exe scripts\\run_seed_study.py --device cuda
  .venv\\Scripts\\python.exe scripts\\run_seed_study.py --dry-run
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

SEED42 = PROJECT_ROOT / "checkpoints" / "mt10" / "seed_42"
CONDITIONS = (("official_clean", 0.0), ("robustness_noise_10", 0.1), ("robustness_noise_40", 0.4))
FINAL_BANK_SEED = 20265010
TRIGGER_THRESHOLD = 0.65
RELEASE_THRESHOLD = 0.05
RELEASE_PATIENCE = 10


def _run(argv: list[str], *, dry_run: bool) -> None:
    printable = " ".join(argv)
    print(f"\n>>> {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True, cwd=PROJECT_ROOT)


def _train_detector(seed: int, device: str, *, dry_run: bool) -> Path:
    config = PROJECT_ROOT / "configs" / "multitask" / f"mt10_detector_seed{seed}.yaml"
    checkpoint = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "failure_detector.pt"
    latest = checkpoint.with_name("failure_detector_latest.pt")
    if not config.is_file():
        raise FileNotFoundError(f"missing detector config: {config}")
    if checkpoint.is_file():
        print(f"[skip] detector seed {seed} already trained: {checkpoint}")
        return checkpoint
    argv = [str(PYTHON), "trainers/train_detector.py", "--config", str(config), "--device", device]
    if latest.is_file():
        argv.append("--resume")  # bare --resume resumes from latest_checkpoint
    _run(argv, dry_run=dry_run)
    return checkpoint


def _train_recovery(seed: int, device: str, *, dry_run: bool) -> Path:
    checkpoint = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "recovery.pt"
    latest = checkpoint.with_name("recovery_latest.pt")
    if checkpoint.is_file():
        print(f"[skip] recovery seed {seed} already trained: {checkpoint}")
        return checkpoint
    # Hyperparameters mirror the canonical pipeline invocation
    # (scripts/run_multitask_pipeline.py stage "train_recovery"; mt10.yaml has
    # no recovery_training overrides, so these are the trainer defaults).
    argv = [
        str(PYTHON), "trainers/train_multitask_recovery.py",
        "--benchmark", "MT10",
        "--data-dir", "datasets/mt10/recovery",
        "--output", str(checkpoint),
        "--history", f"results/tables/mt10_seed{seed}_recovery_training.csv",
        "--curve", f"results/figures/mt10_seed{seed}_recovery_training.png",
        "--summary", f"results/tables/mt10_seed{seed}_recovery_training.json",
        "--log-file", f"results/logs/seed_study/train_recovery_seed{seed}.log",
        "--seed", str(seed),
        "--device", device,
        "--epochs", "60",
        "--batch-size", "1024",
        "--learning-rate", "0.001",
        "--weight-decay", "1e-05",
        "--validation-fraction", "0.2",
        "--hidden-dims", "512", "512", "256",
        "--state-noise-std", "0.005",
        "--grad-clip", "1.0",
        "--patience", "12",
        "--min-delta", "1e-06",
        "--num-workers", "0",
        "--expected-target-per-task", "50",
    ]
    if latest.is_file():
        argv.append("--resume")
    _run(argv, dry_run=dry_run)
    return checkpoint


def _rebind_recovery_provenance(seed: int, *, dry_run: bool) -> None:
    """Rebind recovery provenance hashes to the deployment checkpoints.

    Only the act/detector SHA256 fields change; everything else in the
    provenance (dataset manifest, split, training config) is preserved, and
    both the data-collection and deployment hash pairs are recorded under
    provenance["seed_study"].
    """
    from data.io import file_sha256
    from models.imitation_recovery_policy import ImitationRecoveryPolicy

    checkpoint = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "recovery.pt"
    detector = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "failure_detector.pt"
    act = SEED42 / "act.pt"
    if dry_run:
        print(f"\n>>> rebind provenance: {checkpoint} -> act(seed42)+detector(seed{seed})")
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recovery checkpoint not trained yet: {checkpoint}")

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
    ):
        print(f"[skip] recovery seed {seed} provenance already rebound")
        return

    backup = checkpoint.with_name("recovery_prebind.pt")
    if not backup.exists():
        backup.write_bytes(checkpoint.read_bytes())
    provenance["seed_study"] = {
        "study": "component-seed study (pre-submission P0 item 4)",
        "training_seed": seed,
        "data_collection_act_checkpoint_sha256": old_act,
        "data_collection_detector_checkpoint_sha256": old_detector,
        "deployment_act_checkpoint_sha256": deploy_act,
        "deployment_detector_checkpoint_sha256": deploy_detector,
        "note": (
            "Only the training seed differs from the canonical stack; the "
            "recovery data were collected with the seed-42 ACT+detector. "
            "act/detector hashes are rebound to the deployment checkpoints."
        ),
    }
    provenance["act_checkpoint_sha256"] = deploy_act
    provenance["detector_checkpoint_sha256"] = deploy_detector
    model.save(checkpoint)
    print(f"[rebind] seed {seed}: detector {str(old_detector)[:12]}… -> {deploy_detector[:12]}…")


def _evaluate(seed: int, device: str, *, dry_run: bool) -> None:
    detector = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "failure_detector.pt"
    recovery = PROJECT_ROOT / "checkpoints" / "mt10" / f"seed_{seed}" / "recovery.pt"
    for condition, noise in CONDITIONS:
        summary = PROJECT_ROOT / "results" / "tables" / f"mt10_seed{seed}_{condition}_summary.json"
        csv_path = summary.with_name(summary.name.replace("_summary.json", "_episodes.csv"))
        if summary.is_file():
            print(f"[skip] seed {seed} {condition} already evaluated: {summary.name}")
            continue
        argv = [
            str(PYTHON), "evaluation/evaluate_multitask.py",
            "--benchmark", "MT10",
            "--condition", condition,
            "--benchmark-seed", str(FINAL_BANK_SEED),
            "--act-checkpoint", str(SEED42 / "act.pt"),
            "--mlp-checkpoint", str(SEED42 / "mlp_bc.pt"),
            "--detector-checkpoint", str(detector),
            "--recovery-checkpoint", str(recovery),
            "--output-csv", str(csv_path),
            "--output-summary", str(summary),
            "--methods", "mlp_bc", "act", "heuristic_recovery", "reim",
            "--episodes-per-task", "50",
            "--max-steps", "500",
            "--noise-level", str(noise),
            "--action-std-scale", "0.4",
            "--observation-std-scale", "0.025",
            "--threshold", str(TRIGGER_THRESHOLD),
            "--release-threshold", str(RELEASE_THRESHOLD),
            "--release-patience", str(RELEASE_PATIENCE),
            "--min-recovery-steps", "5",
            "--intervention-cooldown", "10",
            "--recovery-budget", "250",
            "--seed", "42",
            "--device", device,
            "--log-file", f"results/logs/seed_study/eval_seed{seed}_{condition}.log",
            "--resume",
        ]
        _run(argv, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    (PROJECT_ROOT / "results" / "logs" / "seed_study").mkdir(parents=True, exist_ok=True)

    print("Component-seed study: detector+recovery seeds", args.seeds)
    print("Fixed: ACT=seed_42, threshold=0.65, release=0.05/10, bank=20265010")
    print("Conditions:", ", ".join(f"{c}(noise={n})" for c, n in CONDITIONS))

    for seed in args.seeds:
        if not args.skip_training:
            _train_detector(seed, args.device, dry_run=args.dry_run)
            _train_recovery(seed, args.device, dry_run=args.dry_run)
            _rebind_recovery_provenance(seed, dry_run=args.dry_run)
        if not args.skip_evaluation:
            _evaluate(seed, args.device, dry_run=args.dry_run)

    print("\nSeed study plan complete. Summaries land in results/tables/mt10_seed<seed>_<condition>_summary.json")


if __name__ == "__main__":
    main()
