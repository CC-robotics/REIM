# FINAL confirmation-bank runs -- 20266010 (MT10) / 20266050 (MT50).
# Per the professor's review section 2: the 202650xx banks were inspected
# before tuning, so the final numbers must come from fresh one-shot banks.
#
# Frozen configuration (no more parameter changes):
#   - release 0.05 / patience 10 (robustness-first operating point)
#   - detector threshold per precision-floor 0.65 caliber:
#       MT10 canonical horizon=25 -> 0.73 ; MT50 -> 0.71
#     (floor-0.60 numbers are all archived; if the professor picks 0.60,
#      change the two thresholds back to 0.65/0.64 and re-run)
#   - 4 methods (mlp_bc / act / heuristic_recovery / reim) x 5 conditions
#     x 50 episodes per task
#   - MT10: 10,000 episodes; MT50: 50,000 episodes (tens of GPU-hours)
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 misreads UTF-8
# no-BOM scripts as GBK, and a CJK comment can swallow the following code
# line (this bug silently skipped the two official_clean cells before).
#
# Each cell writes its own outputs and supports --resume; rerun the script
# to continue after an interruption.
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts/run_confirmation_banks_202660xx.ps1
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$PY = ".venv\Scripts\python.exe"
$OUTDIR = "results/tables/confirmation_202660xx"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null
New-Item -ItemType Directory -Force -Path "results/logs/confirmation_202660xx" | Out-Null

function Run-Cell($bench, $benchseed, $thr, $condition, $noise) {
  $tag = "${bench}_confirm_${condition}".ToLower()
  Write-Host "=== $tag (bank=$benchseed, thr=$thr, noise=$noise) ==="
  & $PY evaluation/evaluate_multitask.py `
    --benchmark $bench --condition $condition --benchmark-seed $benchseed `
    --act-checkpoint "checkpoints/$($bench.ToLower())/seed_42/act.pt" `
    --detector-checkpoint "checkpoints/$($bench.ToLower())/seed_42/failure_detector.pt" `
    --recovery-checkpoint "checkpoints/$($bench.ToLower())/seed_42/recovery.pt" `
    --mlp-checkpoint "checkpoints/$($bench.ToLower())/seed_42/mlp_bc.pt" `
    --output-csv "$OUTDIR/${tag}_episodes.csv" `
    --output-summary "$OUTDIR/${tag}_summary.json" `
    --methods mlp_bc act heuristic_recovery reim --episodes-per-task 50 --max-steps 500 `
    --noise-level $noise --action-std-scale 0.4 --observation-std-scale 0.025 `
    --threshold $thr --release-threshold 0.05 --release-patience 10 `
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 `
    --seed 42 --device cuda `
    --log-file "results/logs/confirmation_202660xx/${tag}.log" --resume
}

# MT10 confirmation bank 20266010 (threshold 0.73)
Run-Cell "MT10" 20266010 0.73 "official_clean" 0.0
Run-Cell "MT10" 20266010 0.73 "robustness_noise_10" 0.1
Run-Cell "MT10" 20266010 0.73 "robustness_noise_20" 0.2
Run-Cell "MT10" 20266010 0.73 "robustness_noise_30" 0.3
Run-Cell "MT10" 20266010 0.73 "robustness_noise_40" 0.4

# MT50 confirmation bank 20266050 (threshold 0.71)
Run-Cell "MT50" 20266050 0.71 "official_clean" 0.0
Run-Cell "MT50" 20266050 0.71 "robustness_noise_10" 0.1
Run-Cell "MT50" 20266050 0.71 "robustness_noise_20" 0.2
Run-Cell "MT50" 20266050 0.71 "robustness_noise_30" 0.3
Run-Cell "MT50" 20266050 0.71 "robustness_noise_40" 0.4

Write-Host "ALL 10 CONFIRMATION CELLS DONE"
