# Supplement: confirmation-bank robustness_noise_00 cells (disturbed protocol
# at noise 0.0). The publication gate requires ROBUSTNESS_LEVELS = 0.0..0.4,
# i.e. five robustness cells per benchmark, separate from official_clean.
# Same frozen configuration as run_confirmation_banks_202660xx.ps1.
# Keep this file ASCII-only (PS 5.1 GBK misread bug).
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_confirmation_noise00_supplement.ps1
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

Run-Cell "MT10" 20266010 0.73 "robustness_noise_00" 0.0
Run-Cell "MT50" 20266050 0.71 "robustness_noise_00" 0.0

Write-Host "NOISE-00 SUPPLEMENT DONE"
