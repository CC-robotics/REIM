# Backfill horizon closed-loop rerun with per-horizon tuned thresholds.
# PowerShell 版（与 scripts/run_backfill_tuned_closed_loop.sh 等价）。
# 用法（PowerShell，已激活 .venv 或不限）:
#   powershell -ExecutionPolicy Bypass -File scripts/run_backfill_tuned_closed_loop.ps1
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$PY = ".venv\Scripts\python.exe"

function Run-Cell($horizon, $thr, $condition, $noise) {
  $tag = "mt10_horizon${horizon}_tuned_${condition}"
  Write-Host "=== $tag (thr=$thr, noise=$noise) ==="
  & $PY evaluation/evaluate_multitask.py `
    --benchmark MT10 --condition $condition --benchmark-seed 20265010 `
    --act-checkpoint checkpoints/mt10/seed_42/act.pt `
    --detector-checkpoint "checkpoints/mt10/horizon_${horizon}/failure_detector.pt" `
    --recovery-checkpoint "checkpoints/mt10/horizon_${horizon}/recovery.pt" `
    --output-csv "results/tables/${tag}_episodes.csv" `
    --output-summary "results/tables/${tag}_summary.json" `
    --methods reim --episodes-per-task 20 --max-steps 500 `
    --noise-level $noise --action-std-scale 0.4 --observation-std-scale 0.025 `
    --threshold $thr --release-threshold 0.05 --release-patience 10 `
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 `
    --seed 42 --device cuda `
    --log-file "results/logs/${tag}.log" --resume
}

$specs = @(@(0, 0.72), @(10, 0.71), @(50, 0.62))
foreach ($s in $specs) {
  $h = $s[0]; $t = $s[1]
  Run-Cell $h $t "official_clean" 0.0
  Run-Cell $h $t "robustness_noise_10" 0.1
  Run-Cell $h $t "robustness_noise_40" 0.4
}
Write-Host "ALL 9 CELLS DONE"
