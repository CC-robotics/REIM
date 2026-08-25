# Backfill closed-loop rerun with precision-floor 0.65 thresholds (probe values).
# 与 run_backfill_tuned_closed_loop.ps1 同构，但阈值改用 floor-0.65 探针结果：
#   h0=0.79, h10=0.78, h25=0.73, h50=0.71
# 目的：0.60/0.65 两种 precision-floor 口径都做闭环，论文可直接并列呈现，
#       口径最终由导师定夺，选定后无需再跑。
# 条件：clean / 0.1 / 0.4，每任务 20 回合，bank 20265010（与原调阈闭环同库可比）。
# 12 个单元，预计总共约 30-50 分钟（GPU）。
# 用法（PowerShell）:
#   powershell -ExecutionPolicy Bypass -File scripts/run_backfill_floor065_closed_loop.ps1
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$PY = ".venv\Scripts\python.exe"

function Run-Cell($horizon, $thr, $condition, $noise, $ckptdir) {
  $tag = "mt10_horizon${horizon}_floor065_${condition}"
  Write-Host "=== $tag (thr=$thr, noise=$noise) ==="
  & $PY evaluation/evaluate_multitask.py `
    --benchmark MT10 --condition $condition --benchmark-seed 20265010 `
    --act-checkpoint checkpoints/mt10/seed_42/act.pt `
    --detector-checkpoint "checkpoints/mt10/${ckptdir}/failure_detector.pt" `
    --recovery-checkpoint "checkpoints/mt10/${ckptdir}/recovery.pt" `
    --output-csv "results/tables/${tag}_episodes.csv" `
    --output-summary "results/tables/${tag}_summary.json" `
    --methods reim --episodes-per-task 20 --max-steps 500 `
    --noise-level $noise --action-std-scale 0.4 --observation-std-scale 0.025 `
    --threshold $thr --release-threshold 0.05 --release-patience 10 `
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 `
    --seed 42 --device cuda `
    --log-file "results/logs/${tag}.log" --resume
}

$specs = @(@(0, 0.79, "horizon_0"), @(10, 0.78, "horizon_10"), @(25, 0.73, "seed_42"), @(50, 0.71, "horizon_50"))
foreach ($s in $specs) {
  $h = $s[0]; $t = $s[1]; $c = $s[2]
  Run-Cell $h $t "official_clean" 0.0 $c
  Run-Cell $h $t "robustness_noise_10" 0.1 $c
  Run-Cell $h $t "robustness_noise_40" 0.4 $c
}
Write-Host "ALL 12 CELLS DONE"
