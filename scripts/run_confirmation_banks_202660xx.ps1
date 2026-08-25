# FINAL confirmation-bank runs — 20266010 (MT10) / 20266050 (MT50).
# 教授 PDF 第二节：现有 202650xx 库在调参前已被查阅，需全新确认库做一次性盲测。
#
# 冻结配置（参数停止修改）：
#   - release 0.05 / patience 10（robustness-first operating point）
#   - detector threshold 按 precision-floor 0.65 口径：
#       MT10 canonical horizon=25 → 0.73；MT50 → 0.71
#     （floor 0.60 口径数字已全部留档，若导师最终选 0.60，把阈值改回 0.65/0.64 即可）
#   - 4 方法（mlp_bc / act / heuristic_recovery / reim）× 5 条件 × 50 回合/任务
#   - MT10: 10,000 回合；MT50: 50,000 回合 —— 预计需要数十小时 GPU，建议隔夜跑
#
# 每个单元独立输出 + --resume，中断后重跑本脚本即可续跑。
# 用法（PowerShell）:
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
    --output-csv "$OUTDIR/${tag}_episodes.csv" `
    --output-summary "$OUTDIR/${tag}_summary.json" `
    --methods mlp_bc act heuristic_recovery reim --episodes-per-task 50 --max-steps 500 `
    --noise-level $noise --action-std-scale 0.4 --observation-std-scale 0.025 `
    --threshold $thr --release-threshold 0.05 --release-patience 10 `
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 `
    --seed 42 --device cuda `
    --log-file "results/logs/confirmation_202660xx/${tag}.log" --resume
}

# MT10 确认库 20266010（阈值 0.73）
Run-Cell "MT10" 20266010 0.73 "official_clean" 0.0
Run-Cell "MT10" 20266010 0.73 "robustness_noise_10" 0.1
Run-Cell "MT10" 20266010 0.73 "robustness_noise_20" 0.2
Run-Cell "MT10" 20266010 0.73 "robustness_noise_30" 0.3
Run-Cell "MT10" 20266010 0.73 "robustness_noise_40" 0.4

# MT50 确认库 20266050（阈值 0.71）
Run-Cell "MT50" 20266050 0.71 "official_clean" 0.0
Run-Cell "MT50" 20266050 0.71 "robustness_noise_10" 0.1
Run-Cell "MT50" 20266050 0.71 "robustness_noise_20" 0.2
Run-Cell "MT50" 20266050 0.71 "robustness_noise_30" 0.3
Run-Cell "MT50" 20266050 0.71 "robustness_noise_40" 0.4

Write-Host "ALL 10 CONFIRMATION CELLS DONE"
