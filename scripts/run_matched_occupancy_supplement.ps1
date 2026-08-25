# Matched-occupancy supplementary runs (per professor review, section 4.5).
# 两个 REIM 补跑，都在搜索库 seed 20264010 上（不动 202650xx/202660xx），
# seed 42、20 回合/任务、threshold 0.65、canonical horizon=25 checkpoints。
#
# 单元 1 (noise 0.4): release 0.3 / patience 5 —— 网格匹配点
#   （搜索库上 occupancy 49.0% ≈ heuristic 47.6%），补 rescued/harmed 配对统计。
# 单元 2 (noise 0.1): release 0.02 / patience 3 —— 教授建议参数，
#   把 REIM occupancy 从 39.4% 抬向 heuristic 49.6%，验证 0.1 档能否匹配。
#
# 用法（PowerShell）:
#   powershell -ExecutionPolicy Bypass -File scripts/run_matched_occupancy_supplement.ps1
# 预计每个单元约 5-10 分钟（200 回合，GPU）。
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$PY = ".venv\Scripts\python.exe"

function Run-Cell($tag, $condition, $noise, $rel, $pat) {
  Write-Host "=== $tag (noise=$noise, release=$rel, patience=$pat) ==="
  & $PY evaluation/evaluate_multitask.py `
    --benchmark MT10 --condition $condition --benchmark-seed 20264010 `
    --act-checkpoint checkpoints/mt10/seed_42/act.pt `
    --detector-checkpoint checkpoints/mt10/seed_42/failure_detector.pt `
    --recovery-checkpoint checkpoints/mt10/seed_42/recovery.pt `
    --output-csv "results/diagnostics/release_patience_search/${tag}_episodes.csv" `
    --output-summary "results/diagnostics/release_patience_search/${tag}_summary.json" `
    --methods reim --episodes-per-task 20 --max-steps 500 `
    --noise-level $noise --action-std-scale 0.4 --observation-std-scale 0.025 `
    --threshold 0.65 --release-threshold $rel --release-patience $pat `
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 `
    --seed 42 --device cuda `
    --log-file "results/logs/${tag}.log" --resume
}

Run-Cell "mt10_matched_reim_rel030_pat5_noise40" "robustness_noise_40" 0.4 0.3 5
Run-Cell "mt10_matched_reim_rel002_pat3_noise10" "robustness_noise_10" 0.1 0.02 3
Write-Host "ALL 2 CELLS DONE"
