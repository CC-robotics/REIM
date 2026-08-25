#!/usr/bin/env bash
# Backfill horizon closed-loop rerun with per-horizon tuned thresholds.
# Supersedes the preliminary thr=0.65 closed-loop rows (kept on disk).
# Protocol mirrors the original runs (bank 20265010, release 0.05/10,
# min_recovery 5, cooldown 10, budget 250, max steps 500, REIM only) except:
#   threshold <- tuned per horizon (h0=0.72, h10=0.71, h50=0.62)
#   episodes-per-task = 20 (professor's 2026-08-21 directive)
# Usage: bash scripts/run_backfill_tuned_closed_loop.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

run_cell () {
  local horizon="$1" thr="$2" condition="$3" noise="$4"
  local tag="mt10_horizon${horizon}_tuned_${condition}"
  echo "=== ${tag} (thr=${thr}, noise=${noise}) ==="
  "$PY" evaluation/evaluate_multitask.py \
    --benchmark MT10 --condition "${condition}" --benchmark-seed 20265010 \
    --act-checkpoint checkpoints/mt10/seed_42/act.pt \
    --detector-checkpoint "checkpoints/mt10/horizon_${horizon}/failure_detector.pt" \
    --recovery-checkpoint "checkpoints/mt10/horizon_${horizon}/recovery.pt" \
    --output-csv "results/tables/${tag}_episodes.csv" \
    --output-summary "results/tables/${tag}_summary.json" \
    --methods reim --episodes-per-task 20 --max-steps 500 \
    --noise-level "${noise}" --action-std-scale 0.4 --observation-std-scale 0.025 \
    --threshold "${thr}" --release-threshold 0.05 --release-patience 10 \
    --min-recovery-steps 5 --intervention-cooldown 10 --recovery-budget 250 \
    --seed 42 --device cuda \
    --log-file "results/logs/${tag}.log" --resume
}

for spec in "0 0.72" "10 0.71" "50 0.62"; do
  set -- ${spec}
  H="$1"; T="$2"
  run_cell "${H}" "${T}" official_clean 0.0
  run_cell "${H}" "${T}" robustness_noise_10 0.1
  run_cell "${H}" "${T}" robustness_noise_40 0.4
done
echo "ALL 9 CELLS DONE"
