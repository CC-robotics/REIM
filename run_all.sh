#!/usr/bin/env bash
#
# Run the complete REIM protocol.
#
# The canonical invocation is a full Meta-World run.  The explicit smoke
# profile uses the toy backend and isolated artifacts so it can never
# overwrite benchmark data, checkpoints, tables, figures, or paper assets.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

PROFILE="full"
BACKEND="metaworld"
SEED=42
EVAL_SEED=""
TASK_SEED="${REIM_TASK_SEED:-20260726}"
TRAIN_DEVICE="auto"
RECOVERY_DEVICE="${REIM_RECOVERY_DEVICE:-cpu}"
EVAL_DEVICE="cpu"
TORCH_THREADS="${REIM_TORCH_THREADS:-1}"
PYTHON_BIN="${REIM_PYTHON:-}"
RESUME=false
DRY_RUN=false
RUN_PPO_ABLATION=false

SKIP_DEMOS=false
SKIP_ACT=false
SKIP_FAILURES=false
SKIP_DETECTOR=false
SKIP_RECOVERY_STARTS=false
SKIP_RECOVERY=false
SKIP_EPISODE_BANK=false
SKIP_AUDITS=false
SKIP_BASELINE=false
SKIP_ROBUSTNESS=false
SKIP_ABLATION=false
SKIP_COMPARISON=false
SKIP_GATE_SENSITIVITY=false
SKIP_OPERATION_FIGURE=false
SKIP_PLOTS=false
SKIP_MANIFEST=false

usage() {
  cat <<'EOF'
Usage:
  ./run_all.sh                         # full Meta-World protocol (default)
  ./run_all.sh --resume                # safely continue existing artifacts
  ./run_all.sh --smoke                 # fast toy end-to-end integration run
  ./run_all.sh --smoke --resume
  ./run_all.sh full metaworld          # backwards-compatible positional form
  ./run_all.sh smoke toy

Profiles:
  full / metaworld
    500 demonstrations, 200 ACT epochs, 2,000 failure rollouts,
    50 detector epochs, 1,000 training + 200 validation online trigger
    states, a standalone zero-PPO-step imitation recovery actor, a persistent
    1,000-episode 20% CRN bank, 1,000 baseline episodes/method, five robustness
    levels x 200 paired episodes/method, 1,000 ablation and comparison
    episodes/method assembled from the same CRN raw evidence, a separate
    five-threshold gate diagnostic, a rendered simulator operation sequence,
    strict paper-data validation, and PDF build.
    PPO is not part of the default method.

  smoke / toy
    Isolated quick check: 12 demonstrations, 5 ACT epochs, 32 failure
    rollouts, 5 detector epochs, the frozen standalone recovery actor, and
    small evaluations.
    Outputs live under datasets/smoke, checkpoints/smoke, results/smoke,
    and paper_assets/smoke. They are never benchmark results.

Options:
  --full                         Select the canonical full/Meta-World profile.
  --smoke                        Select the isolated smoke/toy profile.
  --profile {full|smoke}         Explicit profile selector.
  --backend {metaworld|toy}      Explicit backend; only canonical pairs are
                                accepted to prevent provenance mistakes.
  --seed N                       Training/data seed (default: 42).
  --eval-seed N                  Held-out episode seed (full default: 8000042;
                                smoke default: seed+1000000).
  --task-seed N                  Frozen Meta-World task-bank seed
                                (default: 20260726).
                                Full runs reject overlap between training/data
                                and evaluation seed ranges.
  --train-device DEVICE          ACT/detector/data device (default: auto).
  --recovery-device DEVICE       Device for the optional PPO negative ablation
                                (default: cpu).
  --eval-device DEVICE           Evaluation device (default: cpu).
  --torch-threads N              PyTorch CPU inference threads (default: 1).
  --python PATH                  Python executable (default: .venv/bin/python).
  --resume                       Resume data collection and any trainer whose
                                latest checkpoint exists; skip completed,
                                audited, or already evaluated artifacts.
  --run-ppo-ablation             Explicitly run the legacy 500k-step PPO
                                recovery as a separate negative ablation. Its
                                checkpoint is never used by formal evaluation.
  --dry-run                      Print every resolved command without running it.

Stage skips:
  --skip-demos                   Skip expert demonstration collection.
  --skip-act                     Skip ACT imitation training.
  --skip-failures                Skip failure rollout generation.
  --skip-detector                Skip LSTM detector training.
  --skip-recovery-starts         Skip exact online trigger-state collection
                                 (full/Meta-World profile only).
  --skip-recovery                Skip standalone imitation-recovery export.
  --skip-episode-bank            Skip persistent CRN episode-bank generation.
  --skip-audits                  Skip CRN determinism/no-intervention audits.
  --skip-baseline                Skip the four-method baseline evaluation.
  --skip-robustness              Skip the five-level robustness experiment.
  --skip-ablation                Skip requesting the paired four-way ablation.
  --skip-comparison              Skip requesting the paired comparison table.
                                In full mode these two derived artifacts are
                                assembled together unless both are skipped.
  --skip-gate-sensitivity        Skip the separate post-freeze five-threshold
                                controller sensitivity diagnostic.
  --skip-operation-figure        Skip searching and rendering the paired
                                ACT-failure/REIM-success simulator sequence.
  --skip-plots                   Skip figures and LaTeX paper assets.
  --skip-manifest                Skip the reproducibility manifest.
  --skip-training                Skip ACT/detector training and recovery export.
  --skip-data                    Skip demonstration, failure, and trigger-state
                                 collection.
  --skip-evaluation              Skip baseline, robustness, ablation, comparison.
  -h, --help                     Show this help.

The runner never deletes datasets and never fabricates results. Without
--resume, collectors fail if their output directory already contains
trajectories. Frozen recovery artifacts and CRN banks are reused when present;
--resume also skips complete per-episode evaluations.
EOF
}

die() {
  printf 'run_all.sh: %s\n' "$*" >&2
  exit 2
}

need_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" && "$value" != --* ]] || die "$option requires a value"
}

while (($#)); do
  case "$1" in
    --full)
      PROFILE="full"
      BACKEND="metaworld"
      shift
      ;;
    --smoke)
      PROFILE="smoke"
      BACKEND="toy"
      shift
      ;;
    --profile)
      need_value "$1" "${2:-}"
      PROFILE="$2"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --backend)
      need_value "$1" "${2:-}"
      BACKEND="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND="${1#*=}"
      shift
      ;;
    --seed)
      need_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --eval-seed)
      need_value "$1" "${2:-}"
      EVAL_SEED="$2"
      shift 2
      ;;
    --eval-seed=*)
      EVAL_SEED="${1#*=}"
      shift
      ;;
    --task-seed)
      need_value "$1" "${2:-}"
      TASK_SEED="$2"
      shift 2
      ;;
    --task-seed=*)
      TASK_SEED="${1#*=}"
      shift
      ;;
    --train-device)
      need_value "$1" "${2:-}"
      TRAIN_DEVICE="$2"
      shift 2
      ;;
    --train-device=*)
      TRAIN_DEVICE="${1#*=}"
      shift
      ;;
    --recovery-device)
      need_value "$1" "${2:-}"
      RECOVERY_DEVICE="$2"
      shift 2
      ;;
    --recovery-device=*)
      RECOVERY_DEVICE="${1#*=}"
      shift
      ;;
    --eval-device)
      need_value "$1" "${2:-}"
      EVAL_DEVICE="$2"
      shift 2
      ;;
    --eval-device=*)
      EVAL_DEVICE="${1#*=}"
      shift
      ;;
    --torch-threads)
      need_value "$1" "${2:-}"
      TORCH_THREADS="$2"
      shift 2
      ;;
    --torch-threads=*)
      TORCH_THREADS="${1#*=}"
      shift
      ;;
    --python)
      need_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --run-ppo-ablation)
      RUN_PPO_ABLATION=true
      shift
      ;;
    --skip-demos)
      SKIP_DEMOS=true
      shift
      ;;
    --skip-act)
      SKIP_ACT=true
      shift
      ;;
    --skip-failures)
      SKIP_FAILURES=true
      shift
      ;;
    --skip-detector)
      SKIP_DETECTOR=true
      shift
      ;;
    --skip-recovery-starts)
      SKIP_RECOVERY_STARTS=true
      shift
      ;;
    --skip-recovery)
      SKIP_RECOVERY=true
      shift
      ;;
    --skip-episode-bank)
      SKIP_EPISODE_BANK=true
      shift
      ;;
    --skip-audits)
      SKIP_AUDITS=true
      shift
      ;;
    --skip-baseline)
      SKIP_BASELINE=true
      shift
      ;;
    --skip-robustness)
      SKIP_ROBUSTNESS=true
      shift
      ;;
    --skip-ablation)
      SKIP_ABLATION=true
      shift
      ;;
    --skip-comparison)
      SKIP_COMPARISON=true
      shift
      ;;
    --skip-gate-sensitivity)
      SKIP_GATE_SENSITIVITY=true
      shift
      ;;
    --skip-operation-figure)
      SKIP_OPERATION_FIGURE=true
      shift
      ;;
    --skip-plots)
      SKIP_PLOTS=true
      shift
      ;;
    --skip-manifest)
      SKIP_MANIFEST=true
      shift
      ;;
    --skip-training)
      SKIP_ACT=true
      SKIP_DETECTOR=true
      SKIP_RECOVERY=true
      RUN_PPO_ABLATION=false
      shift
      ;;
    --skip-data)
      SKIP_DEMOS=true
      SKIP_FAILURES=true
      SKIP_RECOVERY_STARTS=true
      shift
      ;;
    --skip-evaluation)
      SKIP_EPISODE_BANK=true
      SKIP_AUDITS=true
      SKIP_BASELINE=true
      SKIP_ROBUSTNESS=true
      SKIP_ABLATION=true
      SKIP_COMPARISON=true
      SKIP_GATE_SENSITIVITY=true
      SKIP_OPERATION_FIGURE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    full)
      PROFILE="full"
      BACKEND="metaworld"
      shift
      ;;
    smoke)
      PROFILE="smoke"
      BACKEND="toy"
      shift
      ;;
    metaworld|toy)
      BACKEND="$1"
      shift
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ "$PROFILE" == "full" || "$PROFILE" == "smoke" ]] \
  || die "--profile must be full or smoke"
[[ "$BACKEND" == "metaworld" || "$BACKEND" == "toy" ]] \
  || die "--backend must be metaworld or toy"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
if [[ -z "$EVAL_SEED" ]]; then
  if [[ "$PROFILE" == "full" ]]; then
    EVAL_SEED=8000042
  else
    EVAL_SEED=$((SEED + 1000000))
  fi
fi
[[ "$EVAL_SEED" =~ ^[0-9]+$ ]] \
  || die "--eval-seed must be a non-negative integer"
[[ "$TASK_SEED" =~ ^[0-9]+$ ]] \
  || die "--task-seed must be a non-negative integer"
[[ "$TORCH_THREADS" =~ ^[1-9][0-9]*$ ]] \
  || die "--torch-threads must be a positive integer"

if [[ "$PROFILE" == "full" && "$BACKEND" != "metaworld" ]]; then
  die "the full profile must use --backend metaworld"
fi
if [[ "$PROFILE" == "smoke" && "$BACKEND" != "toy" ]]; then
  die "the smoke profile must use --backend toy"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    die ".venv/bin/python is missing; run ./setup.sh or pass --python PATH"
  fi
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python executable not found: $PYTHON_BIN"
fi

if [[ "$PROFILE" == "full" ]]; then
  DEMO_EPISODES=500
  ACT_EPOCHS=200
  FAILURE_EPISODES=2000
  DETECTOR_EPOCHS=50
  RECOVERY_START_TRAIN_EPISODES=1000
  RECOVERY_START_VALIDATION_EPISODES=200
  PPO_TIMESTEPS=500000
  BASELINE_EPISODES=1000
  ROBUSTNESS_EPISODES=200
  ABLATION_EPISODES=1000
  COMPARISON_EPISODES=1000
  MAX_STEPS=200
  BOOTSTRAP_SAMPLES=10000
  GATE_SENSITIVITY_EPISODES=200
  GATE_SENSITIVITY_SEED=$((EVAL_SEED + 200000))
  OPERATION_SEED_START=$((EVAL_SEED + 300000))

  ACT_CONFIG="configs/bc.yaml"
  DETECTOR_CONFIG="configs/detector.yaml"
  RECOVERY_CONFIG="configs/recovery_imitation.yaml"
  PPO_CONFIG="configs/ppo_trigger.yaml"
  DEMO_DIR="datasets/demonstrations"
  FAILURE_DIR="datasets/failures"
  RECOVERY_START_TRAIN_PATH="datasets/recovery_starts/train.npz"
  RECOVERY_START_VALIDATION_PATH="datasets/recovery_starts/validation.npz"
  RECOVERY_START_TRAIN_SEED=$((SEED + 3000000))
  RECOVERY_START_VALIDATION_SEED=$((SEED + 4000000))
  ACT_CHECKPOINT="checkpoints/bc_policy.pt"
  ACT_LATEST="checkpoints/bc_policy_latest.pt"
  DETECTOR_CHECKPOINT="checkpoints/failure_detector.pt"
  DETECTOR_LATEST="checkpoints/failure_detector_latest.pt"
  RECOVERY_CHECKPOINT="checkpoints/imitation_recovery.pt"
  RECOVERY_AUDIT="checkpoints/imitation_recovery.audit.json"
  PPO_ABLATION_CHECKPOINT="checkpoints/ppo_negative_ablation_seed${SEED}.zip"
  PPO_ABLATION_FINAL_CHECKPOINT="checkpoints/ppo_negative_ablation_seed${SEED}_final.zip"
  PPO_CHECKPOINT_DIR="checkpoints/ppo_negative_ablation_seed${SEED}"
  PPO_TENSORBOARD_LOG="results/logs/ppo_negative_ablation_seed${SEED}"
  PPO_MONITOR_DIR="results/logs/ppo_negative_ablation_seed${SEED}_monitor"
  PPO_CURVE_PATH="results/figures/ppo_negative_ablation_seed${SEED}_training_curve.png"
  PPO_METRICS_PATH="results/tables/ppo_negative_ablation_seed${SEED}_metrics.json"
  TABLES_DIR="results/tables"
  FIGURES_DIR="results/figures"
  TRACE_PATH="results/recovery_traces.json"
  ROBUSTNESS_TRACE_PATH="results/robustness_recovery_traces.json"
  PAPER_ASSETS_DIR="paper_assets"
  MANIFEST_PATH="results/run_manifest.json"
  RECOVERY_START_THRESHOLD="${REIM_RECOVERY_START_THRESHOLD:-0.1}"
  FAILURE_THRESHOLD="${REIM_FAILURE_THRESHOLD:-0.2}"
  RECOVERY_EXIT_THRESHOLD="${REIM_RECOVERY_EXIT_THRESHOLD:-0.0}"
  RECOVERY_BUDGET="${REIM_RECOVERY_BUDGET:-150}"
  RECOVERY_MIN_STEPS="${REIM_RECOVERY_MIN_STEPS:-150}"
  RECOVERY_CLEAR_STEPS="${REIM_RECOVERY_CLEAR_STEPS:-200}"
else
  DEMO_EPISODES=12
  ACT_EPOCHS=5
  FAILURE_EPISODES=32
  DETECTOR_EPOCHS=5
  PPO_TIMESTEPS=512
  BASELINE_EPISODES=5
  ROBUSTNESS_EPISODES=3
  ABLATION_EPISODES=5
  COMPARISON_EPISODES=5
  MAX_STEPS=100
  BOOTSTRAP_SAMPLES=100
  GATE_SENSITIVITY_EPISODES=0
  GATE_SENSITIVITY_SEED=0
  OPERATION_SEED_START=0

  ACT_CONFIG="configs/smoke/bc.yaml"
  DETECTOR_CONFIG="configs/smoke/detector.yaml"
  RECOVERY_CONFIG="configs/recovery_imitation.yaml"
  PPO_CONFIG="configs/smoke/ppo.yaml"
  DEMO_DIR="datasets/smoke/demonstrations"
  FAILURE_DIR="datasets/smoke/failures"
  RECOVERY_START_TRAIN_PATH=""
  RECOVERY_START_VALIDATION_PATH=""
  ACT_CHECKPOINT="checkpoints/smoke/bc_policy.pt"
  ACT_LATEST="checkpoints/smoke/bc_policy_latest.pt"
  DETECTOR_CHECKPOINT="checkpoints/smoke/failure_detector.pt"
  DETECTOR_LATEST="checkpoints/smoke/failure_detector_latest.pt"
  # Smoke evaluation reads (but never writes) the same audited recovery actor.
  RECOVERY_CHECKPOINT="checkpoints/imitation_recovery.pt"
  RECOVERY_AUDIT="checkpoints/imitation_recovery.audit.json"
  PPO_ABLATION_CHECKPOINT="checkpoints/smoke/ppo_negative_ablation.zip"
  PPO_ABLATION_FINAL_CHECKPOINT="checkpoints/smoke/ppo_negative_ablation_final.zip"
  PPO_CHECKPOINT_DIR="checkpoints/smoke/ppo_negative_ablation"
  PPO_TENSORBOARD_LOG="results/smoke/logs/ppo_negative_ablation"
  PPO_MONITOR_DIR="results/smoke/logs/ppo_negative_ablation_monitor"
  PPO_CURVE_PATH="results/smoke/figures/ppo_negative_ablation_training_curve.png"
  PPO_METRICS_PATH="results/smoke/tables/ppo_negative_ablation_metrics.json"
  TABLES_DIR="results/smoke/tables"
  FIGURES_DIR="results/smoke/figures"
  TRACE_PATH="results/smoke/recovery_traces.json"
  ROBUSTNESS_TRACE_PATH="results/smoke/robustness_recovery_traces.json"
  PAPER_ASSETS_DIR="paper_assets/smoke"
  MANIFEST_PATH="results/smoke/run_manifest.json"
  # Exercise the controller hand-off in a tiny integration run; the outputs
  # are explicitly stamped as toy/smoke, never as benchmark results.
  RECOVERY_START_THRESHOLD=""
  FAILURE_THRESHOLD="${REIM_FAILURE_THRESHOLD:-0.000001}"
  RECOVERY_EXIT_THRESHOLD="${REIM_RECOVERY_EXIT_THRESHOLD:-0.0}"
  RECOVERY_BUDGET="${REIM_RECOVERY_BUDGET:-24}"
  RECOVERY_MIN_STEPS="${REIM_RECOVERY_MIN_STEPS:-1}"
  RECOVERY_CLEAR_STEPS="${REIM_RECOVERY_CLEAR_STEPS:-1}"
fi

BASELINE_PATH="$TABLES_DIR/baseline.csv"
ROBUSTNESS_PATH="$TABLES_DIR/robustness.csv"
ABLATION_PATH="$TABLES_DIR/ablation.csv"
COMPARISON_PATH="$TABLES_DIR/comparison.csv"
DETECTOR_ONLY_PATH="$TABLES_DIR/detector_only.csv"
BASELINE_EPISODE_PATH="${BASELINE_PATH%.csv}_episodes.csv"
ROBUSTNESS_EPISODE_PATH="${ROBUSTNESS_PATH%.csv}_episodes.csv"
ABLATION_EPISODE_PATH="${ABLATION_PATH%.csv}_episodes.csv"
COMPARISON_EPISODE_PATH="${COMPARISON_PATH%.csv}_episodes.csv"
DETECTOR_ONLY_EPISODE_PATH="${DETECTOR_ONLY_PATH%.csv}_episodes.csv"
EVALUATION_REUSE_MANIFEST="$TABLES_DIR/evaluation_reuse_manifest.json"
DETECTOR_METRICS_PATH="$TABLES_DIR/detector_metrics.json"

if [[ "$PROFILE" == "full" ]]; then
  EPISODE_BANK_DIR="datasets/evaluation"
  MAIN_EPISODE_BANK="$EPISODE_BANK_DIR/pickplace_crn_task${TASK_SEED}_ep${EVAL_SEED}_n${BASELINE_EPISODES}_noise020.json"
  GATE_EPISODE_BANK="$EPISODE_BANK_DIR/pickplace_gate_task${TASK_SEED}_ep${GATE_SENSITIVITY_SEED}_n${GATE_SENSITIVITY_EPISODES}_noise020.json"
  ROBUSTNESS_EPISODE_BANKS=(
    "$EPISODE_BANK_DIR/pickplace_crn_task${TASK_SEED}_ep${EVAL_SEED}_n${ROBUSTNESS_EPISODES}_noise000.json"
    "$EPISODE_BANK_DIR/pickplace_crn_task${TASK_SEED}_ep${EVAL_SEED}_n${ROBUSTNESS_EPISODES}_noise010.json"
    "$MAIN_EPISODE_BANK"
    "$EPISODE_BANK_DIR/pickplace_crn_task${TASK_SEED}_ep${EVAL_SEED}_n${ROBUSTNESS_EPISODES}_noise030.json"
    "$EPISODE_BANK_DIR/pickplace_crn_task${TASK_SEED}_ep${EVAL_SEED}_n${ROBUSTNESS_EPISODES}_noise040.json"
  )
  CRN_DETERMINISM_AUDIT="results/audits/final_episode_bank_determinism.json"
  NO_INTERVENTION_AUDIT="results/audits/final_no_intervention_equivalence.json"
  GATE_HEURISTIC_SUMMARY="$TABLES_DIR/formal_crn/gate_matched_heuristic.csv"
  GATE_HEURISTIC_EPISODES="${GATE_HEURISTIC_SUMMARY%.csv}_episodes.csv"
  GATE_MATCHED_AUDIT="$TABLES_DIR/gate_matched_comparison.json"
else
  MAIN_EPISODE_BANK=""
  GATE_EPISODE_BANK=""
  ROBUSTNESS_EPISODE_BANKS=()
  CRN_DETERMINISM_AUDIT=""
  NO_INTERVENTION_AUDIT=""
  GATE_HEURISTIC_SUMMARY=""
  GATE_HEURISTIC_EPISODES=""
  GATE_MATCHED_AUDIT=""
fi

export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1

if [[ "$PROFILE" == "full" ]]; then
  TRAIN_SEED_MAX=$((SEED + FAILURE_EPISODES - 1))
  DEMO_SEED_MAX=$((SEED + 3 * (DEMO_EPISODES - 1)))
  if ((DEMO_SEED_MAX > TRAIN_SEED_MAX)); then
    TRAIN_SEED_MAX=$DEMO_SEED_MAX
  fi
  EVAL_SEED_MAX=$((EVAL_SEED + BASELINE_EPISODES - 1))
  if ((EVAL_SEED <= TRAIN_SEED_MAX && SEED <= EVAL_SEED_MAX)); then
    die "training/data and evaluation seed ranges overlap; choose --eval-seed outside $SEED..$TRAIN_SEED_MAX"
  fi
  RECOVERY_START_TRAIN_SEED_MAX=$(( \
    RECOVERY_START_TRAIN_SEED + RECOVERY_START_TRAIN_EPISODES - 1 \
  ))
  RECOVERY_START_VALIDATION_SEED_MAX=$(( \
    RECOVERY_START_VALIDATION_SEED \
      + RECOVERY_START_VALIDATION_EPISODES \
      - 1 \
  ))
  if ((
    EVAL_SEED <= RECOVERY_START_TRAIN_SEED_MAX
      && RECOVERY_START_TRAIN_SEED <= EVAL_SEED_MAX
  )); then
    die "evaluation seeds overlap recovery training starts; choose --eval-seed outside $RECOVERY_START_TRAIN_SEED..$RECOVERY_START_TRAIN_SEED_MAX"
  fi
  if ((
    EVAL_SEED <= RECOVERY_START_VALIDATION_SEED_MAX
      && RECOVERY_START_VALIDATION_SEED <= EVAL_SEED_MAX
  )); then
    die "evaluation seeds overlap recovery validation starts; choose --eval-seed outside $RECOVERY_START_VALIDATION_SEED..$RECOVERY_START_VALIDATION_SEED_MAX"
  fi
fi

run_step() {
  local title="$1"
  shift
  printf '\n==> %s\n   +' "$title"
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "false" ]]; then
    "$@"
  fi
}

resume_path_for() {
  local latest="$1"
  local final="$2"
  if [[ -f "$latest" ]]; then
    printf '%s' "$latest"
  elif [[ -f "$final" ]]; then
    printf '%s' "$final"
  fi
}

artifacts_complete() {
  local artifact
  for artifact in "$@"; do
    [[ -s "$artifact" ]] || return 1
  done
}

require_artifact() {
  local artifact="$1"
  local purpose="$2"
  if [[ "$DRY_RUN" == "false" && ! -s "$artifact" ]]; then
    die "missing $purpose: $artifact"
  fi
}

validate_recovery_artifact() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf 'Dry-run validation: %s with %s\n' \
      "$RECOVERY_CHECKPOINT" "$RECOVERY_AUDIT"
    return
  fi
  require_artifact "$RECOVERY_CHECKPOINT" "standalone recovery checkpoint"
  require_artifact "$RECOVERY_AUDIT" "standalone recovery audit"
  "$PYTHON_BIN" - "$RECOVERY_CHECKPOINT" "$RECOVERY_AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
expected = str(audit.get("artifact", {}).get("sha256", ""))
actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if audit.get("audit_result") != "pass":
    raise SystemExit(f"recovery audit did not pass: {audit_path}")
if not expected or actual != expected:
    raise SystemExit(
        f"recovery artifact hash mismatch: expected={expected}, actual={actual}"
    )
equivalence = audit.get("equivalence", {})
if not equivalence.get("all_splits_within_tolerance", False):
    raise SystemExit("recovery action-equivalence audit did not pass")
source = audit.get("source", {})
if int(source.get("num_timesteps", -1)) != 0 or int(
    source.get("n_updates", -1)
) != 0:
    raise SystemExit("formal recovery actor contains PPO interaction/update history")
provenance = audit.get("provenance", {})
input_paths = provenance.get("input_paths", {})
input_hashes = provenance.get("input_sha256", {})
for name, relative_path in input_paths.items():
    input_path = Path(relative_path)
    expected_input_hash = input_hashes.get(name)
    if not input_path.is_file() or not expected_input_hash:
        raise SystemExit(f"missing frozen recovery provenance input: {input_path}")
    actual_input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if actual_input_hash != expected_input_hash:
        raise SystemExit(
            f"recovery provenance hash mismatch for {name}: "
            f"expected={expected_input_hash}, actual={actual_input_hash}"
        )
source_path = Path(str(source.get("checkpoint", "")))
source_hash = str(source.get("checkpoint_sha256", ""))
if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
    raise SystemExit("frozen zero-step source checkpoint is missing or changed")
print(
    "Validated standalone recovery actor: "
    f"sha256={actual}, states={equivalence.get('total_states')}, PPO steps=0"
)
PY
}

episode_evidence_matches() {
  local csv_path="$1"
  local seed_start="$2"
  local episodes="$3"
  local require_bank="${4:-false}"
  [[ -s "$csv_path" ]] || return 1
  "$PYTHON_BIN" - "$csv_path" "$seed_start" "$episodes" "$require_bank" \
    >/dev/null 2>&1 <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed_start = int(sys.argv[2])
episodes = int(sys.argv[3])
require_bank = sys.argv[4].lower() == "true"
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(1)
seeds = {int(row["seed"]) for row in rows}
expected = set(range(seed_start, seed_start + episodes))
if seeds != expected:
    raise SystemExit(1)
if require_bank:
    for row in rows:
        if not row.get("episode_bank_sha256"):
            raise SystemExit(1)
        if not row.get("episode_specification_sha256"):
            raise SystemExit(1)
PY
}

audit_matches_bank() {
  local audit_path="$1"
  local bank_path="$2"
  local audit_kind="$3"
  [[ -s "$audit_path" && -s "$bank_path" ]] || return 1
  "$PYTHON_BIN" - "$audit_path" "$bank_path" "$audit_kind" \
    >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
bank_path = Path(sys.argv[2]).resolve()
kind = sys.argv[3]
audit = json.loads(audit_path.read_text(encoding="utf-8"))
bank = json.loads(bank_path.read_text(encoding="utf-8"))
expected_sha = bank["bank_sha256"]
if kind == "determinism":
    passed = audit.get("all_checks_passed") is True
    recorded_path = audit.get("episode_bank")
    recorded_sha = audit.get("episode_bank_sha256")
else:
    passed = audit.get("passed") is True and audit.get("status") == "passed"
    config = audit.get("configuration", {})
    recorded_path = config.get("episode_bank")
    recorded_sha = config.get("episode_bank_sha256")
if not passed or Path(str(recorded_path)).resolve() != bank_path:
    raise SystemExit(1)
if recorded_sha != expected_sha:
    raise SystemExit(1)
PY
}

reuse_manifest_matches() {
  local manifest="$1"
  shift
  [[ -s "$manifest" ]] || return 1
  "$PYTHON_BIN" - "$manifest" "$EVAL_SEED" "$BASELINE_EPISODES" "$@" \
    >/dev/null 2>&1 <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
seed = int(sys.argv[2])
episodes = int(sys.argv[3])
inputs = [Path(value) for value in sys.argv[4:]]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("strategy") != "paired_episode_reuse":
    raise SystemExit(1)
if int(manifest.get("evaluation_seed_start", -1)) != seed:
    raise SystemExit(1)
if int(manifest.get("episodes_per_condition", -1)) != episodes:
    raise SystemExit(1)
recorded = manifest.get("inputs", {})
for path in inputs:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates = (str(path), str(path.resolve()))
    expected = next((recorded[key] for key in candidates if key in recorded), None)
    if expected != digest:
        raise SystemExit(1)
PY
}

DATA_RESUME_ARGS=()
if [[ "$RESUME" == "true" ]]; then
  DATA_RESUME_ARGS=(--resume)
fi

printf 'REIM protocol: profile=%s backend=%s train-seed=%s eval-seed=%s task-seed=%s\n' \
  "$PROFILE" "$BACKEND" "$SEED" "$EVAL_SEED" "$TASK_SEED"
printf 'Python: %s | train device: %s | eval device: %s | torch threads: %s\n' \
  "$PYTHON_BIN" "$TRAIN_DEVICE" "$EVAL_DEVICE" "$TORCH_THREADS"
printf 'Recovery: audited standalone trigger-aligned imitation actor (zero PPO steps)\n'
if [[ "$RUN_PPO_ABLATION" == "true" ]]; then
  printf 'Optional negative ablation: PPO enabled on device %s; formal evaluation remains actor-only.\n' \
    "$RECOVERY_DEVICE"
fi
printf 'Controller: trigger=%s exit=%s budget=%s min-steps=%s clear-steps=%s\n' \
  "$FAILURE_THRESHOLD" "$RECOVERY_EXIT_THRESHOLD" "$RECOVERY_BUDGET" \
  "$RECOVERY_MIN_STEPS" "$RECOVERY_CLEAR_STEPS"
if [[ "$PROFILE" == "full" ]]; then
  printf 'Recovery curriculum: trigger-state threshold=%s\n' \
    "$RECOVERY_START_THRESHOLD"
  printf 'Formal CRN bank: %s\n' "$MAIN_EPISODE_BANK"
fi
if [[ "$DRY_RUN" == "true" ]]; then
  printf 'Dry run: no commands will be executed.\n'
fi

if [[ "$SKIP_DEMOS" == "false" ]]; then
  run_step "Collect expert demonstrations ($DEMO_EPISODES)" \
    "$PYTHON_BIN" scripts/collect_demonstrations.py \
    --config configs/environment.yaml \
    --backend "$BACKEND" \
    --episodes "$DEMO_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --output-dir "$DEMO_DIR" \
    "${DATA_RESUME_ARGS[@]}"
fi

if [[ "$SKIP_ACT" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] && artifacts_complete "$ACT_CHECKPOINT"; then
    printf 'Resume: completed ACT checkpoint exists; skipping %s.\n' \
      "$ACT_CHECKPOINT"
  else
    ACT_RESUME_ARGS=()
    if [[ "$RESUME" == "true" ]]; then
      ACT_RESUME_PATH="$(resume_path_for "$ACT_LATEST" "")"
      if [[ -n "$ACT_RESUME_PATH" ]]; then
        ACT_RESUME_ARGS=(--resume "$ACT_RESUME_PATH")
      else
        printf 'No partial ACT checkpoint found; starting ACT training from scratch.\n'
      fi
    fi
    run_step "Train state-based ACT ($ACT_EPOCHS epochs)" \
      "$PYTHON_BIN" trainers/train_bc.py \
      --config "$ACT_CONFIG" \
      --data "$DEMO_DIR" \
      --output "$ACT_CHECKPOINT" \
      --epochs "$ACT_EPOCHS" \
      --seed "$SEED" \
      --device "$TRAIN_DEVICE" \
      "${ACT_RESUME_ARGS[@]}"
  fi
fi

if [[ "$SKIP_FAILURES" == "false" ]]; then
  run_step "Generate causal failure rollouts ($FAILURE_EPISODES)" \
    "$PYTHON_BIN" scripts/generate_failures.py \
    --config configs/environment.yaml \
    --backend "$BACKEND" \
    --episodes "$FAILURE_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --output-dir "$FAILURE_DIR" \
    --policy checkpoint \
    --checkpoint "$ACT_CHECKPOINT" \
    --device "$TRAIN_DEVICE" \
    --seed "$SEED" \
    "${DATA_RESUME_ARGS[@]}"
fi

if [[ "$SKIP_DETECTOR" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete "$DETECTOR_CHECKPOINT" "$DETECTOR_METRICS_PATH"; then
    printf 'Resume: completed detector artifacts exist; skipping %s.\n' \
      "$DETECTOR_CHECKPOINT"
  else
    DETECTOR_RESUME_ARGS=()
    if [[ "$RESUME" == "true" ]]; then
      DETECTOR_RESUME_PATH="$(resume_path_for "$DETECTOR_LATEST" "")"
      if [[ -n "$DETECTOR_RESUME_PATH" ]]; then
        DETECTOR_RESUME_ARGS=(--resume "$DETECTOR_RESUME_PATH")
      else
        printf 'No partial detector checkpoint found; starting detector training from scratch.\n'
      fi
    fi
    run_step "Train LSTM failure detector ($DETECTOR_EPOCHS epochs)" \
      "$PYTHON_BIN" trainers/train_detector.py \
      --config "$DETECTOR_CONFIG" \
      --data "$FAILURE_DIR" \
      --output "$DETECTOR_CHECKPOINT" \
      --epochs "$DETECTOR_EPOCHS" \
      --seed "$SEED" \
      --device "$TRAIN_DEVICE" \
      "${DETECTOR_RESUME_ARGS[@]}"
  fi
fi

collect_recovery_start_split() {
  local title="$1"
  local episodes="$2"
  local seed="$3"
  local output="$4"
  local resume_args=()
  if [[ "$RESUME" == "true" ]]; then
    resume_args=(--resume)
  fi
  run_step "$title" \
    "$PYTHON_BIN" scripts/collect_recovery_starts.py \
    --backend metaworld \
    --env-config configs/environment.yaml \
    --episodes "$episodes" \
    --max-steps "$MAX_STEPS" \
    --failure-threshold "$RECOVERY_START_THRESHOLD" \
    --noise-level 0.2 \
    --act-checkpoint "$ACT_CHECKPOINT" \
    --detector-checkpoint "$DETECTOR_CHECKPOINT" \
    --device "$TRAIN_DEVICE" \
    --seed "$seed" \
    --output "$output" \
    "${resume_args[@]}"
}

if [[ "$PROFILE" == "full" && "$SKIP_RECOVERY_STARTS" == "false" ]]; then
  collect_recovery_start_split \
    "Collect exact ACT/LSTM trigger states (training)" \
    "$RECOVERY_START_TRAIN_EPISODES" \
    "$RECOVERY_START_TRAIN_SEED" \
    "$RECOVERY_START_TRAIN_PATH"
  collect_recovery_start_split \
    "Collect exact ACT/LSTM trigger states (validation)" \
    "$RECOVERY_START_VALIDATION_EPISODES" \
    "$RECOVERY_START_VALIDATION_SEED" \
    "$RECOVERY_START_VALIDATION_PATH"
fi

if [[ "$SKIP_RECOVERY" == "false" ]]; then
  if [[ "$PROFILE" == "full" ]]; then
    if artifacts_complete "$RECOVERY_CHECKPOINT" "$RECOVERY_AUDIT"; then
      printf 'Frozen standalone recovery artifacts exist; reusing %s.\n' \
        "$RECOVERY_CHECKPOINT"
    else
      run_step "Export and audit standalone imitation recovery actor" \
        "$PYTHON_BIN" scripts/export_imitation_recovery.py \
        --config "$RECOVERY_CONFIG"
    fi
  else
    printf 'Smoke profile reuses the frozen standalone recovery actor read-only.\n'
  fi
  validate_recovery_artifact
elif [[
  "$SKIP_BASELINE" == "false"
    || "$SKIP_ROBUSTNESS" == "false"
    || "$SKIP_ABLATION" == "false"
    || "$SKIP_COMPARISON" == "false"
    || "$SKIP_GATE_SENSITIVITY" == "false"
    || "$SKIP_OPERATION_FIGURE" == "false"
]]; then
  validate_recovery_artifact
fi

if [[ "$RUN_PPO_ABLATION" == "true" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete "$PPO_ABLATION_CHECKPOINT" "$PPO_METRICS_PATH"; then
    printf 'Resume: completed optional PPO negative ablation exists; skipping %s.\n' \
      "$PPO_ABLATION_CHECKPOINT"
  else
    PPO_RESUME_ARGS=()
    if [[ "$RESUME" == "true" ]]; then
      PPO_RESUME_PATH=""
      if [[ -f "$PPO_ABLATION_FINAL_CHECKPOINT" ]]; then
        PPO_RESUME_PATH="$PPO_ABLATION_FINAL_CHECKPOINT"
      else
        shopt -s nullglob
        PPO_SNAPSHOTS=("$PPO_CHECKPOINT_DIR"/recovery_*_steps.zip)
        shopt -u nullglob
        if ((${#PPO_SNAPSHOTS[@]})); then
          PPO_RESUME_PATH="$(
            printf '%s\n' "${PPO_SNAPSHOTS[@]}" | sort -V | tail -n 1
          )"
        fi
      fi
      if [[ -n "$PPO_RESUME_PATH" ]]; then
        PPO_RESUME_ARGS=(--resume "$PPO_RESUME_PATH")
      else
        printf 'No partial PPO ablation checkpoint found; starting from scratch.\n'
      fi
    fi
    PPO_SMOKE_ARGS=()
    if [[ "$PROFILE" == "smoke" ]]; then
      PPO_SMOKE_ARGS=(
        --n-steps 64
        --batch-size 32
        --n-epochs 2
        --eval-freq 256
        --checkpoint-freq 256
        --eval-episodes 2
        --no-progress-bar
      )
    fi
    run_step \
      "Optional PPO recovery negative ablation ($PPO_TIMESTEPS environment steps)" \
      "$PYTHON_BIN" trainers/train_recovery.py \
      --config "$PPO_CONFIG" \
      --env-config configs/environment.yaml \
      --backend "$BACKEND" \
      --state-mode semantic \
      --output "$PPO_ABLATION_CHECKPOINT" \
      --checkpoint-dir "$PPO_CHECKPOINT_DIR" \
      --tensorboard-log "$PPO_TENSORBOARD_LOG" \
      --monitor-dir "$PPO_MONITOR_DIR" \
      --curve-path "$PPO_CURVE_PATH" \
      --metrics-path "$PPO_METRICS_PATH" \
      --total-timesteps "$PPO_TIMESTEPS" \
      --seed "$SEED" \
      --device "$RECOVERY_DEVICE" \
      "${PPO_SMOKE_ARGS[@]}" \
      "${PPO_RESUME_ARGS[@]}"
  fi
fi

ensure_episode_bank() {
  local title="$1"
  local episodes="$2"
  local noise_level="$3"
  local output="$4"
  local episode_seed_start="${5:-$EVAL_SEED}"
  if artifacts_complete "$output"; then
    printf 'Frozen CRN bank exists; reusing %s.\n' "$output"
    return
  fi
  run_step "$title" \
    "$PYTHON_BIN" scripts/generate_episode_bank.py \
    --backend metaworld \
    --env-name pick-place-v3 \
    --task-bank-seed "$TASK_SEED" \
    --episode-seed-start "$episode_seed_start" \
    --episodes "$episodes" \
    --max-steps "$MAX_STEPS" \
    --retries-per-episode 1 \
    --noise-level "$noise_level" \
    --output "$output"
}

if [[ "$PROFILE" == "full" && "$SKIP_EPISODE_BANK" == "false" ]]; then
  ensure_episode_bank \
    "Generate primary 20% CRN episode bank ($BASELINE_EPISODES episodes)" \
    "$BASELINE_EPISODES" 0.2 "$MAIN_EPISODE_BANK"
  ensure_episode_bank \
    "Generate 0% robustness CRN episode bank ($ROBUSTNESS_EPISODES episodes)" \
    "$ROBUSTNESS_EPISODES" 0.0 "${ROBUSTNESS_EPISODE_BANKS[0]}"
  ensure_episode_bank \
    "Generate 10% robustness CRN episode bank ($ROBUSTNESS_EPISODES episodes)" \
    "$ROBUSTNESS_EPISODES" 0.1 "${ROBUSTNESS_EPISODE_BANKS[1]}"
  ensure_episode_bank \
    "Generate 30% robustness CRN episode bank ($ROBUSTNESS_EPISODES episodes)" \
    "$ROBUSTNESS_EPISODES" 0.3 "${ROBUSTNESS_EPISODE_BANKS[3]}"
  ensure_episode_bank \
    "Generate 40% robustness CRN episode bank ($ROBUSTNESS_EPISODES episodes)" \
    "$ROBUSTNESS_EPISODES" 0.4 "${ROBUSTNESS_EPISODE_BANKS[4]}"
  if [[ "$SKIP_GATE_SENSITIVITY" == "false" ]]; then
    ensure_episode_bank \
      "Generate post-freeze gate CRN episode bank ($GATE_SENSITIVITY_EPISODES episodes)" \
      "$GATE_SENSITIVITY_EPISODES" 0.2 "$GATE_EPISODE_BANK" \
      "$GATE_SENSITIVITY_SEED"
  fi
fi

if [[ "$PROFILE" == "full" && "$SKIP_BASELINE" == "false" ]]; then
  require_artifact "$MAIN_EPISODE_BANK" "primary CRN episode bank"
fi
if [[
  "$PROFILE" == "full"
    && (
      "$SKIP_ABLATION" == "false"
        || "$SKIP_COMPARISON" == "false"
    )
]]; then
  require_artifact "$MAIN_EPISODE_BANK" "primary CRN episode bank"
fi
if [[ "$PROFILE" == "full" && "$SKIP_ROBUSTNESS" == "false" ]]; then
  for episode_bank in "${ROBUSTNESS_EPISODE_BANKS[@]}"; do
    require_artifact "$episode_bank" "robustness CRN episode bank"
  done
fi
if [[
  "$PROFILE" == "full"
    && "$SKIP_GATE_SENSITIVITY" == "false"
]]; then
  require_artifact "$GATE_EPISODE_BANK" "post-freeze gate CRN episode bank"
fi

if [[ "$PROFILE" == "full" && "$SKIP_AUDITS" == "false" ]]; then
  require_artifact "$MAIN_EPISODE_BANK" "primary CRN episode bank"
  require_artifact "$ACT_CHECKPOINT" "ACT checkpoint required by CRN audit"
  if [[ "$RESUME" == "true" ]] \
    && audit_matches_bank \
      "$CRN_DETERMINISM_AUDIT" "$MAIN_EPISODE_BANK" determinism; then
    printf 'Resume: matching CRN determinism audit exists; skipping %s.\n' \
      "$CRN_DETERMINISM_AUDIT"
  else
    run_step "Audit exact CRN replay across constructors and shard boundaries" \
      "$PYTHON_BIN" scripts/audit_episode_bank_determinism.py \
      --episode-bank "$MAIN_EPISODE_BANK" \
      --env-config configs/environment.yaml \
      --state-mode semantic \
      --steps 8 \
      --constructor-seeds "$TASK_SEED" "$((TASK_SEED + 250))" \
      --output "$CRN_DETERMINISM_AUDIT"
  fi
  if [[ "$RESUME" == "true" ]] \
    && audit_matches_bank \
      "$NO_INTERVENTION_AUDIT" "$MAIN_EPISODE_BANK" no-intervention; then
    printf 'Resume: matching no-intervention equivalence audit exists; skipping %s.\n' \
      "$NO_INTERVENTION_AUDIT"
  else
    run_step "Audit bitwise ACT/REIM equivalence when recovery never triggers" \
      "$PYTHON_BIN" scripts/audit_no_intervention_equivalence.py \
      --episode-bank "$MAIN_EPISODE_BANK" \
      --act-checkpoint "$ACT_CHECKPOINT" \
      --output "$NO_INTERVENTION_AUDIT" \
      --offset 0 \
      --count 8 \
      --constructor-seed-act "$TASK_SEED" \
      --constructor-seed-reim "$((EVAL_SEED + 1000049))" \
      --device cpu \
      --torch-threads "$TORCH_THREADS" \
      --env-config configs/environment.yaml
  fi
fi

COMMON_EVAL_ARGS=(
  --profile "$PROFILE"
  --backend "$BACKEND"
  --env-config configs/environment.yaml
  --checkpoint "$ACT_CHECKPOINT"
  --detector-checkpoint "$DETECTOR_CHECKPOINT"
  --recovery-checkpoint "$RECOVERY_CHECKPOINT"
  --device "$EVAL_DEVICE"
  --max-steps "$MAX_STEPS"
  --failure-threshold "$FAILURE_THRESHOLD"
  --recovery-exit-threshold "$RECOVERY_EXIT_THRESHOLD"
  --recovery-budget "$RECOVERY_BUDGET"
  --recovery-min-steps "$RECOVERY_MIN_STEPS"
  --recovery-clear-steps "$RECOVERY_CLEAR_STEPS"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --seed "$EVAL_SEED"
)

BASELINE_BANK_ARGS=()
ROBUSTNESS_BANK_ARGS=()
REQUIRE_BANK_EVIDENCE=false
if [[ "$PROFILE" == "full" ]]; then
  REQUIRE_BANK_EVIDENCE=true
  BASELINE_BANK_ARGS=(
    --episode-bank "$MAIN_EPISODE_BANK"
    --episode-offset 0
  )
  ROBUSTNESS_BANK_ARGS=(--episode-offset 0)
  for episode_bank in "${ROBUSTNESS_EPISODE_BANKS[@]}"; do
    ROBUSTNESS_BANK_ARGS+=(--episode-bank "$episode_bank")
  done
fi

if [[ "$SKIP_BASELINE" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete \
      "$BASELINE_PATH" "$BASELINE_EPISODE_PATH" "$TRACE_PATH" \
    && episode_evidence_matches \
      "$BASELINE_EPISODE_PATH" "$EVAL_SEED" "$BASELINE_EPISODES" \
      "$REQUIRE_BANK_EVIDENCE"; then
    printf 'Resume: complete baseline summary/raw evidence exists; skipping %s.\n' \
      "$BASELINE_PATH"
  else
    run_step "Evaluate four baselines ($BASELINE_EPISODES episodes/method)" \
      "$PYTHON_BIN" evaluation/baseline.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --torch-threads "$TORCH_THREADS" \
      "${BASELINE_BANK_ARGS[@]}" \
      --methods act act_random_reset act_recovery reim \
      --episodes "$BASELINE_EPISODES" \
      --noise-level 0.2 \
      --output "$BASELINE_PATH" \
      --trace-output "$TRACE_PATH" \
      --capture-traces 4
  fi
fi

if [[ "$SKIP_ROBUSTNESS" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete "$ROBUSTNESS_PATH" "$ROBUSTNESS_EPISODE_PATH" \
    && episode_evidence_matches \
      "$ROBUSTNESS_EPISODE_PATH" "$EVAL_SEED" "$ROBUSTNESS_EPISODES" \
      "$REQUIRE_BANK_EVIDENCE"; then
    printf 'Resume: complete robustness summary/raw evidence exists; skipping %s.\n' \
      "$ROBUSTNESS_PATH"
  else
    run_step \
      "Evaluate robustness (5 levels x $ROBUSTNESS_EPISODES episodes/method)" \
      "$PYTHON_BIN" experiments/robustness.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --torch-threads "$TORCH_THREADS" \
      "${ROBUSTNESS_BANK_ARGS[@]}" \
      --methods act reim \
      --noise-levels 0.0 0.1 0.2 0.3 0.4 \
      --episodes "$ROBUSTNESS_EPISODES" \
      --output "$ROBUSTNESS_PATH" \
      --figure "$FIGURES_DIR/robustness.png" \
      --trace-output "$ROBUSTNESS_TRACE_PATH" \
      --capture-traces 4
  fi
fi

if [[
  "$PROFILE" == "full"
    && (
      "$SKIP_ABLATION" == "false"
        || "$SKIP_COMPARISON" == "false"
    )
]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete \
      "$DETECTOR_ONLY_PATH" "$DETECTOR_ONLY_EPISODE_PATH" \
    && episode_evidence_matches \
      "$DETECTOR_ONLY_EPISODE_PATH" "$EVAL_SEED" "$ABLATION_EPISODES" \
      "$REQUIRE_BANK_EVIDENCE"; then
    printf 'Resume: paired detector-only raw evidence exists; skipping %s.\n' \
      "$DETECTOR_ONLY_PATH"
  else
    run_step \
      "Evaluate detector-only component on the primary CRN bank ($ABLATION_EPISODES episodes)" \
      "$PYTHON_BIN" evaluation/baseline.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --torch-threads "$TORCH_THREADS" \
      "${BASELINE_BANK_ARGS[@]}" \
      --methods act_detector \
      --episodes "$ABLATION_EPISODES" \
      --noise-level 0.2 \
      --output "$DETECTOR_ONLY_PATH" \
      --trace-output "results/detector_only_traces.json" \
      --capture-traces 0
  fi

  require_artifact "$BASELINE_PATH" \
    "canonical baseline required for paired evidence reuse"
  require_artifact "$BASELINE_EPISODE_PATH" \
    "canonical baseline raw evidence required for paired reuse"
  require_artifact "$DETECTOR_ONLY_PATH" \
    "detector-only summary required for ablation"
  require_artifact "$DETECTOR_ONLY_EPISODE_PATH" \
    "detector-only raw evidence required for ablation"

  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete \
      "$ABLATION_PATH" "$ABLATION_EPISODE_PATH" \
      "$COMPARISON_PATH" "$COMPARISON_EPISODE_PATH" \
    && episode_evidence_matches \
      "$ABLATION_EPISODE_PATH" "$EVAL_SEED" "$ABLATION_EPISODES" \
      "$REQUIRE_BANK_EVIDENCE" \
    && episode_evidence_matches \
      "$COMPARISON_EPISODE_PATH" "$EVAL_SEED" "$COMPARISON_EPISODES" \
      "$REQUIRE_BANK_EVIDENCE" \
    && reuse_manifest_matches \
      "$EVALUATION_REUSE_MANIFEST" \
      "$BASELINE_PATH" "$BASELINE_EPISODE_PATH" \
      "$DETECTOR_ONLY_PATH" "$DETECTOR_ONLY_EPISODE_PATH"; then
    printf 'Resume: audited paired ablation/comparison assembly is current; skipping rebuild.\n'
  else
    run_step "Assemble paired ablation and comparison from measured CRN evidence" \
      "$PYTHON_BIN" scripts/assemble_cached_evaluations.py \
      --baseline "$BASELINE_PATH" \
      --baseline-episodes "$BASELINE_EPISODE_PATH" \
      --detector-only "$DETECTOR_ONLY_PATH" \
      --detector-only-episodes "$DETECTOR_ONLY_EPISODE_PATH" \
      --ablation "$ABLATION_PATH" \
      --comparison "$COMPARISON_PATH" \
      --manifest "$EVALUATION_REUSE_MANIFEST" \
      --episodes "$ABLATION_EPISODES" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --seed "$EVAL_SEED"
  fi
fi

if [[ "$PROFILE" == "smoke" && "$SKIP_ABLATION" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete "$ABLATION_PATH" "$ABLATION_EPISODE_PATH" \
    && episode_evidence_matches \
      "$ABLATION_EPISODE_PATH" "$EVAL_SEED" "$ABLATION_EPISODES"; then
    printf 'Resume: complete smoke ablation evidence exists; skipping %s.\n' \
      "$ABLATION_PATH"
  else
    run_step "Evaluate smoke four-way ablation ($ABLATION_EPISODES episodes)" \
      "$PYTHON_BIN" experiments/ablation.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --episodes "$ABLATION_EPISODES" \
      --noise-level 0.2 \
      --output "$ABLATION_PATH" \
      --figure "$FIGURES_DIR/ablation.png"
  fi
fi

if [[ "$PROFILE" == "smoke" && "$SKIP_COMPARISON" == "false" ]]; then
  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete "$COMPARISON_PATH" "$COMPARISON_EPISODE_PATH" \
    && episode_evidence_matches \
      "$COMPARISON_EPISODE_PATH" "$EVAL_SEED" "$COMPARISON_EPISODES"; then
    printf 'Resume: complete smoke comparison evidence exists; skipping %s.\n' \
      "$COMPARISON_PATH"
  else
    run_step "Evaluate smoke paired comparison ($COMPARISON_EPISODES episodes)" \
      "$PYTHON_BIN" experiments/comparison.py \
      "${COMMON_EVAL_ARGS[@]}" \
      --methods act act_random_reset act_recovery reim \
      --episodes "$COMPARISON_EPISODES" \
      --noise-level 0.2 \
      --output "$COMPARISON_PATH"
  fi
fi

if [[
  "$PROFILE" == "full"
    && "$SKIP_GATE_SENSITIVITY" == "false"
]]; then
  GATE_THRESHOLDS=(0.100 0.125 0.150 0.175 0.200)
  GATE_STEMS=(tau010 tau0125 tau015 tau0175 tau020)
  for gate_index in "${!GATE_THRESHOLDS[@]}"; do
    gate_threshold="${GATE_THRESHOLDS[$gate_index]}"
    gate_stem="${GATE_STEMS[$gate_index]}"
    gate_output="$TABLES_DIR/gate_calibration_${gate_stem}.csv"
    gate_episode_output="${gate_output%.csv}_episodes.csv"
    if [[ "$RESUME" == "true" ]] \
      && artifacts_complete "$gate_output" "$gate_episode_output" \
      && episode_evidence_matches \
        "$gate_episode_output" "$GATE_SENSITIVITY_SEED" \
        "$GATE_SENSITIVITY_EPISODES" true; then
      printf 'Resume: complete tau=%s gate evidence exists; skipping %s.\n' \
        "$gate_threshold" "$gate_output"
    else
      run_step \
        "Post-freeze gate diagnostic tau=$gate_threshold ($GATE_SENSITIVITY_EPISODES episodes)" \
        "$PYTHON_BIN" evaluation/baseline.py \
        --methods reim \
        --profile full \
        --backend metaworld \
        --env-config configs/environment.yaml \
        --checkpoint "$ACT_CHECKPOINT" \
        --detector-checkpoint "$DETECTOR_CHECKPOINT" \
        --recovery-checkpoint "$RECOVERY_CHECKPOINT" \
        --device "$EVAL_DEVICE" \
        --torch-threads "$TORCH_THREADS" \
        --episodes "$GATE_SENSITIVITY_EPISODES" \
        --seed "$GATE_SENSITIVITY_SEED" \
        --max-steps "$MAX_STEPS" \
        --noise-level 0.2 \
        --episode-bank "$GATE_EPISODE_BANK" \
        --episode-offset 0 \
        --failure-threshold "$gate_threshold" \
        --recovery-exit-threshold "$RECOVERY_EXIT_THRESHOLD" \
        --recovery-budget "$RECOVERY_BUDGET" \
        --recovery-min-steps "$RECOVERY_MIN_STEPS" \
        --recovery-clear-steps "$RECOVERY_CLEAR_STEPS" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
        --output "$gate_output" \
        --trace-output "results/traces/gate_calibration_${gate_stem}.json" \
        --capture-traces 0
    fi
  done

  if [[ "$RESUME" == "true" ]] \
    && artifacts_complete \
      "$GATE_HEURISTIC_SUMMARY" "$GATE_HEURISTIC_EPISODES" \
    && episode_evidence_matches \
      "$GATE_HEURISTIC_EPISODES" "$GATE_SENSITIVITY_SEED" \
      "$GATE_SENSITIVITY_EPISODES" true; then
    printf 'Resume: matched semantic-gate evidence exists; skipping %s.\n' \
      "$GATE_HEURISTIC_SUMMARY"
  else
    run_step \
      "Evaluate semantic heuristic on the post-freeze gate CRN bank" \
      "$PYTHON_BIN" evaluation/baseline.py \
      --methods act_recovery \
      --profile full \
      --backend metaworld \
      --env-config configs/environment.yaml \
      --checkpoint "$ACT_CHECKPOINT" \
      --detector-checkpoint "$DETECTOR_CHECKPOINT" \
      --recovery-checkpoint "$RECOVERY_CHECKPOINT" \
      --device "$EVAL_DEVICE" \
      --torch-threads "$TORCH_THREADS" \
      --episodes "$GATE_SENSITIVITY_EPISODES" \
      --seed "$GATE_SENSITIVITY_SEED" \
      --max-steps "$MAX_STEPS" \
      --noise-level 0.2 \
      --episode-bank "$GATE_EPISODE_BANK" \
      --episode-offset 0 \
      --failure-threshold "$FAILURE_THRESHOLD" \
      --recovery-exit-threshold "$RECOVERY_EXIT_THRESHOLD" \
      --recovery-budget "$RECOVERY_BUDGET" \
      --recovery-min-steps "$RECOVERY_MIN_STEPS" \
      --recovery-clear-steps "$RECOVERY_CLEAR_STEPS" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --output "$GATE_HEURISTIC_SUMMARY" \
      --trace-output "results/traces/gate_matched_heuristic.json" \
      --capture-traces 0
  fi

  run_step "Audit matched learned-vs-semantic gate comparison" \
    "$PYTHON_BIN" scripts/audit_matched_gate.py \
    --heuristic "$GATE_HEURISTIC_EPISODES" \
    --tau-0175 "$TABLES_DIR/gate_calibration_tau0175_episodes.csv" \
    --tau-020 "$TABLES_DIR/gate_calibration_tau020_episodes.csv" \
    --episode-bank "$GATE_EPISODE_BANK" \
    --episodes "$GATE_SENSITIVITY_EPISODES" \
    --task-bank-seed "$TASK_SEED" \
    --seed-start "$GATE_SENSITIVITY_SEED" \
    --noise-level 0.2 \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --bootstrap-seed "$GATE_SENSITIVITY_SEED" \
    --output "$GATE_MATCHED_AUDIT"
fi

if [[ "$SKIP_PLOTS" == "false" ]]; then
  require_artifact "$BASELINE_PATH" "baseline summary for plotting"
  require_artifact "$ROBUSTNESS_PATH" "robustness summary for plotting"
  require_artifact "$ABLATION_PATH" "ablation summary for plotting"
  require_artifact "$DETECTOR_METRICS_PATH" "detector metrics for plotting"
  require_artifact "$TRACE_PATH" "recovery traces for plotting"
  run_step "Generate figures, LaTeX tables, and paper assets" \
    "$PYTHON_BIN" visualization/plot_results.py \
    --baseline "$BASELINE_PATH" \
    --robustness "$ROBUSTNESS_PATH" \
    --ablation "$ABLATION_PATH" \
    --detector-metrics "$DETECTOR_METRICS_PATH" \
    --traces "$TRACE_PATH" \
    --figures-dir "$FIGURES_DIR" \
    --paper-assets-dir "$PAPER_ASSETS_DIR" \
    --only all
  if [[ "$PROFILE" == "full" ]]; then
    if [[ "$SKIP_GATE_SENSITIVITY" == "false" ]]; then
      run_step "Audit and plot post-freeze gate sensitivity" \
        "$PYTHON_BIN" experiments/gate_sensitivity.py \
        --input-dir "$TABLES_DIR" \
        --episodes "$GATE_SENSITIVITY_EPISODES" \
        --seed-start "$GATE_SENSITIVITY_SEED"
    fi
    if [[ "$SKIP_OPERATION_FIGURE" == "false" ]]; then
      if [[ "$RESUME" == "true" ]] \
        && artifacts_complete \
          "$FIGURES_DIR/recovery_operation_sequence.png" \
          "$FIGURES_DIR/recovery_operation_sequence.pdf" \
          "$FIGURES_DIR/recovery_operation_sequence.json" \
          "$PAPER_ASSETS_DIR/Figure5_operation_sequence.png" \
          "$PAPER_ASSETS_DIR/Figure5_operation_sequence.pdf"; then
        printf 'Resume: simulator-rendered operation sequence exists; skipping capture.\n'
      else
        run_step "Render paired ACT-failure/REIM-success simulator sequence" \
          "$PYTHON_BIN" visualization/capture_recovery_rollout.py \
          --seed "$OPERATION_SEED_START" \
          --max-search 200 \
          --noise-level 0.2 \
          --failure-threshold "$FAILURE_THRESHOLD" \
          --recovery-exit-threshold "$RECOVERY_EXIT_THRESHOLD" \
          --recovery-budget "$RECOVERY_BUDGET" \
          --recovery-min-steps "$RECOVERY_MIN_STEPS" \
          --recovery-clear-steps "$RECOVERY_CLEAR_STEPS" \
          --act-checkpoint "$ACT_CHECKPOINT" \
          --detector-checkpoint "$DETECTOR_CHECKPOINT" \
          --recovery-checkpoint "$RECOVERY_CHECKPOINT" \
          --device "$EVAL_DEVICE" \
          --mujoco-gl "${REIM_MUJOCO_GL:-egl}" \
          --output-dir "$FIGURES_DIR" \
          --paper-assets-dir "$PAPER_ASSETS_DIR"
      fi
    fi
    require_artifact "$COMPARISON_PATH" "paired comparison for paper validation"
    require_artifact "$COMPARISON_EPISODE_PATH" \
      "paired comparison raw evidence for paper validation"
    run_step "Validate per-episode evidence and update frozen paper values" \
      "$PYTHON_BIN" scripts/update_paper_results.py \
      --expected-evaluation-seed "$EVAL_SEED"
    run_step "Compile paper results PDF" ./compile_paper.sh
    require_artifact "$PAPER_ASSETS_DIR/reim_results.pdf" \
      "compiled publication PDF"
  fi
fi

if [[ "$SKIP_MANIFEST" == "false" ]]; then
  run_step "Write reproducibility manifest" \
    "$PYTHON_BIN" scripts/reproducibility_manifest.py \
    --profile "$PROFILE" \
    --backend "$BACKEND" \
    --train-seed "$SEED" \
    --evaluation-seed "$EVAL_SEED" \
    --recovery-start-threshold "${RECOVERY_START_THRESHOLD:-0.1}" \
    --failure-threshold "$FAILURE_THRESHOLD" \
    --recovery-exit-threshold "$RECOVERY_EXIT_THRESHOLD" \
    --recovery-budget "$RECOVERY_BUDGET" \
    --recovery-min-steps "$RECOVERY_MIN_STEPS" \
    --recovery-clear-steps "$RECOVERY_CLEAR_STEPS" \
    --output "$MANIFEST_PATH"
fi

printf '\nREIM %s/%s requested stages complete.\n' "$PROFILE" "$BACKEND"
printf '  checkpoints: %s, %s, %s\n' \
  "$ACT_CHECKPOINT" "$DETECTOR_CHECKPOINT" "$RECOVERY_CHECKPOINT"
if [[ "$RUN_PPO_ABLATION" == "true" ]]; then
  printf '  optional PPO negative ablation: %s\n' "$PPO_ABLATION_CHECKPOINT"
fi
printf '  tables:      %s\n' "$TABLES_DIR"
printf '  figures:     %s\n' "$FIGURES_DIR"
printf '  paper assets:%s\n' " $PAPER_ASSETS_DIR"
if [[ "$PROFILE" == "full" ]]; then
  printf '  paper PDF:   %s\n' "$PAPER_ASSETS_DIR/reim_results.pdf"
  printf '  primary CRN: %s\n' "$MAIN_EPISODE_BANK"
fi
printf '  manifest:    %s\n' "$MANIFEST_PATH"
