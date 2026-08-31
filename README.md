# REIM

**Recovery-Enhanced Imitation Learning for Robust Embodied Robot
Manipulation**

REIM is an end-to-end research implementation for studying failure-aware
robot manipulation on Meta-World PickPlace. A state-based Action Chunking
with Transformers (ACT) policy performs the nominal task, a causal LSTM
monitors runtime failure risk from the last 10 observations, and a
trigger-aligned supervised imitation policy takes control early enough to
correct the trajectory and finish the task.

The evidence is organized in two layers. PickPlace remains the mechanism study:
it supports the trigger-state curriculum, detector diagnostics, controller
ablation, and qualitative operation sequence. A separate shared-policy MT10/MT50
protocol tests breadth across known task families. Its official-clean condition
is kept separate from a task-universal action/observation-noise extension. The
multi-task paper-results gate has passed for the frozen configuration
(detector threshold 0.65/0.64, precision-floor 0.60 calibration, release
0.05, patience 10, confirmation bank 20266010/20266050), so the audited
MT10/MT50 numbers are reported below and in the compiled manuscript.

The repository contains data collection, resumable training, closed-loop
evaluation, robustness and ablation studies, plots, LaTeX tables, checkpoints,
and an auditable run manifest. The `toy` backend is only an explicit CI
facility. PickPlace results intended as Meta-World evidence must use
`--backend metaworld --profile full`; MT10/MT50 evidence must use the dedicated
multi-task runner and pass its independent completeness/provenance gate.

## Method

At timestep \(t\), ACT predicts a chunk
\(\hat{a}_{t:t+k}=\pi_{\mathrm{ACT}}(s_t,z=0)\). Predictions from overlapping
chunks for the same execution timestep are combined with the exponential
temporal ensemble from ACT. The detector estimates

\[
p_t = P(\text{labeled event in }[t,t+10]\mid s_{t-9:t}).
\]

Recovery training and deployment use two deliberately separate thresholds.
During curriculum construction, a broad risk threshold of 0.1 captures exact
MuJoCo snapshots from online ACT/LSTM rollouts while the state is still
recoverable. The official Meta-World scripted controller continues from each
snapshot; successful continuations provide 42,386 training and 8,212
trajectory-disjoint validation state--action pairs. A deterministic
21→256→256→4 MLP is trained with Smooth L1 loss. It is a standalone supervised
actor with no critic, optimizer, or rollout state and is trained without
environment interaction.

At deployment, the frozen gate is \(p_t \geq 0.2\). The recovery actor then
owns control until final task success or the 150-step recovery budget is
exhausted (`exit=0`, `min_steps=150`, `clear_steps=200`). This avoids treating
a weak risk-clear or re-lift signal as task recovery. ACT is not advanced while
the recovery actor holds control, and its temporal ensemble is cleared at every
hand-off so unexecuted failure-state chunks cannot leak into future nominal
actions.

The recovery-start audit shows that 98.9% of training triggers and 99.0% of
validation triggers occur before any lift. REIM should therefore be interpreted
as an early pre-grasp/approach correction system, not as evidence of general
post-drop repair.

The ACT implementation follows the key structure of Zhao et al.: a
Transformer CVAE encoder consumes current proprioceptive state and a future
action chunk during training; a Transformer decoder predicts the action
chunk; the encoder is discarded and \(z=0\) is used at inference. This project
adapts ACT to the state observations requested by REIM, rather than claiming
the original multi-camera architecture.

## Architecture

```text
Training:
ACT + LSTM online rollout ─> exact trigger snapshots ─> expert continuations
                                                       │
                                        Smooth-L1 recovery imitation
                                                       │
                                  disjoint validation-loss selection

Deployment:
Meta-World state ─> State-based ACT ─> nominal action ─> Sawyer
       │                                      │             │
       └─ last 10 states ─> LSTM risk ≥ 0.2 ──┴─> imitation recovery
                                                    │
                                    task success or 150-step horizon
```

The default semantic observation is 21-dimensional:

1. Sawyer arm joint positions (7)
2. end-effector position and quaternion (7)
3. object position (3)
4. goal position (3)
5. gripper state (1)

The action is `[dx, dy, dz, gripper]` in `[-1, 1]^4`. The wrapper also exposes
the official 39-dimensional Meta-World observation through `state_mode: raw`.
For MT10/MT50, one shared policy receives that raw observation concatenated with
the benchmark's ordered task one-hot. Task identity is therefore known at
evaluation; this is not an unseen-task or meta-learning protocol.

## Project layout

```text
REIM/
├── configs/                 # PickPlace and MT10/MT50 protocol configurations
├── env/                     # Meta-World/Gymnasium wrapper and disturbances
├── scripts/                 # demonstrations, failures, trigger states, provenance
├── models/                  # ACT, LSTM detector, standalone recovery actor
├── trainers/                # ACT, detector, recovery-imitation training
├── evaluation/              # closed-loop REIM and four baselines
├── experiments/             # robustness, gate sensitivity, component ablation
├── visualization/           # publication plots and LaTeX asset generation
├── run_all.sh               # full protocol and isolated toy smoke runner
├── run_multitask.sh         # dry-run-first MT10/MT50 staged runner
├── datasets/                # trajectories, failure windows, exact trigger states
├── checkpoints/             # learned policies
├── results/                 # episode records, aggregate tables, figures
└── paper_assets/            # paper-ready tables and copied figures
```

## Installation

Core requirements are Python 3.10--3.13, PyTorch, Gymnasium, Meta-World, and
MuJoCo. The setup also installs Stable-Baselines3 for archived experiments; it
is not required to load the exported deployment actor.

**Git LFS is required before cloning.** Binary research artifacts (`*.npz`
datasets, `*.pt` checkpoints) are stored with Git LFS. Without it, clone
yields pointer stub files instead of the real binaries, and every recorded
SHA256 in the manifests will fail to verify:

```bash
git lfs install
git clone https://github.com/CC-robotics/REIM.git
cd REIM
git lfs pull   # fetch the actual binary artifacts
```

```bash
cd REIM
./setup.sh
source .venv/bin/activate
```

On a machine that already has a compatible CUDA PyTorch installation, it can
be reused without installing another copy:

```bash
REIM_PYTHON=/path/to/python \
REIM_USE_SYSTEM_SITE_PACKAGES=1 \
./setup.sh
```

For headless machines, use `MUJOCO_GL=egl` (NVIDIA) or `MUJOCO_GL=osmesa`
(CPU rendering). Non-rendered training does not require a display.

Meta-World 3.x renamed the historical `PickPlace-v2` environment to
`pick-place-v3`. REIM records both names in the configuration and uses the
maintained v3 Gymnasium API.

## Training

### 1. Collect 500 expert trajectories

```bash
python scripts/collect_demonstrations.py \
  --backend metaworld --episodes 500 --seed 42
```

The collector uses Meta-World's scripted Sawyer expert and writes one
trajectory per NPZ, plus `manifest.json` and `statistics.json`.

### 2. Train state-based ACT

```bash
python trainers/train_bc.py --config configs/bc.yaml
```

Despite the compatibility filename `bc_policy.pt`, checkpoint metadata records
`policy_type: ACT`. The default uses 20-action chunks, a Transformer CVAE,
L1 reconstruction, KL weight 10, and temporal ensembling.

### 3. Generate causal failure data

```bash
python scripts/generate_failures.py \
  --backend metaworld --episodes 2000 --seed 42
```

Action, object, and observation disturbances are injected while ACT executes.
Every input window ends at timestep \(t\); its prospective target is positive
when an online failure event occurs in \([t,t+\text{horizon}]\). Thus future
states never enter detector features. Because the target interval is inclusive,
an event observed at \(t\) is also a positive example; causal features alone do
not make this a pure future-event forecast.

### 4. Train the LSTM detector

```bash
python trainers/train_detector.py --config configs/detector.yaml
```

The output includes accuracy, precision, recall, F1, threshold information,
and a confusion matrix. Dataset splits are trajectory-grouped to prevent
near-duplicate temporal windows leaking between train and validation.

Recompute the stricter pre-event audit from raw validation trajectories:

```bash
python scripts/audit_detector_pre_event.py
```

The audit exactly reproduces the stored deployment-threshold confusion matrix,
then separates current-event and future-event windows. At \(\tau=0.2\), 86.1%
of positive validation windows are current-event samples. Excluding those
samples gives future-window precision/recall/F1 of 53.5/68.3/60.0%. The monitor
alerts strictly before the first event in 191/258 (74.0%) event trajectories,
with a median latest-alert lead of one step. For this reason the paper calls
the module a **causal risk monitor**, not a calibrated long-horizon failure
forecaster. Machine-readable outputs are
`results/tables/detector_pre_event_audit.json` and
`results/tables/detector_pre_event_by_offset.csv`.

### 5. Collect exact online recovery starts

The train and validation banks use seed ranges disjoint from imitation,
detector, and final evaluation data:

```bash
python scripts/collect_recovery_starts.py \
  --episodes 1000 --seed 3000042 --noise-level 0.2 \
  --failure-threshold 0.1 \
  --output datasets/recovery_starts/train.npz
python scripts/collect_recovery_starts.py \
  --episodes 200 --seed 4000042 --noise-level 0.2 \
  --failure-threshold 0.1 \
  --output datasets/recovery_starts/validation.npz
```

Each NPZ stores a complete numeric MuJoCo snapshot captured immediately before
the first recovery action, the detector probability and trigger time, and
state/action pairs from successful scripted recoveries. Scripted control is
never called during benchmark evaluation.

### 6. Train and export trigger-aligned recovery imitation

```bash
python trainers/train_recovery.py \
  --config configs/ppo_trigger.yaml --backend metaworld \
  --output checkpoints/recovery_ablation_warmstart_only.zip \
  --pretrain-only
python scripts/export_imitation_recovery.py \
  --config configs/recovery_imitation.yaml
```

The compatibility entry point uses historical filenames, but
`--pretrain-only` runs only 40 Smooth-L1 epochs on 42,386 successful
trigger-aligned pairs and selects on 8,212 validation pairs. It performs no
environment rollout or policy-gradient update. The exporter retains only the
deterministic recovery actor and its normalization/action bounds. The canonical
deployment checkpoint is:

```text
checkpoints/imitation_recovery.pt
SHA-256 c3b4ac353f1146b01edf46d7d324b6f031e8a93c3c5dfb9af373cb1aaca4fa0f
```

`checkpoints/imitation_recovery.audit.json` proves zero environment-training
steps or policy-gradient updates and
bitwise-equivalent clipped and mean actions on all 42,386 training states,
8,212 validation states, and 10,000 deterministic random-support states
(60,598 total). `configs/recovery_imitation.yaml` pins the exact input hashes;
when deliberately retraining a new source actor, update those provenance hashes
before exporting a new artifact.

An independent recovery-only evaluation is available:

```bash
python evaluation/evaluate_recovery.py \
  --backend metaworld --episodes 100 --seed 4242
```

ACT and detector trainers accept `--resume [PATH]`. The canonical
recovery-imitation actor is selected by validation loss and exported with an
immutable audit.

## MT10/MT50 breadth protocol

The multi-task extension trains one task-conditioned stack per suite. Its input
is the official raw 39-dimensional observation plus the ordered MT10 or MT50
task one-hot; the four-dimensional Cartesian/gripper action is unchanged. The
shared comparison contains MT-MLP BC, MT-ACT, heuristic-gated learned recovery,
and MT-REIM. These suites expose known task identities, so they test breadth
across task families rather than unseen-task generalization.

Failure labels are calibrated without using validation or final-evaluation
data. For each task, the failure-training bank fits a threshold at the 0.90
quantile of ACT--expert action L1 disagreement. A deviation crossing is OR'ed
with the final 25 steps of an unsuccessful episode, and the causal target covers
the inclusive current-to-10-step-ahead window. The resulting per-task
thresholds are frozen and reused unchanged for validation and final evaluation.
A fixed raw-disagreement cutoff is diagnostic only, not the multi-task training
label. Expert actions are never available to the deployed detector or policy.

The runner is dry-run by default. Inspect the complete MT10 and MT50 plan
without collecting data, training, evaluating, or writing outputs:

```bash
./run_multitask.sh all both
```

Execute the suites separately so each completed milestone can be audited:

```bash
./run_multitask.sh all MT10 --execute
./run_multitask.sh all MT50 --execute
```

Safely continue collectors, evaluations, and trainers with an existing latest
checkpoint:

```bash
./run_multitask.sh all both --execute --resume
```

Individual subcommands are `collect_demos`, `train_act`, `train_mlp`,
`generate_failures`, `train_detector`, `collect_recovery`, `train_recovery`,
`evaluate_clean`, and `evaluate_disturbed`. Official-clean evaluation uses the
environment success signal and task-macro aggregation. Nonzero disturbance
levels use only task-universal action and observation noise and are a REIM
robustness extension, not official Meta-World scores. Fresh-task retry is kept
outside the primary multi-task comparison.

Partial checkpoints or training curves are engineering artifacts, not paper
evidence. `paper_assets/Table_multitask_clean.tex` is populated through
`paper_assets/multitask_results.tex`, which sets `\REIMMultiTaskResultstrue`
only after the independent publication gate validated the complete MT10 and
MT50 clean and disturbed episode records (input manifest
`d803829126c99866430cdc262c4aa97491a2ab7b345d85e63a2a307097212834`).

### Isolated multitask smoke run (CI)

The multi-task track has an explicit toy CI backend, mirroring the PickPlace
toy smoke convention. `env/toy_multitask.py` reproduces the Meta-World
interface surface (seeded MT10/MT50 banks with 50 variants per class, 39D raw
observations, task one-hot, 4D actions, scripted experts) with deterministic
point dynamics, so the complete 18-stage pipeline can be verified without
MuJoCo. Nothing ever falls back to it silently: every stage requires
`--backend toy`, and the runner selects it through a dedicated profile:

```bash
./run_multitask.sh all MT10 --smoke          # inspect the smoke plan
./run_multitask.sh all MT10 --smoke --execute
```

The smoke profile drives the tiny `configs/multitask/smoke*.yaml` configs and
writes only under `datasets/smoke/multitask`, `checkpoints/smoke/multitask`,
and `results/smoke/multitask`; production MT1/MT10 trees are never touched.
Smoke artifacts verify wiring, provenance, determinism, and file isolation —
including the fail-closed tuned-threshold binding and the five-bank
separation audit — and are never benchmark evidence.

## PickPlace evaluation

Evaluate the full four-method table on 1,000 episodes per method:

```bash
python evaluation/baseline.py \
  --backend metaworld --profile full --episodes 1000 \
  --noise-level 0.2 --seed 8000042 \
  --episode-bank \
    datasets/evaluation/pickplace_crn_task20260726_ep8000042_n1000_noise020.json \
  --failure-threshold 0.2 --recovery-exit-threshold 0 \
  --recovery-budget 150 --recovery-min-steps 150 \
  --recovery-clear-steps 200
```

The main comparison uses a calibrated 20% mixed-disturbance condition. Noise
levels are dimensionless protocol fractions rather than raw values shared
across different units: the 40% endpoint corresponds to action
\(\sigma=0.16\), observation \(\sigma=0.01\), and one object-position displacement
with \(\sigma=0.04\) m. Pass `--noise-level 0` for the clean diagnostic.

For a quick single-method diagnostic, evaluate only REIM:

```bash
python evaluation/evaluate_reim.py \
  --method reim --backend metaworld --profile full \
  --episodes 1000 --noise-level 0.2 --seed 8000042 \
  --failure-threshold 0.2 --recovery-budget 150 \
  --recovery-min-steps 150 --recovery-clear-steps 200
```

The single-method command does not by itself reproduce the serialized
four-method CRN comparison; use the bank-backed command above for formal
paired claims.

Run the diagnostic robustness/component entry points and regenerate plots from
their outputs:

```bash
python experiments/robustness.py \
  --backend metaworld --profile full --seed 8000042
python experiments/ablation.py \
  --backend metaworld --profile full --seed 8000042
python experiments/comparison.py \
  --backend metaworld --profile full --seed 8000042
python visualization/plot_results.py
```

The publication robustness curve uses one serialized CRN bank per disturbance
level, supplied through repeated `--episode-bank` arguments. Seed-only runs are
diagnostics and should not replace the bank-backed formal records.

The repository also retains the original exhaustive experiment runner:

```bash
./run_all.sh
```

It collects demonstrations and failure data, trains ACT and the LSTM, builds
the recovery-start banks, and runs benchmark/robustness/ablation stages.
Because the runner also retains archived experiment stages, the explicit
Steps 1--6 and bank-backed evaluation command above are the canonical paper
path. Training/data use seed 42, recovery-start banks use offsets +3,000,000
and +4,000,000, and the primary held-out CRN bank spans
8,000,042--8,001,041.
Continue a partially completed run without overwriting trajectory files with:

```bash
./run_all.sh --resume
```

`--resume` tells the collectors to retain valid NPZ files and passes an
existing checkpoint to resumable trainers. Evaluation is rerun unless it is
explicitly skipped. Recovery-start resume additionally verifies the curriculum
threshold, seed, episode count, NPZ hash, and the ACT/LSTM checkpoint hashes
before reusing either split.

For a fast end-to-end engineering check that is not a benchmark result:

```bash
./run_all.sh --smoke
```

Smoke artifacts are isolated under `datasets/smoke/`, `checkpoints/smoke/`,
`results/smoke/`, and `paper_assets/smoke/`; they cannot overwrite formal
Meta-World artifacts. The positional forms `./run_all.sh full metaworld` and
`./run_all.sh smoke toy` remain supported.

Every expensive stage can be skipped independently. For example, to reuse
trained checkpoints and regenerate all measured evaluations and assets:

```bash
./run_all.sh --resume --skip-data --skip-training
```

Use `./run_all.sh --help` for every skip flag, device selection, the seed
override, and `--dry-run`, which prints the fully resolved protocol without
executing it.

The full runner uses deployment defaults `threshold=0.2`, `exit=0`,
`budget=min_steps=150`, and `clear_steps=200`. They can be overridden without
editing source through `REIM_FAILURE_THRESHOLD`,
`REIM_RECOVERY_EXIT_THRESHOLD`, `REIM_RECOVERY_BUDGET`,
`REIM_RECOVERY_MIN_STEPS`, and `REIM_RECOVERY_CLEAR_STEPS`. The curriculum
capture threshold is independently controlled by
`REIM_RECOVERY_START_THRESHOLD` and defaults to 0.1.

## Results

### PickPlace mechanism results

Aggregate results are generated from disjoint paired episode seeds and live in:

- `results/tables/baseline.csv`
- `results/tables/ablation.csv`
- `results/tables/robustness.csv`
- `results/tables/recovery_evaluation.json`
- `results/tables/bc_training_summary.json`
- `results/tables/detector_metrics.json`
- `results/tables/gate_sensitivity.csv`
- `results/tables/gate_matched_comparison.json`
- `results/tables/detector_pre_event_audit.json`
- `results/run_manifest.json`

The frozen 20% mixed-disturbance benchmark contains 1,000 paired episodes per
method from one serialized CRN bank (seeds 8,000,042--8,001,041):

| Method | Task success | Intervention outcome | Average steps |
|---|---:|---:|---:|
| ACT | 73.4% | n/a | 93.28 |
| ACT + one fresh-task retry | 81.3% | 74.5% per reset | 93.34 |
| ACT + Heuristic Recovery | 87.7% | 83.3% per intervention | 72.82 |
| REIM | **90.4%** | **85.3% per intervention** | **69.32** |

REIM improves ACT by 17.0 percentage points (paired bootstrap 95% CI
[14.7, 19.3]), the one-retry baseline by 9.1 points [6.1, 12.1], and
heuristic-gated recovery by 2.7 points [0.7, 4.8]. It intervenes in 655 episodes
versus 731 for the heuristic gate: 76 fewer episodes, or a 10.4% relative
reduction. Thus REIM improves both task success and intervention burden in the
primary CRN benchmark.

At disturbance levels 0%, 10%, 20%, 30%, and 40%, ACT/REIM success is
100.0/100.0, 97.0/99.5, 76.5/92.0, 53.5/79.5, and 31.5/63.5%, respectively
(200 paired episodes per condition). The standalone recovery artifact has
72,452 actor parameters, zero environment-training steps, and bitwise-equivalent
outputs on 60,598 audited states.

The post-freeze gate audit uses a separate 200-episode CRN bank. The semantic
heuristic reaches 86.0% success at a 76.0% intervention rate. At
\(\tau=0.175\), REIM reaches 92.5% at 76.5% intervention, a burden-matched
success gain of 6.5 points [2.5, 11.0] (\(p=0.0044\)). At the frozen
\(\tau=0.20\), REIM reaches 90.5% at 63.5% intervention, improving success by
4.5 points [0.5, 8.5] (\(p=0.0490\)). The p-values are exact two-sided
McNemar/binomial tests; this diagnostic was not used to select the primary
threshold.

Rates are stored as fractions and rendered as percentages. The evaluator also
stores all per-episode records. Binary task-success intervals are Wilson
intervals; paired differences, intervention ratios, and step-count intervals
use deterministic episode bootstrap estimates.

Final benchmark provenance includes the full Meta-World protocol, both
recovery-start splits, `imitation_recovery.pt` and its audit, table-shape checks,
and disjoint training/evaluation seeds. Toy, legacy, or partial results must not
be treated as scientific evidence.

### MT10/MT50 breadth results

The independent multi-task paper-results gate has passed on the final
confirmation library: complete MT10 and MT50 clean plus all five disturbed
conditions (action/observation noise 0.0, 0.1, 0.2, 0.3, 0.4; the 0.0
zero-noise control reproduces the official-clean numbers exactly) were
audited with immutable run sidecars, checkpoint hashes, and
task-vocabulary/task-bank provenance. Numbers below are task-macro success
from
`results/tables/confirmation_202660xx/mt{10,50}_confirm_{official_clean,robustness_noise_*}_summary.json`
(confirmation bank seed 20266010 MT10 / 20266050 MT50, 50 episodes per task,
frozen detector threshold 0.65 MT10 / 0.64 MT50 -- precision-floor 0.60
calibration per canonical horizon 25; the earlier precision-floor 0.65
calibration (thresholds 0.73/0.71) is archived under
`results/tables/confirmation_202660xx_floor065/` -- release 0.05,
patience 10); the same values populate `paper_assets/multitask_results.tex`.

Official clean condition:

| Benchmark | MT-MLP BC | MT-ACT | Heuristic recovery | MT-REIM |
|---|---:|---:|---:|---:|
| MT10 | 94.2% | 97.4% | **98.0%** (23.4% interv.) | 97.6% (46.0% interv.) |
| MT50 | 81.5% | 91.6% | **94.0%** (17.8% interv.) | 92.2% (28.6% interv.) |

Robustness extension (task-universal action/observation noise; non-official):

| Noise | MT10 REIM | MT10 ACT | MT10 Heuristic | MT50 REIM | MT50 ACT | MT50 Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 97.6% | 97.4% | **98.0%** | 92.2% | 91.6% | **94.0%** |
| 0.1 | 45.4% | 5.2% | **49.8%** | 34.9% | 1.8% | **35.8%** |
| 0.2 | **53.0%** | 6.0% | 50.6% | **41.3%** | 1.7% | 36.0% |
| 0.3 | **58.4%** | 5.8% | 49.0% | **46.9%** | 1.7% | 35.0% |
| 0.4 | **63.0%** | 6.0% | 46.4% | **49.5%** | 2.0% | 34.0% |

On clean tasks the heuristic gate edges out REIM on the confirmation bank
(MT10 98.0% vs 97.6%; MT50 94.0% vs 92.2%), while REIM is narrowly ahead of
MT-ACT on MT10 (+0.2 pp, 95% CI [+0.0, +0.6]) and stays significantly ahead
of MT-ACT on MT50 (+0.6 pp, 95% CI [+0.0, +1.2]). Under noise the ACT policy
collapses to single digits. At noise 0.1 the heuristic gate leads on MT10
(49.8% vs 45.4%) and is statistically indistinguishable from REIM on MT50
(35.8% vs 34.9%); at noise 0.2 REIM is statistically level with the
heuristic gate on MT10 (53.0% vs 50.6%) but already leads significantly on
MT50 (41.3% vs 36.0%); from noise 0.3 upward REIM leads both baselines on
both benchmarks (MT10 58.4% vs 49.0% and MT50 46.9% vs 35.0% at 0.3; MT10
63.0% vs 46.4% and MT50 49.5% vs 34.0% at 0.4).

Recovery occupancy (share of episode steps spent in recovery) is reported as a
first-class metric alongside success, from the gate-generated
`paper_assets/multitask_clean_statistics.csv` and
`paper_assets/multitask_robustness_statistics.csv` (confirmation bank, seeds
20266010/20266050):

| Noise | MT10 REIM occ. | MT10 Heuristic occ. | MT50 REIM occ. | MT50 Heuristic occ. |
|---|---:|---:|---:|---:|
| clean | 18.7% | 14.2% | 16.8% | 10.6% |
| 0.1 | 37.4% | 49.3% | 45.1% | 60.9% |
| 0.2 | 52.6% | 48.8% | 55.4% | 60.8% |
| 0.3 | 60.7% | 48.4% | 61.8% | 60.3% |
| 0.4 | 69.2% | 47.7% | 67.4% | 60.0% |

The robustness gain comes with a measurable occupancy cost; per-condition
segment and rescued/harmed counts are reported alongside success in the
per-condition episode records under `results/tables/confirmation_202660xx/`.

Matched-occupancy control (directional, 200-episode search bank seed
20264010, 20 episodes per task; heuristic reference re-run on the same bank,
see `results/diagnostics/release_patience_search/mt10_references.csv`): at
MT10 noise 0.4 the heuristic gate occupies 47.6% of steps, and the grid point
release 0.3 / patience 5 matches it at 49.0% occupancy while scoring 49.5%
versus 47.0% success (+2.5 points at equal budget). At noise 0.1 no grid
point reaches the heuristic occupancy of 48.2% (grid maximum 38.3%); the
nearest point is the robustness-first operating point itself (release 0.05 /
patience 10: 38.3% occupancy, 46.0% versus 47.0% success, i.e. -1.0 point at
roughly 10 points lower occupancy). Machine-readable record:
`results/tables/mt10_matched_occupancy_comparison.json`; curve:
`results/figures/mt10_success_occupancy.png`.

Backfill detector-level audit (MT10, terminal-positive horizon ablation).
Thresholds were re-tuned per horizon on the validation bank under a precision
floor of 0.60: horizon 0 -> 0.72, 10 -> 0.71, 25 -> 0.65, 50 -> 0.62.
Pre-event metrics from `results/tables/mt10_backfill_pre_event_summary.csv`:

| Horizon | Threshold | Strict pre-event F1 | Event-traj. early-warning rate | Median lead (steps) |
|---|---:|---:|---:|---:|
| h0 | 0.72 | 0.428 | 30.0% | 1 |
| h10 | 0.71 | 0.414 | 30.7% | 1 |
| h25 | 0.65 | 0.410 | 29.7% | 2 |
| h50 | 0.62 | 0.394 | 32.3% | 1 |

Correction to the 2026-08-23 report: the 0.683 F1 / 74% early-warning figures
quoted there were single-task PickPlace numbers, not MT10; the MT10
detector-level values are the ones in the table above.

Backfill closed loop with tuned thresholds (MT-REIM task-macro success,
`results/tables/mt10_backfill_tuned_closed_loop_summary.csv`; bank 20265010,
release 0.05/10, 20 episodes per task; the preliminary thr=0.65 rows at 50
episodes per task are kept alongside):

| Horizon | Threshold | Clean | Noise 0.1 | Noise 0.4 | Occupancy (clean / 0.1 / 0.4) |
|---|---:|---:|---:|---:|---:|
| h0 | 0.72 | 98.5% | 46.0% | 59.5% | 16.3% / 42.1% / 62.0% |
| h10 | 0.71 | 98.0% | 45.0% | 61.0% | 14.8% / 44.7% / 68.2% |
| h50 | 0.62 | 98.5% | 47.5% | 62.5% | 25.7% / 42.0% / 70.6% |

Against the thr=0.65 runs the tuned thresholds trade nothing away: clean rises
slightly (+1.2 to +1.7 points) and noise-0.4 shifts within ±0.5 points; h50
gains +5.3 / +3.3 points at noise 0.1 / 0.4. Canonical horizon 25 is the main
result reported above.

Multi-seed confirmation (canonical horizon 25 detector+recovery, seeds
42/43/44, bank 20265010, 50 episodes per task,
`results/tables/mt10_multiseed_summary.csv`): MT-REIM task-macro success
96.6% / 44.1% / 60.8% (mean over seeds; per-seed range 96.2-97.0% /
43.0-45.6% / 55.6-65.4%) at noise 0 / 0.1 / 0.4, versus heuristic gate
96.7% / 44.7% / 40.8% and MT-ACT 96.6% / 5.2% / 4.6%. REIM is statistically
level with the heuristic gate at noise 0.1 and clearly ahead at noise 0.4 on
every seed.

## Paper assets

`visualization/plot_results.py` creates:

- `results/figures/framework_architecture.png`
- `results/figures/success_comparison.png`
- `results/figures/robustness.png`
- `results/figures/confusion_matrix.png`
- `results/figures/recovery_examples.png`
- `results/figures/ablation.png`
- `results/figures/gate_sensitivity.png`
- `results/figures/recovery_operation_sequence.png`
- `paper_assets/Table1_baseline.tex`
- `paper_assets/Table2_ablation.tex`
- `paper_assets/Table3_component_diagnostics.tex`
- `paper_assets/Table_multitask_clean.tex` (populated via
  `paper_assets/multitask_results.tex` after the publication gate passed)
- `paper_assets/Figure1_final_framework.png`
- `paper_assets/Figure2_final_results.png`
- `paper_assets/Figure3_detector.png`
- `paper_assets/Figure3_final_ablation.png`
- `paper_assets/Figure4_gate_sensitivity.png`
- `paper_assets/Figure5_operation_sequence.png`

Tables use `booktabs`, explicit metric directions, consistent precision, and
minimal rules. The framework figure uses a closed-loop embodied-AI layout:
perception/state, ACT action chunking, environment feedback, risk gating,
recovery, and return to nominal control.

The multi-task table is structurally present but expands to no LaTeX output
while `\ifREIMMultiTaskResults` is false. This prevents placeholder macros from
appearing in the PDF.

`Figure5_operation_sequence` contains frames rendered from one actual
Meta-World/MuJoCo rollout in this repository: paired ACT failure, LSTM trigger,
supervised-recovery grasp/lift, transport, and task completion. It is a
simulator operation sequence, not a photograph of physical Sawyer hardware.

Compile the measured report after generating the plots and tables:

```bash
./compile_paper.sh
```

The script uses an installed Tectonic binary or downloads a pinned,
checksum-verified local Tectonic 0.16.9 release. It compiles
`paper_assets/reim_results.tex` to:

```text
paper_assets/reim_results.pdf
```

The full `./run_all.sh` protocol performs this compilation automatically after
plot generation.

## Reproducibility and fair reporting

- Every entry point accepts a random seed.
- Train/validation splits are made by trajectory, not individual timesteps.
- Checkpoints include model structure, normalization statistics, optimizer
  state, epoch/timestep, seed, and history.
- PickPlace ACT/detector and the staged multi-task collectors, MLP, ACT,
  detector, recovery trainer, and evaluator expose provenance-checked resume
  paths.
- The final recovery checkpoint is a standalone deterministic actor. Its audit
  records source/data hashes, zero environment-training steps or
  policy-gradient updates, and exact action-equivalence results.
- Full and smoke profiles are explicitly distinguished.
- Plots and LaTeX tables are generated from CSV/JSON files; no result is
  hard-coded.
- The detector pre-event audit reproduces the conventional confusion matrix
  before separately scoring offset-zero and strictly future-event windows.
- Each completed run records package versions, hardware, configuration hashes,
  artifact hashes, backend, and profile. Trigger-state train/validation NPZ and
  JSON hashes and all gate/recovery controller parameters are recorded as
  first-class provenance.
- Training/data seeds and disjoint evaluation seed ranges are checked for
  disjointness before a run can be marked benchmark eligible.

## Legacy / archived controls

Archived files retain the earlier PickPlace MLP task-policy and PPO recovery
controls for provenance and engineering comparison. They are not part of the
manuscript's ACT→LSTM→supervised-recovery method or PickPlace headline table,
and the final evaluator does not silently substitute them for
`imitation_recovery.pt`. The new MT-MLP BC is a separate audited shared-policy,
task-conditioned capacity baseline for MT10/MT50; it is not the archived
single-task policy and is never presented as REIM.

## Future work

The mechanism study remains focused on one state-observed Meta-World task and
one trained ACT/LSTM/recovery stack. The MT10/MT50 breadth protocol covers
known task identities only; its gate has passed for the frozen configuration,
and unseen-task generalization remains open. Stronger evidence should propagate all model-training seeds, test
unseen-task generalization, add camera observations, collect post-grasp and true
post-drop recovery data, calibrate risk under covariate shift, study
uncertainty-aware switching and sim-to-real perturbations, and evaluate a real
Sawyer. ACT chunk size and temporal-ensemble decay should also be ablated
jointly with recovery latency.

## References

- Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. *Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023.
  <https://roboticsproceedings.org/rss19/p016.html>
- Official ACT implementation: <https://github.com/tonyzhaozh/act>
- Meta-World: <https://github.com/Farama-Foundation/Metaworld>
