# REIM manuscript evidence map

This file maps every central manuscript claim to local, inspectable evidence. The draft does not use `paper_assets/reim_results.tex` as a prose source.

## Method and protocol

| Claim | Evidence |
|---|---|
| Paper/project name and expansion | `CITATION.cff` |
| MT10/MT50 state is raw 39-D observation plus official task one-hot | `env/metaworld_multitask.py`; `configs/multitask/mt10.yaml`; `configs/multitask/mt50.yaml` |
| ACT chunk size and architecture | `configs/multitask/mt10_act.yaml`; `configs/multitask/mt50_act.yaml`; `models/bc_policy.py` |
| Detector is a causal one-layer LSTM over 10 states | `models/failure_detector.py`; `configs/multitask/mt10_detector.yaml`; `configs/multitask/mt50_detector.yaml` |
| Task-wise disagreement thresholds are fitted at the 0.90 training-bank quantile, frozen for later banks; prediction horizon 10 and terminal-positive horizon 25 | `configs/multitask/mt10.yaml`; `configs/multitask/mt50.yaml`; `scripts/relabel_multitask_failures.py` |
| Deployment thresholds 0.65 (MT10) and 0.64 (MT50), selected on validation data under task-macro precision floor 0.60 | `results/tables/mt10_detector_threshold.json`; `results/tables/mt50_detector_threshold.json` |
| Recovery collection threshold 0.20 and 50 successful trigger-aligned continuations per task | `datasets/mt10/recovery/manifest.json`; `datasets/mt50/recovery/manifest.json` |
| Recovery MLP, Smooth-L1 loss, task-balanced sampling, and train/validation row counts | `trainers/train_multitask_recovery.py`; `results/tables/mt10_recovery_training.json`; `results/tables/mt50_recovery_training.json` |
| Trigger/release/minimum-hold/cooldown settings 0.65 or 0.64 / 0.05 / 5 / 10; release patience 10; recovery budget 250 | `results/tables/confirmation_202660xx/*noise_40_episodes.csv.run.json`; `evaluation/evaluate_multitask.py` |
| Heuristic gate uses 20-step reward window, tolerance 0.01, and starts after step 30 | same confirmation run manifests; `evaluation/evaluate_multitask.py` |
| Five data stages are payload- and sampling-unit-disjoint | `results/audits/confirmation_gate_staging/mt10_bank_separation.json`; `results/audits/confirmation_gate_staging/mt50_bank_separation.json` |
| Per-suite dataset and evaluation counts in Table II | the two audit files above; `configs/multitask/mt10.yaml`; `configs/multitask/mt50.yaml` |

## Quantitative results

| Claim | Evidence |
|---|---|
| All clean and robustness task-macro success values and bootstrap intervals | `paper_assets/multitask_clean_statistics.csv`; `paper_assets/multitask_robustness_statistics.csv` |
| MT10 strong-noise paired counts: 115 rescued, 32 harmed, N=500 | `results/tables/confirmation_202660xx/mt10_confirm_robustness_noise_40_episodes.csv`; recomputed by `paper_assets/build_figure3_paired_effects.py` |
| MT50 strong-noise paired counts: 610 rescued, 221 harmed, N=2500 | `results/tables/confirmation_202660xx/mt50_confirm_robustness_noise_40_episodes.csv`; recomputed by `paper_assets/build_figure3_paired_effects.py` |
| Per-task improved/tied/worse counts: MT10 7/0/3; MT50 32/3/15 | the same episode CSVs; recomputed by `paper_assets/build_figure4_per_task_analysis.py` |
| Largest task-level positive and negative deltas named in Results | the same episode CSVs, grouped by `task_name` and `method` |
| Detector precision, recall, ECE, and frozen controller settings in Table II | `results/tables/mt10_detector_threshold.json`; `results/tables/mt50_detector_threshold.json`; `results/tables/confirmation_202660xx/*episodes.csv.run.json` |
| Clean and strong-noise intervention/recovery-occupancy results reported in Results | `paper_assets/multitask_robustness_statistics.csv` |
| MT10 seeds 42/43/44 and matched MT50 REIM/heuristic seeds 42/43/44/45 in Table IV | `results/tables/mt10_multiseed_summary.csv`; `results/tables/mt50_four_seed_summary.csv`; audited MT50 per-seed REIM and heuristic episode CSVs/summaries under `results/tables/mt50_seed*_confirm_*` and `results/tables/mt50_seed*_heuristic_confirm_*` |
| Validation-selected occupancy-matched heuristic controls (20 episodes/task; tolerance 0.06 for MT10, 0.02 for MT50) and confirmation comparisons | `results/diagnostics/occupancy_match_20260921/mt10_matched_confirmation_comparison.json`; `results/diagnostics/occupancy_match_20260921/mt50_matched_confirmation_comparison.json`; `results/diagnostics/occupancy_match_20260921/mt50_validation20_tol002_selection.json`; `results/diagnostics/occupancy_match_20260921/mt50_validation20_tol002_shard*_episodes.csv`; `scripts/analyze_occupancy_matched_confirmation.py` |

## Figures used

1. `paper_assets/Figure1_v12_overview.pdf` — overview; audited in `paper_assets/Figure1_v12_truth_audit.md`.
2. `paper_assets/Figure2_multitask_robustness.pdf` — common MT10/MT50 perturbation sweep.
3. `paper_assets/Figure3_paired_effects.pdf` — strong-perturbation success and paired rescued/harmed outcomes.
4. `paper_assets/Figure4_per_task_analysis.pdf` — exact per-task comparison without jitter.

## Scope restrictions carried into the draft

- PickPlace in Figure 1 is qualitative and uses a separate single-task protocol.
- MT10 and MT50 share evaluation settings but use separately trained models.
- Perturbed evaluation is a REIM robustness extension, not an official Meta-World score.
- Detector precision is not task success.
- The draft does not claim state of the art, unseen-task generalization, calibrated probabilities, or real-robot validation.
