# Detector threshold artifact lineage — MT10/MT50

Status of this document: authoritative provenance record for the detector
operating threshold used by all publication-facing MT10/MT50 results.
Written 2026-08-16 in response to the pre-submission review requirement
"统一 threshold artifact"（P0）.

## Verdict

The **current canonical** threshold artifacts are the ones produced by the
single self-consistent full-chain re-run committed in `26e09532` (MT10) and
`afa8fdb8` (MT50) on 2026-08-14:

| Artifact | SHA256 (first 16) | Selected τ | Task-macro precision | Status |
|---|---|---|---|---|
| `results/tables/mt10_detector_threshold.json` | `6aeccc85b28fdd6b…` | 0.65 | 0.6001 | precision_floor_satisfied |
| `results/tables/mt50_detector_threshold.json` | `848d32f99fb47549…` | 0.64 | 0.6004 | precision_floor_satisfied |

Every downstream consumer used exactly these values at runtime, provable from
the fail-closed sidecars: e.g.
`results/tables/mt10_clean_episodes.csv.run.json` records
`protocol.detector_threshold = 0.65`, and
`scripts/run_multitask_pipeline.py::_resolve_tuned_threshold` binds the
tuner artifact, both calibrated-bank manifests, and the detector checkpoint
by SHA256 before any evaluation stage is allowed to run.

## Superseded artifact

| Artifact | SHA256 (first 16) | Selected τ | Task-macro precision | Origin |
|---|---|---|---|---|
| `results/diagnostics/threshold_lineage/mt10_detector_threshold_canonical_prerun_0.77.json` | `db4c960e4e4ac0bd…` | 0.77 | 0.6053 | author's Linux machine (`/home/diy/...`), git commit `88643b1a` |

This earlier artifact was produced by a different detector checkpoint trained
on different hardware (CUDA/torch stack) under the same protocol seed. Its
validation-bank manifest reference (`/home/diy/bonus_track/REIM/...`) does not
exist in this checkout's data banks, so it is **not reproducible from the
current repository state** and is retained here for audit only. It is
superseded by the current canonical artifact and must not be cited as the
operating point of any published number.

Additionally, `results/diagnostics/mt10_gate65_5_per_task.csv.run.json` is a
5-episode-per-task **diagnostic** run (explicitly labeled `diagnostic_gate_65`
by its producer); it never participated in threshold selection and must not be
mixed with publication-grade results.

## Selection rule (both benchmarks)

- Tuner: `evaluation/tune_multitask_detector.py` on the held-out, calibrated
  validation bank only (`bank_role = validation_only`,
  `final_bank_accessed = false`).
- Rule: lowest probability threshold whose task-macro precision satisfies the
  preregistered floor 0.60 (`precision_floor_satisfied`).
- Deployment release threshold (0.15), patience (5), min recovery steps (5)
  and cooldown (10) remain preregistered in `configs/multitask/mt10.yaml` /
  `mt50.yaml`; they were **not** tuned on the validation bank. Per the
  pre-submission review, a principled joint trigger × release × patience
  selection on the gate-validation bank is scheduled as a separate P0 work
  item; until it lands, the numbers above stand as the canonical operating
  point and the hysteresis interval is reported as an engineering choice,
  not a principled one.

## Known calibration caveat

The tuner's own report shows task-macro ECE ≈ 0.138 (MT10) — detector scores
are not fully calibrated, so the numeric distance between trigger (0.65) and
release (0.15) must not be interpreted as a probability difference of 50 pp;
both are empirical operating points. Score calibration (temperature scaling
or isotonic regression, fit on the validation bank only) is tracked as a P1
work item.
