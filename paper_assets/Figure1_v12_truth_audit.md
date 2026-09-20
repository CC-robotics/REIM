# Figure 1 v12 truth audit

| Figure claim | Evidence checked | Assessment / correction |
|---|---|---|
| PickPlace frames show a paired ACT failure and REIM recovery | `results/figures/recovery_operation_sequence.json` records seed 8300042, identical initial state and displacement, ACT failure at 200 steps, and REIM success at 62 steps | Supported as one qualitative paired example only. The figure now labels it as a separate protocol and does not treat it as an aggregate estimate. |
| PickPlace is directly comparable with the MT10/MT50 bars | The PickPlace qualitative trace uses noise level 0.20 and a single-task persistent-recovery rule; the confirmation banks use the multi-task hysteretic rule | Not supported. The panels are now explicitly separated by protocol. |
| Multi-task inputs include task identity | `env/metaworld_multitask.py` appends the official task one-hot to the raw 39-D observation | Supported. “task ID” was corrected to “task one-hot.” |
| Hysteretic switch parameters | MT10/MT50 confirmation `.run.json` files record minimum recovery 5 steps, release probability 0.05 for 10 steps, and cooldown 10 steps; trigger thresholds are 0.65 and 0.64 | Supported for the MT10/MT50 confirmation protocol. |
| Recovery training uses trigger-aligned successful continuations | `scripts/collect_multitask_recovery.py` records online detector-triggered expert continuations and retains successful continuations for supervised training | Supported. Embedded frames are marked as illustrations rather than claimed as literal training samples. |
| Door open and peg insert belong only to MT10 | Both task names occur in the confirmed MT10 and MT50 vocabularies | False in v11. Both are now labeled “shared”; shelf place is labeled “MT50 only.” |
| “40% action noise” | The confirmation protocol has noise level 0.40, action scale 0.40, and observation scale 0.025, giving actual standard deviations 0.16 and 0.01 | The old wording was imprecise. v12 reports the noise level and both actual standard deviations. |
| Bar values | Direct means of the confirmed episode CSVs: MT10 heuristic-gated recovery 46.4%, MT10 REIM 63.0%, MT50 heuristic-gated recovery 33.96%, MT50 REIM 49.52%; 50 episodes per task | Supported. Values are read by the figure script rather than typed manually. |
| MT10 and MT50 use the same perturbation settings but separate models | Run configurations match on evaluation schema, noise settings, episode count, horizon, release rule, and cooldown; checkpoint hashes differ by suite | Supported. |
| Task thumbnails are experimental outcomes | They were produced with official clean scripted experts for visual task selection | They are representative task scenes only, not quantitative evidence; the panel labels them accordingly. |
