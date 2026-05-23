# IK Handoff Journal

Last compacted: 2026-05-23.

The old full running log was copied to `training/ik/JOURNAL_OLD_20260523.md`.
Read this file first after a context refresh. It is meant to be the short,
current truth: what we tried, what survived, how the system works now, and the
rules that should not be broken again.

## Current Goal

- Build and train a contained IK locomotion controller.
- The controller must imitate the dataset accurately under supervised training.
- AE and foot-slide envelope losses are still experimental and must not pollute
  the supervised baseline path.
- Full-dataset long-horizon supervised continuation from the temp baseline
  degraded local Walk_F quality and should not be treated as a trusted
  baseline.

## Hard Rules

- Keep IK work contained under `training/ik`.
- The old root-level FK/non-IK training stack has been removed. Do not recreate
  it.
- Do not create one-off trainer scripts for mini experiments. Change dataset,
  checkpoint, objective weights, curriculum, or labels through the canonical
  trainer arguments/config instead.
- Do not use contact labels anywhere. Foot-slide work is geometry/envelope based.
- Do not use lookahead for the AE. The AE is one-frame/current-transition only.
- Do not make any NPZ silently fail or fall back to Walk_F. Missing paths, empty
  folders, invalid extensions, and skeleton mismatches must raise.
- Checkpoint/run names must start with `YYYYMMDD_HHMMSS_ik_...`.
- `training/runs/` output must not be staged or committed.
- Push only when the user asks. For this cleanup the user explicitly asked to
  push after the journal is done.
- TensorBoard must work for every experiment. Keep it simple and shared.
- Controller TensorBoard loss cards should stay uncluttered. Mixed AE/envelope
  controller runs should show only:
  - `loss/ae_score`;
  - `loss/linear_slide_weighted`;
  - `loss/angular_slide_weighted`;
  - `loss/supervised`.
- Raw diagnostic values can go in viewers/files, but not as dashboard clutter.
- Every new long-run trainer should write a readable config file with the most
  important fields at the top. `train.py` already writes `config_readable.json`;
  port the same idea before relying on other long-run entrypoints.

## Canonical Files

- `training/ik/ik_core.py`
  - IK representation, data loading, input construction, output conversion, FK.
- `training/ik/train.py`
  - Canonical supervised controller trainer.
- `training/ik/train_simple_autoencoder.py`
  - Simple AE trainer.
- `training/ik/train_full_ae_envelope.py`
  - Canonical AE/envelope controller trainer.
- `training/ik/excess_envelope.py`
  - Animation-dependent foot-slide/yaw envelope.
- `training/ik/foot_envelope_viewer.py`
  - Viewer for GT slide, envelope, model slide, and excess.
- `training/ik/watch_supervised_run.ps1`
  - Guarded watchdog/restart script for the current supervised continuation.
- `training/ik/kaggle_prepare.py`, `kaggle_run.py`, `kaggle_start.ps1`,
  `kaggle_sync_tensorboard.py`
  - IK-only Kaggle packaging, launch, and output sync.
- `training/ik/tensorboard_log.py` and `launch_tensorboard_latest.ps1`
  - TensorBoard helpers.

Do not add a new training file when one of these can be extended cleanly.

## Datasets

- Full periodic folder:
  `ue5/animations_omni_only_full/npz_final`
- Full nonperiodic/transition folder:
  `ue5/animations_transitions_only_full_trimmed/npz_final`
- Walk_F assumption:
  `M_Neutral_Walk_Loop_F.npz`
- TurnL45 assumption:
  `M_Neutral_Stand_Turn_045_L.npz`
- CircleL assumption:
  `M_Neutral_Walk_Circle_Strafe_L.npz`

Periodic clips are cyclic. Transition clips are trimmed/noncyclic. The strict
sampler must only sample starts that have the full requested rollout horizon.

## IK Representation

The old endpoint-only IK was rejected because it lost elbow/knee swivel and
hand/foot orientation. Frame 0 could differ from GT before the model predicted
anything.

Current controlled limb payload is 42 dims:

- left arm: hand position root-local 3 + hand rot6 root-local 6 + elbow pole 1.
- right arm: same, 10 dims.
- left leg: foot position root-local 3 + foot rot6 root-local 6 + knee pole 1
  + toe hinge 1.
- right leg: same, 11 dims.

Arms and legs use different anatomical rest poles:

- arms use `-character_forward`;
- legs use `+character_forward`.

Toe is intentionally a single hinge scalar, not a full rotation. The best local
toe hinge axis is chosen from candidates by reconstruction error.

The model output is residual-style where applicable: prediction is added to the
current pose target vector before `output_to_pose`.

Important acceptance rule: encoding a GT frame into the IK payload and decoding
through `fk_from_pose` should reconstruct tracked positions and rotations with
near-zero frame-0 error. If frame 0 is wrong, do not train around it; fix the
representation/pipeline.

## Supervised Controller

`train.py` is the supervised controller path.

The supervised loss is MSE between the cleaned/canonical predicted vector for
the next step and the cached GT target output for that next step. It is not an
end-effector global delta loss.

Important correction from 2026-05-23: this used to compare raw model output and
that caused a regression. The loss must compare the cleaned/canonical
prediction vector, not the raw model output. Raw 6D rotations are redundant: a
raw vector can be numerically far from target while cleaning/normalizing to the
same rotation. The broken raw loss made Walk_F degrade while raw MSE improved.
The current rule is:

```text
raw = model(current_input)
pred_vec = predicted_state_from_raw(raw)
loss = mse(pred_vec, target_vec)
```

Do not change this back to `mse(raw, target_vec)`.

Supervised inside the AE/envelope trainer is a ratio-gated one-step objective,
not an added term on every step. When the ratio selects a supervised step, the
trainer uses K=1 only and does not apply AE or envelope losses on that step.
The intended default experiment ratio is 1 supervised step per 5 objective
steps; the other 4 steps stay AE + weighted slide envelope.

Mixed-trainer supervised K=1 is weighted in the actual loss, not just in logs.
On 2026-05-23 the full-dataset temp baseline
`20260522_075816_ik_full_vanilla_ae_controller_baseline_stall_last.pt`
measured raw cleaned-vector supervised K=1 MSE `0.0001047247` over 10687 valid
rows. With the global controller loss scale `500`, the hard-coded
`SUPERVISED_K1_LOSS_WEIGHT` is `9.548846199989088`, so the baseline supervised
TensorBoard card reads about `0.5`.

Sampling:

- rows are independent;
- the dense/packed GPU-resident row layout is mandatory;
- noncyclic rows require a complete rollout window;
- cyclic rows wrap through root cycle logic.

Curriculum:

- schedule is `1, 2, 8, 16, 32`;
- when K reaches 32, effective K is mixed fractally per row:
  half K32, half lower; of the lower half, half K16, and so on;
- this is per-row and should be logged as effective K mean/max.

Checkpointing:

- save `init`, `latest`, `best`, stage checkpoints, and `last`;
- `latest` must update often enough for crash recovery;
- `config_readable.json` should make the important settings obvious.

Current stability note:

- CUDA graph supervised mode gave illegal-memory-access/hang behavior during
  full overnight work;
- the current full supervised continuation uses `--disable-cuda-graph`;
- prefer stable eager mode over a faster path that risks killing the run.

Current guarded run command shape:

```powershell
python -u training\ik\train.py `
  --run-label full_supervised_from_temp_baseline_continuing `
  --periodic-folder ue5\animations_omni_only_full\npz_final `
  --nonperiodic-folder ue5\animations_transitions_only_full_trimmed\npz_final `
  --init-checkpoint <latest-or-baseline-checkpoint> `
  --resume-step-from-checkpoint `
  --load-optimizer `
  --train-steps 100000 `
  --disable-cuda-graph
```

Watch it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File training\ik\watch_supervised_run.ps1 `
  -Label full_supervised_from_temp_baseline_continuing `
  -StartIfMissing `
  -KillDuplicates
```

The watcher holds a mutex, counts the process list correctly, dedupes only when
asked, and starts from `latest`, then `last`, then the configured baseline.

## Autoencoder Work

We rewrote a simple AE path because older files became too clogged.

The AE row is the same transition row the controller sees: controller input plus
the target/output side of the transition. The controller can then be judged by
feeding current controller input plus predicted output through the frozen AE and
using reconstruction error as `loss/ae_score`.

Important results:

- Vanilla/simple AE reproduced Walk_F well.
- TurnL45 was better than CircleL, but CircleL exposed reenactment weakness.
- Negative/corruption harshness could reduce some metrics but produced hovering
  or ghosty/statue motion in some runs.
- Foot-delta/corruption variants did not clearly beat vanilla for reenactment.
- Before full-dataset AE work, retry vanilla first when behavior looks wrong.

The AE is allowed to be harsher through representation/capacity/weighted error,
but not by adding contact labels or lookahead.

## Foot-Slide Envelope

Foot-slide losses are geometry/envelope based, no contact labels.

The current situation feature is:

```text
[yaw_delta / pi, bend_angle / pi, horizontal_foot_distance_xz_m]
```

Horizontal foot distance is:

```text
norm((foot_l - foot_r)[x,z])
```

`y` is vertical and must not be included.

The envelope is animation-dependent:

- current rollout rows compare against their own animation/clip;
- lookup is by current animation plus runtime situation;
- there is no frame-index/per-frame component in the bound key;
- planted side is part of the lookup;
- GT must be below its own envelope by construction plus margin.

Current envelope code uses cache version `12` and margin `1.05`.

A previous version was wrong because it used a frame-index component in the
bound lookup. That made the loss too frame-specific and violated the intended
"same animation plus situation" design. Do not bring that back.

Performance work already done:

- removed full-body FK from the envelope hot path;
- decode only IK foot/toe state needed for contact geometry;
- carry compact foot/toe state across rollout steps;
- use one shared animation-local KNN distance pass for linear and angular bounds;
- KNN lookup is no longer the bottleneck.

Current rough benchmark:

- AE-only full-dataset K32 update: about `0.25s`;
- animation-dependent envelope full-dataset K32 update: about `0.9s-1.2s`.

Remaining envelope cost is mainly compact contact geometry, decode-next state,
and backward.

## Metrics And Viewers

Useful supervised metric:

- one-step difference against GT catches hovering and bad local reconstruction
  without being confused by phase drift.

Useful foot-slide metric:

- compare model slide/yaw excess against the animation-dependent envelope;
- for visual inspection use the foot envelope viewer showing GT slide,
  envelope, model slide, and excess per frame.

Be careful with full-gait averages. Landing/takeoff can look like sliding if the
metric is not aligned with the pinned interval. For Walk_F, mean-over-interval is
acceptable for quick checks, but do not overfit metric logic to one animation.

## TensorBoard

TensorBoard should stack runs under the shared server.

Useful launch path:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File training\ik\launch_tensorboard_latest.ps1
```

Avoid cards that are not decision-relevant. If a scalar is only useful for a
one-off audit, write it to a diagnostic JSON/viewer or print it, not to the main
dashboard.

## Kaggle

Kaggle support is part of IK now. Do not use the old K111/FK launch path for new
work.

Current IK-local files:

- `kaggle_prepare.py` builds a Kaggle dataset/kernel payload containing:
  - `training/ik`;
  - `fbx_npz_pipeline`;
  - selected `npz_final` datasets;
  - explicitly requested checkpoints/config files.
- `kaggle_run.py` runs inside Kaggle and dispatches to canonical IK trainers:
  - `STEPPER_IK_MODE=supervised` -> `training/ik/train.py`;
  - `STEPPER_IK_MODE=simple_ae` -> `training/ik/train_simple_autoencoder.py`;
  - `STEPPER_IK_MODE=ae_envelope` -> `training/ik/train_full_ae_envelope.py`.
- `kaggle_start.ps1` is the local upload/push wrapper.
- `kaggle_sync_tensorboard.py` downloads Kaggle outputs for local TensorBoard
  and checkpoint review.

Defaults are supervised, full periodic/nonperiodic datasets, eager/no CUDA graph
for stability, and checkpoint mirroring to Kaggle output. Keep future Kaggle
changes in these IK files, not in old root training scripts.

## Tally Of Tried Paths

- Contained IK folder:
  - kept; main trainer should stay untouched.
- Endpoint-only IK:
  - rejected; underdetermined elbows/knees and missing end-effector rotations.
- 42-dim IK payload with poles, foot rotations, and toe hinge:
  - kept.
- NPZ cleaning/re-encoding so model can reconstruct frame 0:
  - kept; frame-0 mismatch means the pipeline is broken.
- Packed/dense row layout:
  - kept and mandatory.
- One-off K=1 trainer special case:
  - rejected; use the same rollout loop for every K.
- Strict full-window sampling:
  - kept; no mid-rollout repair for noncyclic rows.
- Cached target output tensors:
  - kept.
- Fused AdamW on CUDA:
  - useful when supported, fallback otherwise.
- Mixed K at K32:
  - kept; fractal distribution per row.
- AE-only controller:
  - useful but not yet trusted as the full baseline because circle/turn behavior
    exposed reenactment issues.
- AE corruption/negative examples:
  - suspicious; caused hovering/statue behavior in some probes.
- Foot-slide envelope with frame-index lookup:
  - rejected; explicitly wrong.
- Global situation-only envelope across all animations:
  - rejected; mixes incompatible motions.
- Animation-dependent situation envelope:
  - kept.
- Full FK inside envelope loss:
  - rejected as too slow and unnecessary for IK feet.
- Compact IK foot/toe envelope path:
  - kept.
- CUDA graph supervised full run:
  - disabled for now due crash/hang risk.
- Watchdog implemented in heartbeat prompt:
  - rejected; too easy to duplicate process logic.
- Watchdog as one guarded script:
  - kept.
- Old K111 Kaggle scripts:
  - deprecated for future work.
- IK-local Kaggle packaging/run/sync scripts:
  - kept.
- Old root `training/train_locomotion*`, old AE-prior, old visual reporter,
  old K111 Kaggle, and root inspection/sweep scripts:
  - removed after IK became the only future training path.

## Current Open Items

- Keep the current full supervised continuation alive and monitor it through the
  guarded watcher.
- Mixed objective work belongs in `training/ik/train_full_ae_envelope.py`, not
  in a new mini harness.
- Current mixed-objective rule: supervised ratio steps are K=1 only; do not use
  long-horizon supervised rollout in the mixed AE/envelope controller.
- Port `config_readable.json` style to any trainer that will be used for long
  runs beyond `train.py`.
- Do not trust old refined/AE-envelope runs as final baselines unless their
  checkpoint lineage and TensorBoard scalars are verified.
