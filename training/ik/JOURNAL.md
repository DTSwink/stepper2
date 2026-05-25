# IK Handoff Journal

Last compacted: 2026-05-25.

This is the current truth after the context refresh. The older long log is
`training/ik/JOURNAL_OLD_20260523.md`. The removed window-buffer AE experiment is
not part of the active plan.

## Current State

- Future locomotion work is IK-only. Do not recreate the FK/non-IK trainer path.
- The current IK pose vector is the flat IK controller representation used by
  training, HTML viewers, and the standalone model viewer.
- Default controller prediction is non-residual: `predict_residual = false`,
  `zero_init_output = false`. This is intentional for RL/random-policy work.
- Current simple-AE Walk_F sanity checkpoint:
  `training/runs/20260524_052925_ik_walkF_scratch_simple_ae_k32_rawscore_restore/checkpoints/20260524_052925_ik_walkF_scratch_simple_ae_k32_rawscore_restore_last.pt`.
- Do not use:
  `20260524_050034_ik_walkF_scratch_simple_ae_k32_fixedviewer_rerun_last.pt`.
  It was trained with cleaned-output AE scoring and straightened the arms.
- RL all-bounds Walk_F was fixed on 2026-05-25. The working run is:
  `20260525_040544_ik_walkF_fixedK32_rl_datasetmax105_allbounds_noenvelope_lr1e4_fix`.
  The important fix was K32 learning rate `1e-4`; the failed relaunch used the
  later AE-tuned K32 LR `7.5e-6` and looked stuck/high-loss.

## Hard Rules

- Keep IK work in `training/ik`, except explicit IK-facing integration files
  such as `training/model_viewer_app.py`.
- Do not create permanent one-off trainer files for mini experiments. Expose
  dataset, rollout, objective, LR, and logging through canonical trainer args or
  config.
- Do not use clever launch wrappers. If a run cannot be launched as a readable
  CLI/config call, add the missing trainer argument first.
- RL-only means AE-free. If `ae_loss_weight == 0`, trainers must not load,
  validate, or require an AE checkpoint.
- No contact labels anywhere. Foot losses are geometry/envelope based only.
- Missing NPZ paths, empty folders, bad extensions, and skeleton mismatches must
  fail loudly. No silent fallback to Walk_F.
- Run/checkpoint names must start with `YYYYMMDD_HHMMSS_ik_...`.
- Do not stage `training/runs/`, generated screenshots, local viewer settings,
  Kaggle payload/results, or other generated artifacts.
- Push only when the user asks.
- Do not rely on Codex automations/heartbeats. They failed in this workflow. If a
  run needs babysitting, use a direct sleep/check loop in the active turn.

## Launch Checklist

Before saying a run is launched, confirm:

- the process is alive;
- stdout contains run id and TensorBoard logdir;
- stderr is empty or only known harmless framework noise;
- `init` and `latest` checkpoints exist;
- the run appears in `training/runs/tensorboard_stack`;
- at least one real training-step log appeared.

## Canonical Files

- `training/ik/ik_core.py`: IK representation, MotionClip, input/output layout,
  decode, supervised helpers.
- `training/ik/train.py`: canonical supervised controller trainer.
- `training/ik/train_simple_autoencoder.py`: vanilla one-frame AE trainer.
- `training/ik/train_simple_ae_controller.py`: simple AE controller trainer plus
  RL-only/all-bounds experiments.
- `training/ik/rl_loss.py`: separated RL constraint losses.
- `training/ik/train_full_ae_envelope.py`: full AE/envelope/mixed-objective
  trainer.
- `training/ik/excess_envelope.py`: animation-dependent foot-slide/yaw
  envelope.
- `training/ik/checkpoint_runtime.py`: single source of truth for deciding which
  checkpoints use the current flat-vector IK rollout.
- `training/ik/visualize.py` and `training/model_viewer_app.py`: must both use
  the same current IK vector rollout path for controller checkpoints.
- Kaggle files: `kaggle_prepare.py`, `kaggle_run.py`, `kaggle_start.ps1`,
  `kaggle_sync_tensorboard.py`.

## Datasets

- Full periodic: `ue5/animations_omni_only_full/npz_final`.
- Full nonperiodic/transition:
  `ue5/animations_transitions_only_full_trimmed/npz_final`.
- Walk_F: `M_Neutral_Walk_Loop_F.npz`.
- TurnL45/R45: `M_Neutral_Stand_Turn_045_L.npz` /
  `M_Neutral_Stand_Turn_045_R.npz`.
- CircleL/R: `M_Neutral_Walk_Circle_Strafe_L.npz` /
  `M_Neutral_Walk_Circle_Strafe_R.npz`.
- Diamond test clip: `M_Neutral_Walk_Diamond_BL_F_Lfoot.npz`.

Periodic clips are cyclic. Transition clips are trimmed/noncyclic. Training
samplers may reset a row inside the same animation when a noncyclic row reaches
the usable end before requested K is consumed.

## IK Representation

Endpoint-only IK was rejected. Current limb payload is 42 dims:

- arms: hand pos root-local 3 + hand rot6 root-local 6 + elbow pole 1;
- legs: foot pos root-local 3 + foot rot6 root-local 6 + knee pole 1 + toe
  hinge 1.

Arms use rest pole `-character_forward`; legs use `+character_forward`. Pole and
toe scalars are clamped to `[-1, 1]`, alpha `pi/2` (`+/- 90deg`). IK reach is
clamped during decode/state carry.

## AE Objective

The vanilla simple-AE row is:

```text
controller_input + current_root_transition_output
```

When training a controller with frozen AE, score:

```text
controller_input + raw_predicted_output
```

Do not score `clean_output_vector(raw)`. Cleaned-output scoring let the model
predict impossible IK endpoints that the clamp projected back into reach, which
caused the Walk_F straight-arm regression.

Simple AE is one-frame/current-transition only unless the user explicitly
changes the plan.

## Output Reference Contract

Controller output is the next pose expressed in the current root referential.
Frame state vectors are still stored in their own frame/root referential.

- Decode controller predictions with the current root when measuring world
  pose, foot sliding, envelope excess, or rendering the generated next pose.
- Rebase prediction vectors from current root to next/future root before using
  them as the next rollout state.
- Core non-pelvis rotations stay parent-local and are not rebased. Pelvis
  location/rotation and IK endpoint location/rotation are rebased. IK pole and
  toe scalars are preserved.
- Old future-root controller/AE checkpoints are contract-incompatible; new
  checkpoints/schemas carry `output_reference_root = "current"`.

## Supervised Objective

Supervised training compares the cleaned/canonical predicted transition vector
to the GT next-frame vector rebased into the current root:

```text
raw = model(input)
pred_vec = clean_output_vector(raw)
target_vec = transition_target_output(current_idx)
loss = mse(pred_vec, target_vec)
```

Do not compare raw output directly to target; raw 6D rotations are redundant.

## RL Constraints

RL constraints live in `training/ik/rl_loss.py` and are plugged into
`train_simple_ae_controller.py`. Each loss term logs separately. The RL path is
allowed to run with `ae_loss_weight=0` and no AE checkpoint.

Important working Walk_F all-bounds command shape:

```text
python -u -m training.ik.train_simple_ae_controller
  --npz ue5/animations_omni_only_full/npz_final/M_Neutral_Walk_Loop_F.npz
  --rollout-schedule 32
  --rollout-stage-steps 20000
  --rollout-k 32
  --no-mixed-rollout-at-max
  --stage-learning-rate 0.0001
  --log-every 20
  --ae-loss-weight 0
  --linear-slide-loss-weight 0
  --angular-slide-loss-weight 0
  + all dataset-max1.05 RL bound weights/limits
```

Do not use the AE K32 LR (`7.5e-6`) for this RL all-bounds setup; it is too slow
and looks broken.

## TensorBoard And Checkpoints

Controller cards should stay uncluttered. Normally use:

- `loss/train_total` for RL-only/all-bounds;
- `loss/ae_score`;
- `loss/linear_slide_weighted`;
- `loss/angular_slide_weighted`;
- `loss/supervised`;
- individual `loss/rl_*` cards only when debugging RL terms.

Every long run should save `init`, `latest`, `best`, stage checkpoints when K
changes, and `last` at clean end.

## Removed / Rejected

- `training/ik/window_buffer_ae_experiment.py` was removed from the active code.
  Do not resume that path unless the user explicitly asks.
- Do not add contact-label losses back.
- Do not add new permanent mini-harness files.
