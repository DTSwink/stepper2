# IK Handoff Journal

Last compacted: 2026-05-24.

The older running log is archived as `training/ik/JOURNAL_OLD_20260523.md`.
Read this file first after a context refresh. It is the current truth: what the
IK system is, which rules are mandatory, and which recent traps must not be
reintroduced.

## Current State

- Future locomotion work is IK-only.
- The FK/non-IK root training stack is deprecated and should not be recreated.
- The current IK representation reconstructs GT frame 0 exactly enough for
training and viewing. If frame 0 changes before prediction, fix the data/IK
pipeline before training.
- The latest Walk_F simple-AE regression was caused by scoring AE loss on the
cleaned/clamped IK output. That was fixed on 2026-05-24: AE loss must score the
raw controller output.
- Do not use checkpoint
  `20260524_050034_ik_walkF_scratch_simple_ae_k32_fixedviewer_rerun_last.pt`.
  It was trained with the bad cleaned-output AE scoring and straightened arms.
- The corrected Walk_F simple-AE checkpoint is:
  `training/runs/20260524_052925_ik_walkF_scratch_simple_ae_k32_rawscore_restore/checkpoints/20260524_052925_ik_walkF_scratch_simple_ae_k32_rawscore_restore_last.pt`.
  Diagnostic:
  - one-step avg joint error: `0.00421m`;
  - autoregressive avg joint error: `0.02043m`;
  - frame 24 elbows: GT `154.21/150.75`, model `151.92/146.48`.

## Hard Rules

- Keep IK work contained under `training/ik` unless the file is explicitly an
  IK-facing viewer/app integration such as `training/model_viewer_app.py`.
- Do not create one-off trainer files for mini experiments. Change dataset,
  checkpoint, objective weights, curriculum, or labels through canonical trainer
  arguments/config.
- Do not use contact labels anywhere. Foot losses are geometry/envelope based.
- Simple AE is one-frame/current-transition only. Do not add AE lookahead unless
  the user explicitly changes this rule.
- Missing NPZ paths, empty folders, invalid extensions, and skeleton mismatches
  must raise. No silent fallback to Walk_F.
- Run/checkpoint names must start with `YYYYMMDD_HHMMSS_ik_...`.
- Do not stage `training/runs/` outputs or local viewer settings.
- Push only when the user asks.
- TensorBoard must stack runs under the shared server and stay uncluttered.
  Controller cards should normally be only:
  - `loss/ae_score`;
  - `loss/linear_slide_weighted`;
  - `loss/angular_slide_weighted`;
  - `loss/supervised`.
- Do not rely on Codex automations/heartbeats for experiment babysitting. They
  failed in this workflow. Use an explicit blocking sleep/timer loop in the
  active turn if a run needs later checking.

## Canonical Files

- `training/ik/ik_core.py`: IK representation, MotionClip, input/output layout,
  FK/decode, supervised helpers.
- `training/ik/train.py`: canonical supervised controller trainer.
- `training/ik/train_simple_autoencoder.py`: vanilla one-frame AE trainer.
- `training/ik/train_simple_ae_controller.py`: simple AE-only controller
  trainer and fast experiment path.
- `training/ik/train_full_ae_envelope.py`: full AE/envelope/mixed-objective
  controller trainer.
- `training/ik/excess_envelope.py`: animation-dependent foot-slide/yaw
  envelope.
- `training/ik/visualize.py`: HTML rollout viewer. Simple-AE controller
  checkpoints must use the simple-controller vector rollout path.
- `training/model_viewer_app.py`: standalone app integration. It must also use
  the simple-controller vector rollout path for simple-AE controller checkpoints.
- `training/ik/kaggle_prepare.py`, `kaggle_run.py`, `kaggle_start.ps1`,
  `kaggle_sync_tensorboard.py`: IK-only Kaggle packaging/run/sync.

## Datasets

- Full periodic folder: `ue5/animations_omni_only_full/npz_final`.
- Full nonperiodic/transition folder:
  `ue5/animations_transitions_only_full_trimmed/npz_final`.
- Walk_F: `M_Neutral_Walk_Loop_F.npz`.
- TurnL45: `M_Neutral_Stand_Turn_045_L.npz`.
- CircleL: `M_Neutral_Walk_Circle_Strafe_L.npz`.
- Diamond test clip:
  `M_Neutral_Walk_Diamond_BL_F_Lfoot.npz`.

Periodic clips are cyclic. Transition clips are trimmed/noncyclic. Strict eval
samplers require a complete rollout window. Training samplers may use any
one-step-valid start; when a noncyclic row reaches the usable end before its
requested K is consumed, reset that row to a random valid start inside the same
animation and keep going. Do not shrink requested K just because a clip is short.

## IK Representation

Endpoint-only IK was rejected because it loses elbow/knee swivel and
hand/foot orientation.

Current controlled limb payload is 42 dims:

- left arm: hand position root-local 3 + hand rot6 root-local 6 + elbow pole 1;
- right arm: same, 10 dims;
- left leg: foot position root-local 3 + foot rot6 root-local 6 + knee pole 1
  + toe hinge 1;
- right leg: same, 11 dims.

Arms and legs use different rest poles:

- arms: `-character_forward`;
- legs: `+character_forward`.

Toe is a single hinge scalar. The best local toe hinge axis is selected by
reconstruction error.

Reach/pole rules:

- pole and toe scalars are clamped to `[-1, 1]`, with alpha `pi/2`, so this is
  +/- 90 degrees;
- IK endpoint reach is clamped during decode/state carry so positions stay
  physically reachable;
- do not use a straight-through reach projection. It made Walk_F K32 blow up
  late in training.

## Supervised Objective

`training/ik/train.py` is the supervised path.

The supervised loss compares the cleaned/canonical predicted next vector to the
cached GT target vector:

```text
raw = model(input)
pred_vec = predicted_state_from_raw(raw)
loss = mse(pred_vec, target_vec)
```

Do not compare `raw` directly to `target_vec`. Raw 6D rotations are redundant:
a raw vector can be numerically far from target but clean to the same rotation.
The raw-MSE supervised version made Walk_F degrade while the raw MSE improved.

In the mixed AE/envelope trainer, supervised steps are ratio-gated and K=1 only.
When the ratio selects a supervised step, the trainer uses only the supervised
loss for that step. AE/envelope steps use the long-horizon rollout.

Baseline supervised K=1 scaling:

- temp baseline raw cleaned-vector MSE: `0.0001047247`;
- global controller loss scale: `500`;
- `SUPERVISED_K1_LOSS_WEIGHT = 9.548846199989088`;
- this makes `loss/supervised` start around `0.5`.

## Simple AE Objective

The vanilla AE row is:

```text
controller_input + target_output
```

When training a controller with the frozen AE, compute the AE score on:

```text
controller_input + raw_predicted_output
```

This is mandatory. Do not score `clean_output_vector(raw)`.

Why: if the AE sees the cleaned/clamped output, the model can predict impossible
IK endpoints, the clamp projects them back into reach, and the AE stays happy.
That caused the 2026-05-24 Walk_F arm-straightening regression:

- broken run frame 24 elbows: `179.27/179.26`;
- fixed raw-score run frame 24 elbows: `151.92/146.48`;
- GT frame 24 elbows: `154.21/150.75`.

Cleaning/clamping still happens for:

- carried rollout state;
- decoding into visible poses;
- viewer display.

It does not happen before AE scoring.

## Sampling And Curriculum

- Rows are independent.
- Dense/packed GPU-resident layout is mandatory.
- Current maximum schedule is `1, 2, 8, 16, 32, 64`.
- At max K, effective K is mixed geometrically per row: half max K, half lower;
  within the lower group, half next lower, and so on.
- Log effective K mean/max.
- K64 on an 8 GB GPU needs smaller batch. The simple-AE controller caps K64 at
  batch `3328` on CUDA devices with `<=10GB` VRAM; K32 can keep batch `4096`.
- Full-dataset long stages should generally be manual-stop. Save `latest`
  frequently instead of trusting an aggressive stall detector.

## Checkpointing And Configs

Every long-run trainer must save:

- `init` before training;
- `latest` during training;
- `best` when the tracked loss improves;
- stage checkpoints when K changes;
- periodic unique step snapshots for overnight work;
- `last` at clean end.

Every experiment must write an easy-to-read config file. The most important
settings should be at the top. `training/ik/train.py` writes
`config_readable.json`; other long-run entrypoints should follow that style.

## Foot-Slide Envelope

No contact labels.

Current situation feature:

```text
[yaw_delta / pi, bend_angle / pi, horizontal_foot_distance_xz_m]
```

Horizontal foot distance is:

```text
norm((foot_l - foot_r)[x,z])
```

`y` is vertical and must not be included.

The envelope is animation-dependent:

- lookup is by current animation plus current situation;
- planted side is part of the lookup;
- there is no frame-index/per-frame component in the bound key;
- GT must be below its own envelope by construction plus margin.

Current cache version is `12`; default margin is `1.05`.

Performance decisions that survived:

- no full-body FK in the envelope hot path;
- decode only compact IK foot/toe state needed for contact geometry;
- carry compact foot/toe state across rollout steps;
- use one shared animation-local KNN distance pass for linear and angular
  bounds.

## Viewers And Metrics

HTML viewer:

- `training/ik/visualize.py` must detect simple-AE controller checkpoints
  (`metadata.policy.loss == "simple_ae_output_reconstruction"`) and use
  `train_simple_ae_controller` vector-state rollout.
- The old generic `ik_core.build_input`/pose-dict rollout path produced bogus
  drift for simple-AE controller checkpoints.

Standalone viewer:

- `training/model_viewer_app.py` must use the same simple-controller vector
  state, `model_forward`, `clean_output_vector`, and vector handoff for
  simple-AE controller checkpoints.
- If a viewer process was already open before code changes, restart it. It will
  not hot-reload Python code.

Useful metrics:

- one-step joint/global position + rotation + velocity catches local
  reconstruction problems without phase-drift confusion;
- autoregressive rollout catches long-horizon drift;
- elbow/knee angle diagnostics catch the "small hand error, huge elbow error"
  failure mode;
- foot-slide viewer should show GT slide, envelope, model slide, and excess per
  frame.

## Kaggle

Kaggle support is IK-local. Do not use old FK/K111 Kaggle scripts.

Modes:

- `STEPPER_IK_MODE=supervised` -> `training/ik/train.py`;
- `STEPPER_IK_MODE=simple_ae` -> `training/ik/train_simple_autoencoder.py`;
- `STEPPER_IK_MODE=ae_envelope` -> `training/ik/train_full_ae_envelope.py`.

Kaggle packaging should include the current IK code, FBX/NPZ pipeline, selected
NPZ datasets, and requested checkpoints/configs. Do not commit generated
`kaggle_payload` or downloaded `kaggle_results` folders unless explicitly asked.

## Rejected Or Dangerous Paths

- Endpoint-only IK: rejected.
- Contact labels: rejected.
- Per-frame/frame-index envelope lookup: rejected.
- Global situation-only envelope across all animations: rejected.
- Full FK inside envelope loss: rejected for performance.
- One-off mini trainer files: rejected.
- K=1 special-case controller loop: rejected.
- Raw-MSE supervised objective: rejected.
- Cleaned/clamped-output AE scoring: rejected after Walk_F arm regression.
- Straight-through IK reach projection: rejected after K32 blow-up.
- Codex automations/heartbeats for babysitting: rejected.

## Next Good Default

For quick sanity, use the corrected raw-score Walk_F simple-AE run as the known
good local check:

```text
20260524_052925_ik_walkF_scratch_simple_ae_k32_rawscore_restore_last.pt
```

For future AE/controller experiments, first verify:

- frame 0 matches GT;
- one-step error is sane;
- elbows/knees are not being hidden by small endpoint errors;
- TensorBoard shows the expected small set of cards;
- checkpoints are being written from the beginning.
