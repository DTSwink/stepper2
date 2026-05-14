# Training Experiment Journal

This file records the small decisions that are easy to forget after a long
experiment loop. It is intentionally practical: what was run, what happened,
and what to reuse.

## 2026-05-12 - K8 Supervised vs Isaac-Style Agents

Goal: confirm that the IsaacGym-style training loop can match the older
sampled supervised loop on `data/fbx/npz_final/testcasc.npz`.

Shared setup:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py `
  --folder-path data/fbx/npz_final `
  --device cuda `
  --max-epochs 320 `
  --batch-size 64 `
  --rollout-schedule 1,2,4,8 `
  --curriculum-max-epochs-per-stage 70 `
  --curriculum-stall-patience-epochs 35 `
  --curriculum-min-epochs 20 `
  --curriculum-min-delta 1e-5 `
  --no-compile `
  --learning-rate 1e-4 `
  --save-last-every-epochs 10 `
  --save-best-every-epochs 10 `
  --writer-flush-every-epochs 10 `
  --target-loss-reduction 0.90 `
  --profile-timing
```

Results:

| Run | Loop | Reset sampling | Best loss | Avg joint error | End error | Max error | Time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `supervised_k8_contacts_removed_sampled_recheck_20260512_154545` | `sampled` | dataset loader | `0.022496` | `0.01373 m` | `0.02359 m` | `0.02359 m` | `43.2s` |
| `supervised_k8_contacts_removed_agents256_random_reset_20260512_153648` | `agents` | random with replacement | `0.025462` | `0.02558 m` | `0.04332 m` | `0.04332 m` | `71.5s` |
| `supervised_k8_contacts_removed_agents_coverage_nostop_20260512_154711` | `agents` | coverage | `0.020905` | `0.01205 m` | `0.01407 m` | `0.01762 m` | `51.9s` |

Conclusion:

The Isaac-style loop works when its reset distribution matches the dataset
coverage. Random reset with replacement is noisy on this tiny clip because it
can duplicate some starts while missing others. Coverage reset walks through the
valid start indices uniformly, so it behaves like the sampled objective while
keeping the agent-rollout structure we want for later experiments.

Important implementation note:

Final-stage stall stopping is now opt-in with `--stop-on-final-stall`. The
coverage-agent run looked worse when it stopped at epoch 247, but the real
improvements arrived later, around epochs 258-318. For controlled comparisons,
run the full epoch budget unless there is an explicit time cap.

Recommended Isaac-style command:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py `
  --folder-path data/fbx/npz_final `
  --run-name supervised_k8_agents_coverage_$(Get-Date -Format yyyyMMdd_HHmmss) `
  --device cuda `
  --max-epochs 320 `
  --batch-size 64 `
  --rollout-schedule 1,2,4,8 `
  --curriculum-max-epochs-per-stage 70 `
  --curriculum-stall-patience-epochs 35 `
  --curriculum-min-epochs 20 `
  --curriculum-min-delta 1e-5 `
  --no-compile `
  --learning-rate 1e-4 `
  --training-loop agents `
  --agent-sampling coverage `
  --save-last-every-epochs 10 `
  --save-best-every-epochs 10 `
  --writer-flush-every-epochs 10
```

The regenerated viewer for the best coverage-agent checkpoint was written to:

```text
training/runs/model_comparisons/model_comparison.html
```

## 2026-05-12 - Live Training Viewer

Goal: watch agent rollouts during training without making headless training pay
for rendering or snapshot serialization.

Implementation:

- `train_locomotion.py` launches `live_training_viewer.py` by default.
- The viewer starts headless. The UI button says `Visualise`; clicking it asks
  the trainer to write live snapshots. The button then says `Headless`.
- In visualising mode, the viewer shows up to four rollout agents in a 2x2 grid.
  Predicted motion is orange and optional ground truth is blue.
- `Stop experiment` writes a control flag. The trainer exits cleanly after the
  current small unit of work, instead of being hard-killed.
- The bottom graph reads `live_training/loss_history.csv`, one scalar row per
  epoch. This is separate from pose snapshots, so headless mode still avoids FK
  snapshot serialization and OpenGL work.
- `--no-live-viewer` disables the viewer process.
- `--live-viewer-start-visualizing` is available for smoke tests.

Smoke test:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --folder-path data/fbx/npz_final --run-name live_perf_headless --device cuda --max-epochs 40 --batch-size 32 --rollout-schedule 1,2 --curriculum-max-epochs-per-stage 20 --training-loop agents --agent-sampling coverage --no-compile --live-viewer-close-on-exit
```

Performance check on a 40-epoch K1/K2 smoke run:

| Mode | Final best loss | Elapsed |
| --- | ---: | ---: |
| `--no-live-viewer` | `0.045720` | `2.52s` |
| default viewer, headless | `0.045720` | `2.53s` |
| viewer visualising | `0.045720` | `4.49s` |

The scalar losses were identical across modes. Headless mode writes no pose
snapshots; it only appends the scalar loss row needed by the graph.

Stop-button smoke test:

```text
live viewer stop requested before epoch=5
returncode 0
```

### 2026-05-12 live viewer fixes

Issue: starting the viewer in visualising mode worked, but flipping from
headless to visualising could appear blank. The viewer also centered agents by
subtracting the full pelvis position, which visually pushed them into the
ground plane.

Fixes:

- Trainer now rereads the tiny control JSON every training unit and tolerates
  UTF-8 BOMs in that file.
- Viewer reloads/redraws immediately when switching to `Visualise`.
- Viewer preserves vertical height when centering agents into the 2x2 grid.
- Snapshot playback uses a readable 8 FPS ping-pong loop instead of source-FPS
  wrapping over very short K-frame rollouts.
- Added `training/launch_live_k8_visual.ps1` and the desktop shortcut
  `Stepper Live Training`, which launches the known-good K8 coverage-agent
  command with visualisation already enabled.

Verification:

| Run | Purpose | Result |
| --- | --- | --- |
| `live_toggle_fix_smoke_20260512_172155` | Start headless, flip control to visualise during training | snapshot written at epoch 11, shape `[4, 2, 25, 3]` |
| `live_visual_fix_smoke_20260512_172256` | Start visualising and screenshot the OpenGL window | agents visible above the floor, loss graph updating |
| `live_fix_full_k8_capped_20260512_172648` | Full K8 accuracy/perf check with live viewer headless | best `0.020904636` at epoch 318, K8, ~49.1s |

Follow-up fix: the GL canvas was drawing into the back buffer but the manual
viewer tick did not swap buffers. `LiveTrainingViewer.redraw_canvas()` now makes
the GL context current, redraws, and swaps buffers on each visualising tick.
Screenshot comparison on `desktop_live_k8_20260512_173255` confirmed the actor
pixels change one second apart.

Loss graph layout update: the OpenGL view and loss graph now live in a vertical
splitter. Drag the sash above the graph to resize it relative to the character
view. Smoke capture with `--loss-height 280` confirmed the graph renders cleanly
when given more vertical space.

## 2026-05-12 - LR Schedule Sweep

Question: the K8 loss curve with constant `1e-4` bounced around `0.03`; test
whether a scheduled LR reduces that noise and improves final accuracy.

Setup for every run:

- `data/fbx/npz_final`
- agents + coverage sampling
- rollout schedule `1,2,4,8`
- `curriculum-max-epochs-per-stage 70`
- `max-epochs 320`
- batch size `64`
- live viewer disabled

Results:

| Run | LR schedule | Best K8 loss | Best epoch | Wall time |
| --- | --- | ---: | ---: | ---: |
| `lr_baseline_constant_20260512_180529` | constant `1e-4` | `0.020904636` | 318 | 73.0s |
| `lr_cosine_min10_20260512_180529` | whole-run cosine to `1e-5` | `0.017214196` | 319 | 72.4s |
| `lr_stage_decay70_20260512_180529` | stage decay, K8 LR `3.43e-5` | `0.020275569` | 317 | 72.7s |
| `lr_stage_cosine_min20_20260512_180529` | per-stage cosine to `2e-5` | `0.016933307` | 314 | 73.5s |
| `lr_stage_cosine_min10_20260512_181154` | per-stage cosine to `1e-5` | `0.016384257` | 315 | 72.2s |
| `lr_stage_cosine_min30_20260512_181154` | per-stage cosine to `3e-5` | `0.018484160` | 320 | 79.0s |
| `lr_stage_cosine_decay90_min20_20260512_181154` | per-stage cosine, stage reset decayed `0.9x` | `0.018037196` | 317 | 72.6s |
| `lr_stage_cosine_min05_20260512_181707` | per-stage cosine to `5e-6` | `0.015184219` | 320 | 71.7s |
| `lr_stage_cosine_min02_20260512_181900` | per-stage cosine to `2e-6` | `0.017889749` | 320 | 58.0s |

Winner: `--lr-schedule stage_cosine --lr-min-factor 0.05 --lr-stage-decay 1.0`.
It keeps the fast high-LR adaptation at the beginning of every new rollout K,
then calms the optimizer in the later part of the stage. The `2e-6` floor was
too low and underfit; `5e-6` was the best floor in this sweep.

## 2026-05-12 - Autonomous LR Scheduler

Correction: because these runs are placeholder experiments, fixed stage-shaped
LR schedules are too tied to the current clip and curriculum. Prefer an
autonomous scheduler that reacts to the observed loss instead of assuming a
particular stage duration. Keep only broad safety bounds like the initial LR and
minimum LR.

Implementation:

- Added `--lr-schedule adaptive_plateau`.
- Each new rollout K resets LR to the starting LR because the rollout task has
  changed.
- Within a K stage, LR is reduced only after the monitored loss fails to improve
  for `lr_plateau_patience_epochs`.
- Default/autonomous setup is now:
  - `learning_rate = 1e-4`
  - `lr_min_factor = 0.05` (`5e-6` floor)
  - `lr_plateau_patience_epochs = 12`
  - `lr_plateau_factor = 0.7`
  - `lr_plateau_threshold = 0.001`

Adaptive sweep:

| Run | Setup | Best K8 loss | Best epoch | Wall time |
| --- | --- | ---: | ---: | ---: |
| `lr_adapt_plateau_p8_f50_20260512_183151` | patience 8, factor 0.5 | `0.017355395` | 317 | 71.3s |
| `lr_adapt_plateau_p5_f50_20260512_183151` | patience 5, factor 0.5 | `0.017669439` | 318 | 63.8s |
| `lr_adapt_plateau_p5_f70_20260512_183151` | patience 5, factor 0.7 | `0.017384965` | 316 | 71.6s |
| `lr_adapt_plateau_p12_f50_20260512_183647` | patience 12, factor 0.5 | `0.016667884` | 318 | 57.2s |
| `lr_adapt_plateau_p8_f80_20260512_183647` | patience 8, factor 0.8 | `0.017313415` | 320 | 70.9s |
| `lr_adapt_plateau_p12_f70_20260512_183647` | patience 12, factor 0.7 | `0.015092849` | 319 | 72.2s |

Result: `adaptive_plateau` with patience 12 and factor 0.7 slightly beat the
best fixed `stage_cosine` run on this clip (`0.01509` vs `0.01518`) while also
removing the fixed stage-shape assumption.

## 2026-05-12 - Shared Comparison Viewer Refresh

Problem: `training/runs/model_comparisons/model_comparison.html` is a static
HTML file, so refreshing the browser could show an older run unless the page was
regenerated manually.

Fix:

- Training now refreshes the shared comparison HTML at shutdown from the run's
  `checkpoint_best.pt` and first source NPZ.
- `visualize_model.py` no longer has a stale hardcoded checkpoint default. With
  no arguments, it resolves the newest run under `training/runs`, uses that
  run's `checkpoint_best.pt`, and infers the source NPZ from checkpoint metadata.
- Use `--no-update-comparison-on-exit` for sweeps where the shared comparison
  page should not be overwritten.

## 2026-05-12 - Scheduled K8 Reverification

Removed the experimental long-rollout/key-bone reset branch after visual
inspection showed the scheduled autoregressive curriculum worked better for this
clip. Reran the current recommended K8 setup:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --folder-path data/fbx/npz_final --device cuda --training-loop agents --agent-sampling coverage --rollout-schedule 1,2,4,8 --curriculum-max-epochs-per-stage 70 --curriculum-stall-patience-epochs 35 --max-epochs 320 --batch-size 64 --learning-rate 1e-4 --lr-schedule adaptive_plateau --lr-min-factor 0.05 --lr-plateau-patience-epochs 12 --lr-plateau-factor 0.7 --no-compile --no-live-viewer
```

Result:

- Run: `scheduled_k8_adaptive_verify_20260512_201402`
- Best K8 loss: `0.0150928488`
- Best epoch: `319`
- Matches the prior adaptive LR best exactly within logged precision.
- Shared comparison HTML now points at this verification checkpoint.

## 2026-05-12 - Pure Delta-AE Prior Diagnosis

Goal: rerun the delta-transition AE setup without supervised losses in the
model objective, then investigate why it previously failed to match the
supervised autoregressive result.

Findings:

- Old AE normalization used `std_floor = 1e-4`. Many delta-feature channels
  have tiny or zero variance, so generated off-manifold changes in those
  channels dominated the AE gradient and made training brittle.
- Using the AE loss directly with MSE required very small learning rates; the
  model could improve K=1, but K=8 drifted badly.
- Matching the AE training loss shape with `--ae-score-loss huber` helped
  stability, but the decisive fix was raising the AE normalization floor.
- New useful prior setup: delta AE with `--std-floor 0.01`, model AE score with
  `--ae-score-loss huber`, and pure-AE checkpoint selection via
  `--best-metric ae_score`.

Reference metrics from `visualize_model.py` on `testcasc.npz`:

| Run | Objective | Rollout trained | One-step avg joint error | Full autoreg avg joint error | Full autoreg final joint error |
| --- | --- | ---: | ---: | ---: | ---: |
| `scheduled_k8_adaptive_verify_20260512_201402` | supervised | K8 | `0.004423 m` | `0.007902 m` | `0.023608 m` |
| `ae_pure_probe_20260512_212626_k248_huber_lr3e6` | pure AE, old std floor | K8 | not rendered | about `0.06 m` train-window RMSE | poor |
| `ae_pure_probe_20260512_213304_stdfloor001_k248_huber_lr3e6` | pure AE, std floor `0.01` | K8 | `0.002231 m` | `0.017890 m` | `0.034383 m` |
| `ae_pure_probe_20260512_213848_stdfloor001_k16_huber_lr2e6` | pure AE, std floor `0.01` | K16 | `0.002234 m` | `0.013934 m` | `0.026468 m` |
| `ae_pure_probe_20260512_214146_stdfloor001_k32_huber_lr1e6` | pure AE, std floor `0.01` | K32 | `0.002257 m` | `0.009502 m` | `0.020027 m` |

Interpretation: the pure AE framework can now reach supervised-level local
transition accuracy and near-supervised full-rollout accuracy on this clip. The
remaining gap is mostly long-horizon drift: the AE scores transition realism,
not exact phase/state alignment. Increasing rollout K helps because the model
must keep its own generated states inside the AE transition manifold for longer.

The shared comparison HTML currently points to the K32 pure-AE run above.

## 2026-05-12 - UE5 FBX Axis Compatibility

Problem: a UE5 FBX under `ue5/test` reported a Z-up axis system while the
Cascadeur FBX files report Y-up. The old read-time canonicalization only worked
for the Cascadeur-style data and made UE5 FK reconstruction and collider
orientation visibly wrong.

Fix:

- Z-up UE5 sources are now canonicalized as
  `canonical = [source_x, source_z, -source_y]`, which makes source `-Y`
  become training/viewer `+Z` forward.
- Rotation matrices are transformed with the matching row-vector change of
  basis, `R_canonical = P^-1 R_source P`.
- Foot/toe collider axes are now selected from the actual bone basis using the
  foot-to-ball vector plus the vertical axis, instead of hardcoding Cascadeur
  foot axes.
- Hand colliders now choose their width/up axes from the actual hand basis after
  choosing a forward guide from fingers or forearm direction.

Verification:

- Cascadeur `testcasc.npz` FK reconstruction stayed unchanged at about
  `0.0017 m` mean position error.
- UE5 `M_Neutral_Walk_Loop_F.npz` FK reconstruction improved from about
  `1.34 m` mean position error to about `0.0061 m`.
- The regenerated UE5 NPZ now loads with root motion along canonical `+Z` and
  contact counts `contactL=69`, `contactR=68`.

## 2026-05-13 - Contact Metric Clarification

Confirmed the contact pipeline keeps foot height and foot slide as separate
measurements:

- Height/penetration uses `foot_lowest_heights_and_points`, i.e. the absolute
  lowest point on either the foot collider or toe collider.
- Sliding uses `foot_slide_speeds`, which does not reuse the lowest point. It
  solves the continuous 2D sole-rectangle minimization in ground-plane XZ for
  both the foot and toe collider and takes the smaller distance.

This distinction matters for future contact losses: source contact detection can
stay 2D, but a pinned-foot training loss should punish the full velocity of the
chosen persistent contact point.

Follow-up: changed the training contact-slide loss to use
`foot_contact_point_speeds`. It solves the same continuous sole-rectangle
problem over the foot and toe, but minimizes full 3D displacement of the
persistent local point instead of only ground-plane XZ displacement. The
existing pinned height threshold was not tightened.

Next constraint-only pass tightened the pinned contact rule:

- at least one generated contact must reach probability `0.8`;
- pinned contact-point speed threshold is `0.005 m/s`;
- COM must stay within `0.50 m` horizontally of the mean foot location;
- COM must stay within `0.75 m` horizontally of the root.

Pinned hover/slide terms now use a hard generated-contact mask. If a foot is
classified as pinned, it pays the full hover/slide loss; the loss is no longer
discounted by the contact probability.

The hard pin mask is binary/full-price: once contact probability reaches the
`0.8` threshold, hover/slide losses are not discounted by contact probability.
Contact logits are trained by the separate "at least one foot pinned" margin
loss plus a constraint-only bad-contact gate. The bad-contact gate only updates
the contact output: if a generated foot is hovering or sliding, high contact
probability for that foot is discouraged. The foot physics losses themselves
still charge the pose full price whenever the hard pin is on.

An experimental `best_foot` mode was added for the constraint-only trainer. In
that mode the geometry selects the lower-violation foot as the pin candidate,
charges that selected foot the full hover/slide loss, and trains contact logits
to report the selected contact. This is still not supervised by source contact
labels; it is a derived geometric target used to avoid the discontinuous
"which foot should be pinned?" chicken-and-egg during pure constraint training.

## 2026-05-13 - Pose-Aware AE Prior Recheck On Old Clip

After abandoning the constraint-only direction for now, the old `data/fbx`
`testcasc` clip was regenerated into `data/fbx/npz_final` and the AE prior
workflow was rechecked.

Delta-only reference rerun:

- Run: `ae_delta_oldclip_verify_20260513_131331_autoreg_k32`
- Objective: pure delta-transition AE score, no contact physics losses.
- One-step average joint error: `0.002284 m`
- Full autoregressive average joint error: `0.009183 m`
- Full autoregressive final joint error: `0.018618 m`

Pose-aware AE rerun:

- Run: `ae_poseaware_oldclip_verify_20260513_132510_k32`
- Objective: full transition AE score, no contact physics losses.
- AE feature dimension: `877`
- Latent dimension: `128`
- One-step average joint error: `0.002332 m`
- Full autoregressive average joint error: `0.006659 m`
- Full autoregressive final joint error: `0.004274 m`
- Full autoregressive max joint error: `0.011938 m`

Interpretation: adding pose context to the AE prior preserved one-step accuracy
and greatly reduced long-rollout drift on the old single walking clip. This is
the strongest pure-AE result so far on that clip, and it remains free of direct
DeepMimic-style supervised pose loss.

## 2026-05-13 - Pose-Aware AE Prior On UE5 Test Clip

The same pose-aware AE prior workflow was rerun on
`ue5/test/npz_final/M_Neutral_Walk_Loop_F.npz`.

- Run prefix: `ae_poseaware_ue5test_fullroll_20260513_142434`
- Source clip length: `121` frames at `30 FPS`
- Objective: pure pose-aware transition AE score, no direct supervised pose loss
  and no contact physics losses.
- Rollout schedule: `K=1 -> 8 -> 16 -> 32 -> 64 -> 119`, where `K=119` covers
  the full valid autoregressive window for this clip.
- Final checkpoint:
  `training/runs/ae_poseaware_ue5test_fullroll_20260513_142434_k119/checkpoints/checkpoint_best.pt`

Final visualization metrics:

- One-step average joint error: `0.005601 m`
- One-step max joint error: `0.007443 m`
- Full autoregressive average joint error: `0.079561 m`
- Full autoregressive final joint error: `0.157714 m`
- Full autoregressive max joint error: `0.177733 m`

Interpretation: the UE5 test clip contains repeated gait cycles, so exact
long-horizon pose alignment is less important than style continuity. The
one-step prediction is clean, and the full autoregressive rollout keeps the
walking style coherent even though the pose drifts away from the exact source
phase over time. That drift is acceptable for this experiment because the goal
is not frame-locked imitation; it is a reusable AE-style motion prior that can
keep producing a plausible walk.

An optional low-learning-rate polish pass was started afterward, but the
completed `K=119` result above is the accepted conclusion for this experiment.

## 2026-05-13 - Visual Checkpoint Reports

Naive frame distance to ground truth is no longer the main success criterion for
AE-style runs, especially when repeated gait cycles can drift in phase while
still looking correct. To make future experiments easier to judge, an
asynchronous visual report sidecar was added.

Design:

- The trainer periodically writes `checkpoint_last.pt` as it already does.
- `training/visual_reporter.py` runs as a separate process and watches that
  checkpoint.
- When a new checkpoint appears, the sidecar performs an autoregressive rollout
  and writes five static overlay snapshots at `0%`, `25%`, `50%`, `75%`, and
  `100%`.
- The report lives at
  `training/runs/<run_name>/visual_reports/latest/index.html`; older sampled
  reports are copied to epoch-stamped folders.
- The report also writes `metrics.json`, but the visual overlays are intended
  to steer style/coherence decisions when direct ground-truth deltas are no
  longer semantically decisive.

Performance rule: this must stay outside the hot training path. Training never
waits for the visual report process. If the sidecar is too slow, it skips stale
states and renders only the latest saved checkpoint. The only training-side cost
is the normal periodic `checkpoint_last.pt` write, and the whole feature can be
disabled with `--no-visual-reporter`.

## 2026-05-13 - Cyclic Animation Sampling

Added a `--cyclic-animation` flag for loop-clean clips. The point is to avoid
losing random frame initialization when the rollout window approaches the full
clip length.

Semantics:

- The final frame is treated as the duplicated loop-closing frame.
- Trainable starts cover `1..T-2`, i.e. the whole clip minus that duplicate
  final frame.
- Body pose indices wrap modulo `T-1`, so the target after frame `T-2` is frame
  `0`.
- Root motion does not jump back to frame `0`. The root transform past the end
  is extrapolated by repeating the clip's own root-delta sequence. In practice,
  the local root delta immediately after the seam matches the first local root
  delta of the source clip.
- If validation is disabled, the dataset now keeps the full start set instead
  of silently reserving a validation fraction.

Smoke checks on `ue5/test/npz_final/M_Neutral_Walk_Loop_F.npz`:

- `T=121`, cyclic period `120`.
- Cyclic train starts: `119`, matching `T-2`.
- Logical frame `120` uses body pose frame `0` and root transform frame `120`.
- The local root delta from logical `120 -> 121` matches source `0 -> 1` within
  numerical precision.

Quick training check:

- Run prefix: `cyclic_quick_20260513_160336`
- AE prior: cyclic pose-aware AE, `119` transitions, `80` epochs, about `7s`.
- Controller: resumed from accepted
  `ae_poseaware_ue5test_fullroll_20260513_142434_k119`, then ran a short cyclic
  `K=120` polish with coverage agents for `20` epochs, about `113s`.
- Non-cyclic accepted K119 reference:
  - one-step avg `0.005601 m`
  - autoreg avg `0.079561 m`
  - autoreg end `0.157714 m`
  - autoreg max `0.177733 m`
- Cyclic quick K120 result:
  - one-step avg `0.005397 m`
  - autoreg avg `0.076966 m`
  - autoreg end `0.111904 m`
  - autoreg max `0.188114 m`

Correction after visual inspection: this quick cyclic checkpoint is not an
accepted replacement for the K119 result. It improved average/end joint drift,
but it worsened foot clearance. The accepted K119 model has predicted lowest
foot heights close to the source collider baseline:

- Accepted K119 predicted min foot heights: left `-0.0510 m`, right `-0.0445 m`
- Source min foot heights: left `-0.0471 m`, right `-0.0447 m`

The cyclic quick K120 checkpoint went visibly deeper:

- Cyclic quick K120 predicted min foot heights: left `-0.0791 m`, right
  `-0.0862 m`

Interpretation: the cyclic sampling implementation itself passed seam/index
checks, but the short cyclic AE-only polish should not be considered better.
Future acceptance checks must include foot-clearance/contact metrics in addition
to joint drift and visual style.

## 2026-05-13 - Omni Directional Pose-Aware AE Run

Dataset:

- Source folder: `ue5/animations_omni_only`
- Generated folder: `ue5/animations_omni_only/npz_final`
- Clips: 11 total: 8 omni walk directions, 2 extra lateral-foot variants, and
  1 idle loop.
- Skeleton: 26 stored bones including `root`, 25 model body bones.
- FPS: `30` for all clips.
- Cyclic sampling enabled. Shortest trainable cyclic period is `111`, so the
  final rollout schedule used `K=111`.

Important setup note: pure AE controller training must pass
`--no-contact-physics-losses`. The first omni controller attempt accidentally
left contact physics losses enabled, which produced misleadingly bad results and
large AE totals. The accepted run below is pure pose-aware transition AE prior:
no supervised pose loss and no contact/physics losses.

Performance fix:

- The slow first clean attempt used `--agent-batch-clips 0`, which mixed clips
  inside each mini-batch and forced many small per-clip rollout groups.
- The accepted run resumed from the clean K=1 checkpoint and used
  `--agent-batch-clips 1`, so each mini-batch stays on one clip and remains
  vectorized.
- K=8 speed improved from roughly `20s/epoch` to roughly `1-2s/epoch`.

Prior:

- Run: `ae_poseaware_omni_cyclic_20260513_172626`
- Checkpoint:
  `training/runs/ae_poseaware_omni_cyclic_20260513_172626/checkpoints/checkpoint_best.pt`
- AE prior reached target at epoch `214`, reconstruction loss `0.002041`.
- Tier sanity at epoch 200: clean `0.00513`, slight `0.00755`, bad `0.315`,
  noise `1.304`.

Controller:

- Run: `ae_poseaware_omni_pure_cyclic_fast_20260513_181213`
- Checkpoint:
  `training/runs/ae_poseaware_omni_pure_cyclic_fast_20260513_181213/checkpoints/checkpoint_best_k111.pt`
- Resumed from clean K=1:
  `training/runs/ae_poseaware_omni_pure_cyclic_20260513_180204/checkpoints/checkpoint_best_k01.pt`
- Schedule: `K=8 -> 16 -> 32 -> 64 -> 111`.
- Best stage scores:
  - K8: `0.040064`
  - K16: `0.041574`
  - K32: `0.043662`
  - K64: `0.026823`
  - K111: `0.017297`
- The script does not currently auto-stop after the final stage cap, so the run
  was manually stopped after K111 reached the intended stage budget.

Best K111 sampled visualization metrics:

- Forward walk `M_Neutral_Walk_Loop_F`:
  - one-step avg `0.008850 m`
  - autoreg avg `0.036826 m`
  - autoreg end `0.048970 m`
  - autoreg max `0.052801 m`
- Backward walk `M_Neutral_Walk_Loop_B`:
  - one-step avg `0.007204 m`
  - autoreg avg `0.068557 m`
  - autoreg end `0.081624 m`
  - autoreg max `0.117977 m`
- Lateral-left `M_Neutral_Walk_Loop_LL`:
  - one-step avg `0.010364 m`
  - autoreg avg `0.075683 m`
  - autoreg end `0.103676 m`
  - autoreg max `0.118718 m`
- Lateral-right `M_Neutral_Walk_Loop_RR`:
  - one-step avg `0.010717 m`
  - autoreg avg `0.091326 m`
  - autoreg end `0.155555 m`
  - autoreg max `0.160544 m`
- Idle `M_Neutral_Stand_Idle_Loop`:
  - one-step avg `0.003750 m`
  - autoreg avg `0.010861 m`
  - autoreg end `0.013471 m`
  - autoreg max `0.015641 m`

Interpretation: the full omni pure-AE result is a good accepted candidate.
Forward and idle are tight. Lateral and backward motions drift more in exact
pose phase, especially `RR`, but visual overlays remain coherent and gait-like.
This is expected for the AE objective: it is style/transition-prior training,
not direct frame-locked supervised imitation.

## 2026-05-13 - Training Harness Performance Audit

Goal: make the trainers harder to accidentally run in a slow configuration and
remove diagnostic synchronization overhead that does not affect learning.

Environment:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: `2.11.0+cu126`
- CUDA runtime reported by PyTorch: `12.6`
- Installed `triton-windows==3.6.0.post26` so `torch.compile` can be tested on
  Windows.

Findings:

- The multi-clip batch issue was a true absolute speed issue, not just an epoch
  accounting shift. Mixed random-agent batches split one batch into many
  per-clip rollout groups, causing many small FK/loss/backward paths.
- `--agent-batch-clips 1` keeps random-agent batches on one clip and preserves
  vectorization. This is now the default in the supervised trainer too.
- `torch.compile` is technically available after installing `triton-windows`,
  but it is not a speed win for these rollout trainers on this machine. Compile
  cold-start is several seconds, and the K8 steady-state was not faster than
  eager mode. The trainer now treats compile as opt-in and runs a forward plus
  backward probe before accepting it.
- The AE-prior trainer was forcing GPU/CPU synchronization for diagnostic
  metrics inside every rollout step. Those metrics now accumulate on GPU and
  synchronize once per batch. Ground-truth diagnostic RMSEs are sampled every
  `--diagnostic-metrics-every-epochs` epochs by default instead of being
  mandatory every epoch. The AE loss itself is unchanged.
- Validation remains disabled by default; the benchmarked fast path uses
  train-loss driven scheduling/checkpointing.

Benchmarks, reporting/viewers disabled:

- Supervised omni K8, random agents, `agent_batch_clips=1`: `3.74s` total wall
  for 8 epochs, about `0.23-0.32s/epoch` after setup.
- Supervised omni K8, random agents, `agent_batch_clips=0`: `40.84s` total wall
  for 8 epochs, about `2.8-5.8s/epoch`.
- Pure AE omni K8, random agents, `agent_batch_clips=1`: about `2.4s` training
  elapsed for 8 epochs after setup.
- Pure AE omni K8, random agents, `agent_batch_clips=0`: about `25.1s` training
  elapsed for 8 epochs.
- Pure AE omni K64 with diagnostic RMSE every epoch: `7.41s` total wall for 2
  epochs.
- Pure AE omni K64 with diagnostic RMSE disabled: `7.02s` total wall for 2
  epochs. The larger win is avoiding per-step CPU sync; sparse diagnostics are
  mostly a cleanliness/long-run safety improvement.

Current default speed posture:

- Eager CUDA training by default.
- `--agent-batch-clips 1` by default for random-agent batches.
- Live viewer starts headless and writes no pose snapshots until requested.
- Visual reporter remains asynchronous and can be disabled with
  `--no-visual-reporter` for timing sweeps.
- Use `--compile` only for explicit compiler experiments.

## 2026-05-13 - Rollout Compatibility Monitor And Direction Finetune Probe

Problem:

- On the omni dataset, the pure pose-aware AE controller could visually choose
  a nearby body style for a commanded root direction. The observed case was
  lateral motion looking more like a diagonal/nearby walk, which can create
  foot skating even if the generic AE transition score remains plausible.

Benign monitor added:

- `training/inspect_rollout_compatibility.py`
- This is read-only. It does not affect training, checkpoint selection, or
  gradients.
- For a trained controller checkpoint, it rolls out each command clip, keeps the
  generated body transitions fixed, swaps only the AE root-condition slice
  across candidate source clips, and reports which candidate root direction
  scores best.
- This gives a cheap sensor for "the command is pure left, but the generated
  body transition scores like forward-left/back-left/etc." without relying only
  on screenshots.

Prior-side diagnostic added:

- `training/inspect_ae_compatibility.py`
- It prints a root/body score matrix for an AE prior checkpoint.

Baseline accepted checkpoint monitor:

- Controller:
  `training/runs/ae_poseaware_omni_pure_cyclic_fast_20260513_181213/checkpoints/checkpoint_best_k111.pt`
- Compatibility prior probe:
  `training/runs/ae_compat_omni_probe_20260513_02/checkpoints/checkpoint_best.pt`
- Result: all main directions ranked correctly except:
  - `LL -> BL`, rank `3`, gap about `0.434`
  - `RR -> BR`, rank `3`, gap about `0.240`
- This confirmed the monitor catches the exact failure family that was visible
  by eye.

Compatibility prior probe:

- Run: `ae_compat_omni_probe_20260513_02`
- Added an optional compatibility head to the transition AE.
- Positive samples are clean root/body transitions from the same clip.
- Direction negatives keep the root-condition slice from clip A but take body
  transition features from a different clip B.
- Temporal skip negatives are supported but were not emphasized because the old
  AE already punishes frame skips strongly.
- The resulting AE matrix strongly separates wrong direction pairs. Same-root
  lateral variants such as `LL/LR` and `RL/RR` remain intentionally near each
  other because they share the same root motion but differ in lead-foot phase.

Controller finetune probes:

- Aggressive K111 compatibility finetune with score weight `0.1` was unstable
  and tended toward low-motion/static solutions. It is not accepted.
- Softer K8 finetune:
  `ae_compat_finetune_omni_k8_multibatch_20260513_01`
  - Resumed from the accepted K111 controller.
  - Used the compatibility prior with score weight `0.02`, small LR, and more
    batches per epoch.
  - Monitor result for `checkpoint_last.pt`: all commands ranked correctly
    except same-direction lateral alternates:
    - `LL -> LR`, rank `2`, gap about `0.000003`
    - `RR -> RL`, rank `2`, gap about `0.000006`
- Short K111 polish:
  `ae_compat_finetune_omni_k111_polish_20260513_01`
  - Kept the same monitor behavior.
  - LL visualization was regenerated to
    `training/runs/model_comparisons/model_comparison.html`.
  - Exact LL autoreg drift remained worse than the old accepted K111 checkpoint,
    so this should be considered a diagnostic/partial finetune result, not a
    replacement accepted model.

Current interpretation:

- The new compatibility monitor is useful and caught the problem automatically.
- The compatibility prior is direction-aware, but direct controller finetuning
  needs careful balancing because a low compatibility score alone can reward
  low-motion solutions on some sampled batches.
- For future experiments, use this monitor as a read-only alarm alongside visual
  reports. Do not treat the compatibility score as a production loss without
  additional safeguards.

### Model-aware transition AE idea

The next proposed experiment is to keep the accepted K111 omni controller as the
baseline and train a second transition AE with the same feature vector as the
pose-aware AE, but with generated controller transitions used as explicit
negative examples.

Rationale:

- The original pose-aware AE answers: "does this transition look like something
  from the motion dataset?"
- A model-aware AE should answer: "does this transition look like ground truth,
  or like the current controller's own artifacts?"
- This makes the prior specialize to the failure modes the current controller
  actually produces: wrong direction/body pairing, low-motion shortcuts,
  hovering, skating, or other generated-only habits.

Implementation plan:

- Positive transitions: clean ground-truth transitions from the dataset.
- Negative transitions: autoregressive rollouts from the accepted controller.
- Input vector: unchanged from the current pose-aware AE, so the controller
  training path can swap the prior without any input-dimension change.
- Objective: low reconstruction energy on real transitions, and a margin that
  pushes generated transitions to higher reconstruction energy.
- Then freeze this model-aware AE and finetune the controller against its score.
- This can form an iterative loop later:
  `controller -> generated fakes -> model-aware AE -> controller finetune`.

Important guardrail:

- Do not trust this AE score as proof that foot sliding is solved. Foot skating
  must still be judged with the geometric foot-slide monitor and visual reports.
  The AE is a training signal, not the final physicality judge.

Model-aware AE experiment results:

- Run: `modelaware_ae_omni_k111_fakes_20260513_01`
  - Initialized from the accepted pose-aware AE.
  - Positives: clean omni dataset transitions.
  - Negatives: rollouts from accepted K111 controller
    `ae_poseaware_omni_pure_cyclic_fast_20260513_181213/checkpoint_best_k111.pt`.
  - Same feature dimension as the pose-aware AE: no controller input/output
    dimension change.
  - Training separated the static fake set clearly:
    - final real energy about `0.0030`
    - final fake energy about `0.0353`
    - about `99.2%` of fake transitions above the `0.02` margin.

- Finetune: `modelaware_ae_finetune_omni_k111_20260513_01`
  - Resumed from the accepted K111 controller.
  - Used the model-aware AE as the frozen prior.
  - Geometric foot-slide monitor:
    - baseline K111 mean contact p95: `1.2105 m/s`
    - loop-1 best mean contact p95: `1.0313 m/s`
    - loop-1 last mean contact p95: `1.4195 m/s`
    - ground truth mean contact p95: `0.1543 m/s`
  - Interpretation: the model-aware AE gave a real but modest improvement on
    the selected checkpoint. The final checkpoint regressed, so this is not yet
    a robust self-driving loop.

- Second loop:
  - AE: `modelaware_ae_omni_loop2_fakes_20260513_01`
  - Finetune: `modelaware_ae_finetune_omni_k111_loop2_20260513_01`
  - Geometric foot-slide monitor:
    - loop-2 best mean contact p95: `1.0331 m/s`
    - loop-2 last mean contact p95: `1.2092 m/s`
  - Interpretation: loop 2 did not compound the improvement. The approach has
    signal, but the current AE energy is still not a reliable foot-skating
    objective.

Important lesson:

- The model-aware AE can detect that generated transitions differ from real
  transitions, but the current feature/objective still does not directly encode
  "the planted sole point should not slide."
- AE-score checkpoint selection is noisy with one-clip agent batches. Some
  low-score epochs line up with near-zero `motion_rms`, often from easy/idle-like
  sampled batches, so AE score alone must not be treated as acceptance.
- If this path continues, the next useful change is likely to add explicit
  contact geometry features to the AE input/energy, or keep the AE prior but add
  a separate geometric foot-slide loss/monitor for checkpoint acceptance.

Combined original + dynamic AE probe:

- Change: controller finetuning can now average multiple frozen AE priors with
  equal weight. The first use is:
  `0.5 * original_poseaware_AE + 0.5 * current_model_aware_AE`.
- Rationale: keep the original dataset/style manifold as an anchor while the
  dynamic AE keeps learning the controller's current loopholes.
- Resume point requested by inspection: end of cycle 1, not cycle 2, because
  cycle 2 looked worse visually.
- Run: `modelaware_loop_combined_from_cycle1_20260514_01`
  - Initial model:
    `modelaware_loop_continue_20260514_04_model_cycle01/checkpoint_best.pt`
  - Initial dynamic AE:
    `modelaware_loop_continue_20260514_04_ae_cycle01/checkpoint_best.pt`
  - Anchor AE:
    `ae_poseaware_omni_cyclic_20260513_172626/checkpoint_best.pt`
- External foot-slide monitor:
  - start mean contact p95: `1.0061 m/s`
  - combined cycle 1 mean contact p95: `0.9472 m/s`
  - combined cycle 2 mean contact p95: `1.0771 m/s`
  - ground truth mean contact p95: `0.1543 m/s`
- Interpretation: the equal-weight anchor improved the first combined cycle,
  but the next cycle regressed. The best checkpoint from this branch is cycle 1,
  and the loop should not blindly follow dynamic AE loss without foot-slide or
  visual acceptance checks.

Forward-walk diagnostic correction:

- A bad first inspection mistake was to sort isolated worst foot-slide spikes.
  That missed the actual visual story: on the forward walk, the generated motion
  starts mostly forward-walk-like, then progressively drifts into a more lateral
  gait while the root still asks for forward motion.
- For AE/model-aware rollouts, source `.npz` contact labels are not a valid
  generated-contact mask once the model phase drifts. Use generated geometry
  instead for quick inspection.
- Quick forward-only temporal proxy on
  `modelaware_loop_combined_from_cycle1_20260514_01_model_cycle02/checkpoint_last.pt`:
  - support slide mean, first 15 frames: `0.141 m/s`
  - support slide mean, last 15 frames: `0.858 m/s`
  - support slide p95, first 15 frames: `0.255 m/s`
  - support slide p95, last 15 frames: `1.144 m/s`
  - lateral gait excess, first 15 frames: `+0.031`
  - lateral gait excess, last 15 frames: `+0.282`
- Lesson: for non-supervised/phase-free experiments, acceptance monitoring
  needs chronological drift metrics, not only frame-aligned ground-truth deltas
  or top-N worst slide frames.

Baseline correction for foot-slide finetuning:

- For omni finetuning, the correct accepted baseline is the omni K111
  controller, not the forward-only `ue5/test` K119 branch:
  `training/runs/ae_poseaware_omni_pure_cyclic_fast_20260513_181213/checkpoints/checkpoint_best_k111.pt`
- Its matching omni pose-aware AE prior is:
  `training/runs/ae_poseaware_omni_cyclic_20260513_172626/checkpoints/checkpoint_best.pt`
- Its dataset is:
  `ue5/animations_omni_only/npz_final`
- The forward-only K119 branch remains useful for the single forward clip, but
  should not be used as the baseline or AE prior for all-direction omni
  foot-slide finetuning.

Accepted foot-slide-aware omni baseline:

- New best accepted baseline:
  `training/runs/footslide_ae_from_omni_k111_allanims_w20_lr1e6_resume_20260514/checkpoints/checkpoint_best_k111.pt`
- Started from:
  `training/runs/ae_poseaware_omni_pure_cyclic_fast_20260513_181213/checkpoints/checkpoint_best_k111.pt`
- Prior:
  `training/runs/ae_poseaware_omni_cyclic_20260513_172626/checkpoints/checkpoint_best.pt`
- Dataset: `ue5/animations_omni_only/npz_final`
- Objective: `AE prior loss + 20.0 * simple generated-geometry support-foot slide loss`
- Ground-truth zero-loss slide threshold: `0.213529931 m/s`
  (`1.05 * max ground-truth support slide`)
- Cyclic K: `111`
- Best checkpoint metadata:
  - epoch: `1966`
  - best combined objective: `0.0021822865`
  - learning rate: `1e-6`
  - batch size: `256`
  - training loop: `agents`
- This replaces the previous pure-AE omni K111 checkpoint as the preferred
  baseline for future omni experiments unless a later visual inspection proves
  otherwise.
