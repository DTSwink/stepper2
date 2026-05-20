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

## 2026-05-17 - Visual Reporter Retired

The automatic `training/visual_reporter.py` sidecar is disabled in both
`train_locomotion.py` and `train_locomotion_ae_prior.py`. The old
`--visual-reporter` flag is still accepted so older launch commands do not
break, but it now prints a message and does not spawn a process. Visual
inspection should use the standalone model viewer instead.

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

Hybrid periodic + transition dataset rollout rule:

- Periodic folders use cyclic indexing and can contribute the full requested
  rollout K.
- Non-periodic transition folders do not wrap. When a vectorized non-periodic
  cohort reaches a clip end during a long AE rollout, the cohort is reset to a
  fresh valid `(clip, frame)` start instead of producing fake wraparound data.
  This preserves full `active=1.0` training signal while keeping episode
  boundaries clean.
- If a non-periodic clip can support the requested K, starts are sampled only
  from full-K-safe ranges. Reset-at-end is used when requested K is longer than
  that clip can possibly provide or when a sampled start reaches the end.
- Short transition clips are not excluded at K64/K111. Instead, random agent
  clip sampling is weighted by `1 / expected_active_steps(clip, K)`, so short
  clips are sampled more often when K is large. This keeps the distribution of
  active training time per animation roughly uniform rather than letting long
  loops dominate.
- Coverage audit on 2026-05-15 found that `active=1.0` was not enough by
  itself: with 256 rows in one cohort, one row sampled near the end of a
  non-periodic clip could force a reset after one frame. Future runs therefore
  use `agent_min_cohort_steps=8` by default, which avoids near-end starts for
  non-periodic cohorts when the clip is long enough, while still allowing truly
  short clips to participate.
- Follow-up audit found the cleaner fix: keep each vectorized batch tied to one
  animation clip, but reset only the expired rows back into fresh starts in that
  same clip. This preserves the fast one-clip batched path, avoids fake
  wraparound, avoids excluding short clips, and avoids the flimsy whole-cohort
  reset behavior. A small CPU probe at `B16 K32` measured same-clip per-agent
  reset at `0.316s` versus arbitrary cross-clip per-agent reset at `2.866s`.
- Important performance note: fully independent per-row resets into arbitrary
  clips are semantically clean but destroy one-clip vectorization and were far
  too slow. The accepted implementation is same-clip per-agent reset: each
  expired row gets a fresh start in its current clip, so the batch remains
  vectorized by one clip while the agents behave independently at clip ends.
- For noisy AE + footslide fine-tuning, use `gradient_accumulation_batches=2`
  as the first stabilizing test: two separate one-clip cohorts are averaged
  before one optimizer update. This preserves the fast one-clip vectorized
  rollout while making each update less dependent on a single hard/easy clip.

### 2026-05-15 - Turn Clips And Future-Window Safety

- Problem found on `M_Neutral_Walk_Turn_R_180_Lfoot`: root heading was being
  computed from the root local `+Z` row, but in the project UE mannequin NPZs
  local `+Z` is vertical and local `-Y` is the actual character/root heading.
  This made 180-degree turn clips look orientation-invariant to the code even
  though root velocity direction flipped in world space.
- Fix: `heading_yaw_from_root()` now projects root local `-Y` onto the ground
  plane. On the test clip, old yaw span was `0 deg`; corrected yaw span is
  `180 deg`. Stable pre-turn and post-turn root/future features now match:
  frame `5` vs `45` root/future MAE about `0.000001`, while pose features still
  differ as expected.
- Rejected idea: do not extrapolate non-periodic root motion past the clip end.
  That invents command data and can poison transition clips.
- Accepted rule: non-periodic agents must reset before either the next target
  frame or the dense future-root window would pass the clip end. With default
  `future_window=8`, a `55` frame clip has last safe current frame `46`.
- Short non-periodic clips at `K=111` are not forced to start at frame `1`.
  Example: the 55-frame turn clip samples starts up to frame `39` with
  `agent_min_cohort_steps=8`, runs at least 8 valid steps, then same-clip
  per-agent reset keeps it active without clamped/missing future-root features.
- The AE prior clean-transition collector also excludes non-periodic current
  frames whose future-root lookahead would pass the clip end, so the prior is
  trained in the same factual feature space as controller rollouts.
- Baseline safety check: both `ue5/animations_omni_only/npz_final` and
  `ue5/animations_omni_only_full/npz_final` have corrected yaw span `0 deg`
  and old-vs-new root-delta MAE `0.0` for every clip. The accepted omni
  baseline checkpoint is therefore not invalidated by this heading fix.
- Diagnostic file:
  `training/runs/diagnostics/root_heading_feature_diagnostic_20260515.html`.
  On `M_Neutral_Walk_Turn_R_180_Lfoot`, stable pre/post-turn frames `5/45`
  have root/future MAE `0.000001`, while middle-turn frames `25/29` have
  root/future MAE `0.243082`.
- Corrected hybrid AE prior:
  `training/runs/ae_poseaware_hybrid_headingfix_20260515_151658/checkpoints/checkpoint_best.pt`
  (trained from scratch after the heading/schema/future-window fixes).
- Corrected controller restart:
  `training/runs/hybrid_headingfix_from_omni_footslide_baseline_w005_k111_20260515_151853`
  resumes from accepted baseline
  `training/runs/footslide_ae_from_omni_k111_allanims_w20_lr1e6_resume_20260514/checkpoints/checkpoint_best_k111.pt`,
  uses the corrected AE prior above, K111, same-clip per-agent reset,
  `gradient_accumulation_batches=2`, and simple footslide weight `0.05`.
- Replacement scheduled run after user correction:
  `training/runs/hybrid_headingfix_curriculum_from_omni_w005_accum1_omni2x_20260515_152849`.
  This is the preferred run for this branch. It uses the same corrected AE
  prior and same accepted omni footslide baseline, but changes:
  - rollout schedule `1,2,4,8,16,32,64,111` instead of K111-only,
  - `gradient_accumulation_batches=1`,
  - periodic/omni group total sampling weight `2`,
  - non-periodic/transition group total sampling weight `1`,
  - slower K advancement: `curriculum_min_epochs=80`,
    `curriculum_min_eligible_clip_visits=0.5`, stall patience `160`, max stage
    cap `400`.
  With 15 periodic clips and 214 transition clips, the sampler gives the omni
  folder about `66.7%` of one-clip cohorts and the transition folder about
  `33.3%`, instead of letting transition clip count dominate.

### 2026-05-15 - Packed Per-Agent Multi-Clip Rollouts

- The run above exposed a bad statistical side effect of `agent_batch_clips=1`:
  every optimizer step could be 256 agents from the same animation. At epoch
  `1914`, the whole batch sampled `M_Neutral_Walk_Arc_F_Small_L`, producing a
  single-step spike (`ae_score=4.07932`, raw footslide `1.90956`) before the
  next one-clip batch returned to normal. This is unacceptable for the hybrid
  dataset because one hard transition can dominate an update.
- Implemented packed AE-prior rollouts for random-agent training. All clips are
  prepacked into dense frame tensors, while each row carries its own `clip_id`.
  Root state, pose lookup, FK, AE transition features, and footslide are then
  computed on dense batch tensors. Non-cyclic rows reset independently to fresh
  random clips before their target/future-root frames would be missing.
- `--agent-batch-clips 0` is now the default for the AE-prior trainer and uses
  the packed path when contact-physics losses are disabled. `--agent-batch-clips
  1` remains available as the legacy one-clip cohort path for debugging.
- Correctness smoke check: for a same-clip batch, packed and legacy losses
  matched exactly (`abs_diff=0.0`), including AE score and simple footslide.
  A 256-agent packed sample covered 154 unique clips, confirming true per-agent
  random clip sampling.
- K16/B256 benchmark with backward included:
  - packed per-agent mixed clips: `15.6s` train elapsed for 8 epochs
    (`18.2s` total wall including setup),
  - legacy one-clip cohorts: `13.0s` train elapsed for 8 epochs
    (`15.5s` total wall),
  - old un-packed mixed clips: `452.9s` train elapsed for 3 epochs
    (`455.4s` total wall).
  So packed per-agent sampling costs only about 18-20% over the flawed one-clip
  shortcut, while being roughly two orders of magnitude faster than the old
  Python grouped mixed path.
- Added `--final-stage-random-rollout`. Once the curriculum reaches the last
  scheduled K, each optimizer microbatch samples its effective rollout length
  from the schedule rungs themselves, e.g. `{1,2,4,8,16,32,64,111}`, not from
  every integer in between. Earlier stages still train at their exact scheduled
  K. This keeps long-horizon training present without making every final-stage
  update a maximal K111 rollout.
- Smoke checks:
  - forced K111 packed rollout with independent resets completed forward +
    backward,
  - schedule-rung smoke advanced through `1,2,4` and then sampled effective K
    only from those rungs.
- Current run:
  `training/runs/hybrid_headingfix_curriculum_packed_scheduleK_from_omni_baseline_w005_20260515_173112`.
  It resumes from accepted omni baseline
  `training/runs/footslide_ae_from_omni_k111_allanims_w20_lr1e6_resume_20260514/checkpoints/checkpoint_best_k111.pt`,
  uses corrected hybrid AE prior
  `training/runs/ae_poseaware_hybrid_headingfix_20260515_151658/checkpoints/checkpoint_best.pt`,
  packed per-agent random clips, omni/transition group weights `2:1`, simple
  footslide weight `0.05`, and final-stage schedule-rung rollout sampling.
- Added `--initial-rollout-k` so a resumed run can start at the current
  scheduled stage, e.g. K32, while still keeping the full rollout schedule for
  final-stage schedule-rung sampling.
- Dataset refresh on 2026-05-15:
  - Retrained hybrid pose-aware AE after the dataset change:
    `training/runs/ae_poseaware_hybrid_datasetrefresh_20260515_175707/checkpoints/checkpoint_best.pt`.
    It finished epoch 800 with best reconstruction about `0.00305`.
  - The previous controller run was allowed to continue while the AE trained,
    then stopped after its next saved `checkpoint_last.pt` at epoch `2040`,
    K32.
  - Restarted controller from that saved checkpoint with refreshed AE prior,
    `--initial-rollout-k 32`, packed per-agent random clips, full schedule
    `1,2,4,8,16,32,64,111`, final-stage schedule-rung sampling, and doubled
    simple footslide weight `0.10`:
    `training/runs/hybrid_datasetrefresh_packed_scheduleK_from_epoch2040_w010_20260515_180333`.

### 2026-05-15 - Kaggle K111 Fork Plan

- Prepared a Kaggle fork path for testing the noisy final K111/mixed-K stage on
  a free Kaggle GPU without touching the local run.
- Fork point:
  `training/runs/hybrid_datasetrefresh_packed_scheduleK_from_epoch2040_w010_20260515_180333/checkpoints/checkpoint_best_k64.pt`.
  This checkpoint reports `rollout_k=64` and `epoch=443`, so it is the clean
  "beginning of K111" model state for the refreshed hybrid branch.
- Frozen AE prior for the Kaggle run:
  `training/runs/ae_poseaware_hybrid_datasetrefresh_20260515_175707/checkpoints/checkpoint_best.pt`.
- Added Kaggle utilities:
  - `training/kaggle_prepare_k111_fork.py` packages only the needed source
    files, the two current `npz_final` folders, the refreshed AE checkpoint,
    and the K64 fork checkpoint.
  - `training/kaggle_run_k111.py` resumes training at `--initial-rollout-k 111`
    with the same packed per-agent sampling, group sampling weights `2:1`,
    simple footslide weight `0.10`, and optional final-stage schedule-rung
    random rollout.
  - `training/kaggle_sync_tensorboard.py` can download Kaggle kernel outputs
    into a local TensorBoard logdir when Kaggle exposes them.
- Local dry-run packaging succeeded at
  `training/kaggle_k111_fork/`. The payload is about 70 MB and does not include
  full local run history.

### 2026-05-15 - Root Feature Basis Preflight

- Found a real conditioning bug in the root-motion features, exposed by the
  spin/90-degree turn clips. The canonical pose code and FK use the root heading
  matrix as a world-to-root row-vector transform, but `root_delta_feature`,
  `future_root_features`, and the packed AE-prior input path were using its
  transpose. That made a straight-forward tail after a +/-90 degree turn encode
  as local backward motion (`dz=-0.4`) instead of forward (`dz=+0.4`).
- Fixed both `training/train_locomotion.py` and
  `training/train_locomotion_ae_prior.py` so current and future root features
  use the same heading basis as canonical joint positions. The model viewer's
  authored-extension input path now uses the corresponding world-to-local
  transform too; rendering was not changed.
- Added `training/preflight_motion_features.py`. Before any long training on
  hybrid/turn clips, run it against the periodic and non-periodic npz folders.
  It checks that spin/90-degree turn tails condition as local forward motion and
  that the packed input builder matches the unpacked builder. Do not restart a
  long run unless this preflight passes.
- Current passing diagnostic was written to
  `training/runs/diagnostics/motion_feature_preflight.json`.
- Consequence: hybrid/transition checkpoints trained before this fix should be
  treated as contaminated by wrong root conditioning. Restart from the accepted
  baseline only after retraining the AE/controller under this corrected feature
  convention.

### 2026-05-15 - Rootbasis Restart Hygiene

- Arc sanity check before restart: all six `M_Neutral_Walk_Arc_F_*` transition
  clips now encode local forward motion consistently (`dz ~= +0.39/+0.40`,
  0% backward frames). The old transposed-basis formula produced backward local
  motion on 23-94% of arc frames depending on the clip, so the arc hover bug
  was the same root-feature basis bug, not a separate arc-data issue.
- TensorBoard cleanup: old event logs were moved out of `training/runs` into
  `training/tensorboard_archive/20260515_before_rootbasis_restart/`. Checkpoints
  and run artifacts remain in their original run folders. TensorBoard should now
  index only new experiments created after this cleanup.
- Trainers now date-prefix new run names by default, e.g.
  `YYYYMMDD_HHMMSS_<run_name>`, so TensorBoard sorts future runs
  alphabetically by creation time. Pass `--no-date-prefix-run-name` only when a
  caller has already provided a date-first name.
- Added 30-minute trail checkpoints to the supervised, AE-prior, and transition
  AE trainers. They are saved as
  `checkpoint_time_YYYYMMDD_HHMMSS_epoch_XXXXXX.pt` in addition to existing
  `last`, `best`, and numbered checkpoint rules.
- Corrected AE prior trained from scratch under the fixed root-feature basis:
  `training/runs/20260515_235409_ae_poseaware_hybrid_rootbasis_refresh/checkpoints/checkpoint_best.pt`.
  It used the previous dataset-refresh AE setup and reached best reconstruction
  about `0.00313`.
- Restarted local controller run, not Kaggle:
  `training/runs/20260515_235740_hybrid_rootbasis_from_omni_baseline_w010_mixedk111`.
  It resumes from accepted omni baseline
  `training/runs/footslide_ae_from_omni_k111_allanims_w20_lr1e6_resume_20260514/checkpoints/checkpoint_best_k111.pt`,
  uses the corrected AE prior above, packed per-agent random clips,
  periodic/non-periodic sampling weights `2:1`, simple footslide weight `0.10`,
  starts at K111, and uses final-stage schedule-rung mixed K sampling.

### 2026-05-16 - Canonical Body Basis Restart

- Found a second basis mismatch: root/future features were fixed, but
  canonical body positions and joint canonical velocities were still using the
  transposed heading basis in one path. This made the hidden body-state input
  inconsistent with the visible root speed/yaw on +/-90 degree spin clips.
- Fixed canonical body projection in both trainers so body canonical positions,
  root deltas, and future root deltas use the same row-vector world-to-heading
  convention. `training/preflight_motion_features.py` now includes a canonical
  heading-invariance check for the spin-tail case.
- Current preflight:
  `training/runs/diagnostics/motion_feature_preflight.json`.
  Canonical invariance max abs was about `3.6e-7`, and spin/turn tails encode
  local forward as `dz ~= +0.4`.
- Invalidated live run archived out of TensorBoard:
  `training/runs/20260516_001619_hybrid_rootbasis_from_omni_baseline_w010_curriculum_to_mixedk64/tb`
  was moved under
  `training/tensorboard_archive/20260516_canonical_basis_invalidated/`.
- Retrained the AE prior from scratch under the corrected canonical body basis:
  `training/runs/20260516_022731_ae_poseaware_hybrid_canonbasis_refresh/checkpoints/checkpoint_best.pt`.
  Best reconstruction reached about `0.00287`.
- Restarted the controller with fresh weights, not resumed old weights, because
  older controller checkpoints were trained against the contaminated input
  convention:
  `training/runs/20260516_023559_hybrid_canonbasis_scratch_w010_curriculum_to_mixedk64`.
  It starts at K1, uses packed per-agent random clips, periodic/non-periodic
  sampling weights `2:1`, simple footslide weight `0.10`, schedule
  `1,2,4,8,16,32,64`, and switches to schedule-rung mixed K only at the final
  K64 stage.

### 2026-05-16 - Rotation Audit Harness

- Added `training/rotation_audit.py` as a reusable rotation/basis tripwire for
  the hybrid periodic + non-periodic dataset. It is intentionally broader than
  the earlier motion-feature preflight:
  - random and dataset 6D rotation roundtrips;
  - degenerate/collinear 6D fallback conversion;
  - dataset global rotation orthonormality and determinant checks;
  - UE Z-up to canonical Y-up vector/rotation consistency;
  - FK rotation reconstruction from stored local rotations;
  - synthetic global-yaw equivariance for FK positions/rotations/canonical
    positions across all 226 clips;
  - root delta and dense future-root feature invariance under global yaw;
  - packed trainer path vs unpacked reference for build-input, root-state, FK,
    and transition features;
  - footslide-speed invariance under global yaw.
- Full current audit command:
  `.tools/python310/python.exe training/rotation_audit.py --frames-per-clip 8 --equivariance-clips 226 --output training/runs/diagnostics/rotation_audit.json`.
  It passed with `loaded_clips=226` and `failures=0`.
- Found and fixed one rotation edge case: exact/nearly collinear predicted 6D
  rows could produce a collapsed matrix in Gram-Schmidt. `rotation_6d_to_matrix`
  now uses a stable orthogonal fallback for degenerate rows. Normal non-degenerate
  rotations are unchanged.
- Inertness check after the fix: dataset stored 6D rotations, one-step model
  outputs, and sampled K64 rollouts from both
  `20260516_024659_hybrid_canonbasis_from_k8_w050_curriculum_to_mixedk64/checkpoints/checkpoint_best_k64.pt`
  and `checkpoint_last.pt` had `fallback_count=0`, `near_1e-5_count=0`,
  and `old_new_max_abs=0`. So the fix does not change current checkpoints or
  normal rollout paths; it only catches pathological future outputs.
- The audit also exposed a non-rotation representation caveat: several source
  clips contain small animated local translations on non-root bones, especially
  spine links, while the controller FK intentionally uses fixed rest offsets for
  all non-pelvis bones. Stored global positions can therefore differ from
  fixed-offset FK by up to about `0.030 m` on the current full dataset, even
  though FK rotation reconstruction is exact (`~6.6e-7`). Treat this as a model
  representation envelope, not a basis bug.
- Packed-vs-unpacked comparisons must use future-safe frames for non-periodic
  clips, because the packed training path assumes rows reset before future-root
  windows would pass the clip end.

### 2026-05-16 - AE-Prior Monitoring Rule

- For AE-prior / non-supervised controller runs, do not rank model quality by
  ground-truth RMSE unless the question is explicitly about frame-locked
  imitation. RMSE is misleading here because good behavior may phase-shift or
  choose a different valid continuation.
- Use the actual active objective terms for numerical "struggle" checks:
  frozen AE transition score, generated support-foot slide loss, and their
  weighted total. Ground-truth RMSE may still be logged as a debug diagnostic,
  but it should not decide whether an AE-prior model is improving.

### 2026-05-16 - Synthetic Root Boundary Agents

- Added synthetic root-motion clips as a controller-only stress source. These
  clips live under
  `ue5/animations_synthetic/npz_final` and are deliberately not visually valid
  human motion, so they must not be fed into AE prior training.
- Implementation rule:
  - `transition_autoencoder.py` remains unchanged and still trains only from
    `folder_path` / periodic / non-periodic real motion folders.
  - `train_locomotion_ae_prior.py` accepts `--synthetic-folder-path` and
    `--synthetic-agent-fraction`; those clips are appended only for controller
    rollout root trajectories.
  - Checkpoint metadata keeps real `npz_folders` separate from
    `synthetic_npz_folders`, so future AE retraining helpers do not inherit the
    synthetic stress clips by accident.
  - Random pose initialization samples only from real clips. Synthetic rows keep
    synthetic roots but borrow a real pose when
    `--init-pose-sampling random_dataset` is enabled.
  - During non-periodic reset, synthetic rows reset back into the synthetic pool
    and real rows reset back into the real pool, preserving the fixed agent
    fraction throughout long K rollouts.
- Smoke test:
  `synthetic_sampler_smoke_delete` used a 16-agent CPU batch with 25% synthetic
  rows and verified `real_per_batch=12`, `synthetic_per_batch=4`, real metadata
  paths did not include synthetic clips, and `synthetic_clip_count=9`.
  The throwaway run was deleted after verification.
- Current live experiment:
  `training/runs/20260516_161746_hybrid_synthetic20_k50_w050_from_localbestk32`.
  It resumes the local pushed best K32 checkpoint
  `20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`,
  uses `K=50`, `--init-pose-sampling random_dataset`, simple footslide weight
  `0.50`, and synthetic fraction `0.20`. With batch size 256, this is the fixed
  count `51/256 = 19.9219%` synthetic rows per update.
- First visual report screenshot saved at
  `training/runs/20260516_161746_hybrid_synthetic20_k50_w050_from_localbestk32/visual_reports/latest/browser_screenshot.png`.

## 2026-05-16 - Root-Conditioned Foot Support Envelopes

- Added `training/support_envelope.py` for cached root-conditioned zero-loss bounds used by geometry losses.
- The envelope conditions on two horizontal root-motion features:
  - root yaw delta from `t-1` to the future-window end;
  - signed horizontal bend angle between the current root step and the current-to-window-end root displacement.
- Real clips define the envelope; synthetic clips get cached per-frame features and queried bounds, but synthetic values do not expand the bounds.
- Real ground-truth transitions are explicitly guarded with their own measured value times the margin, so real dataset transitions have zero excess for both support slide and vertical foot/toe yaw.
- `train_locomotion_ae_prior.py` now uses the packed-frame envelope for `simple_footslide_loss_weight` when the packed random-agent path is active. The old scalar threshold remains only as fallback for non-packed runs.
- Added vertical foot-yaw loss: measure world foot/toe angular velocity, project onto global vertical Y, take the max over foot/toe and both feet, and apply linear excess above the cached bound.
- Added auto foot-yaw scaling: with `--foot-yaw-loss-scale-radps <= 0`, the trainer rolls the current resumed checkpoint on the forward-walk clip and sets the scale so that checkpoint's maximum forward-walk yaw excess equals loss `1` before the user weight is applied.
- Smoke checks on the full hybrid set: 226 real clips + 9 synthetic clips, 10,687 real transitions defining the envelope, 13,315 cached target transitions. Real GT slide excess max = 0 and real GT vertical-yaw excess max = 0.

## 2026-05-16 - K32 Support-Envelope Relaunch

- Requested run launched from the latest accepted local K32 baseline:
  `training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`.
- AE prior:
  `training/runs/20260516_022731_ae_poseaware_hybrid_canonbasis_refresh/checkpoints/checkpoint_best.pt`.
- New live run:
  `training/runs/20260516_191627_hybrid_supportenv_k32_synth20_w002_from_k32`.
- Setup: fixed `K=32`, same-clip initialization, `20%` synthetic root clips, support-envelope simple footslide weight `0.02`, support-envelope vertical foot-yaw weight `0.02`.
- A first attempted launch exposed NaNs from the differentiable SO(3) log/acos vertical-yaw path near sharp foot rotations. The metric was changed to a stable vertical-axis `atan2` extraction in `contact_physics.foot_vertical_yaw_speeds`, preserving pure-yaw detection and keeping pitch at zero in the sanity check. Support-envelope cache version was bumped so yaw bounds rebuild with the same metric used in training.
- Ground-truth support-envelope verification after the fix: real slide excess max `0`, real vertical foot-yaw excess max `0`; synthetics are cached but do not affect the bounds.
- Frozen baseline AE scores for turn-in-place clips are recorded in the console output from this run handoff; the largest mean score was on `M_Neutral_Stand_Turn_180_L` at about `0.00514`.

## 2026-05-17 - Window Transition AE Side Probe

- Added `training/window_transition_autoencoder.py` as a contained side experiment. It does not replace the production 1-frame transition AE.
- The script reuses the current foot-motion-aware transition feature schema from
  `training/runs/20260516_200756_ae_poseaware_hybrid_footmotion/checkpoints/checkpoint_best.pt`.
  Each per-frame transition feature is normalized by that AE's stored mean/std, then flattened into W-frame windows.
- Trained W=`8`, `16`, and `32` variants in
  `training/runs/20260517_041727_windowae_w8_w16_w32_vs_transitionae_neutral90`.
- Comparison target: best checkpoint from
  `training/runs/20260517_031133_hybrid_footmotionae_k220_only_synth050_slide0_yaw0_firstframe_from_k32/checkpoints/checkpoint_best.pt`,
  evaluated on `M_Neutral_Stand_Turn_090_L/R`.
- Mean generated scores over the two neutral 90 turn clips:
  - current 1-frame AE: `0.02717` (GT mean `0.00320`);
  - W8 AE: `0.05808` (GT mean `0.01171`);
  - W16 AE: `0.11813` (GT mean `0.02389`);
  - W32 AE: `0.19232` (GT mean `0.01645`).
- CSV details are saved at
  `training/runs/20260517_041727_windowae_w8_w16_w32_vs_transitionae_neutral90/neutral_stand_90_window_compare.csv`.

## 2026-05-17 - Root-Conditioned Window AE Probe

- Vanilla window AEs failed an important sanity check: a visually bad
  idle/ghost turn rollout could score lower than real GT on some turn clips.
- Minimal change tested: keep the same normalized transition feature window,
  but train a conditional window predictor with only the root-motion slice as
  input and all non-root transition channels as target. No model-specific hard
  negatives were used.
- Baseline no-foot-motion AE schema:
  `training/runs/20260516_022731_ae_poseaware_hybrid_canonbasis_refresh/checkpoints/checkpoint_best.pt`.
- Bad rollout checkpoint used only for evaluation:
  `training/runs/20260517_050255_hybrid_windowae16_turninplace_real_pureae_k32_randomframe_sameclip_from_k32/checkpoints/checkpoint_best.pt`.
- Conditional W16:
  `training/runs/20260517_053600_conditional_root_windowae_w16_vs_ghostturn`.
  It caught most 90/135/180 failures but still had 45-degree loopholes
  (`generated_mean < gt_mean` on both 45-degree clips).
- Conditional W8/W32:
  `training/runs/20260517_053938_conditional_root_windowae_w8_w32_vs_ghostturn`.
  W8 still had the 45-degree loophole; W32 scored generated worse than GT on
  all 8 turn-in-place clips. Mean over all 8 clips:
  - W32 GT mean: `0.15772`;
  - W32 generated mean: `0.39891`;
  - W32 generated p95: `0.78631`.
- Trainer smoke test with conditional W32 prior succeeded:
  `training/runs/smoke_conditional_windowae_w32_k32_delete`.
  This confirms `train_locomotion_ae_prior.py` can use conditional window
  checkpoints through the packed K rollout path.

## 2026-05-17 - Cheap Denoising 1-Frame Transition AE Probe

- Tested a cheaper alternative to W8/W16/W32 priors: keep the original
  1-frame transition AE shape, keep root/future-root channels clean, and add
  Gaussian noise only to non-root/body transition channels during AE training.
  This keeps inference as a single transition AE call rather than a windowed
  prior.
- Added `input_noise_mask` to `training/transition_autoencoder.py`.
  The useful setting here is `--input-noise-mask nonroot`, with
  `--input-noise-std 0.05`.
- Best practical checkpoint from this probe:
  `training/runs/20260517_153309_denoise_nonroot_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`.
- Tier report for that checkpoint:
  clean GT mean `0.01272`, clean GT p95 `0.02591`, slight-noise mean
  `0.01517`, bad-perturbation mean `0.3235`, random-noise mean `1.4321`.
- Ghost-turn comparison used the bad rollout checkpoint
  `training/runs/20260517_031133_hybrid_footmotionae_k220_only_synth050_slide0_yaw0_firstframe_from_k32/checkpoints/checkpoint_best.pt`
  on all neutral stand turn-in-place clips. Mean over clips:
  GT `0.00961`, generated ghost rollout `0.09350`, generated p95 `0.26613`.
  Per-clip generated scores are worse than GT on every 45/90/135/180 turn.
- A latent-16 denoising AE was stricter on ghost turns, but raised the real GT
  floor too much. Latent-32 is the better default candidate unless we want a
  deliberately harsher critic.

## 2026-05-17 - No-Slide W16 Continuation From Promising Checkpoint

- User visual pick / current best visual seed:
  `training/runs/20260517_194453_structured_w16_curriculum_walkF45_to_k32_from_baseline/checkpoints/checkpoint_best_k08.pt`.
- This checkpoint used the structured denoise/root-lookahead W16-style transition
  AE:
  `training/runs/20260517_190041_structured_denoise_rootlook16_fullae_lat32_damped035_w1_n0p05_e300/checkpoints/checkpoint_best.pt`.
- Important constraint from the user: do not add slide loss here. These runs are
  trying to solve the turn/ghost/skate issue from AE priors and conditioning, not
  by final-stage footslide penalties.
- Diagnostic tool used only for monitoring:
  `training/visualize_rollout_foot_skating.py`. It reports support-foot and
  source-contact sole sliding over autoregressive rollouts; it is not part of
  training loss.
- Baseline K32 on `M_Neutral_Stand_Turn_045_R/L`:
  - R45 source-contact p95 pred `0.2340`, GT `0.0463`;
  - L45 source-contact p95 pred `0.3913`, GT `0.1109`.
- Promising structured W16 K8 checkpoint:
  - R45 support p95 pred `0.1312`, source-contact p95 pred `0.2624`;
  - L45 support p95 pred `0.1113`, source-contact p95 pred `0.2800`.
  This matches the user's visual read: not solved, but better than several later
  branches.
- Continuing that checkpoint with only the structured W16 AE to K16/K32 lowered
  AE loss but worsened skating:
  `training/runs/20260517_210754_structured_w16_resume_from_k08_to_k32_noslide_freshadam`.
  K32 R45 source-contact p95 pred `0.4277`, L45 `0.3440`.
- Adding the old pose-aware AE as a second prior did not help:
  `training/runs/20260517_212243_dualprior_oldw1_structw16_from_promising_k08_noslide`.
  K16 R45 source-contact p95 pred `0.3552`, L45 `0.2738`.
- Adding the 1-frame denoising/rootlook AE as a second prior also did not beat
  the user's selected W16 K8 checkpoint in the early K8 stage:
  `training/runs/20260517_213255_dualprior_denoisew1_structw16_from_promising_k08_noslide`.
- Working conclusion: the W16 K8 checkpoint is currently the best visual seed
  among these no-slide branches. Longer AE-only rollout optimization can game
  the priors by skating, so future no-slide work should improve the prior itself
  or the conditioning/negative-space tests before trusting lower AE loss.

## 2026-05-17 - Corrected No-Slide Best Checkpoint

- User visual inspection found that this checkpoint was much better than my
  foot-skating monitor suggested:
  `training/runs/20260517_221536_official1f_plus_compatfixed_k32_strongercompat_noslide/checkpoints/checkpoint_best_k32.pt`.
- Treat this as the current best no-slide mini-dataset checkpoint unless later
  visual inspection disproves it. It uses:
  - primary 1-frame denoising/root-lookahead prior:
    `training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`,
    weight `1.0`;
  - corrected rootlook16 compatibility prior:
    `training/runs/20260517_215153_denoise_rootlook16_compatfixed_mini_e240/checkpoints/checkpoint_best.pt`,
    weight `0.3`;
  - compatibility head penalty weight `0.10`;
  - no supervised, contact, simple footslide, foot-yaw, or motion-floor loss.
- Important lesson: `visualize_rollout_foot_skating.py` is useful for spotting
  likely skating, but it over-penalized this checkpoint relative to the actual
  visual rollout. Do not reject a visually good no-slide checkpoint only because
  the source-contact skating metric is high.
- Direct K32 fork from the accepted local K32 baseline:
  `training/runs/20260517_223536_directk32_official1f_plus_compatfixed_from_baseline_noslide`.
  It resumed
  `training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`
  and started immediately at K=32 with the same prior pair. Best AE score after
  120 epochs was `0.011312724`, worse than the visually accepted checkpoint's
  `0.008649662`. This suggests the previous continuation/curriculum path did
  help reach a better basin, even though the final accepted run itself was
  K32-only.
- Control experiment for "is the second AE necessary?":
  `training/runs/20260517_225556_curriculum_single1f_denoise_from_baseline_noslide`.
  Same baseline and K8->K16->K32 curriculum, but only the 1-frame denoising
  prior at weight `1.0`; no rootlook16 compatibility prior. It reached very low
  single-prior AE loss (`checkpoint_best_k32.pt`, epoch `201`, best
  `0.003438556`), which is lower than the double-AE run only because it is an
  easier objective.
- Short GT-initialized viewer diagnostics still favor the double-AE checkpoint:
  - R45 autoregressive mean joint error over 50 frames:
    single-prior curriculum `0.019192`, double-AE accepted `0.008783`;
  - L45 autoregressive mean joint error over 50 frames:
    single-prior curriculum `0.015630`, double-AE accepted `0.009067`.
  This supports the current hypothesis: curriculum helps, but the corrected
  rootlook16 compatibility prior is doing real work that the 1-frame denoising
  prior cannot do by itself.

## 2026-05-17 - Mini Coefficient Sweep For Dual No-Slide AE

- Goal: tune the two non-primary coefficients in the accepted no-slide setup
  without silently damaging the normal forward gait.
- Fixed ingredients:
  - baseline model:
    `training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`;
  - 1-frame denoising/root-lookahead prior:
    `training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`,
    weight kept at `1.0`;
  - corrected rootlook16 compatibility prior:
    `training/runs/20260517_215153_denoise_rootlook16_compatfixed_mini_e240/checkpoints/checkpoint_best.pt`;
  - no supervised, contact, footslide, foot-yaw, or motion-floor losses.
- Sweep script:
  `training/run_mini_coeff_sweep.py`.
  It trains K8->K16->K32 curriculum runs on the mini `WalkF + StandTurn45`
  dataset and evaluates direct GT-initialized autoregressive joint error with
  `training/visualize_model.py`.
- Overhead smoke:
  - single 1-frame AE, K32 35 epochs:
    `training/runs/20260517_230944_bench_overhead_single1f_k32_e35`,
    about `28.7s` at epoch 35;
  - dual AE, K32 35 epochs:
    `training/runs/20260517_230944_bench_overhead_doubleae_k32_e35`,
    about `36.0s` at epoch 35.
  The second AE costs roughly `25%` in this small benchmark, which is noticeable
  but not the dominant problem compared with failed training branches.
- Critical evaluation rule: include `M_Neutral_Walk_Loop_F` in the score. A
  coefficient set that improves turns but damages WalkF is rejected. The summary
  with WalkF is:
  `training/runs/coeff_sweeps/20260517_231222_mini_doubleae/summary_with_walkf.csv`.
- Results, direct autoregressive mean joint error (`R45`, `L45`, `WalkF`,
  combined average):
  - extra `0.30`, compatibility `0.05`:
    `0.008949`, `0.008793`, `0.018080`, combined `0.011941`;
  - extra `0.30`, compatibility `0.10`:
    `0.009783`, `0.010500`, `0.018082`, combined `0.012788`;
  - extra `0.15`, compatibility `0.05`:
    `0.013684`, `0.009077`, `0.016881`, combined `0.013214`;
  - extra `0.30`, compatibility `0.20`:
    `0.010674`, `0.010846`, `0.018144`, combined `0.013221`;
  - extra `0.60`, compatibility `0.10`:
    `0.032467`, `0.009026`, `0.019131`, combined `0.020208`;
  - extra `1.00`, compatibility `0.10`:
    `0.035091`, `0.011797`, `0.018104`, combined `0.021664`.
- Accepted checkpoint reference:
  `training/runs/20260517_221536_official1f_plus_compatfixed_k32_strongercompat_noslide/checkpoints/checkpoint_best_k32.pt`
  scored `R45=0.008964`, `L45=0.009251`, `WalkF=0.018140`, combined
  `0.012118` with the same direct comparison method.
- Current coefficient recommendation for this mini setup:
  keep primary 1-frame prior weight `1.0`, use rootlook16 extra prior weight
  `0.30`, and reduce compatibility-score weight from `0.10` to `0.05`.
  The gain is modest, but it is the fastest/cleanest setting found here and it
  preserved WalkF.
- Follow-up overhead pass:
  - Tried sharing a single superset transition feature tensor across the
    1-frame and rootlook16 priors, slicing it for the shorter prior. Tensor
    values were identical (`feature max_abs=0`, score diff `0`, gradient diff
    about `1e-9`), but the real training smoke got slower. This was reverted.
    Likely reason: the shared autograd graph made backward scheduling worse
    even though forward feature work was reduced.
  - `torch.compile` on the frozen priors was also slower in the local
    microbenchmark, so it was not kept.
  - Kept two strictly equivalent cleanups:
    no-foot-loss packed rollouts no longer compute the current-pose FK that was
    only needed by footslide/foot-yaw losses, and monitor-only RMS values detach
    their inputs before doing logging math. These do not change the loss,
    gradients, model weights, or generated motion.
  - No-checkpoint microbenchmark after cleanup on the mini K32 path:
    single prior `~0.86s`/forward+backward batch, dual prior `~1.79s`.
    Full smoke runs are noisier because best-checkpoint disk writes and laptop
    thermal state dominate short 35-epoch timings.

## 2026-05-18 - Mixed Rollout Cohorts

- Added `--mixed-rollout-cohorts` to `training/train_locomotion_ae_prior.py`.
  Instead of sampling one random K for the whole agent batch, it can split rows
  into fixed K cohorts, for example `--mixed-rollout-cohort-schedule 2,4,8,16,32`.
- Implementation detail: the outer GPU loop still runs at the current scheduled
  K, but rows whose cohort length is shorter reset inside that loop. With
  power-of-two cohorts, all rows end on the same outer K boundary. Starts are
  sampled using each row's own cohort K, so short non-periodic clips are not
  excluded just because the outer loop is K32/K64.
- Sanity check on the mini `WalkF + StandTurn45` setup:
  cohort row counts for batch 64 were `K2=13`, `K4=13`, `K8=13`, `K16=13`,
  `K32=12`, with zero start-range violations. Expected per-outer-K32 resets:
  `K2=15`, `K4=7`, `K8=3`, `K16=1`, `K32=0`.
- Timing on the current double-AE mini K32 path, forward+backward only:
  fixed K32 `0.847s`/batch, mixed `2,4,8,16,32` cohorts `0.886s`/batch.
  This is about `4.5%` overhead in the measured path, so the idea is viable
  enough to try in an actual experiment.
- Random same-animation frame initialization was checked with cohorts
  `5,10,15,20,50`: `init_pose_sampling=same_clip`,
  `agent_fixed_start_frame=0`, varied start frames inside each cohort, zero
  start-range violations.
- Live mini run launched from the old K32 baseline:
  `training/runs/20260518_003040_mixedcohort_5_10_15_20_50_minidoubleae_from_oldk32`.
  Setup: mini `WalkF + StandTurn45`, outer K50, cohort distribution
  `5/10/15/20/50`, primary 1-frame prior weight `1.0`, rootlook16 prior weight
  `0.30`, compatibility weight `0.05`, no foot/contact/motion-floor losses.
- Mixed cohorts now also accept normalized weights via
  `--mixed-rollout-cohort-weights`. The sampler uses each row's own cohort K
  for start sampling and, in mixed-cohort mode, requires sampled clips to support
  that full cohort K. This prevents long-K rows from silently choosing short
  transition clips and immediately resetting.
- Replaced the uniform K50 test with weighted cohorts
  `K=2,4,8,16,32,64` and weights `5,10,15,20,25,35` in
  `training/runs/20260518_005420_mixedcohort_pct_2_4_8_16_32_64_minidoubleae_from_oldk32`.
  The weights are normalized; for batch size 64 this gives counts
  `3/6/9/12/14/20`, mean effective K `31.6`. On the mini dataset, K64 rows can
  only sample the periodic walk loop, while the 45-degree spin clips remain
  eligible for K2-K32.
- Important repro caveat: the strong mini result above used the mini-specific
  secondary AE
  `training/runs/20260517_215153_denoise_rootlook16_compatfixed_mini_e240/checkpoints/checkpoint_best.pt`.
  A follow-up run with the exact intended big-training AE stack instead
  (`20260517_161820` primary + `20260517_184310` secondary) was
  `training/runs/20260518_222908_repro_bigAE_mixedcohort_pct_2_4_8_16_32_64_from_oldk32`.
  It reached a lower AE scalar (`best_ae_score ~= 0.00665` by epoch 239) but
  did not reproduce the visual R45 quality: R45 frame-17 mean joint error was
  about `0.058 m` versus `0.0048 m` for the mini-specific AE checkpoint. Do not
  use the mini-specific AE result as evidence that the big AE stack is adequate;
  the visual/direct-GT check caught a scalar-loss blind spot.
- General-AE repair test:
  - The failed big secondary AE
    `training/runs/20260517_184310_denoise_rootlook16_fullae_lat32_dampedcompat035_n0p05_e300/checkpoints/checkpoint_best.pt`
    scored the bad R45 rollout slightly *better* than GT on reconstruction
    (`bad ~= 0.00634`, GT `~= 0.00721`), explaining the ghost-turn loophole.
  - A clean full-dataset replacement trained with the mini-like damped
    structured denoise/compat settings
    `training/runs/20260518_225429_denoise_rootlook16_full_generalcompat_dampedstruct_e240/checkpoints/checkpoint_best.pt`
    improved bad-vs-GT separation but was still too weak for model training.
  - The best general replacement so far is the existing conditional full-dataset
    prior
    `training/runs/20260517_165428_cond_rootw16_body_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`.
    It is trained on the big omni+transition dataset, not on the mini set.
  - Repro run:
    `training/runs/20260518_231611_repro_condBodyW16_general_mixedcohort_from_oldk32/checkpoints/checkpoint_best_k64.pt`
    with primary 1-frame AE weight `1.0`, conditional W16 prior weight `0.30`,
    mixed K cohorts `2,4,8,16,32,64` with weights `5,10,15,20,25,35`.
  - Direct-GT rollout check:
    R45 avg `0.00995`, f17 `0.00760`;
    L45 avg `0.00930`, f17 `0.01220`;
    WalkF avg `0.01640`.
    This is much better than the failed broad big-AE repro and preserves WalkF.
    It is close to, but still not perfectly identical to, the mini-specific AE
    checkpoint (R45 avg `0.00777`, WalkF avg `0.01507`).
  - Weight tests: conditional W16 weight `0.45` and `0.60` did not improve the
    overall mini repro. `0.60` helped WalkF and some isolated phases but hurt
    R45 average; `0.45` was a middle-ground but not a clear win. Keep `0.30`
    unless a later visual check says otherwise.

## 2026-05-19 - Big-Dataset AE Versus Mini-AE Repro Gate

- Added `--sample-mode uniform_clip` to `training/transition_autoencoder.py`.
  Default remains `rows`, so older AE runs are reproducible. Uniform-clip mode
  samples an animation first, then a transition row inside that animation. This
  was tested because the mini-specific AE saw the short 45-degree turn clips
  often, while big-dataset row sampling dilutes those clips among 10k+ rows.
- The gate for these experiments is direct rollout geometry on the mini repro,
  not scalar AE loss alone:
  - `R45`, `L45`, and `WalkF` are all checked.
  - WalkF must stay good; a turn fix that damages normal gait is rejected.
- Known reference:
  `training/runs/20260518_005420_mixedcohort_pct_2_4_8_16_32_64_minidoubleae_from_oldk32/checkpoints/checkpoint_best_k64.pt`
  used the mini-specific secondary AE and scored:
  `R45 avg=0.00777 f17=0.00482`,
  `L45 avg=0.00856 f17=0.00806`,
  `WalkF avg=0.01693`.
- Best clean big-dataset/general candidate remains:
  `training/runs/20260517_165428_cond_rootw16_body_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`
  as the secondary prior at weight `0.30` with the 1-frame prior at weight
  `1.0`. Its mini repro checkpoint:
  `training/runs/20260518_231611_repro_condBodyW16_general_mixedcohort_from_oldk32/checkpoints/checkpoint_best_k64.pt`
  scored:
  `R45 avg=0.00995 f17=0.00760`,
  `L45 avg=0.00930 f17=0.01220`,
  `WalkF avg=0.01782`.
  This is close to the mini AE result and preserves WalkF, but is still a bit
  softer on the 45-degree turns.
- Rejected attempts:
  - `20260519_001600_cond_rootw16_body_balancedclip_full_n0p05_e300`:
    clip-balanced conditional AE sharpened R45 bad-vs-GT scoring but made WalkF
    GT reconstruction less happy. Controller repro with this prior alone was
    visually/metric worse (`R45 avg ~= 0.0327`), so clip balancing by itself is
    not the fix.
  - `20260519_002900_fullae_rootlook16_balancedclip_minirecipe_e300`:
    full-transition mini-recipe AE trained on the full dataset scored the bad
    R45/L45 rollout better than GT under reconstruction. Its compatibility head
    did separate bad from GT, but using that compatibility score in controller
    training (`20260519_011000...`) worsened R45 and WalkF, so do not use it as
    a controller loss in this setup.
  - `20260519_003500_cond_rootw16_body_dampeddenoise_full_n0p05_e300`:
    conditional W16 plus damped denoising did not improve the bad-vs-GT
    separation over the existing row-trained conditional AE.
  - `20260519_005800_cond_rootw16_body_lat8_full_n0p05_e360`:
    smaller latent bottleneck over-penalized GT and still did not reliably
    penalize the ghost turn more than ground truth.
  - A three-prior ensemble
    `20260519_004200_repro_general_cond_old_plus_balanced_w030_w010_from_oldk32`
    reached a decent scalar loss but failed the direct rollout gate
    (`R45 f17 ~= 0.054 m`), another reminder that scalar AE loss is not enough.
- Current conclusion:
  for a clean big-dataset-trained AE, the conditional root-window body AE is the
  strongest working choice. The mini-specific AE's extra tightness appears to
  come from its restricted training distribution, not from a recipe that
  transfers directly to the big dataset. Any future improvement should be judged
  by the direct R45/L45/WalkF gate before being trusted.

## 2026-05-19 - Strict No-Tradeoff AE Neighborhood Search

- User constraint for this pass: only accept a checkpoint if it is a strict
  improvement over the current best, not a tradeoff.
- Current checkpoint to keep:
  `training/runs/20260518_231611_repro_condBodyW16_general_mixedcohort_from_oldk32/checkpoints/checkpoint_best_k64.pt`.
- Nearby attempts checked:
  - `20260519_013000_repro_general_cond_w025_from_oldk32`: rejected. It fell
    into the bad 45-degree turn basin (`R45 f17 ~= 0.083 m`).
  - `20260519_013700_repro_general_cond_w030_huber05_from_oldk32`: rejected.
    Huber AE scoring looked calm but the rollout gate failed
    (`R45 f17 ~= 0.078 m`).
  - `20260519_015600_polish_general_cond_w030_lr1e5_from_good`: rejected as a
    replacement. It improved L45/WalkF a little but worsened R45 frame 25.
  - `20260519_021100_repro_general_cond_w040_long_from_oldk32`: useful but not
    a strict replacement. It reduced worst-case/overall average error
    (`worst ~= 0.02594 m` versus current `0.02913 m`) but worsened early R45
    and a few L45 values. Under the no-tradeoff rule, keep the current
    checkpoint.
- Decision: move on with the current `w030` checkpoint unless the acceptance
  rule changes from strict per-gate improvement to an aggregate metric.

## 2026-05-19 - Accepted Strict-Win Checkpoint Soup

- Confirmed best general conditional/root-window AE remains:
  `training/runs/20260517_165428_cond_rootw16_body_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`.
  The full best controller setup uses it together with the primary 1-frame
  denoising AE:
  `training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`.
- New accepted controller checkpoint:
  `training/runs/20260519_023300_soup_strictwin_doubleae_general_from_w030/checkpoints/checkpoint_best_k64.pt`.
- Method: post-hoc checkpoint soup / linear weight interpolation. This has no
  runtime overhead: the saved result is just a normal controller checkpoint.
  Base checkpoint was:
  `training/runs/20260518_231611_repro_condBodyW16_general_mixedcohort_from_oldk32/checkpoints/checkpoint_best_k64.pt`.
- Soup weights are stored in:
  `training/runs/20260519_023300_soup_strictwin_doubleae_general_from_w030/SOUP_WEIGHTS.json`.
- Gate result versus previous best, direct 80-frame rollout:
  - Previous best:
    `R45 avg=0.00994950 f17=0.00759503 f25=0.01854316 end=0.01170241 max=0.01868471`;
    `L45 avg=0.00929694 f17=0.01219531 f25=0.01066126 end=0.01143346 max=0.01468577`;
    `WalkF avg=0.01782222 f17=0.01838074 f25=0.01739129 end=0.02234678 max=0.02913017`;
    worst `0.02913017`, mean-all `0.01532125`.
  - New soup checkpoint:
    `R45 avg=0.00985577 f17=0.00751158 f25=0.01844196 end=0.01132380 max=0.01856698`;
    `L45 avg=0.00911379 f17=0.01180393 f25=0.01064015 end=0.01108168 max=0.01428456`;
    `WalkF avg=0.01765805 f17=0.01827932 f25=0.01725379 end=0.02205193 max=0.02885311`;
    worst `0.02885311`, mean-all `0.01511469`.
- Strict acceptance check: all 15 gate metrics improved. The largest delta was
  still negative (`max_delta=-2.11e-05 m`), so this is not a tradeoff.

## 2026-05-19 - CircleL Isolation Test From K32 Baseline

- User correction: controller training experiments should start from the K32
  baseline, not from the soup checkpoint, unless explicitly requested.
- Created contained mini dataset:
  `training/runs/mini_datasets/circleL_only/nonperiodic/M_Neutral_Walk_Circle_Strafe_L.npz`.
- Training setup for the quick test:
  - start checkpoint:
    `training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`
  - mixed K schedule `2,4,8,16,32` with weights `5,15,20,30,40`
  - no footslide/contact/motion-floor losses
  - direct GT rollout used only for evaluation, not training.
- Results on CircleL:
  - K32 start: avg `0.101523`, f17 `0.119181`, f25 `0.099672`,
    end `0.089856`, max `0.158635`.
  - old 1-frame AE only
    (`20260519_041000_circleL_only_old1f_from_oldk32_K32mix`):
    avg `0.082483`, f17 `0.021739`, f25 `0.046899`,
    end `0.076076`, max `0.192583`.
  - new conditional/root-window AE only
    (`20260519_041400_circleL_only_newcond_from_oldk32_K32mix`):
    avg `0.082686`, f17 `0.012384`, f25 `0.027812`,
    end `0.060732`, max `0.219316`.
  - both AEs
    (`20260519_040000_circleL_only_doubleae_from_oldk32_K32mix`):
    avg `0.037010`, f17 `0.009648`, f25 `0.011568`,
    end `0.071382`, max `0.092565`.
- Interpretation:
  the new conditional AE does not intrinsically kill CircleL. On the isolated
  CircleL task, the best result is using both AEs together. The failure in the
  broader mini dataset is a multi-clip balancing/conflict problem: CircleL can
  be learned, but single-clip training overfits and damages R45/WalkF, while
  broad mini training did not allocate or shape enough useful pressure to solve
  circles and preserve turns simultaneously.

## 2026-05-19 - R45 Missing 0.3 Blend Ablation

- User asked why the `0.3` blend had not been tested on R45. That was a real
  missing row in the pure-AE ablation.
- Important correction: several false starts were not comparable:
  - adding `--compatibility-score-weight` changed the objective and loss scale;
  - forcing a K32 continuation from the fragile K16 run exposed the long-rollout
    failure mode but was not the clean curriculum result;
  - the clean result below uses AE reconstruction priors only, no extra
    compatibility-head penalty.
- Final clean R45 `0.3` run:
  `training/runs/20260519_075640_ablate_R45_blend_w03_stage16_32_from_cleanK8/checkpoints/checkpoint_best_k32.pt`.
- Setup:
  - start checkpoint:
    `training/runs/20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt/checkpoints/checkpoint_best_k32.pt`
  - AE1 old 1-frame denoising prior weight `1.0`:
    `training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`
  - AE2 conditional/root-window prior weight `0.3`:
    `training/runs/20260519_043759_compat_rootw16_yawbody_lat64_e180/checkpoints/checkpoint_best.pt`
  - mixed rollout cohorts over `2,4,8,16,32` with weights `5,15,20,30,40`.
- Diagnostic output:
  `training/runs/diagnostics/ablation_R45_pure_vs_blend_20260519.html`.
- Trainable-horizon R45 foot-skate metrics:
  - pure AE1: support p95 `0.246864`, source-contact p95 `0.349284`,
    excess p95 `0.326238`.
  - pure AE2: support p95 `0.238798`, source-contact p95 `0.263928`,
    excess p95 `0.208829`.
  - equal blend: support p95 `0.248924`, source-contact p95 `0.247546`,
    excess p95 `0.215955`.
  - clean `0.3` blend: support p95 `0.105055`, source-contact p95 `0.204683`,
    excess p95 `0.159388`.
- Interpretation:
  on R45, clean `0.3` blend is the best row in this ablation by the foot-skate
  diagnostics. It also keeps AE1 dominant, matching the earlier observation that
  AE1 is still useful for gait preservation while AE2 helps root/body
  compatibility.

## 2026-05-19 - R45 Direct Mixed-K Speed Audit

- User asked whether the R45 `0.3` blend can skip rollout curriculum and train
  directly with mixed K. The first direct run looked absurdly slow.
- Root cause:
  `train_locomotion_ae_prior.py` was still enabling the older
  contact-physics auxiliary losses by default. For pure AE ablations this is
  wrong: it both adds non-AE loss terms and disables the packed rollout path.
- Code change:
  pure AE prior training now leaves contact-physics auxiliary losses disabled by
  default. They must be opted into explicitly with `--contact-physics-losses`.
  `--no-contact-physics-losses` remains accepted and explicit.
- Timing probes on the single R45 clip, mixed K32, batch 64:
  - legacy/contact-on path: 3 epochs took about `189s`; `train_total=3.00578`
    while `ae_score=0.08202`, proving non-AE loss dominated.
  - packed pure-AE path: 3 epochs took about `54s`; `train_total == ae_score`.
  - packed pure-AE with one agent batch per epoch: roughly `3.5-4.0s` per
    optimizer update after setup.
  - `torch.compile` was not useful for this short ablation because compile
    setup cost dominated and per-update cost did not improve enough.
- Fast direct mixed-K run:
  `training/runs/20260519_092931_ablate_R45_blend_w03_directmixedK32_fastpure_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  It reached best AE `0.010951` in about `176s` wall time for 40 optimizer
  updates.
- Diagnostic comparison against the curriculum result:
  `training/runs/diagnostics/ablation_R45_directmixed_fast_compare_20260519.html`.
  Trainable-horizon support/source-contact/excess p95:
  - curriculum `0.3`: `0.1051 / 0.2047 / 0.1594`
  - direct mixed-K fast `0.3`: `0.2555 / 0.2730 / 0.2689`
- Interpretation:
  the speed bug is fixed, but skipping the curriculum did not match the R45
  curriculum result in this 40-update test. Direct mixed-K is much faster than
  the broken run, but curriculum still appears to provide useful stabilization
  for this turn-in-place mini ablation.
- Follow-up fairer budget test:
  resumed the direct mixed-K run from its 40-update checkpoint and let it
  plateau with the same pure-AE objective:
  `training/runs/20260519_094241_ablate_R45_blend_w03_directmixedK32_longplateau_from_fast40/checkpoints/checkpoint_best_k32.pt`.
  It stopped at epoch `120`, best AE `0.009223`, in about `7 min`.
- Diagnostic output:
  `training/runs/diagnostics/ablation_R45_directmixed_longplateau_compare_20260519.html`.
- Trainable-horizon foot-skate metrics:
  - curriculum `0.3`: support p95 `0.1051`, source-contact p95 `0.2047`,
    excess p95 `0.1594`.
  - direct mixed-K 40-update: support p95 `0.2555`, source-contact p95
    `0.2730`, excess p95 `0.2689`.
  - direct mixed-K plateau: support p95 `0.2563`, source-contact p95
    `0.3290`, excess p95 `0.2978`.
- Trainable-horizon GT rollout error:
  - curriculum `0.3`: mean joint `0.007418 m`, p95 joint `0.011158 m`,
    mean end-effector `0.010003 m`.
  - direct mixed-K plateau: mean joint `0.039467 m`, p95 joint `0.086105 m`,
    mean end-effector `0.029524 m`.
- Conclusion:
  the direct mixed-K run optimized the AE scalar slightly better than the
  curriculum checkpoint, but produced worse GT alignment and worse foot-skate.
  For this R45 one-clip ablation, the rollout curriculum is doing real
  stabilization work that is not captured by AE loss alone.

## 2026-05-19 - R45 Fast Pure-AE Curriculum Repro

- Clarification:
  the successful R45 curriculum run was scheduled up to `K=32`, not `K=64`.
  The schedule was `2,4,8,16,32`.
- Fast rerun:
  `training/runs/20260519_095707_ablate_R45_blend_w03_fastpure_curriculumK32_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  This used the cleaned pure-AE path where contact-physics auxiliary losses are
  disabled by default and packed rollout is active.
- Setup:
  AE1 old 1-frame denoising prior weight `1.0`, AE2 conditional/root-window
  prior weight `0.3`, old K32 baseline initialization, R45-only mini dataset,
  curriculum `2,4,8,16,32`, one agent batch per epoch, no diagnostics/live
  viewer during training.
- Timing:
  reached K32 and finished in about `4 min` wall time. Internal timing profile
  was roughly `61.8%` forward/loss and `32.8%` backward.
- Best checkpoint:
  epoch `412`, K32, best AE `0.0085025`.
- Diagnostic output:
  `training/runs/diagnostics/ablation_R45_fastpure_curriculum_compare_20260519.html`.
- Trainable-horizon GT rollout error:
  - old curriculum `0.3`: mean joint `0.007418 m`, p95 joint `0.011158 m`,
    mean end-effector `0.010003 m`.
  - direct mixed-K plateau `0.3`: mean joint `0.039467 m`, p95 joint
    `0.086105 m`, mean end-effector `0.029524 m`.
  - fast pure-AE curriculum `0.3`: mean joint `0.009651 m`, p95 joint
    `0.017854 m`, mean end-effector `0.011572 m`.
- Trainable-horizon foot-skate metrics:
  - old curriculum `0.3`: support p95 `0.1051`, source-contact p95 `0.2047`,
    excess p95 `0.1594`.
  - direct mixed-K plateau `0.3`: support p95 `0.2563`, source-contact p95
    `0.3290`, excess p95 `0.2978`.
  - fast pure-AE curriculum `0.3`: support p95 `0.1011`, source-contact p95
    `0.2732`, excess p95 `0.1965`.
- Interpretation:
  the fast cleaned curriculum reproduces the important part of the old result:
  it is close to GT and far better than direct mixed-K. It is not bit-for-bit
  the same result; source-contact slide is worse than the old curriculum, while
  support-p95 slide is similar and AE scalar is better. For future R45-style
  ablations, default to curriculum up to K32 first, then only extend beyond K32
  once the K32 result looks stable.

## 2026-05-19 - Six-Clip AE1/AE2 Blend Sweep

- User requested a 10-point blend sweep from `100% AE1 / 0% AE2` to
  `0% AE1 / 100% AE2`.
- Important correction:
  the mini dataset is exactly `idle + walk forward + circle R/L + turn45 R/L`,
  staged at `training/runs/mini_datasets/idle_walkF_circle_stand45`.
  This is not the older `walkF_stand45_circle` set because that one omitted
  idle.
- Confirmed priors:
  both AE checkpoints used here are full-dataset priors, not mini-dataset
  priors.
  - AE1:
    `training/runs/20260517_161820_denoise_rootlook1_lat32_n0p05_e360/checkpoints/checkpoint_best.pt`
    trained with `periodic_folder_path=ue5/animations_omni_only_full/npz_final`
    and `nonperiodic_folder_path=ue5/animations_transitions_only_full_trimmed/npz_final`.
  - AE2:
    `training/runs/20260519_043759_compat_rootw16_yawbody_lat64_e180/checkpoints/checkpoint_best.pt`
    trained on the same periodic/nonperiodic full folders.
- Sweep setup:
  old K32 baseline initialization, fast packed pure-AE path, curriculum
  `2,4,8,16,32`, no contact-physics losses, no live viewer, no visual reporter.
  Blend weights were normalized percentages:
  `1.000/0.000`, `0.889/0.111`, ..., `0.000/1.000`.
- Diagnostic output:
  `training/runs/diagnostics/blend_sweep_idle_walkF_circle_stand45_20260519/summary.csv`
  and
  `training/runs/diagnostics/blend_sweep_idle_walkF_circle_stand45_20260519/per_clip.csv`.
- Aggregate trainable-horizon results:
  - best GT joint/EE alignment: blend00, AE1 `1.0`, AE2 `0.0`;
    joint mean `0.01314 m`, EE mean `0.02106 m`,
    source-contact excess p95 `0.17595`.
  - best aggregate source-contact excess p95: blend03, AE1 `0.667`,
    AE2 `0.333`; excess p95 `0.16222`, joint mean `0.01344 m`,
    EE mean `0.02370 m`.
  - AE2-heavy blends degraded badly on circles; pure AE2 had joint mean
    `0.02570 m`, EE mean `0.05235 m`, source-contact excess p95 `0.38069`.
- Per-clip caveat:
  blend03 and blend04 improve aggregate source-contact excess, but their
  Circle_R worst-clip excess is much worse than blend00. Since the practical
  preference is "all clips decent" rather than "one excellent average with one
  bad clip", blend00 is the safer winner of this sweep.
- Conclusion:
  on this six-clip mini set, adding AE2 does not produce a clean win. AE1-only
  is currently the best default by conservative rollout metrics. AE2 remains
  useful as a targeted compatibility idea, but this sweep says it should not be
  automatically blended into the baseline without a stronger guard against
  circle/strafe degradation.
- Follow-up correction:
  the first sweep endpoint commands still loaded the zero-weight prior, so
  `AE1=1.0 / AE2=0.0` used only AE1 in the loss but still inherited AE2's
  `root_lookahead_steps=15` for model input/truncation. AE1 itself is a
  1-frame transition prior with `root_lookahead_steps=1`. The training harness
  now filters out zero-weight priors before loading them, so future endpoint
  sweeps are real endpoints and do not accidentally shorten non-periodic tails
  or inflate the future-root horizon.
- Verified rerun:
  `training/runs/20260519_135347_pure_ae1_verified_idle_walkF_circle_stand45_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  Metadata confirms only AE1 is loaded, `ae_prior_weights=[1.0]`, and
  `root_lookahead_steps=1`.
- Verified pure AE1 metrics versus the earlier bugged endpoint:
  - bugged endpoint with hidden AE2 horizon: joint mean `0.01314 m`, EE mean
    `0.02106 m`, support p95 `0.13309`, source-contact excess p95 `0.17595`.
  - verified AE1-only endpoint: joint mean `0.01459 m`, EE mean `0.02373 m`,
    support p95 `0.17615`, source-contact excess p95 `0.20155`.
  - conclusion: true AE1-only remains in the same ballpark and uses the correct
    short AE horizon, but the accidental longer future horizon slightly helped
    the earlier endpoint. Circle_R is still the worst clip and should be used
    as the first stress test when changing priors or model capacity.

## 2026-05-19 - Generated-Negative AE Loop, Fresh07-Fresh11

- Evaluation rule update:
  for long motions, absolute world-space drift is not considered a failure by
  itself. The primary metrics are local/root-relative motion plausibility and
  support/foot-skating style metrics. Direct GT world overlap is mainly useful
  for short clips such as 45-degree turns and circles where phase/trajectory
  ambiguity is limited.
- Robust worst-row AE reduction branch:
  `training/runs/fresh09_robusttop25_controller_full_from_oldk32/checkpoints/checkpoint_best_k32.pt`
  used `ae_row_top_fraction=0.25` and `ae_row_top_weight=1.0`.
  It did not help: support metrics worsened on most key clips, so this branch
  should not be used as a default.
- Fresh10 loop:
  AE:
  `training/runs/fresh10_exact_footmotion_compat_accum_against_fresh09_fakes/checkpoints/checkpoint_best.pt`.
  Controller:
  `training/runs/fresh10_exact_controller_full_footmotion_modelaware_recononly_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  Fresh10 AE cleanly separated GT from Fresh09 generated fakes in lab tests,
  but the Fresh10 controller found a new low-AE skating region. Key trainable
  support p95 values:
  - Walk_F `0.1454` vs GT `0.0518`
  - Circle_L `0.3422` vs GT `0.1280`
  - Circle_R `0.3763` vs GT `0.0469`
  - Turn045_R `0.2259` vs GT `0.0012`
  - Box_L `0.7832` vs GT `0.1404`
- Fresh11 loop:
  AE:
  `training/runs/fresh11_exact_footmotion_compat_accum_against_fresh10_fakes/checkpoints/checkpoint_best.pt`.
  Controller:
  `training/runs/fresh11_exact_controller_full_footmotion_modelaware_recononly_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  Fresh11 improved some clips but did not solve the dataset:
  - Idle support p95 `0.0049` vs GT `0.0031`
  - Walk_F `0.1029` vs GT `0.0518`
  - Circle_L `0.3731` vs GT `0.1280`
  - Circle_R `0.3633` vs GT `0.0469`
  - Turn045_R `0.1734` vs GT `0.0012`
  - Box_L `0.5213` vs GT `0.1404`
- Compatibility-head check:
  Fresh11's compatibility head is very sensitive in lab tests. With
  `compatibility_score_weight=1.0`, GT remains near `0.006-0.010`, while
  Fresh11 rollouts score roughly `5-14` depending on the clip. However,
  controller training with a small `compatibility_score_weight=0.01` did not
  improve the support metrics overall; it helped some Box contact excess but
  worsened several clips. Do not blindly add compatibility loss at this weight.
- Current interpretation:
  the generated-negative loop is moving the needle, but reconstruction-only AE
  loss can still be gamed by skating-compatible poses. The compatibility head
  has useful information, but its controller weighting needs a more principled
  schedule or a better calibration before it becomes a default training term.

## 2026-05-19 - Harsh Simple AE1 Test

- Question:
  can the old simple 1-frame denoising AE be made harsher without changing the
  framework by weighting important features and penalizing worst reconstruction
  dimensions?
- Code branch:
  `training/transition_autoencoder.py` now supports feature weights for pelvis,
  lower body, feet/toes, velocity channels, and an optional top-fraction
  reconstruction penalty. Old checkpoints remain loadable because missing
  metadata falls back to uniform reconstruction weights.
- Harsh AE checkpoint:
  `training/runs/20260519_harsh_ae1_lat16_n0075_top10_foot4_lower2_pelvis2/checkpoints/checkpoint_best.pt`.
  Settings: latent dim 16, non-root input noise `0.075`, pelvis/lower-body/
  velocity weighting, foot/toe weighting `4.0`, top 10 percent feature penalty
  weighted `0.5`.
- Controller checkpoint:
  `training/runs/20260519_harsh_ae1_controller_full_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  Trained from the old K32 baseline with only the harsh AE prior.
- Benchmark folder:
  `training/runs/diagnostics/harsh_ae1_vs_baseline_keyclips_20260519`.
- Result:
  the harsher AE separated synthetic AE training tiers better, but it did not
  improve the controller. Compared to the old K32 baseline, key-clip support
  p95 got worse on walk forward, circles, and 45-degree turns. The only notable
  improvement was source-contact excess on circle clips, but it came with worse
  support p95 and no meaningful GT improvement.
- Key numbers, baseline -> harsh:
  - Idle support p95 `0.0034 -> 0.0032`, joint mean `0.0075 -> 0.0202`.
  - Walk_F support p95 `0.0741 -> 0.1296`, contact excess p95
    `0.3188 -> 2.0423`, joint mean `0.0259 -> 0.0472`.
  - Circle_L support p95 `0.3739 -> 0.7791`, contact excess p95
    `2.2049 -> 1.1064`, joint mean `0.1010 -> 0.1049`.
  - Circle_R support p95 `0.4898 -> 0.6931`, contact excess p95
    `1.2956 -> 0.8968`, joint mean `0.0959 -> 0.0948`.
  - Turn045_L support p95 `0.1698 -> 0.3255`, joint mean
    `0.0267 -> 0.0288`.
  - Turn045_R support p95 `0.1713 -> 0.3546`, joint mean
    `0.0350 -> 0.0418`.
- Conclusion:
  this "harsher AE1" idea is not a default. It makes the AE numerically more
  sensitive, but the controller still finds bad low-loss skating regions. Keep
  the branch as a diagnostic option only.

## 2026-05-19 - Longer Baseline-Style AE Windows

- Goal:
  retry the longer-window idea while keeping the old successful baseline spirit:
  denoising reconstruction, no compatibility head, no foot-motion extras, no
  feature weighting, no supervised loss.
- Compatibility/code note:
  `window_transition_autoencoder.py` now supports baseline-style denoising via
  `input_noise_std` and `input_noise_mask`. AE loaders now tolerate old
  transition-AE checkpoints that lack the newer non-learned
  `reconstruction_weights` buffer; learned parameters are unchanged.
- Literal W16 window AE:
  `training/runs/20260519_w16_like_baseline_denoise_lat512/checkpoints/checkpoint_best_w16.pt`.
  This reconstructs a 16-transition normalized feature window. Latent was set
  to 512, matching the old AE's rough per-frame compression ratio
  (`32 * 16`).
- Literal W16 controller:
  `training/runs/20260519_w16_like_baseline_controller_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  It was trained from the old K32 baseline with a K16 -> K32 curriculum because
  a W16 window prior cannot score rollouts shorter than 16.
- Literal W16 result:
  not a win. Compared with the old K32 baseline, support p95 worsened on
  Walk_F, both circles, and both 45-degree turns. Example baseline -> W16:
  - Walk_F support p95 `0.0741 -> 0.1041`.
  - Circle_L `0.3739 -> 2.1224`.
  - Circle_R `0.4898 -> 1.9143`.
  - Turn045_R `0.1713 -> 0.5279`.
- Root-lookahead-16 AE:
  `training/runs/20260519_ae1_rootlook16_like_baseline_lat32_n0p05/checkpoints/checkpoint_best.pt`.
  This is the closer analogue to the old AE1 baseline: one transition at a
  time, latent 32, denoising `0.05` on non-root features, but with 16 future
  root deltas appended instead of 1.
- Root-lookahead-16 controller:
  `training/runs/20260519_ae1_rootlook16_controller_from_oldk32/checkpoints/checkpoint_best_k32.pt`.
  Trained from old K32 with normal K2/4/8/16/32 mixed-cohort curriculum.
- Root-lookahead-16 key benchmark:
  `training/runs/diagnostics/rootlook16_vs_baseline_keyclips_20260519`.
  Compared to the old K32 baseline, it reduces some source-contact excess but
  still worsens the main support p95 on walk/circle/45-turn clips:
  - Idle support p95 `0.0034 -> 0.0017`.
  - Walk_F support p95 `0.0741 -> 0.1253`, contact excess p95
    `0.3188 -> 0.1475`.
  - Circle_L support p95 `0.3739 -> 0.7700`, contact excess p95
    `2.2049 -> 1.1220`.
  - Circle_R support p95 `0.4898 -> 0.5416`, contact excess p95
    `1.2956 -> 1.1179`.
  - Turn045_L support p95 `0.1698 -> 0.3553`.
  - Turn045_R support p95 `0.1713 -> 0.3619`.
- Conclusion:
  longer root context does carry useful information, since contact-excess
  improves on several moving clips, but by itself it does not remove skating.
  The current 1-frame denoising AE remains the better default unless the longer
  context is paired with another constraint that specifically preserves support
  quality.

## 2026-05-20 - Self-Adversarial Repro Gap Was A Loss-Helper Drift

- Question:
  the regenerated mini self-adversarial AE/controller loop was slightly worse
  than `20260519_152748_selfadv_mini_recipe02_reachK32`. We tested whether this
  was statistical luck.
- Fast battery:
  12 fresh AE-cycle-1 reproductions from the same controller/prior/data all
  landed exactly at `0.008295293897390366`, while the original AE-cycle-1 score
  was `0.008291669189929962`. That is deterministic, not a 50/50 stochastic
  fluke.
- Cause:
  the newer generic reconstruction-loss helper always routed through weighted
  row losses, even when all reconstruction weights were 1 and top-k weighting
  was disabled. Numerically that was not identical to the older direct
  `F.huber_loss` / rowwise `F.mse_loss(...).mean(dim=-1)` path used by the
  original run.
- Fix:
  `transition_autoencoder.reconstruction_loss*` now fast-paths the default
  unweighted/no-top case through the old exact PyTorch loss expressions. The
  weighted/top-k behavior remains available only when requested.
- Verification:
  after the fix, 3 AE-cycle-1 repros matched the original bit-exactly. A full
  self-adversarial two-cycle repro at
  `training/runs/20260520_023146_repro_selfadv_after_lossfix_mini_recipe02_reachK32`
  matched every main artifact bit-exactly:
  AE cycle 1, controller cycle 1, AE cycle 2, and controller cycle 2 all had
  `max_abs_diff = 0.0` against the original saved run.

## 2026-05-20 - GT-Diff Cycles Converge When Kept Reconstruction-Only

- Problem:
  the GT-difference hard-negative cycle had a promising cycle-2 checkpoint, but
  replay-buffer cycle 3 diverged visually and numerically.
- Structural fix:
  `train_model_aware_transition_ae.py` fake rollout collection now uses the
  same non-cyclic start support as controller training. Generated fakes must
  leave room for the controller's future-root lookahead, instead of sampling
  tail starts that the controller cannot legally train through at the same K.
- Negative result:
  the support fix alone stabilized sampling but did not solve convergence.
  Keeping the compatibility/BCE head still failed to beat cycle 2 reliably.
- Clean positive result:
  remove the compatibility classifier from model-aware AE cycles and keep only:
  real reconstruction + fake reconstruction hinge, with `low_energy_high_gtdiff`
  hard-negative selection. No footslide loss was used in the controller.
- Recipe:
  start from the promising GT-diff cycle-2 controller/prior, then for each
  cycle train:
  - AE: `compatibility_real_weight=0`,
    `compatibility_fake_weight=0`, `fake_margin=0.04`,
    `real_weight=1.25`, `fake_weight=1.0`,
    `hard_negative_mode=low_energy_high_gtdiff`,
    `hard_negative_keep_fraction=0.7`,
    `fake_rollout_steps=32`, `fake_starts_per_clip=10`.
  - Controller: pure AE prior, `compatibility_score_weight=0`,
    no contact/footslide physics losses, K schedule `2,4,8,16,32` with mixed
    cohorts `5,15,20,30,40`.
- Monitor result on the mini set
  `idle + walk forward + circle L/R + stand turn 45 L/R`:
  - Old recipe:
    mean support p95 `0.179434`, mean source-excess p95 `0.305586`,
    max source-excess p95 `0.867848`.
  - GT-diff cycle 2:
    mean support p95 `0.178629`, mean source-excess p95 `0.303543`,
    max source-excess p95 `0.837216`.
  - Reconstruction-only GT-diff cycle 3:
    mean support p95 `0.143514`, mean source-excess p95 `0.215739`,
    max source-excess p95 `0.548820`.
  - Reconstruction-only GT-diff cycle 4:
    mean support p95 `0.136991`, mean source-excess p95 `0.206674`,
    max source-excess p95 `0.365415`.
  - Reconstruction-only GT-diff cycle 5:
    mean support p95 `0.135287`, mean source-excess p95 `0.168158`,
    max source-excess p95 `0.339076`.
- Current best checkpoint from this lab:
  `training/runs/20260520_gtdiff_convergence_lab/controller_cycle05_recononly_supportfix_fresh/checkpoints/checkpoint_best_k32.pt`.
- Interpretation:
  the divergence was not inherent to GT-diff hard negatives. The unstable part
  was the separate compatibility classifier/reward channel interacting with
  repeated cycles. Reconstruction-only fake hinge gives a smoother energy
  landscape and the cycle improved monotonically over three measured cycles.
