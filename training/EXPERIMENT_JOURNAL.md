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
