# Kinematic Locomotion Imitator

This folder trains a lightweight autoregressive kinematic controller from the
project NPZ motion format.

Edit the top of `train_locomotion.py`:

```python
folder_path = "data/npz"
```

Then run from the project root:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py
```

For a quick smoke test:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --max-epochs 1 --batch-size 4 --future-window-seconds 0.25 --device cpu
```

For GPU training:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --device cuda
```

Recommended K8 Isaac-style supervised run:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --folder-path data/fbx/npz_final --device cuda --training-loop agents --agent-sampling coverage --rollout-schedule 1,2,4,8 --curriculum-max-epochs-per-stage 70 --curriculum-stall-patience-epochs 35 --max-epochs 320 --batch-size 64 --learning-rate 1e-4 --lr-schedule adaptive_plateau --lr-min-factor 0.05 --lr-plateau-patience-epochs 12 --lr-plateau-factor 0.7 --no-compile
```

Use `--agent-sampling coverage` for small clips. It resets rollout agents across
the valid start frames uniformly, which avoids the noisy duplicate/missed starts
you get from pure random reset on tiny datasets.

For AE-prior multi-clip random-agent experiments, prefer the packed per-agent
path: leave `--agent-batch-clips 0` and `--packed-agent-rollout` on. Each agent
samples its own clip and resets independently when a non-cyclic clip would run
out of valid target/future-root frames, while the trainer keeps the rollout in
dense tensors. `--agent-batch-clips 1` is the older one-clip cohort shortcut;
it is fast but can make one hard animation dominate a whole optimizer step.

At the end of every training run, `training/runs/model_comparisons/model_comparison.html`
is refreshed from that run's `checkpoint_best.pt` and source NPZ. Use
`--no-update-comparison-on-exit` for batch sweeps where you do not want the
shared comparison page overwritten.

The recommended learning-rate schedule is `adaptive_plateau`: each new rollout
K starts at `1e-4`, then the trainer lowers LR only when the monitored loss
stops improving for several epochs. This keeps the schedule useful for
placeholder experiments where future datasets and clip lengths are still
unknown. The LR floor is `5e-6`.

Live training viewer:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --folder-path data/fbx/npz_final --device cuda --training-loop agents --agent-sampling coverage
```

The trainer launches `live_training_viewer.py` by default. The window starts in
headless mode with a `Visualise` button. While headless, no pose snapshots are
serialized and no OpenGL rendering is requested by the training process. Click
`Visualise` to see up to four rollout agents in a 2x2 grid; the button then
changes to `Headless`, which disables snapshot capture again.

The desktop shortcut `Stepper Live Training` starts the recommended K8
coverage-agent run with visualisation already enabled. It uses
`training/launch_live_k8_visual.ps1` and writes a timestamped run under
`training/runs/desktop_live_k8_*`.

The viewer also has a `Stop experiment` button. It asks the trainer to exit
cleanly after the current small training unit, preserving normal shutdown and
checkpoint behavior. A tiny loss graph at the bottom reads one scalar row per
epoch from `live_training/loss_history.csv`; this is intentionally separate from
the heavier pose snapshot path. Drag the horizontal splitter above the graph to
give it more vertical room while inspecting a run. Use `--no-live-viewer` for
fully unattended/scripted runs.

`torch.compile` is opt-in. On this Windows RTX 4060 Laptop setup, eager mode is
currently faster for the rollout trainers once compile cold-start is included.
If you want to test a future PyTorch/Triton build, use:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --device cuda --compile --compile-mode reduce-overhead
```

The trainer runs a forward/backward probe before accepting a compiled model and
falls back to eager mode if compilation fails.

To time a run until validation loss drops by a target ratio:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --device cuda --target-loss-reduction 0.98 --stop-at-target-loss-reduction
```

TensorBoard:

```powershell
.\.tools\python310\Scripts\tensorboard.exe --logdir .\training\runs
```

Refresh-on-demand model viewer:

```powershell
.\.tools\python310\python.exe .\training\live_model_viewer_server.py --run-dir .\training\runs\YOUR_RUN_NAME --checkpoint-name checkpoint_last.pt
```

Then open:

```text
http://127.0.0.1:8017/model_comparison.html
```

The HTML is regenerated only when the browser requests or refreshes that URL,
so training no longer pays the cost of repeatedly rendering the viewer.

Asynchronous visual checkpoint reports:

```powershell
.\.tools\python310\python.exe .\training\visual_reporter.py --run-dir .\training\runs\YOUR_RUN_NAME --checkpoint-name checkpoint_last.pt
```

The supervised and AE-prior trainers launch this sidecar by default. It watches
`checkpoint_last.pt`, rolls out the newest checkpoint in a separate process, and
writes five static overlay snapshots at `0%`, `25%`, `50%`, `75%`, and `100%`
to:

```text
training/runs/YOUR_RUN_NAME/visual_reports/latest/index.html
```

This is diagnostic only. The trainer never waits for the sidecar; if rendering
falls behind, the sidecar skips stale checkpoints and catches the newest saved
state. Use `--no-visual-reporter` to disable it entirely, or
`--visual-report-interval-seconds`, `--visual-report-device`, and
`--visual-report-max-frames` to tune cost.

Standalone side viewer:

```powershell
.\training\launch_model_viewer_app.ps1
```

Use Open File to add NPZ actors or checkpoint actors. Checkpoint actors generate
motion on demand during playback; select a model actor in the outliner and use
Set Model Source NPZ to choose the source trajectory it should condition on.

Checkpoints are written to:

```text
training/runs/locomotion_mlp/checkpoints
```

The trainer uses the dataset root trajectory as authoritative. The network only
predicts the next body pose; during rollout, the next root comes from the NPZ,
FK recomputes joint canonical positions, and cleaned 6D rotations are fed back
into the next step.

For loop-clean gait clips, pass `--cyclic-animation`. In that mode, valid start
frames cover the whole clip minus the duplicated final frame: `1..T-2`. Body
pose targets wrap modulo `T-1`, while root transforms past the end continue by
repeating the clip's per-frame root deltas. This keeps random frame
initialization available even for full-clip rollouts, instead of forcing every
long rollout to start near frame 0.

By default, positions are scaled by `position_unit_scale = 0.01`, because the
FBX/NPZ data is in Unreal-style centimeters while the speed constants are easier
to reason about in meters/sec. Set it to `1.0` in `TrainConfig` if you want raw
FBX units.

For experiment notes and known-good run names, see
`training/EXPERIMENT_JOURNAL.md`.
