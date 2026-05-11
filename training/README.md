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

`torch.compile` is enabled by default and self-tests before training. If the
local PyTorch build cannot compile, for example because Triton is unavailable on
Windows, the trainer prints the reason and falls back to normal eager GPU
training. You can skip the compile probe explicitly:

```powershell
.\.tools\python310\python.exe .\training\train_locomotion.py --device cuda --no-compile
```

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

Checkpoints are written to:

```text
training/runs/locomotion_mlp/checkpoints
```

The trainer uses the dataset root trajectory as authoritative. The network only
predicts the next body pose; during rollout, the next root comes from the NPZ,
FK recomputes joint canonical positions, and cleaned 6D rotations are fed back
into the next step.

By default, positions are scaled by `position_unit_scale = 0.01`, because the
FBX/NPZ data is in Unreal-style centimeters while the speed constants are easier
to reason about in meters/sec. Set it to `1.0` in `TrainConfig` if you want raw
FBX units.
