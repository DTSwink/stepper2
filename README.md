# Stepper

Stepper is a local pipeline for game-oriented kinematic locomotion experiments:

- Convert Autodesk FBX animation into NPZ motion data.
- Strip helper bones into a final training skeleton.
- Visualize NPZ motion with mannequin-like collider volumes.
- Train a supervised autoregressive PyTorch locomotion imitator.
- Compare generated motion against ground truth in an HTML viewer.

The current stable path is the supervised DeepMimic-style training framework. The experimental AMP/adversarial work is intentionally not part of the saved GitHub snapshot.

## Main Folders

- `fbx_npz_pipeline/` - FBX to NPZ, NPZ to FBX, skeleton pruning, and NPZ HTML visualization tools.
- `training/` - supervised locomotion trainer, timed runners, and model comparison viewers.

Generated data, UE/Cascadeur animation assets, checkpoints, TensorBoard logs, and local dependency installs are ignored by Git.

## Typical Flow

1. Convert FBX files into NPZ and final stripped NPZ:

   ```powershell
   .\fbx_npz_pipeline\convert_fbx_to_npz.ps1
   ```

2. Visualize an NPZ:

   ```powershell
   .\fbx_npz_pipeline\view_npz.ps1
   ```

3. Train the current fast supervised setup:

   ```powershell
   .\training\run_fast_locomotion_timed.ps1 -Polish 1 -MaxK 8 -HiddenDim 256
   ```

4. Visualize checkpoints:

   ```powershell
   python .\training\visualize_best.py
   ```

## Dependencies

Install the Python dependencies listed in `training/requirements.txt`. Autodesk FBX SDK Python bindings are required for FBX import/export.
