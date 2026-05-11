# Stepper

Stepper is a local pipeline for game-oriented kinematic locomotion experiments:

- Convert Autodesk FBX animation into NPZ motion data.
- Strip helper bones into a final training skeleton.
- Visualize NPZ motion with mannequin-like collider volumes.
- Train a delta-transition-AE guided autoregressive PyTorch locomotion imitator.
- Compare generated motion against ground truth in an HTML viewer.

The current default training path is the scratch delta-AE prior workflow. It
trains a fresh transition autoencoder, a K=1 controller warmup, a K=1 polish,
then a K=2/4/8 autoregressive controller. The older supervised DeepMimic-style
runner is still available for comparison. The experimental AMP/adversarial work
is intentionally not part of the saved GitHub snapshot.

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

3. Train the current default scratch delta-AE setup:

   ```powershell
   .\run_training.ps1
   ```

   This delegates to:

   ```powershell
   .\training\run_delta_ae_scratch.ps1
   ```

   It does not load any older model checkpoints. By default it refreshes the
   HTML viewer once after training; pass `-LiveViewer -SaveLiveEveryEpochs 20`
   if you want live HTML updates during the autoregressive stage.

4. Optional: run the older fast supervised setup:

   ```powershell
   .\training\run_fast_locomotion_timed.ps1 -Polish 1 -MaxK 8 -HiddenDim 256
   ```

5. Visualize checkpoints:

   ```powershell
   python .\training\visualize_best.py
   ```

## Dependencies

Install the Python dependencies listed in `training/requirements.txt`. Autodesk FBX SDK Python bindings are required for FBX import/export.
