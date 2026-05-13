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

1. Put source FBX files in a dataset folder. By default the training runner
   uses:

   ```powershell
   .\ue5\example_cascadeur
   ```

   The runner looks for `npz_final` inside that FBX folder. If it is missing,
   it automatically creates sibling `npz`, `npz_final`, and `reports` folders
   with the FBX-to-NPZ pipeline.

   To build or rebuild that folder manually:

   ```powershell
   .\fbx_npz_pipeline\ensure_npz_final.ps1 -FbxPath .\ue5\example_cascadeur
   ```

   Add `-Force` to rebuild an existing `npz_final` folder.

2. Visualize an NPZ:

   ```powershell
   .\fbx_npz_pipeline\view_npz.ps1
   ```

3. Train the current default scratch delta-AE setup:

   ```powershell
   .\run_training.ps1
   ```

   To train from another FBX dataset folder:

   ```powershell
   .\run_training.ps1 -FbxPath .\ue5\animations
   ```

   Add `-RebuildNpzFinal` if you want training to regenerate the derived NPZ
   files from the FBX source before starting.

   This delegates to:

   ```powershell
   .\training\run_delta_ae_scratch.ps1
   ```

   It does not load any older model checkpoints. By default it uses
   `<FbxPath>\npz_final` and refreshes the HTML viewer once after training; pass
   `-LiveViewer -SaveLiveEveryEpochs 20` if you want live HTML updates during
   the autoregressive stage.

4. Optional: run the older fast supervised setup:

   ```powershell
   .\training\run_fast_locomotion_timed.ps1 -Polish 1 -MaxK 8 -HiddenDim 256
   ```

5. Visualize checkpoints:

   ```powershell
   python .\training\visualize_best.py
   ```

## Foot Contact Metrics

Foot height and foot slide are deliberately different measurements:

- Height/penetration uses the absolute lowest point on the foot or toe collider.
- Source contact labels use 2D slide speed, which does not use that lowest
  point. It solves the best same-local-point ground-plane displacement over the
  continuous foot sole rectangle and toe sole rectangle, then uses the smaller
  distance.
- Training contact slide loss uses the same continuous sole-point idea, but
  minimizes full 3D velocity for the persistent point on the foot or toe sole.
  This penalizes upward/lateral/forward motion together when the model predicts
  contact.
- Constraint-only experiments currently use a much stricter pinned contact-point
  speed threshold (`0.005 m/s`) than the source contact labelling threshold
  (`0.350 m/s`), because the former is a planted-foot constraint while the
  latter is only a permissive label detector.

## Dependencies

Install the Python dependencies listed in `training/requirements.txt`. Autodesk FBX SDK Python bindings are required for FBX import/export.
