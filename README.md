# Stepper

Stepper is now an IK-first locomotion workspace.

The active code lives in:

- `training/ik/` - IK representation, supervised controller training, AE
  experiments, foot-slide envelope tools, TensorBoard helpers, and Kaggle
  packaging.
- `fbx_npz_pipeline/` - FBX/NPZ conversion, skeleton pruning, model-
  reconstructable NPZ baking, and NPZ HTML viewers.

The old non-IK/FK locomotion training stack has been removed. Future training
work should extend the canonical IK trainers instead of adding parallel harnesses.

## Common Commands

Full supervised IK training:

```powershell
.\.tools\python310\python.exe .\training\ik\train.py `
  --periodic-folder .\ue5\animations_omni_only_full\npz_final `
  --nonperiodic-folder .\ue5\animations_transitions_only_full_trimmed\npz_final `
  --run-label full_supervised
```

Single NPZ supervised sanity run:

```powershell
.\.tools\python310\python.exe .\training\ik\train.py `
  --npz .\ue5\animations_omni_only_full\npz_final\M_Neutral_Walk_Loop_F.npz `
  --run-label walkF_supervised
```

Launch TensorBoard:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\ik\launch_tensorboard_latest.ps1
```

Prepare an IK Kaggle payload:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\ik\kaggle_start.ps1
```

## Rules

- Keep IK training work under `training/ik`.
- Keep run outputs/checkpoints out of commits.
- Do not add one-off trainer scripts for mini experiments; add general arguments
  to the canonical trainer instead.
- For current system behavior and gotchas, read
  `training/ik/JOURNAL.md` before changing training code.

Generated data, UE/Cascadeur animation assets, checkpoints, TensorBoard logs,
and local dependency installs are ignored by Git.
