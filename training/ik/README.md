# IK Locomotion

This folder owns the active locomotion system.

Current policy:

- Run ids and checkpoint files use `YYYYMMDD_HHMMSS_ik_<label>`.
- IK payload is the 42-dim hand/foot position+rotation+pole/toe representation.
- GPU-resident dense rows are mandatory.
- Batch rows are sampled independently.
- The supervised trainer uses one rollout loop for every K.
- Noncyclic rows require a complete rollout window.
- Full-dataset runs use both:
  - `--periodic-folder ue5\animations_omni_only_full\npz_final`
  - `--nonperiodic-folder ue5\animations_transitions_only_full_trimmed\npz_final`
- TensorBoard cards should stay uncluttered.
- Do not create one-off trainer files for mini experiments.

Before editing training logic, read `JOURNAL.md`.

## Supervised

```powershell
.\.tools\python310\python.exe training\ik\train.py `
  --periodic-folder ue5\animations_omni_only_full\npz_final `
  --nonperiodic-folder ue5\animations_transitions_only_full_trimmed\npz_final `
  --run-label full_supervised
```

## Walk_F Sanity

```powershell
.\.tools\python310\python.exe training\ik\train.py `
  --npz ue5\animations_omni_only_full\npz_final\M_Neutral_Walk_Loop_F.npz `
  --run-label walkF_supervised
```

## TensorBoard

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File training\ik\launch_tensorboard_latest.ps1
```

## Kaggle

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File training\ik\kaggle_start.ps1
```
