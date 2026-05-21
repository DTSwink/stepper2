# Contained IK Locomotion

This folder owns the IK-marker experiment code. The main `training/train_locomotion.py`
file is intentionally left untouched.

Policy baked into this path:

- Run ids and checkpoint files use `YYYYMMDD_HHMMSS_ik_<label>`.
- IK markers are always enabled.
- Periodic clips are treated as cyclic.
- GPU-resident rows/rollouts are mandatory. IK code should stay on the contained fast path.
- Batch rows are sampled independently from the dataset; there is no one-animation cohort mode.
- The supervised trainer uses the same rollout loop for every rollout length.
- Full-dataset runs should pass `--periodic-folder` and `--nonperiodic-folder` instead of pretending every clip is the same cyclic walk loop.

Before a real training run, run:

```powershell
.\.tools\python310\python.exe training\ik\perf_audit.py --periodic-folder ue5\animations_omni_only_full\npz_final
```

Simple supervised walk-forward entrypoint:

```powershell
.\.tools\python310\python.exe training\ik\train.py --npz ue5\animations_omni_only_full\npz_final\M_Neutral_Walk_Loop_F.npz
```

Full-dataset supervised entrypoint:

```powershell
.\.tools\python310\python.exe training\ik\train.py --periodic-folder ue5\animations_omni_only_full\npz_final --run-label full_supervised
```
