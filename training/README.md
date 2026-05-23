# Training

The active training code is IK-only and lives in `training/ik`.

Use these entry points:

- `training/ik/train.py` - supervised IK controller training.
- `training/ik/train_simple_autoencoder.py` - one-frame simple AE training.
- `training/ik/train_full_ae_envelope.py` - AE/envelope controller experiments.
- `training/ik/foot_envelope_viewer.py` - foot-slide envelope diagnostics.
- `training/ik/kaggle_prepare.py` / `kaggle_run.py` / `kaggle_sync_tensorboard.py`
  - IK Kaggle workflow.

The old root-level FK/non-IK training scripts were removed to prevent accidental
use. If a new experiment needs trainer behavior that does not exist yet, extend
the canonical IK trainer path instead of copying a new harness.

For the current handoff, rules, and known pitfalls, read:

```text
training/ik/JOURNAL.md
```
