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
