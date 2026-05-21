# Stepper Training Journal

Last cleaned: 2026-05-20

This is the compact handoff journal. The full historical journal was archived to:

`C:\Users\singerie\Documents\Cursor\stepper\training\archive\EXPERIMENT_JOURNAL_FULL_20260520.md`

Keep this file tight. Do not re-add abandoned branches unless they become active again.

## Current Direction

We are working on the self-adversarial transition-prior loop for kinematic locomotion.

The winning direction is:

1. Train / reuse a stabilizing one-frame reconstruction AE prior.
2. Generate controller rollouts.
3. Select bad generated rollouts using GT-difference / skating-style monitors, not naive world-space drift.
4. Train a reconstruction-only model-aware AE so generated bad transitions reconstruct worse than real transitions.
5. Train the controller with pure AE loss, no supervised pose loss and no direct slide-excess loss for this recipe.
6. Repeat cycles.

The currently planned next change is fake replay with geometric age decay:

```text
fresh fake generation       coeff = 1.0
previous cycle fakes        coeff = 0.5
two cycles old              coeff = 0.25
three cycles old            coeff = 0.125
...
```

Reason: naive equal replay made the AE remember old mistakes but weakened learning on the current mistake set. Fresh-only can forget old failure modes. Geometric replay should keep past mistakes alive without letting stale fakes dominate the critic.

## Do Not Reopen

These were useful but are not the active recipe:

- AMP / discriminator PPO imitation.
- PCA/GMM priors.
- Physics-only geometric loss experiments.
- Direct supervised DeepMimic-style frame imitation for this current line.
- Compatibility / BCE classifier head in the model-aware AE.
- Direct slide-excess loss as the main training signal for the current self-adversarial recipe.
- Big-window pure root AE variants that made ghost turning look good numerically but bad visually.

## Important Paths

Project root:

`C:\Users\singerie\Documents\Cursor\stepper`

Main compact runner:

`C:\Users\singerie\Documents\Cursor\stepper\training\train_locomotion_self_adversarial.py`

Model-aware AE trainer:

`C:\Users\singerie\Documents\Cursor\stepper\training\train_model_aware_transition_ae.py`

Main controller trainer:

`C:\Users\singerie\Documents\Cursor\stepper\training\train_locomotion_ae_prior.py`

Current old K32 controller baseline:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt\checkpoints\checkpoint_best_k32.pt`

Current stabilizing one-frame prior:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260517_161820_denoise_rootlook1_lat32_n0p05_e360\checkpoints\checkpoint_best.pt`

Best current saved self-adversarial controller cycle:

Legacy cycle 5 fresh checkpoint from `20260520_gtdiff_convergence_lab`.

## Run Hygiene

- New experiment folders should start with a full timestamp: `YYYYMMDD_HHMMSS_name`.
- Date-only prefixes such as `YYYYMMDD_name` are too ambiguous. The training entrypoints now replace that with a full timestamp automatically.
- Flushing TensorBoard means restarting the TensorBoard viewer and clearing `%TEMP%\.tensorboard-info`. Do not delete event files under `training\runs`.
- In the AE-prior trainer, random-agent experiments use mixed per-row clips on the packed rollout path automatically. Do not add packed-rollout or batch-clip flags to new commands.
- Vocabulary: "linear" means the foot-slide / slide-excess term. "Angular" means the yaw-excess term.

## Canonical Self-Adversarial Recipe

The compact runner currently hard-codes the recipe to avoid flag mistakes.

Exposed arguments should remain minimal:

- `--periodic-folder-path`
- `--nonperiodic-folder-path`
- `--start-controller-checkpoint`
- `--start-prior-checkpoint`
- `--run-name`
- `--cycles`
- `--device`
- `--eval-device`
- `--key-clip-names`
- `--no-evaluate`

Hard-coded AE settings:

- reconstruction-only
- no compatibility / BCE head
- `compatibility_real_weight = 0`
- `compatibility_fake_weight = 0`
- `fake_margin = 0.04`
- `real_weight = 1.25`
- `fake_weight = 1.0`
- `fake_starts_per_clip = 10`
- `fake_rollout_steps = 32`
- `hard_negative_mode = low_energy_high_gtdiff`
- `hard_negative_keep_fraction = 0.7`
- `max_epochs = 80`
- `lr = 3e-4`
- `batch_size = 1024`
- `noise_std = 0.02`

Hard-coded controller settings:

- pure AE prior training
- no supervised loss
- no direct contact / slide-excess physics losses
- no compatibility score
- no `torch.compile`
- mixed per-row random clips on the packed rollout path
- K schedule `2,4,8,16,32`
- mixed cohort windows `2,4,8,16,32`
- mixed cohort weights `5,15,20,30,40`
- `max_epochs = 620`
- `lr = 5e-5`
- `batch_size = 64`
- MLP hidden dim `512`
- hidden layers `2`

## Exact Reproduction Check

This command reproduced the saved cycle-2-to-cycle-5 recipe exactly:

```powershell
.\.tools\python310\python.exe training\train_locomotion_self_adversarial.py `
  --periodic-folder-path training\runs\mini_datasets\idle_walkF_circle_stand45\periodic `
  --nonperiodic-folder-path training\runs\mini_datasets\idle_walkF_circle_stand45\nonperiodic `
  --start-controller-checkpoint training\runs\20260520_041312_selfadv_gtdiffweighted_mini_cycle1_from_oldk32\controller_cycle_02\checkpoints\checkpoint_best_k32.pt `
  --start-prior-checkpoint training\runs\20260520_041312_selfadv_gtdiffweighted_mini_cycle1_from_oldk32\ae_cycle_02\checkpoints\checkpoint_best.pt `
  --run-name clean_exact_repro_from_cycle2 `
  --cycles 3 `
  --device cuda `
  --eval-device cpu `
  --no-evaluate
```

Repro run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260520_214701_clean_exact_repro_from_cycle2`

Expected controller best values:

| Cycle | Meaning | Expected best value |
| --- | --- | ---: |
| 1 | equivalent to saved recon-only cycle 3 | `0.010423826985061169` |
| 2 | equivalent to saved recon-only cycle 4 | `0.01020110584795475` |
| 3 | equivalent to saved recon-only cycle 5 | `0.011211628094315529` |

Physical monitor equality was exact against saved cycle 5:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\diagnostics\clean_exact_repro_from_cycle2_20260520`

Expected monitor:

- mean slide-excess p95: `0.13528687157668173`
- mean source-contact excess p95: `0.16815690422663465`
- max source-contact excess p95: `0.33907386660575867`

## From Old K32 Check

Starting from old K32 baseline and the one-frame prior:

Controller:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260516_044547_hybrid_canonbasis_from_bestk32_constantk32_w010_resumeopt\checkpoints\checkpoint_best_k32.pt`

Prior:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260517_161820_denoise_rootlook1_lat32_n0p05_e360\checkpoints\checkpoint_best.pt`

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260520_221233_clean_from_oldk32`

Controller AE loss by cycle:

| Cycle | best value |
| --- | ---: |
| 1 | `0.010220642201602459` |
| 2 | `0.012548768892884254` |
| 3 | `0.01189617719501257` |
| 4 | `0.0140564925968647` |
| 5 | `0.012971676886081696` |

Physical monitor from old K32:

| Run | mean slide-excess p95 | mean source-excess p95 | max source-excess p95 |
| --- | ---: | ---: | ---: |
| cycle 1 | `0.246490` | `0.413391` | `1.161697` |
| cycle 3 | `0.166809` | `0.240746` | `0.528019` |
| cycle 5 | `0.131833` | `0.191911` | `0.441061` |

Interpretation:

- It improves versus its own early cycles.
- It does not reach the saved canonical cycle-5 shelf.
- Do not conflate exact cycle-2-start reproduction with from-old-K32 convergence.

## GT-Diff Monitor

The fake selector GT metric is:

`32-frame sum of global joint-position RMS vs GT`.

From old K32 run:

| Cycle | mean GT-diff | p95 GT-diff |
| --- | ---: | ---: |
| 1 | `0.469048` | `0.828293` |
| 2 | `0.408659` | `0.713986` |
| 3 | `0.346784` | `0.623724` |
| 4 | `0.336598` | `0.525606` |
| 5 | `0.357363` | `0.622987` |

This improves until cycle 4, then regresses at cycle 5. Saved exact cycle 5 had:

- mean `0.360685`
- p95 `0.590725`

Important: GT-diff alone is not sufficient. Cycle 4 can be better on GT-diff but worse on skating. Always pair it with the physical skating/slide-excess monitor.

## Replay Ablation

Question tested: should older bad fakes be kept?

Naive equal replay test from old-K32 cycle 4:

Start controller:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260520_221233_clean_from_oldk32\controller_cycle_04\checkpoints\checkpoint_best_k32.pt`

Start prior:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260520_221233_clean_from_oldk32\ae_cycle_04\checkpoints\checkpoint_best.pt`

Replay AE:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\replay_ablation_from_oldk32_cycle4\ae_replay_old123_plus_c4\checkpoints\checkpoint_best.pt`

Replay controller:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260520_231521_replay_ablation_from_oldk32_cycle4\controller_replay_cycle5_from_c4\checkpoints\checkpoint_best_k32.pt`

Results:

| Run | mean slide-excess p95 | mean source-excess p95 | max source-excess p95 |
| --- | ---: | ---: | ---: |
| normal fresh-only cycle 5 | `0.131833` | `0.191911` | `0.441061` |
| equal replay old1-3 + fresh4 | `0.143507` | `0.227044` | `0.514735` |

AE margin audit:

| Fake source | fresh-only AE margin success | replay AE margin success |
| --- | ---: | ---: |
| old cycle 2 fakes | `0.297` | `0.874` |
| old cycle 3 fakes | `0.577` | `0.847` |
| fresh cycle 4 fakes | `0.842` | `0.795` |

Interpretation:

- Equal replay helps remember old mistakes.
- Equal replay steals capacity/weight from current failures.
- Next test should use geometric fake replay decay.

## Next Experiment: Geometric Fake Replay

Implement fake replay such that all fake generations are retained but weighted by age:

```text
age 0: fresh fakes        weight = 1.0
age 1: previous cycle     weight = 0.5
age 2: two cycles old     weight = 0.25
age 3: three cycles old   weight = 0.125
...
```

Implementation notes:

- `training\train_model_aware_transition_ae.py` already handles fake buffers with `features` and optional `weights`.
- It currently concatenates buffered fakes with fresh fakes.
- It already applies fake weights in the fake hinge term.
- Add cycle-aware replay in `training\train_locomotion_self_adversarial.py`.
- Store fake generation metadata: source cycle and age.
- Avoid drowning fresh fakes by old samples. Either:
  - sample fake rows proportional to their geometric weights and use an unweighted hinge, or
  - use a grouped weighted loss where each generation contributes according to its geometric coefficient, independent of row count.
- Prefer grouped weighted loss if practical, because it matches the intended math and avoids old generations dominating due to count.

Test plan:

1. Reproduce the exact cycle-2-start run unchanged.
2. Add geometric replay only.
3. Re-run the same cycle-2-start continuation.
4. Compare:
   - controller AE loss,
   - GT-diff mean and p95,
   - mean slide-excess p95,
   - mean source-contact excess p95,
   - max source-contact excess p95.
5. Only then try from old K32.

Expected success condition:

- Does not worsen fresh fake margin success like equal replay did.
- Keeps old fake margin success above fresh-only.
- Improves or preserves the physical monitor relative to fresh-only.

## Geometric Replay From Old K32 Result

Implementation:

- Added cycle-aware fake replay metadata in `training\train_model_aware_transition_ae.py`.
- Fake generations are retained in one buffer and sampled by normalized generation weight:
  age `0,1,2,3,4` use coefficients `1,0.5,0.25,0.125,0.0625`.
- The compact runner now hard-codes decay `0.5`, keeps all fake rows, disables live checkpoint churn, and stops on final K32 stall.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_000021_geo_replay_from_oldk32`

Diagnostics:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\diagnostics\geo_replay_from_oldk32_20260521`

Final cycle-5 replay mix:

| Source cycle | coefficient | sample probability |
| --- | ---: | ---: |
| 1 | `0.0625` | `0.032258` |
| 2 | `0.125` | `0.064516` |
| 3 | `0.25` | `0.129032` |
| 4 | `0.5` | `0.258065` |
| 5 | `1.0` | `0.516129` |

Controller AE loss by cycle:

| Cycle | best K32 value | epoch |
| --- | ---: | ---: |
| 1 | `0.010220642201602459` | 574 |
| 2 | `0.010302556678652763` | 524 |
| 3 | `0.010150065645575523` | 487 |
| 4 | `0.0114653455093503` | 527 |
| 5 | `0.011200749315321445` | 532 |

Training-time note:

- The final cycle entered K32 at epoch 468.
- K32 improved from `0.018676838` to `0.011200749`.
- The run stopped on final stall at epoch 587, saving the remaining epochs up to 620.
- Diagnostic RMSE and live checkpoints stayed disabled, so no obvious avoidable time sink was left in the loop.

GT-diff selector metric:

| Cycle | mean GT-diff | p95 GT-diff |
| --- | ---: | ---: |
| 1 | `0.468437` | `0.829944` |
| 2 | `0.313264` | `0.572237` |
| 3 | `0.355713` | `0.578210` |
| 4 | `0.347719` | `0.626963` |
| 5 | `0.342375` | `0.520779` |

Physical monitor:

| Cycle | mean slide-excess p95 | mean source-excess p95 | max source-excess p95 |
| --- | ---: | ---: | ---: |
| 1 | `0.246496` | `0.413395` | `1.161700` |
| 2 | `0.194682` | `0.274332` | `0.680314` |
| 3 | `0.196465` | `0.268645` | `0.647390` |
| 4 | `0.149382` | `0.217561` | `0.379245` |
| 5 | `0.126830` | `0.196525` | `0.344971` |

Interpretation:

- Compared with fresh-only from-old-K32 cycle 5 (`0.131833`, `0.191911`, `0.441061`), geometric replay improves mean slide-excess and worst-case source-excess, but mean source-excess is slightly worse.
- Compared with equal replay (`0.143507`, `0.227044`, `0.514735`), geometric replay is better on all physical monitor metrics.
- GT-diff p95 is much better than fresh-only from-old-K32 cycle 5 (`0.520779` vs `0.622987`).
- Verdict: geometric replay is a real improvement over equal replay and a credible from-baseline K32 recipe, but it has not matched the saved canonical source-excess shelf (`0.168157`) yet.

## Correction: Cycle Fuel Is GT-Diff

Mistake corrected:

- I temporarily treated the physical monitor (`mean slide-excess p95` and `mean source-excess p95`) as the stop rule.
- That was wrong for this recipe. The self-adversarial cycle is fueled by GT-difference hard negatives, so GT-diff is the acceptance signal.
- Slide-excess/source-excess remain important diagnostics for skating, but they should not stop the cycle unless the experiment explicitly changes the fuel.

Decay-0.25 run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_005404_geo_replay025_from_oldk32`

Physical diagnostics:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\diagnostics\geo_replay025_from_oldk32_continue_guard_20260521`

Continuation result:

- Cycle 6 was incorrectly stopped by the physical source-excess rule.
- Under the corrected GT-diff fuel criterion, cycle 6 improved and should have been accepted.
- The physical diagnostic still matters: cycle 6 reduced worst-case source-excess but worsened mean source-excess slightly.

Controller AE loss by cycle:

| Cycle | best K32 value | epoch |
| --- | ---: | ---: |
| 1 | `0.010220642201602459` | 574 |
| 2 | `0.010358504951000214` | 469 |
| 3 | `0.01007064152508974` | 520 |
| 4 | `0.010731321759521961` | 480 |
| 5 | `0.012896726839244366` | 518 |
| 6 | `0.011503607034683228` | 537 |

Physical monitor:

| Cycle | mean slide-excess p95 | mean source-excess p95 | max source-excess p95 |
| --- | ---: | ---: | ---: |
| 1 | `0.246496` | `0.413395` | `1.161700` |
| 2 | `0.207166` | `0.307550` | `0.852120` |
| 3 | `0.178840` | `0.252516` | `0.587271` |
| 4 | `0.140499` | `0.211577` | `0.405262` |
| 5 | `0.138046` | `0.182714` | `0.410856` |
| 6 | `0.138043` | `0.188249` | `0.334958` |

GT-diff selector metric:

| Cycle | mean GT-diff | p95 GT-diff | max GT-diff |
| --- | ---: | ---: | ---: |
| 1 | `0.468437` | `0.829944` | `0.930270` |
| 2 | `0.372813` | `0.605357` | `0.771048` |
| 3 | `0.348090` | `0.574869` | `0.704175` |
| 4 | `0.320418` | `0.563327` | `0.686559` |
| 5 | `0.333379` | `0.547061` | `0.772648` |
| 6 | `0.282791` | `0.540445` | `0.641656` |

Interpretation:

- GT-diff p95 keeps improving through cycle 6 (`0.547061 -> 0.540445`), so the corrected rule says the decay `0.25` continuation was still productive.
- Compared with decay `0.5`, decay `0.25` cycle 6 is worse on GT-diff p95 (`0.540445` vs `0.520779`), better on mean slide-excess than cycle 5 only by noise, and better on max source-excess than its own cycle 5.
- Compared with fresh-only from-old-K32 cycle 5 (`0.131833`, `0.191911`, `0.441061`), decay `0.25` improves source-excess and worst-case source-excess, but slide-excess is slightly worse.
- Compared with equal replay (`0.143507`, `0.227044`, `0.514735`), decay `0.25` is better on all physical monitor metrics.
- Compared with the warm-checkpoint successful shelf (`0.135`, `0.168` from the reference table), decay `0.25` nearly matches slide-excess but is still above the source-excess diagnostic target.
- Follow-up tested below: adding a foot-slide term inside the fake selector hurt GT-diff fuel, so the selector change was removed instead of being preserved as a config option.

Unsafe partial continuation, discarded:

- Cycle 7 AE had already completed during the earlier partial continuation, and its replay buffer source cycle was already `7`.
- The interrupted cycle 7 controller was resumed from its K8 partial checkpoint into `controller_cycle_07_resume_k08`.
- Best K32 controller value: `0.01196411345154047` at epoch `326`.
- GT-diff p95 broke the rule: cycle 6 `0.540445` -> cycle 7 `0.594814`.
- Mean GT-diff also worsened: `0.282791` -> `0.306863`.
- Max GT-diff worsened sharply: `0.641656` -> `0.917178`.
- This evidence is not used as the official stop point because the controller cycle was resumed from a partial K8 state.

Clean restart to GT break:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_025705_geo_replay025_puregt_restart_clean`

This restart began again from the original old K32 controller and the original one-frame prior. It recomputed each AE/controller cycle fully and judged only the completed K32 controller checkpoint with GT-diff p95. Tolerance was `0.003`.

| Cycle | controller best K32 | controller epoch | AE best | AE epoch | mean GT-diff | p95 GT-diff | max GT-diff | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.0102206422` | 574 | `0.0050500440` | 80 | `0.469048` | `0.828293` | `0.931016` | baseline |
| 2 | `0.0103585050` | 469 | `0.0039902031` | 80 | `0.372851` | `0.606245` | `0.773284` | accept |
| 3 | `0.0100706415` | 520 | `0.0034980946` | 80 | `0.348495` | `0.574774` | `0.704120` | accept |
| 4 | `0.0107313218` | 480 | `0.0030063272` | 78 | `0.320166` | `0.563698` | `0.689673` | accept |
| 5 | `0.0128967268` | 518 | `0.0037277006` | 77 | `0.333265` | `0.548382` | `0.770962` | accept |
| 6 | `0.0115036070` | 537 | `0.0027567595` | 78 | `0.282857` | `0.540945` | `0.640404` | accept |
| 7 | `0.0132301906` | 519 | `0.0026133577` | 80 | `0.413886` | `0.602150` | `0.888085` | break |

Official stop point:

- Cycle 6 is the last accepted decay `0.25` pure-GT checkpoint.
- Clean cycle 7 broke the GT-fuel monotonic rule by `+0.061205` p95 (`0.540945` -> `0.602150`), far beyond the `0.003` tolerance.
- Do not continue to cycle 8 from this branch unless intentionally overriding the GT-fuel rule.

Cycle-7 spike investigation:

- Recomputed GT-diff directly from the saved checkpoints; the CSV was correct.
- The metric is deterministic for the configured starts. With denser starts (`20` per clip), cycle 7 still worsened: cycle 6 p95 `0.586675` -> cycle 7 p95 `0.638557`.
- The biggest starts-10 per-clip p95 jumps from cycle 6 to clean cycle 7 were:
  - `Stand_Turn_045_R`: `0.187482` -> `0.518604`
  - `Walk_Loop_F`: `0.621372` -> `0.854009`
  - `Circle_Strafe_L`: `0.302631` -> `0.515805`
- A real weighting bug was found in the GT-diff hard-negative trainer: new GT-diff fake weights kept reusing the cycle-1 severity reference (`2.349512`). By cycle 7, selected GT-diff severity was only `0.324920`, so fresh selected fakes averaged weight `0.138292`.
- Patched this so `low_energy_high_gtdiff` recomputes the severity reference from the current selected fake batch; the carried `slide_excess` reference is kept only for the legacy foot-slide mode.
- Probe AE with the fixed weighting separated current fakes much better: final `fresh_ok` improved from `0.440` to `0.920`.
- However, the fixed-weight controller probe made GT worse, not better: cycle 7 p95 became `0.833850` (starts 10) / `0.885999` (starts 20).
- Therefore the stale reference was a real bug but not the root cause of this cycle-7 break.
- Root cause: the controller can exploit the AE prior. Under clean AE7, cycle-6 controller rollouts had mean AE rollout energy `0.039302`; the cycle-7 controller drove that down to `0.015819` while GT p95 rose to `0.602150`. The highest-GT-diff walk rollouts had low AE energies around `0.012`-`0.014`, so AE score was no longer a reliable proxy for GT fuel.
- Mechanism: clean AE7 was saved with `root_lookahead_steps=1` but `conditional_root_window=false`. That means the transition AE reconstructs the whole supplied transition vector, including the root lookahead, instead of predicting the target next-pose fields from the current/root condition. It is a plausibility/reconstructability critic, not a paired GT correctness critic.
- Worst-rollout nearest-real-feature checks back a phase-slip loophole. The worst `Walk_Loop_F` cycle-7 rollout (`GT-diff=0.888085`) matched nearest real walk frames from the same clip, but after step 6 the nearest phase jumped by roughly `+90` frames and then `+30` frames. Another bad walk start (`GT-diff=0.812361`) jumped by about `-30`, `-120`, then `-90` frames. So the controller found plausible walk transitions that the AE likes, while drifting to the wrong phase relative to the exact GT rollout.
- Implication: stronger fake weighting alone is not the natural fix; the probe made that worse. The next targeted test should make the critic conditional on the cycle fuel, e.g. `conditional_root_window=true`, or add GT-diff checkpoint selection inside controller training so AE score cannot choose a phase-slipped controller.

## Conditional Root-Window Restart Result

Tested `conditional_root_window=true` from the original old K32 controller and original one-frame prior. The compact runner temporarily passed `--conditional-root-window` to the model-aware AE, kept geometric replay decay `0.25`, and judged completed K32 controller checkpoints by GT-diff p95 with tolerance `0.003`.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_042356_geo_replay025_condroot_restart_from_oldk32`

GT-diff fuel metric:

| Cycle | controller best K32 | controller epoch | AE best | AE epoch | mean GT-diff | p95 GT-diff | max GT-diff | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.0366126075` | 525 | `0.0515598208` | 80 | `1.547952` | `4.303680` | `4.386157` | baseline |
| 2 | `0.0148491506` | 584 | `0.0108453101` | 80 | `0.407472` | `0.761532` | `1.010741` | accept |
| 3 | `0.0185371805` | 458 | `0.0086185941` | 79 | `0.341963` | `0.578889` | `0.778584` | accept |
| 4 | `0.0194877442` | 516 | `0.0073726648` | 78 | `0.355574` | `0.650527` | `0.695832` | break |

Result:

- Conditional root-window did not rescue the cycle; it broke the GT-fuel rule at cycle 4 (`0.578889 -> 0.650527`, delta `+0.071638`).
- Cycle 1 was much worse than the unconditional clean restart. This is probably because the conditional AE changes the reconstruction scale/architecture: the old one-frame prior can only partially initialize it (`net.0` and final decoder layer are shape-mismatched), and the old `fake_margin=0.04` was too low for the first conditional AE. Cycle-1 fake energy was around `0.415`, so the fake hinge was almost entirely inactive (`fresh_ok=0.990`) even before the controller learned against it.
- Cycle 3 recovered to a normal-looking shelf, but cycle 4 still regressed. Per-clip cycle 3 -> 4 p95: Walk_F improved slightly (`0.703173 -> 0.677457`), Circle_Strafe_L improved (`0.567662 -> 0.385300`), but Circle_Strafe_R worsened (`0.459071 -> 0.599823`) and the global p95 broke.
- Conclusion: `conditional_root_window=true` alone is not the next successful recipe. If revisited, it needs retuning around the conditional AE scale/fake margin or a separately warm conditional prior; do not leave it as the compact runner default.

## Simple Conditional AE + Walk Scratch Micro

Switched gears to a simple one-frame root-conditioned transition AE:

- Train the AE on the full dataset, not just walk.
- Use `root_lookahead_steps=1`, `conditional_root_window=true`, reconstruction-only, no compatibility head.
- Then train only the controller/model on `M_Neutral_Walk_Loop_F`.
- Controller must start from scratch, not from old K32.

AE run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_051229_simple_condroot1_full_recon_e160`

AE setup:

- periodic: `ue5\animations_omni_only_full\npz_final` (`15` clips)
- nonperiodic: `ue5\animations_transitions_only_full_trimmed\npz_final` (`211` clips)
- latent `32`, hidden `512`, LR `3e-4`, batch `1024`, epochs `160`, input noise `0.02`, uniform-clip sampling
- best train loss `0.0163799`
- final tier report: clean `0.0374035`, slight `0.0408204`, bad `0.546742`, random noise `1.66217`

Walk-only scratch controller run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_051502_micro_walkF_simple_condroot1_scratch`

Result:

| Model | K32 epoch | AE score | 32-step GT mean | 32-step GT p95 | 32-step GT max | trainable source-excess p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch conditional-AE controller | 234 | `0.0277024` | `1.146909` | `1.387386` | `1.428030` | `3.0471` full / `2.6581` trainable |
| old K32 baseline reference | 7620 | `0.0031466` | `0.543699` | `0.835545` | `0.935658` | `0.3188` full / `0.2849` trainable |

Takeaway:

- The simple conditional AE is trainable and separates clean/bad tiers on the full dataset.
- But using it alone as a from-scratch walk controller objective is not enough. It reaches K32 quickly, yet the learned walk skates/drifts badly.
- This is a useful negative control: the conditional critic may be a scoring component, but pure from-scratch AE-prior training does not bootstrap a good walk policy by itself.

Next direction:

- Reuse this simple conditional AE as a standing stabilizer/checkpointed prior, not as the only bootstrap objective.
- Canonical reusable AE checkpoint:
  `C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`
- First train the controller supervised against ground-truth pose/transition targets, with the excess-envelope slide-excess loss that was dropped earlier restored as an auxiliary constraint.
- After the supervised/foot-envelope warm start behaves, perturb the controller/poses and train recovery with:
  - the simple conditional AE stabilizer,
  - excess-envelope slide-excess loss,
  - perturbations that force the model to learn that weird poses still cannot solve the rollout by slide-excess.
- Motivation: pure AE-prior training from scratch did not bootstrap walk, but the AE can still be useful as a reusable "stay near plausible root-conditioned transitions" regularizer once GT supervision and the slide-excess envelope provide the primary behavioral target.

## Excess-Envelope GT Audit

Before bringing the excess-envelope slide-excess loss back into the supervised/perturbation plan, reran the packed-envelope ground-truth invariant check on the full real dataset used for the simple conditional AE.

Audit artifacts:

- Summary: `C:\Users\singerie\Documents\Cursor\stepper\training\runs\audits\20260521_excess_envelope_full_dataset\summary.json`
- Per-clip CSV: `C:\Users\singerie\Documents\Cursor\stepper\training\runs\audits\20260521_excess_envelope_full_dataset\per_clip.csv`

Setup:

- periodic: `ue5\animations_omni_only_full\npz_final` (`15` clips)
- nonperiodic: `ue5\animations_transitions_only_full_trimmed\npz_final` (`211` clips)
- valid real transitions: `10687`
- future window: `8`
- excess-envelope margin `1.05`, KNN `32`
- cache: legacy pre-rename envelope cache artifact

Result:

| Check | max GT value | max bound | max excess | nonzero excess count |
| --- | ---: | ---: | ---: | ---: |
| slide-excess | `1.862085 m/s` | `1.955189 m/s` | `0.0` | `0` |
| yaw-excess | `18.202225 rad/s` | `19.112335 rad/s` | `0.0` | `0` |

Historical scalar context, now removed from the trainer:

- old fallback slide-excess threshold would have been `1.9551896 m/s` (`1.05 * max GT slide-excess`).
- This is permissive because the full transition dataset contains sharp legitimate foot motions, e.g. `M_Neutral_Walk_Spin_LL_to_F_Lfoot`.
- Removed the scalar-threshold fallback from the trainer so restored slide-excess can only use packed excess-envelope bounds.

Conclusion:

- Ground-truth is below both packed-envelope thresholds again: slide-excess excess max `0`, yaw-excess excess max `0`.
- Safe to reintroduce the excess-envelope slide-excess loss, with the usual caution that the envelope is a root-conditioned zero-loss bound and should be used in packed mode.

## Walk Micro: Corrected Envelope Calibration

The first envelope-resume run `20260521_walkF_env010_from_bestk32` was miscalibrated for the intended test and should not be used as evidence. It scaled the slide-excess loss from the logged mixed-cohort rollout mean, not from the full K32 generated rollout scored by the AE prior.

Correct calibration target:

- Start controller: `20260521_051502_micro_walkF_simple_condroot1_scratch\checkpoints\checkpoint_best_k32.pt`
- AE prior: `20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`
- Roll out the controller checkpoint for full K32 over valid walk starts.
- Score the generated rollout with the AE prior.
- Scale envelope terms so their K-averaged contribution starts at `0.1 * AE prior score`.
- Do not use GT drift/skating diagnostics to choose these weights.

Correct full-K32 generated-rollout calibration:

| source checkpoint | K-avg AE prior score | K-avg slide-excess raw | slide-excess weight for `0.1 * AE` | K-avg yaw-excess raw |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint_best_k32.pt` | `0.0368001` | `0.0883263` | `0.04166379` | `0.0` |
| `checkpoint_last.pt` | `0.0377239` | `0.0984654` | `0.03831189` | `0.0` |

Vertical yaw-excess has zero envelope excess on this generated walk rollout, so no finite weight can make it start at `0.1 * AE`. Treat it as inactive for this exact micro unless training creates yaw excess later.

Corrected run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_walkF_env010_aeprior_fullk32_from_bestk32`

Setup:

- full K32 rollout, no mixed short cohorts
- slide-excess envelope weight `0.04166379`
- yaw-excess weight `0.04166379`, scale `1.0`, but raw yaw stayed `0`
- capped at `60` epochs

Training result:

| epoch | total | raw AE | raw slide-excess | raw yaw-excess |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0400066` | `0.0363501` | `0.0877622` | `0.0` |
| 58 best | `0.0258092` | `0.0254499` | `0.00862338` | `0.0` |

Diagnostics after training:

| Model | GT mean | GT p95 | GT max | rollout slide-excess p95 | trainable source-excess p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| corrected envelope resume | `1.931325` | `2.342181` | `2.438082` | `0.0579087` | `5.6592` |
| scratch conditional-AE controller | `1.134544` | `1.387777` | `1.449384` | `0.168095` | `2.6581` |
| old K32 reference | `0.589845` | `0.945574` | `0.948876` | `0.0406236` | `0.2849` |

Interpretation:

- Negative result. The corrected AE-prior/full-K32 calibration reduced the training excess-envelope raw term and the AE objective, but the actual walk rollout got much worse on GT drift and source-contact skating.
- This confirms the excess-envelope mean alone is not enough to protect the walk cycle when optimizing the AE-prior objective.
- Do not use `20260521_walkF_env010_aeprior_fullk32_from_bestk32` as a good checkpoint.

## Full Dataset Scratch AE Calibration

Goal:

- Reuse the full-dataset simple conditional AE as the controller prior.
- Train a controller from scratch on the whole real dataset.
- Measure the trained controller's generated full-K32 rollouts against the packed excess-envelope losses.
- Use the average excesses to set carried auxiliary weights. GT drift/skating diagnostics are not used for this calibration.

AE prior:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`

Dataset:

- periodic: `ue5\animations_omni_only_full\npz_final` (`15` clips)
- nonperiodic: `ue5\animations_transitions_only_full_trimmed\npz_final` (`211` clips)

Training:

- Initial scratch run: `20260521_full_simpleAE_scratch`
- The initial run used the default long curriculum and only reached K4 by epoch `280`; do not use it as the final calibration checkpoint.
- Continuation from scratch K4 with shorter stage limits: `20260521_full_simpleAE_scratch_k04_to_k32`
- K32 best checkpoint:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_full_simpleAE_scratch_k04_to_k32\checkpoints\checkpoint_best_k32.pt`

K32 best:

- epoch `138`
- train best AE objective `0.0622488`

Calibration artifacts:

- all valid starts: `C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_full_simpleAE_scratch_k04_to_k32\full_dataset_envelope_calibration_all_valid_starts.json`
- K32-eligible trainer starts only: `C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_full_simpleAE_scratch_k04_to_k32\full_dataset_envelope_calibration.json`

All-valid-start calibration, full K32 generated rollout:

| quantity | value |
| --- | ---: |
| valid start rows | `10687` |
| K-avg AE prior score | `0.0728193` |
| K-avg slide-excess excess loss | `0.753669` |
| K-avg yaw-excess excess loss | `0.0467158` |
| target per auxiliary (`0.1 * AE`) | `0.00728193` |

Carried weights from all valid starts:

| loss | weight |
| --- | ---: |
| slide-excess envelope | `0.00966197` |
| yaw-excess envelope | `0.155877` |

Important:

- These weights assume `--yaw-excess-loss-scale-radps 1.0`.
- The K32-eligible trainer-start subset gave smaller weights (`0.00787004` slide-excess, `0.116087` yaw-excess), but the all-valid-start full dataset calibration is the one to carry forward.
- The average vertical-yaw excess is nonzero on the full dataset, unlike the walk-only micro.

## Idle Random-Init AE-Only Sanity Check

Goal:

- Check whether the reusable simple AE prior can train the controller to settle toward idle from arbitrary body poses.
- Target motion: only `M_Neutral_Stand_Idle_Loop`.
- Initial body pose: random frame from the full real dataset.
- Loss: AE prior only, no slide-excess or yaw-excess envelope terms.

Implementation note:

- Patched `--init-pose-sampling random_dataset` so the initial pose can come from any loaded clip, including synthetic/init-only clips. This lets the idle clip be the only trainable real target while the full dataset is loaded as the random init pool.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_idle_randominit_AEonly_from_fullscratch`

Setup:

- trainable real clip: `training\runs\micro_datasets\idle_only\periodic\M_Neutral_Stand_Idle_Loop.npz`
- random-init pool: idle target clip plus full real dataset loaded as synthetic/init-only clips (`227` clips total)
- start checkpoint: `20260521_full_simpleAE_scratch_k04_to_k32\checkpoints\checkpoint_best_k32.pt`
- prior: `20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`
- 32-frame rollout, AE-only, no slide-excess/envelope weights

Best:

- epoch `55`
- best AE value `0.0435977`

Sanity metric artifact:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_idle_randominit_AEonly_from_fullscratch\idle_randominit_sanity_summary.json`

Fixed 256 random init poses, average distance to the idle target:

| Model | body-pose error at start | body-pose error after 32 frames | body-pose error still left | hands/feet error at start | hands/feet error after 32 frames | hands/feet error still left |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before idle tune | `0.163028` | `0.0723242` | `0.443631` | `0.295750` | `0.0909693` | `0.307589` |
| after idle tune | `0.163028` | `0.0512103` | `0.314120` | `0.295750` | `0.0615551` | `0.208132` |

Interpretation:

- Positive sanity check. The AE-only idle tune does naturally pull random full-dataset poses toward idle.
- The tuned model improves over the full-dataset scratch checkpoint on the same fixed random init set.
- This does not yet prove the recovery recipe is good under perturbations, but the simple AE prior is at least capable of learning an idle basin without a slide-excess loss.

## Idle Random-Init Slide-Excess Check, Plain-English Result

Question:

- If the idle controller starts from a random pose, does adding slide-excess loss teach it to lift a foot instead of dragging both feet?
- Before answering that, remove the hidden idle-only rule that forced both feet to behave as planted feet. That old rule made the previous slide-excess idle test unfair.

Code cleanup:

- Removed the idle-only "pin both feet" rule.
- Idle now uses the same slide-excess rule as the other motions: judge the foot that is sliding less at that instant.
- Updated the viewer diagnostics to show the same rule.
- Forced the trainer to rebuild the slide-excess limits instead of reusing the old idle-pinned limits. This was just cache bookkeeping, not an experiment result.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_idle_randominit_slide_min_from_fullscratch`

Setup:

- Same random-pose-to-idle test as the AE-only sanity check.
- Start checkpoint: `20260521_full_simpleAE_scratch_k04_to_k32\checkpoints\checkpoint_best_k32.pt`
- Prior: `20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`
- Losses: AE prior plus slide-excess only. No yaw-excess term.

Result:

- Not a win.
- The slide-excess loss made the model lift a foot slightly more often.
- It also lowered the slide-excess number a little.
- But the actual idle recovery got worse than the AE-only run.
- The "both feet dragging at once" check also got worse than AE-only.

Numbers worth keeping:

| Check | AE-only idle run | with slide-excess | Better result |
| --- | ---: | ---: | --- |
| Final body-pose error after 32 frames | `0.029654` | `0.032276` | AE-only |
| Final hands/feet marker error after 32 frames | `0.035436` | `0.044914` | AE-only |
| Time where both feet drag at once | `45.0%` | `49.5%` | AE-only |
| Slide-excess amount | `0.201874` | `0.189594` | slide-excess |
| Foot-lift signal | `0.006343 m` | `0.013564 m` | slide-excess |

How to read those checks:

- "Final body-pose error" means how close the body pose is to idle after the 32-frame recovery. Lower is better.
- "Final hands/feet marker error" means how close the hands and feet are to the idle target after the same recovery. Lower is better.
- "Time where both feet drag at once" means how often both feet were sliding faster than 5 cm/s at the same time. Lower is better.
- "Slide-excess amount" means how much the less-sliding foot still exceeded the learned safe sliding limit. Lower is better.
- "Foot-lift signal" means the high-end foot height during the rollout. Higher means it is lifting a foot more.

Decision:

- Keep the idle pinning removal.
- Do not treat this slide-excess idle run as a successful recipe.
- AE-only is still the cleaner idle recovery baseline.

## Idle Random-Init Three-Loss Rerun

Question:

- Redo the idle random-pose-to-idle experiment with all three loss terms visible separately.
- Use the full timestamp naming rule so the run is easy to identify in TensorBoard.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_074317_idle_randominit_all3loss_from_fullscratch`

Setup:

- Same idle target and random full-dataset initial-pose pool as the previous idle checks.
- Start checkpoint: `20260521_full_simpleAE_scratch_k04_to_k32\checkpoints\checkpoint_best_k32.pt`
- Prior: `20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`
- Loss weights:
  - AE prior: `1.0`
  - slide-excess: `0.00966197`
  - yaw-excess: `0.155877`
- 32-frame rollout, packed random-agent path, no visual reporter.

TensorBoard curves to inspect:

- `loss/ae_score`
- `loss/slide_excess`
- `loss/yaw_excess`
- Raw versions are under `monitor/raw_ae_score`, `monitor/raw_slide_excess`, and `monitor/raw_yaw_excess`.

Training scalar snapshot:

| Curve | first | best/min | last |
| --- | ---: | ---: | ---: |
| total | `0.126404` | `0.071478` | `0.075604` |
| AE score | `0.061589` | `0.047171` | `0.049200` |
| weighted slide-excess | `0.003277` | `0.001451` | `0.001715` |
| weighted yaw-excess | `0.061538` | `0.022474` | `0.024689` |
| raw slide-excess | `0.339160` | `0.150154` | `0.177462` |
| raw yaw-excess | `0.394787` | `0.144177` | `0.158385` |

Immediate read:

- This run is good for inspecting the three-loss balance.
- Yaw-excess is a real part of the total here.
- Slide-excess is visible and decreasing, but after calibration its weighted contribution is much smaller than AE and yaw.

## Idle Random-Init Linear x100 Rerun

Question:

- Repeat the three-loss idle random-init experiment, but multiply the linear foot-slide weight by `100`.
- Keep the angular yaw weight unchanged.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_075943_idle_randominit_linear100_angular1_from_fullscratch`

Weights:

- AE prior: `1.0`
- linear / foot-slide: `0.966197`
- angular / yaw: `0.155877`

Training scalar snapshot:

| Curve | first | best/min | last |
| --- | ---: | ---: | ---: |
| total | `0.450823` | `0.136883` | `0.144793` |
| AE score | `0.061589` | `0.061589` | `0.067886` |
| weighted linear / foot-slide | `0.327695` | `0.038168` | `0.043504` |
| weighted angular / yaw | `0.061538` | `0.032258` | `0.033403` |
| raw linear / foot-slide | `0.339160` | `0.039504` | `0.045026` |
| raw angular / yaw | `0.394787` | `0.206942` | `0.214290` |

Immediate read:

- The x100 linear weight strongly suppresses foot sliding.
- It also worsens the AE score versus the balanced three-loss run.
- Treat this as a useful diagnostic for the foot-slide/yaw/AE tradeoff, not yet as a clean recipe.

## Excess-Envelope Meaning Correction

Correction:

- The foot-slide envelope should be an upper allowance for each root-motion situation, not a strict planted-foot target.
- For idle, if every frame has the same root-motion situation, the foot-slide envelope should cover the highest GT foot-slide seen in that idle situation, plus the configured margin.
- The temporary lower-bound idea that made idle almost zero was wrong for an envelope. That would be a separate "force planted feet" target, not the restored envelope loss.

Code correction:

- `training/excess_envelope.py` now uses the upper nearest same-situation bound for both linear / foot-slide and angular / yaw.
- It still depends on the two situation axes: root yaw change over the future window and future path bend.
- It no longer lets an individual frame legalize itself separately; the allowance comes from the matching situation neighborhood.

Idle exception:

- For clips with `idle` in the filename, the envelope is now hard-coded to `0` for both linear / foot-slide and angular / yaw.
- This intentionally treats idle as planted feet plus no foot yaw, rather than as permission to reproduce tiny GT jitter.
- Verified on `M_Neutral_Stand_Idle_Loop`: linear bound min/mean/max `0/0/0`, angular bound min/mean/max `0/0/0`.

## GT-Diff + Slide Selector Result

Tested direct patch: keep the existing `low_energy_high_gtdiff` mode name but mix foot-slide excess into the fresh fake selector by normalized rank. No new user-facing flag was added.

Run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_015210_geo_replay025_gtdiff_slide_from_oldk32`

GT-diff fuel metric:

| Cycle | mean GT-diff | p95 GT-diff | max GT-diff |
| --- | ---: | ---: | ---: |
| 1 | `0.490822` | `0.651546` | `0.820216` |
| 2 | `0.439948` | `0.745376` | `0.927120` |

Result:

- The p95 GT-diff fuel broke the monotonic rule from cycle 1 to cycle 2 (`0.651546 -> 0.745376`).
- This is not close enough to keep training.
- The run was stopped during cycle 3 controller training.
- The slide-rank selector patch was removed. `low_energy_high_gtdiff` is back to GT-diff-only severity.
- Do not preserve this as a config option unless there is a new reason to re-test it.

## Walk-Forward Supervised Scratch

Implementation note:

- `training/train_locomotion.py` now caches fixed GT local rotation matrices in each `MotionClip`.
- The supervised rotation losses fetch those cached matrices instead of rebuilding the target GT rotation matrices from 6D every loss call.
- The predicted side is still computed live, as intended.

Smoke:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_093106_walkF_supervised_cachedgt_smoke`

- One CPU epoch passed on `M_Neutral_Walk_Loop_F`.
- This verifies the cached GT fields survive the training path.

Full scratch run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_093649_walkF_supervised_cachedgt_scratch`

- Dataset: `training\runs\micro_datasets\walk_forward_only\periodic`
- Training: supervised GT-only locomotion loss, no contact/envelope extras.
- K schedule: `1,2,4,8,16,32`
- Result at epoch 300: best K32 supervised loss `0.073163`.
- Best checkpoint: `C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_093649_walkF_supervised_cachedgt_scratch\checkpoints\checkpoint_best_k32.pt`
- This run is added to the existing TensorBoard compare folder without flushing older runs.

## Idle Recovery One-Planted Loss Lab

Goal:

- Solve the foot-sliding issue in a contained fast idle-recovery lab.
- Start from walk-forward poses, target idle, keep the simple conditional-root AE as stabilizer.
- Validate first on the two selected walk-forward starts, then on random full-dataset starts.
- Desired behavior: one foot may move, but both feet should not move together; at least one foot is always planted.

Contained runner:

`C:\Users\singerie\Documents\Cursor\stepper\training\idle_recovery_slide_lab.py`

Important metric contract:

| Metric | Pass line |
| --- | ---: |
| final joint RMSE to idle | `<= 0.075 m` |
| both-feet-moving rate | `<= 0.05` |
| both-feet-low-and-moving rate | `<= 0.02` |
| planted-foot availability | `>= 0.90` |
| left solo / right solo motion | each `>= 0.05` |

What worked:

- K32 can hit either target recovery or clean feet, but it kept trading one for the other on random full-dataset starts.
- K64 is the natural horizon for the strict one-planted rule; it gives the controller enough time to move one foot, settle it, then move the other.
- The useful new loss term is not another threshold flag. It is a smooth one-foot rule:
  - penalize `left_speed * right_speed`, so one foot can move fast only when the other is slow;
  - add a contact-aware version of the same product, so "both feet low and moving" is punished harder than airborne swing.
- The final polish used mixed fixed+random init batches so the two hand-picked walk starts and random starts had to pass together.

Best run:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_143000_idle_recovery_k64_mixed_polish_v18`

Best checkpoint:

`C:\Users\singerie\Documents\Cursor\stepper\training\runs\20260521_143000_idle_recovery_k64_mixed_polish_v18\checkpoints\checkpoint_best.pt`

Recipe at the successful checkpoint:

- Rollout K: `64`
- Init mix: half fixed two walk-forward starts, half random full-dataset starts
- AE stabilizer: `0.3`
- Slide loss: `6`
- both-fast loss: `2`
- speed-product loss: `0.08`
- both-ground-product loss: `1.5`
- foot-plan loss: `25`
- target loss: `0.2`
- terminal target loss: `70`
- target foot global weight: `0.5`

Independent reload check for the saved best checkpoint:

| Eval batch | success | final RMSE | both-moving | both-low-moving | planted | solo left/right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed two walk starts | `1` | `0.07446` | `0.008` | `0.008` | `1.000` | `0.070 / 0.281` |
| random seed 1234 | `1` | `0.07198` | `0.012` | `0.009` | `0.997` | `0.124 / 0.114` |
| random seed 2345 | `1` | `0.07480` | `0.014` | `0.010` | `0.994` | `0.140 / 0.127` |
| random seed 3456 | `1` | `0.06953` | `0.017` | `0.013` | `1.000` | `0.082 / 0.114` |
| random seed 4567 | `1` | `0.07163` | `0.018` | `0.015` | `0.999` | `0.100 / 0.131` |
| random seed 5678 | close miss | `0.07577` | `0.028` | `0.022` | `0.997` | `0.123 / 0.208` |

Read:

- This is the first checkpoint that passes the official fixed+random metric contract and visibly follows the one-foot-at-a-time rule.
- The extra random seed 5678 miss is tiny, but it is still a miss; do not call this exhaustive full-dataset proof.
- The failed hard-seed polish run `20260521_144000_idle_recovery_k64_hardseed_polish_v19` made target recovery worse. Keep `v18` as the answer checkpoint for now.

## Fresh Context Instructions

If this file is read by a new Codex context:

1. Work in `C:\Users\singerie\Documents\Cursor\stepper`.
2. Do not browse old runs randomly. Start from this journal.
3. Verify `git status --short` before edits.
4. Preserve the compact runner's minimal flag surface.
5. Do not add new loss families unless the user explicitly redirects.
6. The slide-excess selector test has already failed and was removed; do not reintroduce it without a new reason.
7. Latest official pure-GT decay `0.25` stop point is clean restart cycle 6 from `20260521_025705_geo_replay025_puregt_restart_clean`; clean cycle 7 broke GT p95.
8. Cycle-7 investigation found an AE-prior exploit, not a metric typo: the unconditional transition AE can reward plausible wrong-phase walk transitions. Do not trust controller AE score alone as the cycle-success signal.
9. The conditional-root-window restart from old K32 also failed; it stopped at cycle 4. Do not switch the compact runner default to conditional mode unless the margin/initialization problem is intentionally retuned.
10. Simple full-dataset conditional AE plus walk-only scratch controller is a negative control: the AE tier separation is good, but pure from-scratch AE-prior training produces a bad walk.
11. New direction: supervised GT warm start with excess-envelope slide-excess loss, then perturbation/recovery training using the simple conditional AE stabilizer plus excess-envelope slide-excess. Reuse `20260521_051229_simple_condroot1_full_recon_e160\checkpoints\checkpoint_best.pt`.
12. Excess-envelope GT audit passed on the full real dataset: slide-excess excess max `0`, yaw-excess excess max `0`; use packed envelope mode, not scalar fallback, for the restored loss.
13. When judging this self-adversarial cycle, use GT-diff as the fuel/acceptance metric. Use skating/slide-excess metrics as diagnostics unless the recipe is explicitly changed to optimize them.
14. Ignore both walk micro envelope-resume checkpoints as candidates. The first used the wrong calibration target; the corrected full-K32 AE-prior calibrated run `20260521_walkF_env010_aeprior_fullk32_from_bestk32` is a negative result: training total improved, but GT p95 worsened to `2.342181` and trainable source-excess p95 worsened to `5.6592`.
15. Full-dataset scratch AE calibration produced carried envelope weights from all valid starts: slide-excess `0.00966197`, yaw-excess `0.155877` with `--yaw-excess-scale-radps 1.0`. Calibration checkpoint is `20260521_full_simpleAE_scratch_k04_to_k32\checkpoints\checkpoint_best_k32.pt`.
16. Idle random-init AE-only sanity check passed: from random full-dataset poses, the idle-tuned controller reduces mean joint RMSE to idle from `0.163028` to `0.0512103` over 32 steps, better than the full-dataset scratch checkpoint.
17. The idle-specific "both feet pinned" slide-excess rule was removed. Later, the planted-foot selector changed again: slide-excess and yaw-excess now use the foot with the lowest custom foot/toe collider point. The earlier idle slide-excess run is mixed/negative historical context only.
18. Envelope correction: foot-slide and yaw now both use the upper nearest same-situation GT bound. Do not use the temporary lower-bound foot-slide idea; it was stricter than an envelope and made idle look like it should have a near-zero threshold.
19. New idle exception: idle envelope bounds are hard-coded to linear `0` and angular `0`; this is an intentional planted-idle rule, not a GT-jitter allowance.
20. Planted-foot selector changed: the planted foot is now the foot with the lowest custom foot/toe collider point. Linear / foot-slide and angular / yaw both use that same planted foot for envelope building, training loss, viewer diagnostics, and slide/yaw analysis.
21. Walk-forward supervised scratch baseline is `20260521_093649_walkF_supervised_cachedgt_scratch`; it trained from scratch to K32 with best supervised loss `0.073163`. Fixed GT rotation matrices are cached on clip load and fetched during supervised loss.
22. Idle recovery one-planted lab success checkpoint is `20260521_143000_idle_recovery_k64_mixed_polish_v18\checkpoints\checkpoint_best.pt`. It uses K64 plus smooth speed-product and contact-aware speed-product losses. It passes the official fixed+random validation; an extra random seed 5678 is a tiny close miss, so treat it as a strong contained solution, not exhaustive proof.
