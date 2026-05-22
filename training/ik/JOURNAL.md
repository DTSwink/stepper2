# IK Journal

## 2026-05-21

- Moved the IK-marker experiment into `training/ik` so the main locomotion trainer is clean again.
- Restored the tracked main training/AE/visualizer files to their original versions.
- Removed old generated IK/supervised experiment outputs from `training/runs`.
- New naming rule: runs and checkpoint files created by the contained IK trainer use `YYYYMMDD_HHMMSS_ik_<label>`.
- New policy for the contained trainer: IK markers always on, periodic clips cyclic, supervised rollout rows on the fast GPU-resident path, batch rows sampled independently.
- Performance concern: the generic rollout trainer was slow for a single motion because it still exercised the full generic K-rollout geometry path. Added `perf_audit.py` to measure the supervised rollout path before any real training.
- Audit result on `M_Neutral_Walk_Loop_F.npz` / RTX 4060 Laptop GPU:
  - One-step supervised optimizer step: mean `1.22 ms` over 5 measured iterations after warmup.
  - Generic K32 rollout optimizer step: mean `5083.87 ms` over 3 measured iterations after warmup.
  - Conclusion: single-motion supervised IK runs should start from the supervised rollout path. The generic K-rollout path is roughly 4000x slower for this sanity check and should only be used after a specific rollout-loss need is proven.
- Course correction: the fast GPU-resident row layout is not a one-clip trick. The IK path now treats it as mandatory for full-dataset work too.
  - `train_ae_prior.py` forces IK markers, random per-row rollout starts, no legacy contact-physics fallback, and dense multi-clip tensors.
  - Its row sampler now samples independent starts for ordinary real batches, not only synthetic/mixed-cohort batches.
  - `train.py` and `perf_audit.py` accept periodic/nonperiodic dataset folders so the next full-dataset run stays on the same row layout from the first step.
- Updated performance audit:
  - Single Walk_F, batch 119, K8 rollout state/backward step: mean `357.47 ms`.
  - 15-clip periodic folder, batch 256, K8 rollout state/backward step: mean `283.95 ms`.
  - 15-clip periodic folder, batch 256, one-step supervised step: mean `2.28 ms`.
  - The remaining rollout cost is FK/IK plus backprop through rollout state, not clip I/O or per-clip Python sampling.

## 2026-05-21 Follow-up

- Dataset resolution is now fail-fast:
  - Requested empty NPZ folders raise instead of falling back to Walk_F.
  - Missing paths and non-`.npz` files raise.
  - Mixed skeletons raise if either bone names or parent topology differs.
- Removed the `training_loop`/row-sampling switch from the IK AE-prior entrypoint; it now always builds row-independent rollout batches.
- Performance audit on full local dataset (`15` periodic + `214` nonperiodic clips, batch `512`):
  - Before position-only FK: K8 rollout state/backward step `586.34 ms`.
  - After position-only FK: K8 `207.82 ms`, K32 `767.82 ms`.
  - One-step supervised step stayed around `1.4 ms`.
- Remaining bottleneck profile after the speedup:
  - Backward through rollout state: about `103 ms` at K8.
  - Position-only IK FK/canon: about `52 ms` at K8.
  - Input construction: about `22 ms` at K8.
  - Model/output conversion: about `12 ms` at K8.

## 2026-05-21 Uniform Rollout Update

- Removed the separate one-step row-table path from `train.py` and `perf_audit.py`.
- The supervised trainer now uses one sampled rollout loop for every rollout length; `ROLLOUT_K = 1` is just the shortest case.
- First version kept non-cyclic row repair inside the rollout loop; the next pass replaced it with strict full-window sampling.
- `perf_audit.py` now reports the same `supervised_rollout_optimizer_step` path for whatever rollout length is requested.

## 2026-05-21 Rollout Hot Path Pass

- Kept the existing per-clip cyclic policy; did not force all clips to non-cyclic.
- Replaced mid-rollout row repair with strict full-window start sampling. Non-cyclic samples now start only where the whole requested rollout fits; cyclic samples still wrap.
- The trainer now fails loudly if any clip cannot provide a full requested rollout window instead of silently dropping that clip.
- Precomputed eligible clips plus strict max starts, then sample clips uniformly and starts uniformly within each clip. This keeps the previous random-clip-per-row behavior instead of weighting longer clips more heavily.
- Cached GT target-output tensors in `ClipStore` so supervised rollout steps gather the target vector directly.
- Moved final-step pose/FK work behind the final-step check; the last supervised step only needs the output loss.
- Requested fused AdamW on CUDA with a plain AdamW fallback.
- Controlled before/after comparator on full local dataset, batch `128`:
  - K=1: old `43.39 ms`, new `28.72 ms`.
  - K=2: old `200.33 ms`, new `172.27 ms`.
  - K=8: old `1099.55 ms`, new `1065.24 ms`.
- `perf_audit.py` after the pass:
  - K=1: `25.28 ms`.
  - K=2: `151.67 ms`.
  - K=8: `914.69 ms`.
- Safety checks passed for K=1/K=2/K=8/K=32:
  - Every sampled non-cyclic row has enough future frames for the requested rollout and input horizon.
  - Cached target outputs exactly match rebuilding `pose_target_output(get_pose(...))`.

## 2026-05-21 Walk_F Supervised Run

- Ran contained supervised rollout training on `M_Neutral_Walk_Loop_F.npz`.
- Runtime: `6000` steps in `165.6 s`.
- Best supervised rollout mean joint error: `0.003542 m`; final reported mean: `0.005080 m`.
- Checkpoints:
  - `training/runs/20260521_204504_ik_walkF_supervised/checkpoints/20260521_204504_ik_walkF_supervised_best.pt`
  - `training/runs/20260521_204504_ik_walkF_supervised/checkpoints/20260521_204504_ik_walkF_supervised_last.pt`
- Added TensorBoard scalar logging to `train.py` and backfilled this run with `25` scalar points from the text log.

## 2026-05-21 Mixed-K Walk_F Probe

- Added hardcoded final-stage mixed rollout support to `train.py`.
- Current max rollout is `K=32`; mixed rollout activates only when the requested rollout reaches that max.
- Distribution is exact fractal-by-remainder per batch:
  - For Walk_F batch `119`: `59` rows at `K=32`, `30` at `K=16`, `15` at `K=8`, `7` at `K=4`, `4` at `K=2`, `4` at `K=1`.
- Valid-window sampling is per row:
  - Each row samples its own effective K.
  - Each effective K has its own start pool.
  - The sampled start is checked against that row's K and clip, not a global max-start table.
- Walk_F validity probe passed with the exact distribution above.
- Walk_F mixed-K performance audit, batch `119`, K max `32`: `4755.33 ms` per optimizer step.
- Fixed K=32 comparator was not better in a controlled local probe (`4558.64 ms` fixed vs `4885.35 ms` mixed), so the requested mixed distribution still effectively costs max-K time on Walk_F.
- Did not launch a full `6000` step mixed-K Walk_F run because the measured speed implies roughly `8` hours.

## 2026-05-21 Mixed-K Performance Fix

- Root cause for the IK slowdown:
  - The supervised IK rollout was still calling the IK/FK reconstruction path every continued step just to produce `pred_canon`.
  - In the IK representation, the next training input uses predicted pelvis/core/marker values directly; `canon_pos` is not used by `store_build_input` when `ik_marker_pos` exists.
  - After skipping that redundant solve for IK rollout state, Walk_F mixed K=32 batch `119` improved from `4408.06 ms` to `1550.75 ms`.
- Second hot spot:
  - `store_build_input` recomputed deterministic root/future trajectory features with several `root_state` calls on every rollout step.
  - Added a `ClipStore.input_root_features` cache and changed input construction to gather those features by `(clip_id, cur_idx)`.
  - Cache equivalence check on Walk_F wraparound frames: max absolute difference from the old formula was `3.78e-6`.
- Final performance:
  - Walk_F mixed K=32, batch `64`: `887.64 ms` per optimizer step.
  - Walk_F mixed K=32, capped batch `119`: `838.95 ms` per optimizer step.
  - Full local dataset (`15` periodic + `214` nonperiodic clips), mixed K=32, batch `128`: `912.73 ms` per optimizer step.
  - Old non-IK journal comparator was `886 ms` for mini mixed K=32 batch `64`, so the contained IK supervised path is now in the same ballpark.
- Safety reflection:
  - The IK rollout-state shortcut does not change the supervised target or sampled rows; it removes FK work whose `canon_pos` output is unused by IK inputs.
  - Validation still uses FK/IK to compute joint-space error, so evaluation remains physically grounded.
  - The root feature cache changes repeated deterministic math into a gather. The measured difference is float noise, not a semantic change.

## 2026-05-21 Lean IK Harness Pass

- Removed contacts from the active IK model surface:
  - `make_batch_dims()` no longer allocates contact inputs or contact outputs.
  - `pose_target_output()` and `ClipStore.target_output` no longer append contacts.
  - `output_to_pose()` no longer parses contact logits.
  - `MotionClip`, `ClipStore`, and `get_pose_from_clip()` no longer expose contact state.
  - Contact-state flags were removed from the IK core config/CLI; contact losses now fail fast if someone tries to re-enable them.
  - `train.py` and `perf_audit.py` have no contact-state flags.
- Simplified the supervised rollout hot path:
  - Training now rolls forward compact IK tensors directly: target vector, pelvis position, and IK marker positions.
  - The full pose dictionary path remains for validation/FK error only.
  - Removed per-step effective-K stat sync; the mixed-K distribution is fixed by batch size and logged from the schedule.
- Removed unnecessary active harness flags:
  - No device flag; CUDA is used when available.
  - No batch-size flag; training and audit both use `BATCH_SIZE = 4096`.
  - No rollout-K flag; training and audit both use `ROLLOUT_K = 32`.
  - No mixed-rollout toggle; mixed rollout is the hardcoded policy at max K.
- Trusted post-clean performance with other GPU work stopped:
  - Walk_F, mixed K=32, capped batch `119`: `131.63 ms` per optimizer step.
  - Full local dataset (`15` periodic + `214` nonperiodic clips), mixed K=32, batch `4096`: `141.27 ms` per optimizer step.
  - This is roughly `6.7x` faster than the previous `887.64 ms` Walk_F same-harness comparator, and far past the requested additional 2x cut.

## 2026-05-21 Separate Restore Cost Audit

- Benchmarked one-at-a-time reversions on the full local dataset (`229` clips), mixed K=32, batch `4096`, CUDA graph static masked step, fp16 model forward.
- Current baseline for this audit: width `128`, no LayerNorm, detached raw predicted rollout state.
- Results:
  - Current baseline: `16.21 ms` per optimizer step.
  - Restore point 2, width `512`: `36.18 ms` (`2.23x` baseline).
  - Restore point 3, LayerNorm: `23.70 ms` (`1.46x` baseline).
  - Restore point 4, full BPTT through predicted rollout state: `20.62 ms` (`1.27x` baseline).
  - Restore point 5, clean/normalize 6D predicted rollout state before feedback: `21.65 ms` (`1.34x` baseline).
- Interpretation:
  - Width is the expensive one.
  - LayerNorm is moderate cost.
  - Full BPTT and 6D cleaning are surprisingly cheap under the captured static graph compared with the earlier eager path.
  - If training quality needs it, points 4 and 5 are affordable to restore first.

## 2026-05-21 Restore FK-Like Training Assumptions

- Restored all four training-affecting items from the separate audit:
  - `HIDDEN_DIM = 512`.
  - `LayerNorm` after each hidden linear layer.
  - Full BPTT through predicted rollout state.
  - Clean/normalize predicted 6D rotations before feeding rollout state forward.
- Kept the pure performance shell:
  - CUDA graph static masked rollout step.
  - Fixed full-dataset batch size.
  - Mixed K=32 fractal distribution.
  - No contact input/output surface.
  - fp16 autocast for model forward.
- File-backed performance after restore:
  - Walk_F, mixed K=32, capped batch `119`: `24.87 ms` per optimizer step.
  - Full local dataset (`15` periodic + `214` nonperiodic clips), mixed K=32, batch `4096`: `81.38 ms` per optimizer step.
- This is slower than the stripped `16.21 ms` full-dataset benchmark, as expected, but keeps the training behavior closer to prior FK experiments while retaining the large CUDA-graph performance gain.

## 2026-05-21 Walk_F Full Mixed-K Run

- Ran `train.py` on `M_Neutral_Walk_Loop_F.npz` with restored FK-like assumptions and current mixed K=32 harness.
- Runtime: `6000` steps in `146.8 s`.
- Throughput including validation/checkpoint overhead: `40.9` optimizer steps/s, about `24.5 ms` per step.
- Mixed-K distribution: mean effective K `21.24`, max K `32`.
- Best rollout mean joint error: `0.015853 m` at step `3500`.
- Final step `6000`: rollout mean `0.020431 m`, rollout max `0.048330 m`.
- Checkpoints:
  - `training/runs/20260521_222539_ik_walkF_full_K32_mix/checkpoints/20260521_222539_ik_walkF_full_K32_mix_best.pt`
  - `training/runs/20260521_222539_ik_walkF_full_K32_mix/checkpoints/20260521_222539_ik_walkF_full_K32_mix_last.pt`

## 2026-05-21 Payload42 IK Representation Fix

- Replaced endpoint-only IK state with a `42`-dim limb payload:
  - hands: root-local endpoint position, root-local endpoint rotation6, elbow pole scalar.
  - feet: root-local endpoint position, root-local endpoint rotation6, knee pole scalar, toe hinge scalar.
- The encoder/decoder now preserves hand and foot endpoint rotations explicitly instead of rebuilding them from endpoint direction.
- Added pole-based two-bone decode for elbows/knees and a one-axis toe hinge decode.
- Added per-clip calibration for limb local pole axes and toe hinge axes, selected from the source clip so full-dataset row mixing does not silently reuse clip 0's axes.
- Added descendant recomputation after IK overrides so fingers/twist children inherit corrected endpoint/mid rotations instead of stale pre-IK transforms.
- Added `roundtrip_check.py`.
  - `M_Neutral_Walk_Loop_F.npz`: hand/foot endpoint position max below `5e-8 m`, endpoint rotation max `0.0816 deg`; toe position max below `1e-6 m`, toe rotation max `5.31 deg`.
  - `M_Neutral_Walk_Arc_F_Wide_R.npz`: hand/foot endpoint position max below `8e-8 m`, endpoint rotation max `0.0816 deg`; toe position max below `1e-6 m`, toe rotation max `1.60 deg`.
- Generated decoder sanity screenshots with all pole/toe floats forced to `0` and `1`:
  - `_tmp/ik_payload_float_0.png`
  - `_tmp/ik_payload_float_1.png`
- Training audit:
  - Starting the new state directly at mixed K=32 learned poorly (`6000` steps, best K32 rollout mean `0.251924 m`), even though smoke tests showed valid gradients.
  - K=1 overfit was healthy (`3000` quick-test steps reached about `0.007 m` one-step rollout mean), so the issue was curriculum hardness, not representation wiring.
- Restored the staged working recipe in the contained harness:
  - hardcoded K schedule `1, 2, 4, 8, 16, 32`;
  - hardcoded stage steps `500, 500, 750, 1000, 1250, 3000`;
  - fractal mixed K only at the final K=32 stage.
- Walk_F staged Payload42 run:
  - Runtime: `7000` steps in `198.7 s`.
  - Best final K=32 mixed rollout mean: `0.012117 m` at step `6500`.
  - Final step `7000`: rollout mean `0.013027 m`, rollout max `0.033349 m`.
  - Checkpoints:
    - `training/runs/20260521_233943_ik_walkF_payload42_staged/checkpoints/20260521_233943_ik_walkF_payload42_staged_best.pt`
    - `training/runs/20260521_233943_ik_walkF_payload42_staged/checkpoints/20260521_233943_ik_walkF_payload42_staged_last.pt`

## 2026-05-22 K1 Reconstruction Fix

- User-visible issue: the staged Payload42 controller learned the walk shape but did not reconstruct even one step tightly enough in the HTML viewer.
- Diagnosis:
  - Long K=1 run with the previous absolute-output trainer plateaued around `0.00176 m` best mean joint error.
  - GT IK target payload still decoded to near-zero position error, so the representation was not the blocker.
  - Deterministic residual one-step fitting reached effectively zero, while fp16 autocast plateaued at millimeter scale.
- Trainer changes:
  - IK supervised controller now predicts residuals from the current pose (`predict_residual=True`, zero-initialized output).
  - Supervised loss now applies the same residual convention as rollout/evaluation.
  - AMP is disabled for this IK supervised path because residual deltas need fp32 precision.
  - Tiny datasets now use a true full batch when the requested batch covers the whole start pool, instead of sampling rows with replacement.
  - Added hardcoded per-stage LR decays and graph recapture at decay points.
- Verification run:
  - Run: `20260522_011428_ik_walkF_k1_residual_fixed`.
  - Best checkpoint: `training/runs/20260522_011428_ik_walkF_k1_residual_fixed/checkpoints/20260522_011428_ik_walkF_k1_residual_fixed_best.pt`.
  - One-step viewer metrics: mean `0.000011 m`, max frame-mean `0.000048 m`.
  - Autoregressive viewer metrics over the whole walk: mean `0.000262 m`, max `0.000660 m`.
  - Unfloored one-step rotation angle: mean `0.000354 deg`, max `0.0685 deg`.

## 2026-05-22 Simple Controller IO Autoencoder

- Added `train_simple_autoencoder.py` as a clean replacement experiment for the clogged AE files.
- Feature definition is intentionally minimal:
  - exact controller input (`current pose`, `previous pose`, pose delta, root/future root features);
  - exact supervised controller target output (`next pose output`);
  - AE target is the same concatenated vector.
- No compatibility head, denoising branch, contact logic, windowing, model-aware loop, or foot-slide losses.
- Walk_F sanity run:
  - Run: `20260522_013146_ik_simple_ae_walkF_probe`.
  - Feature rows: `119`; feature dim: `413`; latent dim: `32`.
  - Best validation reconstruction at step `750`: `0.004462` normalized MSE.
  - Synthetic diagnostic at best step:
    - slight noise: `1.51x` clean;
    - bad statue/no-next-motion: `5.87x` clean;
    - bad shuffled output: `78.46x` clean;
    - random noise: `280.45x` clean.
- Interpretation: the plain bottleneck AE does reconstruct clean rows and its old-style bad synthetic metric separates bad rows from clean rows on the mini Walk_F test.

## 2026-05-22 Simple AE Controller Rewrite

- Rewrote `train_simple_ae_controller.py` so the pure-AE controller trainer no longer imports rollout/storage helpers from the older training files.
- The file now owns its small local clip store, root-feature cache, per-row valid start pools, fractal mixed-K sampler, rollout loss, validation, optimizer setup, and checkpoint policy.
- Kept `ik_core.py` as the shared IK representation/math layer and `train_simple_autoencoder.py` as the AE definition/checkpoint format.
- Checkpoint policy now writes `init` before training, `latest` at every log point, `best` on any validation improvement, and `last` at clean completion.
- Sanity check on Walk_F:
  - local controller input vs `ik_core.build_input` max abs delta: `5.96e-08`;
  - one small K=2 pure-AE loss/validation pass completed;
  - one K=1 FK diagnostic pass completed.

## 2026-05-22 Simple AE Controller Speed Audit

- Audited `train_simple_ae_controller.py` on the real full local split:
  - periodic: `ue5/animations_omni_only_full/npz_final` (`15` clips);
  - nonperiodic: `ue5/animations_transitions_only_full_trimmed/npz_final` (`211` clips);
  - total frames: `12616`, controller input dim `302`, output dim `111`.
- Baseline measured full-dataset training step times:
  - K1: `32.5 ms`;
  - K8: `103.9 ms`;
  - K32 mixed: `696.2 ms`.
- Critical diagnostic issue:
  - FK rollout diagnostic was not training-critical and was extremely slow on the 226-clip set.
  - K8 diagnostic with 256 validation rows took `128.0 s`; K32 logs would stall training badly.
  - Default pure-AE controller training now skips this FK diagnostic. AE validation remains on.
- Hot-path optimization:
  - Replaced the rollout-state projection with a local fast 6D cleaner in the pure-AE controller trainer.
  - On normal nondegenerate 6D rows, it matched the old cleaner exactly in the audit (`max delta 0.0` on real zero-init controller outputs).
- After the local optimization and FK diagnostic removal:
  - K1: `30.1 ms`;
  - K8: `77.1 ms`;
  - K32 mixed: `249.7 ms`;
  - K32 AE validation: `0.069 s`.
- Tested `torch.compile` on the controller and AE:
  - first compile/warmup included a `47 s` compile hit;
  - measured K32 mixed step was slightly slower (`440.9 ms` vs `414.7 ms` in that paired run);
  - not adopted.

## 2026-05-22 Walk_F Pure-AE Controller Speed Fix

- Problem run:
  - `20260522_023223_ik_walkF_simple_ae_controller_fast`.
  - Long staged run reached K32 mixed at step `8001` after `798.2 s`.
  - At step `8500`, K32 mixed `ae_loss=0.620046` after `1185.2 s`.
  - It was stopped; latest checkpoint is preserved.
- Diagnosis:
  - Tiny Walk_F batch has only `119` valid rows.
  - Eager K32 mixed pure-AE rollout stayed around `0.7 s/step`; larger replacement-sampled batches did not help.
  - CUDA graph replay on a static masked K32 pure-AE step reduced this to about `118 ms/step`.
  - The old LR reset to `3e-4` at each harder K was also wasteful; K32 often improved, then got pushed upward.
- Compressed schedule experiments:
  - `(K1 80, K2 80, K32 420)` with K32 LR `1e-5`: K32 probe `0.0689` in `19.9 s`.
  - `(K1 80, K2 80, K32 420)` with K32 LR `3e-5`: K32 probe `0.1243` in `19.5 s`.
  - `(K1 80, K2 80, K8 160, K32 360)` with K32 LR `3e-5`: K32 probe `0.3284` in `20.6 s`.
  - `(K1 80, K2 80, K8 160, K16 220, K32 320)` with K16 LR `5e-5`, K32 LR `2e-5`: K32 probe `0.2308` at `19.9 s`; later `0.3796` at `25.7 s`.
- Adopted local trainer changes:
  - CUDA graph static masked stepper for pure-AE controller training on CUDA.
  - Removed fake validation/test-set loss logging and fake global best checkpointing.
  - Checkpoints now use `init`, `latest`, `stage_K*`, and `last`.
  - Walk_F schedule is now `K=(1,2,8,16,32)` with steps `(80,80,160,220,240)` and LR `{1:1e-4, 2:1e-4, 8:1e-4, 16:5e-5, 32:2e-5}`.
- Verification run:
  - Run: `20260522_030813_ik_walkF_simple_ae_controller_graph_quick`.
  - Completed in `15.4 s`.
  - Final K32 mixed `loss/train_ae=0.20057`.
  - Final checkpoint: `training/runs/20260522_030813_ik_walkF_simple_ae_controller_graph_quick/checkpoints/20260522_030813_ik_walkF_simple_ae_controller_graph_quick_last.pt`.
- Carry-over caveat:
  - CUDA graph static masking is a general performance fix.
  - The compressed schedule/LR is proven only on Walk_F and may need re-expansion or retuning for the full dataset.

## 2026-05-22 Pure-AE Repro And TensorBoard Fix

- Stopped the foot-harshness option sweep after visual suspicion that bad audit variants might be confused with the earlier simple-AE result.
- Reproduced the known pure simple-AE controller path with the original AE checkpoint:
  - original checkpoint: `20260522_041619_ik_walkF_ae_output_only_pure_last.pt`;
  - fresh repro run: `20260522_051508_ik_walkF_ae_output_only_pure_repro`;
  - fresh repro final `loss/train_ae=0.004774`;
  - IK viewer repro one-step joint error: start `0.000000`, avg `0.003234`, end `0.002549`;
  - IK viewer autoregressive joint error: start `0.000000`, avg `0.037866`, end `0.039689`.
- The generic `training/visualize_model.py` is not valid for these IK simple-AE checkpoints because it imports the non-IK trainer and tries to build the old `569 -> 155` model shape. Use `training/ik/visualize.py` or the model viewer IK path for these checkpoints.
- TensorBoard issue:
  - the server process was still watching an old static `logdir_spec` ending before the `04:16` pure-AE run, so newer runs could not appear;
  - `launch_tensorboard_latest.ps1` now reports the true `/data/runs` count;
  - `train_simple_ae_controller.py` and the foot-harshness audit trainer now refresh TensorBoard after creating a new event file, so future runs should not be invisible behind a stale TensorBoard process.

## 2026-05-22 Foot Harshness AE Audit

- Built a foot-pin metric that compares predicted foot slide/rotation against GT over the selected pinned interval, plus a separate teacher-forced one-step reconstruction metric to catch hover/drift failures without phase sensitivity.
- Rejected variants:
  - denoising corruption examples: one-step and slide degraded badly after rollout training;
  - output-only AE and score weighting/top-k scoring: often reduced rotation by freezing, but created hover/slide failures;
  - tighter/wider bottlenecks: did not beat the clean simple-AE baseline on slide.
- Best current variant is the foot-delta AE:
  - no contact labels and no extra temporal window beyond the existing controller row;
  - augments the AE row with the derived next-foot displacement in the current-root frame, then scores output reconstruction plus that derived foot-delta reconstruction.
- K16 checkpoint:
  - run `20260522_063420_ik_foot_audit_opt7_foot_delta_controller_k16`;
  - foot slide ratio `1.519` vs clean final `1.775` and clean K16 `1.729`;
  - one-step position `0.00339 m`, rotation `0.819 deg`, foot position `0.00255 m`;
  - viewer export: `training/runs/model_comparisons/foot_delta_ae_k16_063420.html`.
- Full K32 promotion:
  - run `20260522_063829_ik_foot_audit_opt7_foot_delta_controller_full`;
  - one-step improves to `0.00293 m`, `0.731 deg`, foot position `0.00215 m`;
  - foot slide ratio is worse than K16 at `1.713`, but still better than clean final;
  - viewer export: `training/runs/model_comparisons/foot_delta_ae_full_063829.html`.
- Current recommendation:
  - use K16 foot-delta as the anti-slide winner for inspection;
  - keep full K32 as a cleaner overall rollout candidate, but not the best slide candidate.

## 2026-05-22 Overnight Full-Dataset AE Envelope Setup

- Pushed the contained IK work before starting overnight changes:
  - `9913f59 Save IK AE training framework`;
  - `5c694d9 Add IK full AE envelope runner`;
  - `768770c Make IK full curriculum stall-only`.
- Rechecked the suspicious one-motion results with vanilla AE:
  - CircleL foot-delta AR joint error avg `0.2193`, upper-body speed ratio `0.36x`;
  - CircleL vanilla AR joint error avg `0.0087`, upper-body speed ratio `1.00x`;
  - TurnL vanilla/foot-delta were close, but CircleL makes vanilla the safer full-dataset reenactment baseline.
- Added `training/ik/excess_envelope.py`:
  - IK-local geometry envelope;
  - situation feature is `[yaw_delta/pi, bend_angle/pi, horizontal_foot_distance_xz_m]`;
  - horizontal foot distance uses world `X/Z` only, not vertical `Y`;
  - GT self-values are included in the bound before margin so GT envelope excess is exactly zero.
- Added `training/ik/train_full_ae_envelope.py`:
  - full-dataset vanilla AE baseline;
  - stall-only K curriculum for controller training;
  - envelope weight calibration from baseline means;
  - refined continuation from baseline;
  - final random-init continuation from refined baseline.
- Smoke checks:
  - full-dataset envelope rows: `10687`;
  - GT linear excess mean/p95/max: `0.0/0.0/0.0`;
  - GT angular excess mean/p95/max: `0.0/0.0/0.0`.
- First overnight attempt used a safety max-stage timer and advanced K8 by cap; stopped it because that was too close to a fixed stage length.
- Restarted the real overnight run with stall-only labels:
  - baseline label `full_vanilla_ae_controller_baseline_stall`;
  - refined label `full_vanilla_ae_controller_refined_stall`;
  - final label `full_vanilla_ae_controller_random_init_stall`;
  - stdout log `training/runs/overnight_ik_ae_envelope_stall_20260522_075753.out.log`;
  - stderr log `training/runs/overnight_ik_ae_envelope_stall_20260522_075753.err.log`.
