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
  - `train_ae_prior.py` forces IK markers, random per-row agents, no legacy contact-physics fallback, and dense multi-clip tensors.
  - Its agent sampler now samples independent starts for ordinary real batches, not only synthetic/mixed-cohort batches.
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
- Removed the `training_loop`/`agent_sampling` switch from the IK AE-prior entrypoint; it now always builds row-independent agents.
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
