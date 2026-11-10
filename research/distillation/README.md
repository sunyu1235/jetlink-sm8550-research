## Latest completed result

The trained scaled-SiLU single-block student passes the unchanged full160 synthetic replay and a held-out reconstructed real-road320-frame replay (also passes last160 after warmup). The road data was not used for fitting/selection. Its standalone block compiles for HTPv73 and has host-functional numeric evidence. No mobile latency or full63-block distillation is claimed. See long-student-results.md and long-silu-student/acceptance.json; earlier failures and intermediate status below are retained as history.

## Latest update: scaled-SiLU single-block student passes160 synthetic frames

See long-student-results.md. Both earlier ReLU candidates remain rejected. A new trained scaled-SiLU MLP passes the full original160-frame checker. This is NOT a fully distilled model or target-latency result: only one block is replaced,62 other GELUs remain. Real-road replay and HTP execution remain unverified.

# JetLink Light Speed: offline block-distillation experiment

Status update: real teacher downloaded and hash-verified; one real transformer MLP has been distilled using separate synthetic sequences and reinserted into a complete CPU model. The same-width student passed an initial FOUR-frame screen but FAILED the expanded original acceptance sequence at a 26-frame prefix; the candidate is rejected. **Not a validated driving model**, not 160-frame acceptance, not HTP speed evidence. No device accesses or workloads were run. Existing evidence is unchanged. Earlier setup notes below are superseded by this update where stated.

## What the repository actually supports

Inspected snapshot commit `8f2bc5d31e06decaaa90190486e08c468e781c1f` in sunyu1235/jetlink-sm8550-research, especially:

- `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/dot-review-request.txt`: 52.4086 ms accepted 160-frame replay versus 32.475 ms inaccurate old path; original weights/captures absent.
- `.../prepare_last_query.py` and `.../last-query-partition-manifest.json`: three separate policy transformer branches, temporal tokens 288, residual width 512, MLP expansion 1024. Final attention queries already reduced to one; K/V retained. This is not a new optimization opportunity.
- `vendor/jetlink/scripts/verify_parity.py`: original per-head and per-column tests, including quiet-column absolute errors and recurrent feedback. Synthetic road-like frames are explicitly stand-ins, not a real driving distribution.
- `vendor/jetlink`: inference/export/runtime integration. Search did not find a reusable torch training loop, optimizer or distillation recipe. The ONNX export is not a trainable source checkpoint.

All widths and branch topology above come from supplied manifests/scripts, not from opening unavailable original ONNX weights. The raw teacher is publicly linked by the repository, SHA256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`; this branch did not download it. It is ~792 MB, so access alone does not supply training data or a differentiable implementation.

## Focused hypothesis

Replace selected GELU-containing MLP blocks with an HTP-friendly `Gemm → Relu → Gemm` student, initially at unchanged expansion width. Fit against the accurate block's **post-second-linear, pre-residual** output. Only after establishing held-out error should width drop, e.g. policy 1024→768→512. This seeks both removal of CPU activation boundaries and reduced dense compute, not merely fewer parameters.

This differs from the already-failed `Min(Gelu(x),Relu(x))` rewrite: weights are explicitly recalibrated using teacher block targets. It also avoids assuming the native HTP GELU is an adequate teacher. Vision and policy must both be treated: prior evidence shows independent errors in each.

Why a cheap first experiment: the first affine projection can be reused and the second affine fitted with regularized least squares, so a single block can be rejected without reconstructing the whole training system. This constrained student may be unable to preserve accuracy; failure is valuable. If it fails, do not sweep all 63 activations blindly. End-to-end fine-tuning, a higher-capacity student, or the custom-op route would be a separate escalation.

## Implemented artifact

`fit_relu_student.py` selects channels deterministically by normalized feature/target correlation, fits a normalized ridge regression, folds normalization into the second affine weights, and exports FP16 weight storage plus a JSON diagnostic report. It refuses nonfinite arrays, invalid dimensions, nonpositive regularization, overwrites, and FP16 overflow. It does not modify any model.

Input NPZ keys:
- `x_train[N,D]`, `y_train[N,O]`, `x_validation[M,D]`, `y_validation[M,O]`;
- `w1[D,H]`, `b1[H]`, in mathematical right-multiply convention, regardless of ONNX Gemm transB storage;
- inputs must be the actual MLP input, targets the accurate teacher block output before adding residual; no old inaccurate NPU targets;
- entire routes must be separated before token sampling. Do not randomly split adjacent frames or tokens from one rollout.

Example, once real paired captures exist:

```
OPENBLAS_NUM_THREADS=1 python fit_relu_student.py block_pairs.npz \
  --width 1024 --ridge 0.0001 --route-split-confirmed --output student_1024.npz
```

The confirmation flag records an operator assertion, not an automatic leakage check. Fit each width to training routes, choose settings on validation routes, and reserve a separate untouched test set for final comparison. Do not train on the existing 160-frame acceptance set.

Tests: `python -m unittest -v` passes four tests (recover representable synthetic teacher, deterministic compression, constant-input stability, invalid-input rejection). `synthetic-smoke.json` records a CPU fit with actual policy-like dimensions 512→1024→512 and 4096 random training tokens. It took ~0.445 s on this executor with one BLAS thread. Its random-GELU teacher validation RMSE ~0.228 is **not** an accuracy success. This only demonstrates the algorithm is executable. FP16-weight/FP64-compute diagnostics are explicitly not FP16 backend validation.

## Data and compute required next

1. Materialize and hash-check the original ONNX; identify exact MLP input/output tensors and Gemm weight layouts. Reproduce accurate CPU reference and inspect whether export-to-autodiff conversion is needed. Conversion success must be numerically verified, not assumed.
2. Use owner-supplied, authorized, representative *offline* warped camera sequences plus scalar/context inputs, resets and recurrence. Include turns, lane changes, merges, leads/braking, camera exposures, curves, road markings and quiet straight stretches. Preserve route/session identity and timestamps. Existing final-output `.npy` files cannot reconstruct MLP inputs and are not a training corpus.
3. For an initial go/no-go screen, gather roughly 8k–32k token rows per selected block from many disjoint sequences, not one contiguous scene. This is an engineering pilot size, not a sufficient driving-data claim. At D=O=512, X/Y float32 alone take ~32–128 MiB per block. Capturing every layer/token of every frame would balloon storage; stream or subsample within training routes, retaining independent validation routes.
4. CPU ridge fitting can start immediately once captures exist. Teacher replay and accurate intermediate captures likely dominate pilot cost. No GPU reservation or paid compute is necessary for the implemented initializer. Full sequence-aware joint training needs trainable graph reconstruction and memory profiling before a GPU-hour estimate is credible.
5. Keep residual/attention topology, I/O names, scalars, state shape, and head layout unchanged. Distill features and all named outputs, with training-only scale normalization per head/column so large outputs do not drown quiet plan columns. Include hidden-state/temporal consistency. For joint training, unroll the student's own states, with teacher targets on matching real input sequences; teacher-forced one-frame loss alone conceals drift. Independent recurrent states are mandatory in evaluation.

## Acceptance and stop rules

- First benchmark a single candidate block on the actual QNN 2.42 / v73 stack, with matched input, after offline fitting. Verify placement/fusion: ReLU support in principle does not prove a one-partition result.
- Replace one block at a time, run complete unchanged upstream parity on independent recurrent rollouts. Local RMSE is only a screening metric, not acceptance. A block that breaks quiet columns or hidden-state drift is rejected.
- Passing the original saved/synthetic 160-frame suite is necessary regression evidence, not road safety or general accuracy certification. Evaluate additional held-out recorded sequences; all learned candidates remain offline research.
- Recompile the whole candidate and measure complete single-frame native Run p50/p99 and end-to-end pipeline separately under matching thermal/warmup conditions. Do not compare training CPU timing with phone latency or sum isolated kernels as a whole-model result.
- The 52.4086→32 ms goal requires ~20.41 ms (38.9%) savings. A 25% narrower MLP cannot be converted into a 25% whole-model latency promise. Partition-boundary removal and actual target profiling determine whether the goal is plausible.
- If unchanged-width ReLU students already fail the unchanged full-model checker, do not proceed to narrower students under a claim of preserved parity. If reaching the target requires accepting changed behavior, that is a new model-validation goal, not a successful implementation-only fix.

## Exact current blockers

No paired real block inputs/teacher outputs, no usable training corpus, no trainable checkpoint/source, and no local QNN/HTP target are supplied to this branch. These are missing prerequisites, not denied actions. The only tests run are the four initializer unit tests and synthetic CPU shape smoke test. No actual model training, ONNX export, full-model parity, HTP compile, or latency benchmark has been performed for this student.

## Real-model experiment update

Original teacher now available at `/workspace/shared/jetlink-models/light-speed.onnx`, verified size 792305602 bytes and exact published SHA256. `teacher-graph.json` is the inspected original graph, not inferred architecture. `capture_teacher_block.py` instruments the actual off-policy temporal-transformer layer-0 MLP, input `layer_norm_49`, output `linear_99`, activation `gelu_41`. Original custom `org.tinygrad::Contiguous` is explicitly lowered to `Identity` as a storage-layout hint. This is logged, not hidden.

`real-block/teacher-block.onnx` preserves FP16 teacher weights. Full-model CPU replay succeeded for 8 new-seed training frames and 4 separately reset validation frames; all outputs were finite. `real-block/sequences.npz` preserves frame shapes; `pairs.npz` flattens tokens for regression. These are synthetic generator sequences (seeds713/991), not real route captures. The original 160-frame acceptance sequence was not used for training.

`run_real_block_pilot.py` fits widths1024/768/512 using actual teacher activations. Held-out local errors with FP16 weight storage and FP64 compute:
- 1024: RMSE0.015704, max absolute1.07987.
- 768: RMSE0.028529, max absolute1.34920.
- 512: RMSE0.066229, max absolute4.25252.

`evaluate_student.py` reinserts the 1024-wide ReLU student into the complete teacher graph, runs both models with independent recurrent states, and saves 4 complete outputs. The unchanged upstream checker passes per-frame and pooled tests (`real-block/student-1024-parity.log`); lead max error0.5625, plan max error0.3047. These errors are nonzero and a four-frame pass is not acceptance. No meaningful device speed has been measured.

Public data checks:
- https://huggingface.co/datasets/commaai/commaCarSegments/blob/main/README.md documents CAN/rlog data only, not paired camera video. Its large hour count must not be claimed as available dual-camera training input.
- https://github.com/commaai/comma2k19/blob/master/README.md describes an old single-road-camera EON highway dataset; it does not directly furnish the model's synchronized dual-camera inputs.
- Environment catalog lists only this cloud computer. No NVIDIA device was found and no SM8550 target is registered. CPU teacher execution is viable at roughly1.3–1.9 seconds/frame here, unrelated to target-device latency.

## Expanded validation: candidate REJECTED

The full160 replay was started but stopped after26 frames once the unchanged checker already failed; further frames cannot repair per-frame failures. The final26-frame prefix log is `real-block/student-prefix-final-parity.log`. Failed heads: plan (worst single-frame correlation0.973172), lead(0.998384), desire_state pooled and lead_prob pooled. The original acceptance sequences were never used to train this student.

This falsifies the idea that the implemented same-width fixed-projection ReLU ridge fit on eight synthetic frames is adequate. It does not prove all distillation impossible. Narrower students are not advanced to latency claims. More representative data and joint/sequence-aware optimization are needed before this can be a competitive model candidate. No HTP speed testing is available, and this student is not accurate enough to warrant a claimed speed/accuracy success.

Causal control completed: reran the current unmodified teacher on the same26 inputs, which passes against archived references. Student versus this same-runtime teacher still fails the same four heads (plan0.973160, lead0.998393). Therefore the rejection is not explained by ORT-version mismatch. Logs: `real-block/teacher-prefix26-parity.log` and `real-block/student-vs-current-teacher26-parity.log`.
