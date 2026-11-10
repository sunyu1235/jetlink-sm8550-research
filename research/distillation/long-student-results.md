# Long-history student experiments

Training uses actual first off-policy transformer MLP teacher activations from160 independently reset synthetic training frames (seed713). Validation uses160 separate synthetic frames (seed1777). Neither original acceptance seed1 targets nor public-road inputs were used for fitting. This is one MLP, not a completely distilled model.

## ReLU student: rejected again

Both affine layers now trained, not just a fixed first projection. Width1024,2000 Adam steps, batch512, cosine learning rate1e-4→1e-5; ridge output initialization. Independent long-history validation block RMSE0.06688→0.05672 using FP16 stored weights/FP32 computation. Exported whole model still fails the original checker on32-frame acceptance prefix: plan correlation0.998802 on one frame. Thus more training/history alone did not make this architecture accurate enough. No target speed claims.

## Scaled-SiLU student: full160 numerical replay PASS

New hypothesis: x*sigmoid(1.7021484375*x) starts much closer to GELU than ReLU and uses ordinary Mul/Sigmoid operators, avoiding reliance on QNN's problematic native GELU. Both affine layers trained1000 steps, LR2e-5→2e-6, width1024; scale is exactly the FP16 constant exported. Independent validation block RMSE0.0061593, max absolute0.288849. These are block metrics, not driving-accuracy claims. Analytical activation gradient checked against finite differences, max discrepancy1.14e-10. ONNX block and full candidate pass structural checker.

Full160-frame unchanged original-checker replay PASSED with independently fed-back recurrent state. Pooled worst plan column corr0.999976 and lead column0.999989; max absolute plan0.1875, lead0.8125. Log: long-silu160-parity.log. This establishes only this single-block replacement on synthetic inputs; the other62 GELUs remain teacher operations, so neither 32ms nor elimination of all CPU boundaries is established. No HTP measurements yet.

Artifacts:
- train_long_student.py and long-student/training.json
- train_long_silu_student.py and long-silu-student/training.json
- build_student_model.py; long-silu-student/student-block.onnx
- real-block/student-long-silu.onnx (shares unchanged teacher external weight file)
- long-relu32-parity.log

## HTP compilation and host-function evidence

The independent trained SiLU block was converted with QNN2.42, compiled into model.so, and finalized for HTP SoC43/v73. Its15 tensors are FLOAT16; graph operations are FullyConnected/Reshape/ElementWiseNeuron/Eltwise_Binary, without CPU custom nodes. Context size2,166,784 bytes. This does not establish whole-model ORT partition placement or mobile latency.

Actual QNN HTP-host execution on16 selected long-validation block inputs is finite and closely matches ORT student outputs: aggregateRMSE0.00143075, max absolute0.125, every sample correlation at least0.9999998838. It is not bit-exact and is not device execution. Evidence produced by the quantization branch: `/workspace/shared/jetlink-exact-integration/student-htp-host16-comparison.json`.

## Public recorded-image validation pending

A legitimate official public dual-camera CI route was found and converted into320 aligned inputs under `/workspace/shared/jetlink-public-road/inputs320`. Geometry/source checked against the original Light Speed PR, not guessed from current dimensions. Reconstructed context is explicit; reset160-frame warmup precedes160-frame scoring. Five of six tinygrad warp sample comparisons are exact; one differs by1 in2 pixel values. Existing logged outputs are from openpilot0.10.2 and are not Light Speed labels: a fresh original teacher is run instead.

The first student process was system-killed with exit137 after116 frames, without Python traceback. Resource pressure is suspected, not proven. That is an execution failure, not an accuracy result. Runs are now serialized and recurrent checkpoints added. A small stateful checkpoint/resume test passed (outputs0,1 then resumed2,3). The first64 warmup frames passed original numerical checks, but final road scoring is not complete. Current teacher capture also saves block activations for reuse. Its restarted first86 outputs equal the earlier teacher outputs exactly.

## Final recorded-image replay: PASS

The serial student retry completed all320 frames. The unchanged upstream checker passes both all320 and the predeclared last160 after160 warmup. This public-road sequence was not used for training or weight selection. Teacher and student independently propagated their own recurrent states. Student retry first116 outputs equal the interrupted parallel run exactly; teacher restart first86 equal the earlier teacher exactly. All outputs are finite.

Post-warmup pooled worst correlated plan column0.999959; lead0.999999. Max absolute errors are nonzero (plan0.5, lead0.5, desire_state0.3281, lead_prob0.1406), within the original head/column-specific rules. Some quiet columns are assessed by absolute error, not correlation. This is teacher numerical agreement on one short recorded-image sequence, not a ground-truth driving metric or safety certification. Scalar context is source/log reconstructed, not original scheduling state.

Evidence: real-block/road-silu-scored/summary.json, all_frames.log, after_warmup.log; full outputs under road320-teacher-capture and road320-student-serial. Preprocessing sources/tests/frozen hashes are under /workspace/shared/jetlink-public-road.

Current boundary: only one MLP has been replaced. Other62 GELUs remain unchanged. No complete63-block student, whole-model mobile execution,32ms latency, thermal/p99 device verification, or general driving-accuracy conclusion has been produced. Further public day-road sources were verified but not used as train/test evidence.
