# Full mixed deployment graph now prepares for SM8550/v73

The complete graph containing the selected A16/W8 projection, 63 custom FP16 GELUs, faithful image-input promotion, and FP16 remainder successfully passed backend node validation and QnnGraph_finalize, producing a 785,444,864-byte context. See `exact-qnn-results/mixed-fullgraph/summary.json`. This is larger than the sibling all-FP16 context by 901,120 bytes; no size or speed improvement is claimed.

Two converter issues were addressed explicitly: truthful static metadata avoids a native ONNX shape-inference crash; early converter validation of a temporary FP32/FP16 pre-fallback graph was disabled for serialization, then generated C++ node validation was restored to 1 before compiling and preparing the final graph. The final graph's former failing Add has matching FP16 inputs/output, all 63 custom GELUs have FP16 I/O, and the selected projection retains its existing U16/U8 encodings. The validated backend preparation, not disabled intermediate validation, is the successful gate.

**Whole-context execution is not yet passed.** The sibling FP16 full context crashes during host execution between the vision and policy sections; this mixed context was not blindly executed. The whole-model accuracy PASS results below use actual QNN execution for the selected projection with CPU remainder, not this fully accelerated context. Device speed remains unmeasured.

# Latest result: actual native-QNN projection passes recorded-road whole-model parity

The independently validated QNN-host bridge now replaces the selected A16/W8 projection inside the entire exact-GELU-repaired model. Its real QNN HTP graph execution, with ONNX nearest-even input quantization and FP16 output boundary, completed 320 synchronized dual-camera recorded-road frames with its own recurrent state. Both all 320 frames and the predeclared last 160 after warmup pass the unchanged upstream checker. No weights, encodings, thresholds, or calibration were refitted to this road sequence. See `exact-qnn-results/qnn-bridge-road320-summary.json`, `all_frames.log`, and `after_warmup.log`.

This is stronger than the earlier ORT-QDQ simulation: the selected projection actually runs through QNN graphExecute, and matching native inputs were independently verified against official net-run with zero output mismatches. The rest of the model and exact GELUs still execute on CPU. This does not establish a fully accelerated graph, phone parity, device latency, driving safety, or the requested 32 ms target. Only one public nighttime/cone sequence is covered; broader driving-distribution validation remains needed. The original synthetic 160-frame acceptance through this actual QNN bridge also completed and passed unchanged. See `exact-qnn-results/qnn-bridge-synthetic160-summary.json` and its full parity log.

# Current status: exact-GELU integration passes CPU parity; native QNN block builds

Updated November 9, 2026. These results supersede older timeline statements below.

- **CPU whole-model numerical gate passes:** first-projection A16/W8 QDQ, with an immediate FP16 cast before the original first bias, the original FP16 second projection, and all 63 GELUs repaired by the repository exact CPU implementation. All 160 original synthetic acceptance frames pass the unchanged checker using independently propagated state. See `exact-qnn-results/fp16-rest-full160-summary.json` and `build_fp16_rest_exact.py`.
- **Actual QNN build succeeds for the selected projection:** QNN 2.42 converter with supplied calibration and preserved QDQ encodings, Clang host model library, and HTP offline context preparation for SoC 43 / v73 (SM8550). Context size is 593,920 bytes. This is one projection, not a complete accelerator graph.
- **Actual x86 HTP host execution succeeds for 160 block inputs:** compared with ORT QDQ, RMSE is 0.0005745245 and maximum absolute difference is 0.05840874. Therefore the compiled block is not numerically identical to the CPU candidate. See `exact-qnn-results/qnn-htp-host160-comparison.json`. Integrating this actual compiled execution back into recurrent whole-model validation remains necessary.
- **Independent exact CPU plugin tests:** all 63,488 finite FP16 patterns match mathematical reference rounding for each erf/tanh mode; 100,005 finite FP32 samples per mode also match. This does not prove Android libm/device table equivalence.
- **SiLU sibling route compilation check:** its standalone FP16 block compiles to a 2,166,784-byte SM8550/v73 context. Actual HTP-host versus ORT comparisons on 16 selected long-validation inputs have RMSE 0.001430751 and maximum difference 0.125. This is a block diagnostic, not full deployed-model qualification.

No phone was contacted. No on-device speed improvement, 32 ms result, full-QNN model accuracy, or complete-QNN context is claimed here. Public real dual-camera replay preparation is underway separately; this quantization calibration and completed acceptance remain synthetic.

The `exact-qnn-results` directory includes independently obtained numerical reports and a local launcher as a setup example. The launcher uses explicit workspace dependency paths and must be adapted for a different machine. SDK binaries, headers, model weights, generated contexts, and raw calibration tensors are intentionally excluded from the repository. Obtain vendor components under their own licenses. Generated reports contain CPU/host timings only, not device benchmarks.

## Earlier experimental timeline

# Latest completed gate: selective A16/W8 passes original 160-frame CPU parity

**PASS:** The first-only mixed-precision candidate completed all 160 original synthetic acceptance frames with independently fed recurrent state and passed the unchanged upstream per-frame, per-head and per-column checker. `real-block-results/first-only-whole160-summary.json` records exit 0 and `first-only-whole160-parity.log` contains the complete result.

Exact scope: only the first MatMul in original policy transformer layer-0 MLP (`node_MatMul_546`, weight `val_645`) is A16/W8. The second projection and correct tanh GELU remain floating point in this CPU diagnostic. Calibration used 160 frames from seed 713; selection used block outputs from 160 separate seed-1777 validation frames. Original seed-1 acceptance outputs were never fitted. The whole-model replay uses ORT CPU 1.23.2 and recorded layout-only Contiguous-to-Identity lowering; it is not the device's ORT/QNN implementation.

**Still not achieved:** an HTP-compiled accurate candidate or measured phone speed improvement. No 32 ms claim is supported. This is not full-model blanket quantization. The previous both-MatMul candidate remains rejected by the full 160-frame gate despite its earlier 16/32-frame passes. Deployment must incorporate the exact-GELU repair, then perform fresh QNN compilation, identical-input device parity, and controlled A/B/A latency measurements.

The remaining blocker is matching HTP/QNN execution and the accurate deployable graph/runtime context, not access to the original public model or synthetic inputs: those were recovered and verified. Representative real-road dual-camera calibration is still missing. CPU wall times saved in the report are execution diagnostics only, not acceleration evidence.

## Re-run the final selective candidate

Use the explicit CLI paths for your checkout/environment. The producer scripts in the distillation deliverable create `teacher-block.onnx` and `teacher-capture.onnx` from the hash-verified original model. Then:

```sh
# All following commands require ONNX, ORT CPU, NumPy and the pinned upstream checker.
python research/requantization/capture_long_history.py --model <teacher-capture.onnx> --spec <spec.json> --out <new-long-capture>
python research/requantization/quantize_block.py --block <teacher-block.onnx> --data <new-long-capture/sequences.npz> --exclude-node node_MatMul_548 --a16-only --output <new-qdq>
python research/requantization/run_candidate_replay.py --source <original.onnx> --inputs <regenerated-inputs> --reference-dir <original-compare-dir> --qdq <new-qdq/QUInt16-w8-pc0.onnx> --frames 160 --out <new-whole-replay>
```

Use fresh output directories. The executed candidate's calibration tensors are exactly the same 160 training frames in `early-pilot.npz` and final `sequences.npz`; the former used the earlier four-frame seed-991 set only for preliminary block reporting, never for scale fitting. Final selection additionally evaluated all 160 seed-1777 frames. No bias correction is used in the passing first-only candidate. Captured block weight/tensor names and hashes are retained in the reports. Some helpers retain clearly visible experiment-specific default paths; supply explicit paths when moving environments.

The following sections retain the experiment timeline, including rejected candidates. A later completed result above supersedes earlier in-progress wording below.

---

# Long-history calibration and selective precision

The long-history data capture completed: 160 training frames at seed 713 and 160 separate validation frames at seed 1777, each with its own reset and recurrent state. Actual coverage audit confirms the old short calibration was inadequate: first-MatMul teacher activations reach [-14.736, 14.045] versus the old encoding's [-7.022, 8.359], with 44,277 values outside that old range. This explains a concrete quantization risk, but is not alone complete attribution of the head failures.

Four fixed block candidates were evaluated on all 160 held-out seed-1777 frames:

| Candidate | Block normalized RMSE | Correlation |
| --- | --- | --- |
| Both MatMuls A16/W8, long calibration | 2.8502% | 0.99959552 |
| Both, plus long-training bias correction | 1.8627% | 0.99982660 |
| First MatMul A16/W8 only | **0.5879%** | **0.99998272** |
| Second MatMul A16/W8 only | 2.7853% | 0.99961380 |

The first-only candidate is selected for full-model testing based on this independent block validation, not acceptance-set fitting. The second projection and GELU remain floating point in this CPU diagnostic; the extraction promotes them to float32 and separately measures promotion drift. Consequently the ONNX artifact is not a claim of optimal device size/layout. Full original seed-1 160-frame replay is now running. All existing thresholds remain unchanged. No device speed has been measured.

Evidence: `real-block-results/long-block-validation.json`, `long-history-short-range-audit.json`, and `first-only-quantization.json`. The numbers above remain block diagnostics until the full-model gate completes.

# Latest result: short calibration candidate rejected by full 160-frame gate

The corrected candidate's earlier 16/32-frame pass did **not** survive the full original 160-frame replay. Final unchanged-checker result: FAIL in four heads, with worst per-frame plan correlation 0.990958 and lead 0.997491; desire_state and lead_prob also fail pooled checks. Full logs and summary are in `real-block-results/bias-whole160-*`. No accuracy/speed-qualified quantized model has been produced.

A concrete next hypothesis is insufficient history coverage in calibration: only eight initial frames trained the first quantizer, whereas feature history has capacity 128 and the first observed plan frame-level failure appeared at frame 64. This is a hypothesis, not established causality. `capture_long_history.py` is now collecting 160 recurrent training frames at seed 713 and 160 validation frames at new seed 1777. The original seed-1 acceptance inputs/targets are not used to fit quantization scales or bias corrections. This extends synthetic history coverage; it does not make the calibration representative of real roads.

**Deployment guard:** The current full QDQ graph is a CPU numerical-screening artifact based on the source ONNX. Do not deploy it directly with QNN and assume that precision survives: unqualified standard GELU placement could recreate the original HTP GELU error. A device candidate must incorporate the existing exact-erf/exact-tanh repair or a validated new HTP implementation, retain correct partition placement, and undergo fresh full-model device checks. Quantization does not repair the operator mapping by itself.

## Bias-corrected full-model 16-frame gate

The bias-corrected A16/W8 candidate now **passes the unmodified checker on the original 16-frame prefix**, with its own recurrent state. It passes both archived-reference comparison and comparison against the separately executed current CPU baseline. The earlier uncorrected candidate fails this same gate. Correction still used only seed-713 calibration targets; no acceptance outputs were fitted.

This is a short synthetic CPU numerical result for **one** quantized MLP, not acceptance of a fully quantized model, a 160-frame pass, representative road accuracy or HTP speed. A 160-frame independently recurrent replay is underway. Corrected 16-frame evidence is in `real-block-results/bias-whole16-*`.

## Latest whole-model rejection and next correction

Actual A16/W8 replacement of the selected policy MLP **failed** 16-frame independently recurrent whole-model CPU replay. Both archived reference comparison and this executor's separate original-model baseline comparison fail the lead head: `mu[2]` pooled correlation 0.998820, below 0.999. The current original-model CPU baseline passes the archived-reference checker (not byte-identical; max absolute difference 0.125). This isolates a real candidate failure rather than accepting the block's high aggregate correlation as full-model success. Evidence: `real-block-results/whole16-summary.json` and adjacent original checker logs.

`bias_correct_block.py` then fits a standard PTQ output-bias correction **only** to calibration sequence seed 713, without acceptance target fitting. On separate synthetic validation seed 991, block normalized RMSE improves from 2.3203% to 2.0086% and correlation from 0.999731 to 0.999798. This corrected block still requires full-model checking; neither this improvement nor the original candidate is a qualified accuracy/speed result. `run_candidate_replay.py` provides the independent recurrent test path.

## Continuation: actual model and QDQ block now executed

The initial missing-model/synthetic-input blocker below has since been removed. The exact 792,305,602-byte public ONNX was downloaded and its recorded SHA256 verified. `regenerate_history.py` recreated 960 files for all 160 original synthetic replay frames, with every byte length and SHA256 matching the committed manifest. These are not real road calibration data.

The cooperating block experiment captured the original policy transformer layer-0 MLP (`layer_norm_49` to `linear_99`) from full-model independent synthetic recurrent sequences: eight calibration frames, seed 713; four validation frames, seed 991. `quantize_block.py` creates actual ONNX QDQ candidates with the original weights, preserving tanh GELU. Float32 promotion drift is measured separately, not assumed equivalent to FP16.

Measured on CPU using ORT 1.23.2 (not the device's 1.29.0):
- A16/W8: local block normalized RMSE 2.3203%, correlation 0.99973086, max absolute error 0.44046 versus original FP16 block.
- A8/W8: normalized RMSE 13.8367%, correlation 0.99051742, max absolute error 1.66420.
- QNN's configuration helper in this ORT version explicitly disables per-channel MatMul. Both requested variants produce identical candidates with scalar weight scales; a `per_channel=True` flag is not evidence of per-channel deployment.
- CPU execution and ONNX validation succeeded for all four requested configurations, representing two distinct graphs. Neither local correlation is a full-model parity pass or hardware-speed result.

The synthetic blocks/data live outside Git under `/workspace/shared/jetlink-distillation/real-block/`; QDQ candidates under `/workspace/shared/jetlink-qdq-real-block-v2/`. Summary is retained in `real-block-results/summary.json`. Those workspace paths are transient execution artifacts, not portable download links. Original-model publication/license terms must be checked before redistributing derived weight files.

`replay_qdq_whole.py` implements insertion of the A16/W8 block into the full original model with explicit FP16/FP32 boundary casts, and independent recurrent CPU replay against archived references and a current CPU baseline. The only unrelated graph lowering is the layout-only `org.tinygrad::Contiguous` to `Identity` needed by the CPU runtime; this is recorded, not silently treated as original byte-identical execution. Full-model results will be recorded separately when that run completes. No HTP package, device compile, or device timing has been performed.

---

# Requantization feasibility: measured boundary study and QDQ pilot

## What is actually implemented

`boundary_sensitivity.py` implements unsigned asymmetric affine quantize/dequantize (QDQ), using zero-inclusive min/max calibration, explicit integer zero points and round-to-nearest-even. It replays the **unmodified upstream checker** against simulated output-boundary quantization of the accepted saved outputs. It does not load or execute the model.

Calibration uses accepted output frames 0–79. Evaluation uses frames 80–159 against the saved independent reference outputs. The baseline passes this 80-frame suffix. Every source tensor and the checker are SHA256 recorded. No input/result arrays were altered. Seven unit tests pass, including exact zero, error bound, clipping, invalid data and exact coverage by diagnostic output groups.

## Measured results

For 10% range margin (`results/summary.json`):

| Scale grouping | UINT8 | 12-bit diagnostic | UINT16 |
| --- | --- | --- | --- |
| Whole output tensor | FAIL, 11 heads | FAIL, 4 heads | PASS |
| Per named head | FAIL, 5 heads | FAIL, plan | PASS |
| Per semantic column | FAIL, plan | FAIL, plan | FAIL, plan |

The whole-tensor UINT16 case adds at most 0.001995 absolute error to the accepted output and has no clipped elements. Whole-tensor UINT8 adds up to 0.512556 and fails even with no clipping. Narrower column scales are **not always better**: the held-out column distribution leaves the calibrated range. Column UINT16 clips 219/1,476,160 elements and plan `std[6]` correlation drops to 0.998784, below 0.999. This is a different failure from the old GPU quiet-column failure.

A follow-up **exploratory sensitivity run**, not an independently validated choice, raises the margin to 25% (`results-margin125/summary.json`). The pass/fail matrix is unchanged. Column UINT16 still clips 146 elements and plan `std[6]` is 0.998805. Increasing the margin alone is not enough in these tested configurations.

Interpretation: preserve output/head precision in the initial quantization pilot; do not begin with global 8-bit output encoding. More representative calibration is necessary. This does **not** establish that every internal activation needs 16 bits, that internal UINT8 cannot work, or that UINT16 internal activations will pass. Quantized weights and error accumulation are absent here. The temporal split shares one trajectory, and simulated hidden-state output quantization is **not fed back**. Neither full recurrent accuracy nor road safety is validated. 12-bit and semantic-column scales are diagnostics, not claimed QNN layouts.

## Concrete next pilot

The goal is to reduce heavy linear/Conv cost without reintroducing inaccurate native GELU. Requantizing the old fast graph does not automatically fix its two distinct activation formulas. Keep the 36 erf and 27 tanh choices separate throughout preprocessing and compilation.

1. Obtain the uncompiled original model, verify SHA256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`, and retain an untouched source. Extract one Conv/linear block and a policy block before whole-model work. Start with the audited `[32,2048] × [2048,8192]` MatMul and its second projection as a local sensitivity target, but do not assume those two nodes dominate total runtime.
2. Capture real model inputs **and** accepted block boundaries across independent drives, day/night, transitions and long recurrent rollouts, with exact preprocessing and reset/state metadata. Split by entire sequence, not adjacent frames. Reserve held-out sequences before tuning. The snapshot outputs cannot substitute for activation calibration.
3. Construct a QNN-targeted QDQ pilot with UINT16 activations and UINT8 weights for the selected Conv/MatMul regions. Use per-channel weights only where the exact operator/backend accepts them. Keep head outputs, recurrent state, normalization, attention softmax and correct GELU paths at the baseline precision initially. This is a conservative experiment design, not evidence those exclusions are all necessary.
4. ORT's public QNN workflow exposes `qnn_preprocess_model`, `get_qnn_qdq_config` and `quantize`; `init_overrides` specifies promoted tensors. Pin and inspect the deployed ORT/QNN versions before use. Documentation examples start from float32, whereas this source is FP16: conversion must preserve formulas, cast boundaries and an independently checked float baseline. Do not simply cast initializer buffers and declare semantic equivalence. No EPContext cache is a calibration source.
5. Compare three fresh-compiled candidates: baseline FP16, selected UINT16-activation/UINT8-weight blocks, then selectively lowered UINT8 activations. Quantize one region at a time and rank head/column error changes. Record every CPU fallback and partition boundary. Failed QNN coverage or increased boundary count can destroy theoretical integer throughput gains. Add the custom correct GELU route only when its independent correctness is established.
6. Run the unchanged checker on 160-frame same-input recurrent replays at minimum, followed by independent longer held-out sequences. Teacher and candidate must each feed their own previous state. Do not feed teacher state to the candidate for final acceptance. Record all nonfinite values, saturation, head failures and sequence worst cases, not just average correlation.
7. Benchmark A/B/A on the same SM8550 with matched warmup, thermal state, QNN/ORT versions, allocator and shared-memory settings. Record mean, P50/P99, transfer cost, CPU/NPU partition count and complete single-frame latency. Do not report throughput as latency. Reuse 52.4086 ms only as the historical accurate baseline, not today's measured baseline.

## Performance gate and stop conditions

Reaching 32 ms from 52.4086 ms needs about **20.41 ms / 38.94%** reduction in the full inference path. Output-only quantization saves none of that measured time. W8 compression could reduce memory traffic but cannot remove the existing CPU activation crossings by itself. A correct accelerator GELU and quantization may be complementary; neither presently has measured combined speed.

Stop a candidate when a head fails, saturation is unexplained, fallback increases, or A/B/A shows no repeatable whole-model benefit. Do not relax the parity checker. A failed UINT16 pilot calls for sensitivity analysis and possibly QAT/distillation, not an automatic further reduction to UINT8.

## Blockers, exact scope

Completed: 18 boundary experiments, baseline checks, source hashes, original checker logs, seven unit tests and the pilot design. No phone was contacted. No model was quantized, trained, compiled or benchmarked.

The current checkout omits original/derived driving weights, raw replay input captures, intermediate activations, QNN context binaries and runtime libraries. This executor also lacks `onnx`, `onnxruntime`, and `torch`. Therefore the blocked action is **constructing and validating a real quantized JetLink model and timing it on SM8550**, due to missing model/data/toolchain/device context, not an access denial or failed device execution. The public original ONNX link is in the repository review request; obtaining that alone does not supply calibration/replay data. The next useful handoff is model plus sequence-separated calibration/held-out captures and an authorized matching QNN test environment, not more saved outputs.

## Reproduce

From the repository root:

```sh
python -m unittest discover -s research/requantization -p 'test_*.py' -v
python research/requantization/boundary_sensitivity.py --output /tmp/jetlink-boundary-new
python research/requantization/boundary_sensitivity.py --margin 1.25 --output /tmp/jetlink-boundary-margin125-new
```

Use fresh output directories. NumPy is the only added runtime dependency for this experiment. Generated original-checker logs include every evaluated frame and pooled result; `*.encodings.json` records all scales and zero-points.

## Sources

- Repository `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/dot-review-request.txt` and `README.md`: source hash, missing artifacts, exact formulas and historical accuracy/latency.
- `reports/jetlink-operator-audit-20261101/README.md`: real sampled block sizes and precision evidence.
- `vendor/jetlink/scripts/verify_parity.py`: untouched acceptance implementation.
- [Official ONNX Runtime QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html), consulted for QDQ APIs and mixed integer precision. Its examples do not certify this model or SM8550 configuration.
