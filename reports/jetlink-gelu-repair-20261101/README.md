# Targeted GELU repair — 2026-11-01

This experimental standalone repair follows the same-input operator audit in
`../jetlink-operator-audit-20261101`. It does not modify the Jetlink APK or device
system libraries, and is not yet an accepted 30 ms implementation.

## Evidence and implementation

The tested native HTP GELU (`approximate=none`) has substantially greater local
error than the sampled convolution, normalization and matrix products. For some
positive finite inputs its output exceeds its input, whereas
`GELU(x) = x/2 * (1 + erf(x/sqrt(2)))` cannot do so.

`exact_gelu.c` implements an explicit CPU custom operator. It enumerates all
65,536 FP16 bit patterns once, evaluates the erf formula with double precision,
rounds to FP16 and stores a 128 KiB table. Evaluation is a direct lookup using
the input's bits; no interpolation, training, calibration or fitted output
correction is involved. The table does not expand model weights or change the
FP16 interfaces. `ExactGelu32` evaluates the formula directly for FP32 inputs.
The library was subsequently extended with `ExactGeluTanh` and
`ExactGeluTanh32` to preserve the original model's separate tanh GELU definition.
Each FP16 variant has its own 128 KiB table. The tanh variant also matched all
63,488 finite inputs bit for bit against its independently generated reference;
see `tanh-finite-domain.json`.

All 63,488 finite FP16 input patterns were replayed on the phone and matched
the independently generated PC double-formula/FP16-rounding reference bit for
bit: no mismatches and no non-finite outputs. This is a finite-domain operator
check, not a full-model or real-road accuracy claim. Non-finite input behavior
and FP32 custom-op coverage are outside that check.

## Causal block check

Only one GELU was replaced in the actual final-stage ConvNeXt block. Its input,
weights, other nodes, and comparison reference are unchanged.

| Error vs CPU FP16 block | Original HTP | HTP + CPU ExactGelu |
| --- | ---: | ---: |
| Mean absolute error | 0.0141739 | 0.0004780 |
| Root mean squared error | 0.0371596 | 0.0030058 |
| Maximum absolute error | 1.4375 | 0.5 |

Mean absolute error decreased by about 96.6%. The saved QNN context graph
contains an NPU EPContext, the CPU ExactGelu, then a second NPU EPContext. This
checks actual placement rather than treating a CPU fallback as NPU execution.
Single first-run execute times were 2.094 ms originally and 3.200 ms with the
replacement; no steady-state timing or thermal conclusion follows from them.
The remaining block error is not zero.

## Full-model candidate and acceptance

`replace_gelu.py` changes exact GELU nodes to an explicit custom domain while
preserving graph edges, weights and output metadata. The whole prepared Light
Speed candidate is generated under the ignored `outputs` directory; its manifest
records every replacement and model hash. Structural checking cannot validate
custom operator semantics, which are checked separately as above.

The first full candidate replaces the 36 vision exact-erf GELUs and preserves
the 27 policy tanh GELUs. Its 16-frame full-phone replay fails three heads
(`lane_lines`, `lead`, `plan`), down from the original six, and averages 71.819 ms
in an un-warmed replay. Its context contains exactly 37 NPU EPContext nodes and
36 CPU ExactGelu nodes. This candidate is not a full precision pass.

An independent control feeds those captured vision features to the PC FP32
policy with its own recurrent states; every upstream check passes. See
`full-replay-policy-fp32`. This supports that the targeted vision correction
addresses the observed vision parity failure, and localizes the remaining
failure to the policy path in this controlled replay.

Source review found that ORT v1.29.0 registers Gelu with SimpleOpBuilder, maps
it to `QNN_OP_GELU`, and does not forward the ONNX `approximate` attribute.
The prepared model has 36 `none` and 27 `tanh` GELUs, all FP16. The second
candidate therefore preserves both formulas explicitly rather than replacing
tanh with erf. See `light-speed-gelu-both.json` for the 63 replacements.

The second candidate reuses byte-verified unchanged floating weights through
standard ONNX external-data offsets into the first candidate already on the
phone. Shape/index constants and small initializers remain inline because the
phone ORT shape inferrer cannot parse external `Slice` constants during model
loading. `both-external-weights-inline-shapes.json` records all reused offsets,
lengths and hashes; the loaded graph is 4,370,514 bytes and reuses 787,283,968
weight bytes. This changes storage only, with no arithmetic or weight changes.

`gelu_probe.c` and `qnn_jl_custom.c` load the explicit custom library and preserve
the previous runner's recurrent state logic and strict input validation. CPU
fallback is deliberately enabled only for this hybrid experiment; the exported
context graph must be inspected to see what actually remains on the CPU. This
is distinct from the prior GPU/HTP-only diagnostics, which forbid CPU fallback.

The second candidate passes unchanged upstream per-frame and pooled per-column
parity over all 16 recurrent frames. The saved context has 45 NPU EPContext
nodes and 63 CPU custom nodes (36 exact-erf and 27 specified-tanh), with no other
CPU nodes. Mean execution time is 73.372 ms; P99 is 74.615 ms. See
`both-full-replay-summary.md` and the complete comparator log beside it.

An instrumented 16-frame run measures about 16.418 ms in CPU activation nodes
and 58.876 ms in QNN nodes per frame. QNN durations include host execution,
transfers and RPC, not just accelerator compute. Profiling itself changes time
(mean 75.929 ms), so these figures are diagnostic, not a replacement for the
unprofiled result. Detailed trace and aggregation are retained in `cost-profile*`.

A separate independent-seed 160-frame replay and roughly 60-second stress run
are documented in `../jetlink-gelu-history-20261101`. The 800 stress frames average
73.751 ms, P99 76.631 ms, and 13.335 observed loop frames/s. Battery temperature
is 22 -> 23 C; the GPU thermal sensor is 31.9 -> 71.8 C (maximum 72.2 C).
That sensor is not an NPU-core junction measurement. Every stress-frame output
is finite; an all-finite stress run is distinct from numerical parity validation.
No thermal protection or frequency setting was changed.

Full-model latency must be measured separately. CPU/GPU handoffs may cost too
much for 30 ms even if precision passes. If a DSP custom operator is subsequently
implemented, the same mathematical table and acceptance tests can be reused;
no DSP implementation or timing is claimed here.

## Artifacts

The committed artifacts are source code, manifests, comparison logs and CSV
telemetry. Generated tensor captures, model weights, runtime libraries and
executables remain local and are not included in this commit. Some older audit
scripts reference the original machine's capture directories; regenerate or
provide those datasets before replaying those comparisons.

`qnn_jl_custom.c` and `jl_ort.h` derive from Jetlink's COrt wrapper at commit
`a863c3801da6fd925d14eb1f5b1e250f8efcc649`. Its MIT notice is preserved in
`Jetlink-LICENSE.txt`. Building the standalone probes requires the Android NDK
and the official ONNX Runtime 1.29 C headers, with their `onnxruntime/` include
layout. Link `qnn_jl_custom.c` with `gelu_probe.c` (or `history_probe.c`) using
`-DJL_ORT_DLOPEN -ldl`; build `exact_gelu.c` as an Android shared library with
`-fPIC -shared -lm`. The tested builds use `-O2 -Wall -Wextra -Werror`.

- `finite-domain.json`, `finite-domain.log`: complete finite-input operator check.
- `block-causal-result.json`, `block-hybrid.log`: block before/after error and placement.
- `block-exact-gelu.json`: block rewrite manifest and hashes.
- `light-speed-exact-gelu.json`: full-model rewrite manifest and hashes.
- `light-speed-gelu-both.json`: both-formula rewrite manifest and hashes.
- Raw models, inputs, output buffers and saved context are retained in
  `outputs/jetlink-gelu-repair-20261101` at the workspace root.

## Source evidence for policy GELU handling

- [ORT v1.29.0 operator factory](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/builder/op_builder_factory.cc)
- [ORT v1.29.0 SimpleOpBuilder](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/builder/opbuilder/simple_op_builder.cc)
- [ORT v1.29.0 QNN operator mapping](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/builder/opbuilder/base_op_builder.h)
- [ONNX GELU definitions](https://onnx.ai/onnx/operators/onnx__Gelu.html)
