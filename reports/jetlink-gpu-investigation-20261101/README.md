# Light Speed precision investigation — 2026-11-01

Target: Moto X40 / SM8550 / HTP v73 / Adreno 740. Original model SHA-256:
`6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`.
Repository examined: zoompilot/jetlink, commit
`a863c3801da6fd925d14eb1f5b1e250f8efcc649`.

## What is established

The upstream split profile already uses HTP for vision and QNN GPU for policy.
It is a serial dependency within a frame; throughput pipelining must not be
reported as reduced single-frame latency. Prior GPU graph initialization failed;
no working GPU+NPU precision result has been obtained.

The 16-frame controlled replay localizes errors to both portions. Substituting
an FP32 CPU policy for the NPU policy removes several failed heads but leaves
lane lines and road edges failing. CPU vision followed by NPU policy fails
three heads. A PC FP32 vision + FP32 policy replay passes the unchanged upstream
thresholds. Full phone FP32 replay also passes those checks (16 frames,
maximum absolute output difference 0.2101). The phone vision and policy stages
averaged 608.229 ms and 150.245 ms respectively in separate replay runs; their
sum, 758.474 ms, is not an integrated application latency measurement. These are
synthetic frames and numerical parity results, not measured driving accuracy.

See the adjacent `jetlink-precision-audit-20261101` directory for the complete
upstream comparison logs, exact configurations and exit codes. Do not use a
pooled correlation to override a per-head or per-column failure.

## Hardware and source constraints

Qualcomm documents that HTP executes floating-point models with FP16 math,
including models whose inputs and outputs are FP32. FP32 math is unsupported.
The precision option is deprecated for QAIRT >= 2.35. Therefore, changing the
model's declared type or disabling one ORT FP16 flag is not a repair on this
device/runtime.

QNN GPU supports FP16 and FP32 tensors. Its native graph configuration also
documents HYBRID (FP16 storage and FP32 calculations). ORT v1.29.0's examined
QNN execution provider passes HTP graph options but exposes no corresponding
GPU precision-mode option. Merely passing a made-up GPU option is not a test.

The vision graph has 41 LayerNormalization nodes, all with `stash_type=1`.
ONNX specifies FP32 for the first normalization stage with this setting.
The examined ORT QNN layer-normalization builder forwards axis and epsilon,
but does not forward `stash_type`. This is a concrete investigation lead,
not proof that normalization alone accounts for the observed output errors.
The prior sampled layer outputs showed increasing differences along vision;
the full-input error and local operator error must be distinguished.

## Chosen diagnostic and decision gates

1. Establish a phone FP32 CPU numerical baseline with all recurrent states
   updated independently; verify every input's byte count before replay.
2. Extract the final ConvNeXt block (`add_34` -> `add_35`), capture its input
   from frame zero of the unchanged replay and reset image history. Preserve
   weights, epsilon and the operation structure. Create FP16 and exactly
   widened FP32 variants. This shrinks the GPU initialization problem from
   the whole model to one actual representative block.
3. Feed exactly the same captured input to each backend. Forbid CPU fallback,
   record model/input hashes and raw outputs. Separate initialization failure
   from numerical disagreement. These block metrics are diagnostics; they
   cannot be called a full-model precision pass.
4. If FP32 GPU runs and matches the same-input CPU result, investigate a bounded
   graph partition to avoid the whole-graph failure, then run complete recurrent
   model parity. If it fails, reduce this reproducible block to the failing
   operator before changing the full model or runtime.
5. Only after complete numerical parity passes: measure the integrated frame's
   total time, including processor handoffs. Keep the existing 30 ms objective
   separate from correctness. A 32 ms HTP result that fails parity is not an
   accepted optimized model.

Do not loosen the upstream thresholds, fit output corrections to these frames,
or use overclocking, thermal changes or disabled Android services to address
numerical errors. Quantization/retraining would need a representative calibration
and validation set and is not the first precision-preserving repair.

## Reproduction artifacts

`build_block_case.py` produces both model variants, inputs, PC reference outputs
and a hash manifest under the ignored workspace `outputs` directory.
`strict_probe.c` is derived from the earlier standalone probe, with mandatory
non-state input files, exact byte counts, and CPU fallback forbidden for QNN.
It preserves the prior state ping-pong logic and does no warmup in capture mode.
The existing final-frame finite check is not a parity or all-frame finite check;
raw outputs must be independently compared.

## Same-input block result

The extracted final-stage ConvNeXt block runs on both GPU and HTP with CPU
fallback forbidden. All 65,536 output elements are finite. The following are
errors against each variant's PC CPU reference on exactly the same captured
block input; the FP16/FP32 inputs are exact representations of the same values.

| Backend | Reference | Mean absolute error | Maximum absolute error | Single first-run execute time |
| --- | --- | ---: | ---: | ---: |
| GPU FP32 | CPU FP32 | 0.000003126 | 0.001526 | 10.835 ms |
| GPU FP16 | CPU FP16 | 0.003251 | 0.7500 | 6.296 ms |
| HTP FP16 | CPU FP16 | 0.014174 | 1.4375 | 2.094 ms |

These are one-block, one-input diagnostics, with initialization outside the
reported execution time and no warmup. They establish that GPU FP32 can run
this real block accurately and that the earlier whole-graph failure is not
universal GPU incompatibility. They do not establish steady-state speed,
whole-model correctness, or a 30 ms hybrid path. Because this block includes
depthwise convolution, normalization, two matrix products, GELU and residual
addition, a further same-input per-operator check is required before assigning
the difference to any one operator.

`block-errors.json` contains both FP16- and FP32-reference comparisons for all
three backends. The three `block-*.log` files retain runtime messages and
execution statistics. `outputs/jetlink-block-case-20261101/manifest.json` records
the exact model and input hashes; the same directory holds all raw outputs.

## Further precision-preserving route under investigation

The stock QNN HTP limitation is a runtime path limitation, not proof that every
part of Hexagon lacks FP32 instructions. Qualcomm's Hexagon documentation and
official XNNPACK article describe HVX FP32 arithmetic and GEMM/reduction kernels.
Custom HVX code or a custom QNN operator package could therefore implement
sensitive reductions with higher precision while avoiding GPU handoffs. The
installed Android NDK compiler lists no Hexagon target; this path needs a proper
Hexagon toolchain and a compatible DSP operator package. No such kernel has been
built or run here, and no timing claim is made for it.

First determine which local operators disagree. If normalization accounts for
most of the error, a precision-preserving normalization implementation is a
bounded candidate. If convolution/matrix products dominate, moving normalization
alone is insufficient; graph partitions or higher-precision arithmetic for those
products must be measured. FP32 labels alone do not imply bit-identical arithmetic
across backends, so complete recurrent parity remains the acceptance gate.

## Primary references

- [Jetlink split processor profile](https://github.com/zoompilot/jetlink/blob/a863c3801da6fd925d14eb1f5b1e250f8efcc649/JetlinkKit/Sources/JetlinkORT/OrtBackend.swift)
- [Qualcomm HTP precision](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)
- [ORT QNN GPU support](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#running-a-model-with-qnn-eps-gpu-backend)
- [Qualcomm GPU graph precision modes](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/api-rst_file_include_QNN_GPU_QnnGpuGraph_h.html)
- [ORT QNN execution provider v1.29.0](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/qnn_execution_provider.cc)
- [ORT QNN layer-normalization builder v1.29.0](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/builder/opbuilder/layer_norm_op_builder.cc)
- [ONNX LayerNormalization semantics](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)
- [Qualcomm HVX FP32 kernels in XNNPACK](https://www.qualcomm.com/developer/blog/2026/03/bringing-xnnpack-hexagon-npu)
- [Qualcomm HVX precision/compiler guidance](https://github.com/qualcomm/learn-ripple/blob/main/opt/hexagon/src/hvx-opt.md)
- [Hexagon v73 HVX reference manual](https://docs.qualcomm.com/doc/80-N2040-54/80-N2040-54.pdf)
