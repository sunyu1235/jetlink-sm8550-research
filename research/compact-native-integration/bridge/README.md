# Saved-context compact native bridge

This Linux host-functional ORT custom op executes the existing compact upstream and three-tail QNN HTP saved contexts. It is not a phone implementation or target-speed benchmark.

## Interface

Domain `jetlink.qnn`, op `QnnCompactPipeline`, version 1.

Input order: `gelu_43`, `gelu_54`, `gelu_61` (FP16 [1,288,1024]), then `add_50`, `add_70`, `add_86` (FP16 [1,288,512]). Output order: `select_4`, `select_7`, `select_10`, each FP16 [1,512].

QnnSystem parses saved-context descriptors; tensor names, counts, shapes and FP16 types are checked. The six compact outputs are bound to tail inputs by name, not by context tensor order. The upstream context emits interleaved residual/projection outputs, while the tails list residuals before projections. Six shared host buffers total 9,216 bytes. There is no zero-padding, FP16/FP32 conversion, quantization or host numerical approximation inside this bridge.

The backend is opened with RTLD_GLOBAL because the custom host package expects symbols from it. The package uses the prior symbol-isolated native21 build. This repairs an initial package-load failure, not an arithmetic change. The initial failed initialization performed no inference.

## Build and run

`bash build.sh` builds `libjetlink_compact_bridge.so`. `bash run.sh SCRIPT [args]` supplies existing QNN/dependency library paths and Python dependencies. The full model must register this library through ORT SessionOptions and run through CPUExecutionProvider, whose custom callback invokes real host QNN graphExecute for the two contexts. The rest of the full model remains outside this bridge.

Defaults point to the existing workspace artifacts. Override `QNN_SDK_ROOT` and `ORT_HEADERS` when building; override `JETLINK_QNN_BACKEND`, `JETLINK_QNN_SYSTEM`, `JETLINK_GELU_PACKAGE`, `JETLINK_COMPACT_UPSTREAM` and `JETLINK_COMPACT_TAILS` at runtime when moving corresponding verified dependencies. The launcher itself retains explicit dependency paths and requires adaptation on another executor.

## Validation completed

`test_bridge.py` compares eight existing deterministic samples against official net-run native FP16 outputs: 12,288 values, zero bit mismatches. Wrong input shape, dtype and nonfinite inputs are rejected. A clean ninth invocation after the rejected calls reproduces official sample7 exactly, confirming rejection did not poison subsequent execution. All outputs and intermediate compact buffers are checked finite.

Evidence: `parity.json`, `parity.log`, `artifact-hashes.json`. No full-model recurrent test was run by this bridge task; the parent performs that separately. No phone/device execution, performance claim, deployment or publication occurred.
