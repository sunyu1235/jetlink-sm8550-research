# Single operator QNN HTP FP16 audit

Five nodes were lifted from the same FP16 block, each as a one-node ONNX graph with its original initializer tensors. The input to each node was captured during one ONNX Runtime CPU execution of the complete block using the same `add_34` FP16 tensor. Shape inference filled missing metadata before extraction. The CPU reference for each mini graph matches the corresponding full-block CPU node output exactly (`max_abs=0`, `RMSE=0` for all five).

The HTP runs used the already-installed strict probe with ORT 1.29.0, QNN HTP and `enable_htp_fp16_precision=1`. `session.disable_cpu_ep_fallback=1` was set, each input file was checked on-device for exact byte size, and all five runs exited 0 with output dtype enum 10 (FP16) and zero non-finite values. The existing ADB device connection was used; the daemon was not restarted. Only diagnostic ONNX/input/output files in `/data/local/tmp/jetlink-operator-audit-20261101` were written on the phone.

| Node | Exact node / operation | Output shape | HTP FP16 vs CPU FP16 MAE / max abs / nRMSE | HTP FP16 vs CPU FP32 math MAE / max abs / nRMSE | CPU FP16 vs CPU FP32 math nRMSE |
|---|---|---|---|---|---:|
| Depthwise Conv | `node_conv2d_39`, `Conv`, group 2048, weights `[2048,1,7,7]` | `[1,2048,4,8]` | 2.18e-7 / 0.003906 / 1.89e-6 | 9.92e-4 / 0.230530 / 2.13e-4 | 2.13e-4 |
| LayerNormalization | `node_layer_norm_39`, axis -1, epsilon 1e-5, stash type 1 | `[1,4,8,2048]` | 3.88e-8 / 0.000488 / 1.26e-5 | 1.91e-5 / 0.001830 / 1.97e-4 | 1.97e-4 |
| MatMul 1 | `val_521__gemm__matmul`, `[32,2048] × [2048,8192]` | `[32,8192]` | 3.28e-7 / 0.003906 / 1.38e-5 | 2.45e-4 / 0.003210 / 2.46e-4 | 2.46e-4 |
| MatMul 2 | `val_523__gemm__matmul`, `[32,8192] × [8192,2048]` | `[32,2048]` | 1.61e-7 / 0.000977 / 7.69e-6 | 1.21e-4 / 0.012028 / 2.12e-4 | 2.12e-4 |
| GELU | `node_gelu_35`, `approximate="none"` | `[1,4,8,8192]` | 2.22e-4 / 0.007813 / **3.91e-3** | 2.22e-4 / 0.007813 / **3.91e-3** | 1.71e-4 |

nRMSE is `RMSE / std(reference)`; it helps compare errors across outputs with different scales. GELU has the largest local HTP-versus-reference error by normalized RMSE, about 16–20× the other isolated nodes. Its CPU FP16 reference is very close to CPU FP32 math, so this difference is specific to the HTP result for this sampled GELU input. Of the 1,407 positive GELU inputs, HTP returned an output greater than the input for 48 (3.41%); the largest overshoot was `0.0078125` at input `7.24609375` (output `7.25390625`). Neither CPU FP16 nor CPU FP32 math had a positive-input overshoot. This makes GELU a clear follow-up target, but does not establish it as the sole source of full-model drift. Details are in `gelu-overshoot.json`. The Conv's maximum absolute error against FP32 math looks larger (`0.2305`), but its output standard deviation is `13.09`, and its nRMSE is only `2.13e-4`.

## Limits

This is an operator-level diagnostic on one captured activation per node, not attribution of whole-model error. The node inputs all come from the CPU-executed block; the HTP output of one node is not fed into the next HTP node. That deliberately removes upstream error accumulation and may not represent the input distribution seen in a full HTP chain. The result supports prioritizing isolated GELU behavior for a follow-up check; it does not prove GELU is the cause of the full-model difference or that moving LayerNorm to another provider would help. The LayerNormalization node tested here has very small local error.

## Reproduction artifacts

- `extract_operator_audit.py` shape-infers the input block, captures selected node inputs/outputs on CPU, and writes exact one-node graphs, input tensors, FP16 references, and node metadata.
- `add_fp32_references.py` creates CPU FP32-math references from each exact same captured FP16 input and the original FP16 constants cast to FP32.
- `run_htp_probes.py` uses the pre-existing ADB device and strict probe; it refuses to overwrite remote/local result directories and verifies the strict-probe hash and input lengths.
- `compare_operator_audit.py` writes the error metrics without changing the source model or thresholds.
- `analyze_gelu_overshoot.py` counts positive GELU inputs for which each backend returns `output > input`.
- Full node, shape, attribute, hash, timing, reference and comparison records are in `../outputs/jetlink-operator-audit-20261101/manifest.json`, `operator-errors.json`, and `htp16-runs.json`. Per-operation strict-probe logs are `*.htp16.log`.
- HTP raw outputs and all diagnostic one-node models/inputs/references are in `../outputs/jetlink-operator-audit-20261101/<operator>/`.

The probe header used for ONNX tensor element type IDs is `C:\Users\sunbo\Documents\moto-x40-root\jl_ort.h`; it defines `JL_FLOAT16=10` and `JL_FLOAT=1`. The strict runner source is [strict_probe.c](../jetlink-gpu-investigation-20261101/strict_probe.c), which uses QNN HTP, sets `session.disable_cpu_ep_fallback=1`, and verifies exact input byte counts. Official references: [ONNX Runtime QNN EP options and CPU fallback](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html), [Qualcomm QAIRT HTP operators and FP16 precision](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html).
