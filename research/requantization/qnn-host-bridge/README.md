# Linux host-functional QNN first-projection ORT bridge

This bridge actually executes the compiled A16/W8 first-projection graph through the x86 QNN HTP backend. It is not a fake quantized MatMul or an ORT replacement for QNN execution. It does **not** run on a phone, measure device latency, establish end-to-end accuracy, or establish production deployment readiness.

## Build and integration

```
bash build.sh
bash run.sh test_bridge.py
bash run.sh test_contract.py
bash verify_native.sh
```

`run.sh` supplies this workspace's QNN/Hexagon host runtime dependencies and Python ORT packages. For integration, launch the parent numerical runner through `bash run.sh /path/to/runner.py ...`. Register `libjetlink_qnn_bridge.so` using `SessionOptions.register_custom_ops_library`. Use domain `jetlink.qnn`, version 1, op `QnnFirstProjection`, input FLOAT16 [1,288,512], output FLOAT16 [1,288,1024]. `single_node.onnx` is the minimal example. The bridge uses CPUExecutionProvider as its ORT dispatch boundary, but inside invokes QNN graphExecute on libQnnHtp.so, not ORT CPU MatMul.

The QNN model shared library is loaded from the existing local integration folder. `JETLINK_QNN_MODEL` and `JETLINK_QNN_BACKEND` can override paths; callers must supply the same intended graph and backend. Required SDK headers are referenced in place, not redistributed. Graph composition uses the model's actual `QnnModel_composeGraphs`/`QnnModel_freeGraphsInfo` ABI and runtime tensor metadata. Device settings are SoC43, v73, graph optimization 3, matching the net-run configuration. Graph/API versions, input/output counts, tensor dtype, shape, and quantization encoding are checked. Scales and offsets are read from composed tensors, not hardcoded.

## Numerical contract

1. Decode each FP16 input to FP32 exactly. Reject NaN/Inf.
2. Set the calling thread's floating-point mode to FE_TONEAREST, restoring its previous mode afterward.
3. Divide in FP32 by the graph's input scale; round to nearest integer, ties to even; subtract graph offset; saturate [0,65535]; convert to U16.
4. Execute the real QNN graph with native U16 buffers.
5. Compute `(int32(native_output) + output_offset) * output_scale` in FP32, then round to FP16.

This matches the intended ONNX QuantizeLinear nearest-even input boundary. The historical qnn-net-run float-input converter uses a different quantization convention, so exact parity is asserted only when both graph executors receive identical native input codes. Do not treat a loose allclose check as proof of equivalent quantization.

Runtime input scale is 7.6622942287940532e-05, offset -38745; output scale .00043916353024542332, offset -33554. These values are observed/logged from the graph.

## Verified checks

- 63,488 finite FP16 bit patterns exercised, covering every finite FP16 value. All 147,456 native input elements matched NumPy FP32 divide + rint + zero point + clipping exactly.
- Native execution compared with independent qnn-net-run on the same model/config and identical nearest-even U16 sample-0 input: zero native-U16 mismatches across 294,912 outputs.
- Wrong shape and dtype rejected; NaN and Inf rejected.
- Concurrent calls to one session returned identical FP16 bits. QNN initialization, execution, and cleanup are serialized globally; separate kernels own separate backend/device/context and buffers.
- Session teardown and recreation returned identical FP16 bits.
- Library domains intentionally live for process lifetime, as required by ORT session references. Runtime handles and model/backend libraries are released with the kernel. Exceptions become ORT errors.

`contract-results.json` and `native-parity.json` record these tests. `nearest160-results.json` records the final nearest-even bridge versus all160 historical net-run float outputs rounded to FP16. That is a diagnostic quantizer-discrepancy report, not an accuracy comparison to the original full driving model. `round-away-results.json` are the initial round-away experiment, not the final numerical contract. `modes*.log` document rejected quantizer hypotheses.

Debug-only `JETLINK_DUMP_INPUT`/`JETLINK_DUMP_OUTPUT` dump the latest call's native U16 buffers to explicit local paths; leave unset for normal integration. `LIMIT` limits `test_bridge.py` samples and `RESULTS` chooses its report filename. No timing claims are made.

Portability: source and scripts contain this research workspace’s default dependency/data paths. Configure QNN_SDK_ROOT, ORT_HEADERS, CXX, JETLINK_QNN_MODEL and JETLINK_QNN_BACKEND and adjust run/test fixture paths in another environment. No SDK headers/libraries, compiled bridge, raw tensors, model weights or ONNX files are included in the source-only deliverable.

Final all160 diagnostic: 95,857 FP16 bit mismatches / 47,185,920 elements (0.203147%), maximum absolute difference 0.0078125, aggregate RMSE 2.758386e-5 versus historical float-input net-run outputs rounded FP16. The native same-input check is exactly equal; these residuals come from the separately defined input quantization convention. Observed whole-test host wall time was 336.38 seconds for160 frames (2.10 seconds/frame), including initialization, comparisons and competing work, not device timing.
