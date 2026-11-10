# Execution update

The parent executed the real splice, static metadata completion, converter, host library build and backend-validated SM8550/v73 preparation successfully. See the main requantization README and `exact-qnn-results/mixed-fullgraph/summary.json` for exact scope. `build_validated_context.sh` consolidates the successful commands; its shell syntax was checked, but the consolidated wrapper itself has not been rerun end-to-end. The equivalent component commands were executed.

The original draft converter command below fails intermediate pre-fallback dtype validation. The executed flow serializes through that stage, restores generated C++ `DO_GRAPH_NODE_VALIDATIONS` to 1, and only then treats a successful backend finalize as a compile result. Never use an unchecked context or claim numerical qualification based on conversion alone. The complete mixed context has not passed execution or device speed tests.

## Original script-development notes

# Prepare the mixed first-projection full HTP graph

The initial task prepared scripts and tested tiny synthetic fixtures only. After the parent ran the splice and encountered a converter shape-inference crash, this task performed the explicitly authorized metadata-only repair described below. No external full-model weights were loaded and no full conversion or model execution was run by this worker. Heavy stages still require a parent-granted slot.

## 1. Splice after the slot is granted

```
PYTHONPATH=/workspace/shared/jetlink-python python splice_first_projection.py \
  --base /workspace/shared/jetlink-models/light-speed-htp-gelu-images16.onnx \
  --qdq /workspace/shared/jetlink-qdq-first-only-final-data/QUInt16-w8-pc0.onnx \
  --outdir /workspace/shared/jetlink-mixed-first-full
```

The new output directory must not already exist. The script replaces only node_MatMul_546, checking its layer_norm_49/val_645→val_646 boundary. It imports the selected source's first six nodes (input Q, weight DQ, input DQ, MatMul, output Q, output DQ), retaining their attributes, domains and encoded constants. A leading FP16→FP32 Cast and trailing FP32→FP16 Cast reconnect to the original bias Add. No later source MLP nodes are imported. The original exact custom GELU mapping, image promotion, bias and every other node remain byte-identical. The unused original weight is removed only when no node still consumes it.

Imported values are prefixed to avoid collisions. The com.microsoft Q/DQ opset is preserved/added; no unrelated domain is upgraded. The ONNX model and all initializer tensor payloads are saved to mixed-first-projection.onnx and mixed-first-projection.data, followed by the path-based ONNX checker. A splice-report.json records sources, unchanged-node proof and imported initializer hashes. Checker success is structural validation, not numerical or converter proof.

## 2. Draft converter command

```
bash /workspace/shared/htp-gelu/integration/run_preserve_converter.sh \
  --input_network /workspace/shared/jetlink-mixed-first-full/mixed-first-projection.onnx \
  --output_path /workspace/shared/jetlink-mixed-first-full/qnn/mixed_first.cpp \
  --op_package_config /workspace/shared/htp-gelu/integration/Gelu.xml \
  --float_fallback --float_bitwidth 16 --float_bias_bitwidth 16 \
  --act_bitwidth 16 --weights_bitwidth 8 \
  --no_simplification --preserve_io layout
```

`convert_mixed_fullgraph.sh INPUT_ONNX OUTPUT_CPP` prints this command only. Passing `--run` explicitly executes it. The launcher and XML paths can be overridden with PRESERVE_CONVERTER_LAUNCHER and GELU_OP_CONFIG. The existing version-specific preserve-arithmetic launcher prevents folding separate arithmetic into NN nodes; it does not itself prove numerical parity.

Flags were verified using the real launcher's `--help` without converting a model. Float fallback consumes carried QDQ encodings and leaves missing encodings in floating point after math-invariant propagation. It explicitly forbids `--input_list` and `--ignore_encodings`, neither of which is supplied. No overriding encodings, calibration, per-channel re-quantization or QDQ simplification is requested. The U16/U8 constants in the selected QDQ are preserved by the splice. Actual post-conversion tensor encodings, FP16 fallback scope, boundary casts and separate bias arithmetic must still be inspected; the draft command is not evidence that the converter preserves every desired boundary.

## 3. Repair only proven FP16 biases, if needed after conversion

```
PYTHONPATH=/workspace/shared/jetlink-python python repair_preserved_bias.py \
  --net-json /workspace/shared/jetlink-mixed-first-full/qnn/mixed_first_net.json \
  --cpp /workspace/shared/jetlink-mixed-first-full/qnn/mixed_first.cpp \
  --bin /workspace/shared/jetlink-mixed-first-full/qnn/mixed_first.bin \
  --out-prefix /workspace/shared/jetlink-mixed-first-full/qnn/mixed_first_fixed
```

The helper finds only FC signatures [FP16 input, FP16 weight, FP32 bias, FP16 output]. It does not assume nineteen or any fixed number of biases. Optional `--expect-count N` makes a caller-specified count an assertion. Every consumer of a repaired constant must be the compatible FC bias position. Before writing final files it verifies raw byte counts against tensor dimensions, unique archive members, finite values, exact FP32→FP16→FP32 **bitwise** roundtrip including signed zero, and exactly one unambiguous C++ datatype declaration. Validation failures leave no final artifacts. It updates the C++ type, raw storage, JSON dtype and output references, and records payload hashes and zero numerical changes. Quantized FC biases and incompatible shared constants are never silently repaired.

The repair remains a format fix, not a full-model parity proof. Converted library construction, full-model functional parity, recurrent accuracy and phone latency remain later stages.

## Lightweight tests completed

`PYTHONPATH=/workspace/shared/jetlink-python python test_helpers.py` exercises only synthetic models/archives: splice topology, unchanged bias/custom-GELU nodes, external-data save and ONNX checker; bias counts zero/one/two; exact signed-zero preservation; rejection of nonrepresentable values and shared consumers before writes. It never opens the real full driving model.

## Metadata-only repair after converter shape-inference failure

The updated splice script now invokes complete_qdq_metadata.py before saving. It derives initializer declarations from TensorProto dtype/dims and propagates the eight inserted Cast/QDQ/MatMul nodes from the original static boundary and weight shapes. Scalar scales/zero points remain genuine rank-zero tensors. It rejects unknown, dynamic, conflicting or missing metadata rather than padding declaration counts. Every referenced tensor must have a verified type/static shape.

For an existing spliced model, the standalone repair is:

```
PYTHONPATH=/workspace/shared/jetlink-python python complete_qdq_metadata.py \
  --source /workspace/shared/jetlink-mixed-first-full/mixed-first-projection.onnx \
  --output /workspace/shared/jetlink-mixed-first-full/mixed-first-projection-shapes.onnx
```

The new file must reside beside its source, sharing unchanged external-data locations/offsets/lengths. The helper loads metadata only, adds value_info, verifies node and initializer protos remain byte-identical, writes only the ONNX protobuf, and runs the path-based structural checker without shape inference.

Actual repair completed: 56 declarations added (7 QDQ/Cast outputs, 7 imported initializers and 42 preexisting GELU shape initializers). All 1,784 SDK-referenced tensor names have static type/shape metadata. There are 1,786 unique declarations, including two preexisting unused names; no dummy names were added. The repaired model passed ONNX checker. Calling the installed SDK's two shape-check methods, extracted directly from its source with infer_shapes replaced by a failure sentinel, returned the same model with zero inference calls. This narrow test is not a full converter retest. See metadata-repair-report.json and sdk-shape-skip-test.json.
