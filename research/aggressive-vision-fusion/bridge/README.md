# Single-context vision bridge

## Validated scope

Scalar and vector GELU packages each passed exactly one full-model inference on
recorded-night frame 0 with independent zero initial states. All four outputs
match the split-native scalar control byte-for-byte. The saved vector full-model
head also passes the unchanged repository `scripts/verify_parity.py compare`
against the saved CPU baseline: every slice/column satisfies the checker.
These are one-frame host functional gates, not recurrent or device latency gates.

Evidence:
- `full-frame-check/result.json`: scalar full-model parity.
- `full-frame-check/provenance.json`: dependency hashes collected **after** scalar execution.
- `vector-full-frame-check/result.json`: vector parity and dependency hashes.
- `vector-full-frame-check/cpu-baseline-check/checker.log` and `evidence.json`:
  unchanged checker applied to saved actual vector output, without new inference.
- `negative-tests.json`: four malformed ORT inputs rejected before QNN initialization.

The focused source tests comprise 3 model/closure tests, 4 ORT invalid-input
cases and 9 tensor descriptor cases. Runner shell syntax and refusal to overwrite
an existing output directory also pass.

## Contract and lifetime

Custom domain `jetlink.qnn` v1, operator `QnnVisionPipeline`: one FP16 input
`[1,24,128,256]`, one FP16 output `[1,32,512]`. The source's exact uint8-to-FP16
cast `_to_copy_4` is retained; the QNN boundary is named `cat_2`. No transpose or
reinterpretation is added. The saved graph is `vision_merged`, output `view_39`.
The builder replaces exactly 421 source nodes, preserves the remainder, all four
outputs/states and all three `view_39` consumers, and runs ONNX checker.

Dense, fixed, FP16 descriptors and finite input/output values are checked.
The system handle and binary storage remain alive until backend context release,
so shallow descriptor pointers remain valid. RAII cleans partial initialization.
QNN initialization, execution and destruction are serialized by one mutex.
Initialization is lazy; first valid inference includes initialization. Errors fail
the ORT invocation instead of silently falling back to CPU.

Android requires explicit paths, uses HTP package target, and remains unqualified.
Linux host uses CPU package target. Never interpret simulator elapsed time as
phone inference performance.

## Portable reproduction

Provide your licensed SDK/context/weights and existing Python environment.
No software installation or vendor files are included. Set these variables to
absolute local paths (the example labels below must be replaced):

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.42.0.251225
export ORT_HEADERS=/path/to/ort/include
export PYTHON=/path/to/python
export PYTHONPATH=/path/to/python/dependencies
export QNN_EXTRA_LIBRARY_PATH=/path/to/host/dependencies:/path/to/hexagon/tools/lib
export JETLINK_VENDOR=/path/to/jetlink/repository
export JETLINK_ROAD=/path/to/recorded-night/inputs320
export JETLINK_SPLIT_CONTROL=/path/to/native-first-frame/native
export JETLINK_VISION_CONTEXT=/path/to/vision/context/vision.bin
SOURCE=/path/to/teacher-compact-boundaries.onnx
SCALAR_PACKAGE=/path/to/libJetlinkGeluHostHidden.so
VECTOR_PACKAGE=/path/to/vector/libJetlinkGeluHost.so
```

Run from this bridge directory. QNN_EXTRA_LIBRARY_PATH is only needed where the
host requires extra SDK dependencies. QNN_HOST_LIB_DIR, JETLINK_QNN_BACKEND and
JETLINK_QNN_SYSTEM can override library discovery. `--road`, `--vendor`,
`--split-control` and `--context` override corresponding JETLINK_* settings.

```bash
bash build.sh
"$PYTHON" ../build_model.py --source "$SOURCE" --output ./model-new
JETLINK_SOURCE_MODEL="$SOURCE" "$PYTHON" -m unittest discover -s .. -p test_build_model.py
"$PYTHON" test_bridge.py

# Each command below executes one full native model; each output must be fresh.
JETLINK_GELU_PACKAGE="$SCALAR_PACKAGE" bash run_full_frame.sh \
  --model ./model-new/vision-native.onnx \
  --bridge-library ./libjetlink_vision_bridge.so \
  --weights ./model-new/teacher-capture.weights --output ./scalar-reproduction
JETLINK_GELU_PACKAGE="$VECTOR_PACKAGE" bash run_full_frame.sh \
  --model ./model-new/vision-native.onnx \
  --bridge-library ./libjetlink_vision_bridge.so \
  --weights ./model-new/teacher-capture.weights --output ./vector-reproduction
```

The model builder symlinks the original local external weight file, refuses an
existing output directory, and does not modify source files. The full-frame
runner likewise refuses existing outputs before creating an ORT session. It
hashes selected package, backend/system libraries, model, context, external weights,
road inputs and split-control arrays. It does not retry inference automatically.

To reproduce only the saved-array CPU checker (no inference):

```bash
PYTHONPATH="$JETLINK_VENDOR:$PYTHONPATH" "$PYTHON" \
  "$JETLINK_VENDOR/scripts/verify_parity.py" compare \
  --dir ./vector-full-frame-check/cpu-baseline-check
```

`test_contract.cpp` is a standalone source-only descriptor test: compile with the
same include flags as build.sh plus `-ldl -pthread`, then execute it. No QNN
library is loaded by that test. `test_bridge.py --valid` is a separate opt-in
boundary-only comparison; it was not invoked for the full-model validation.

## Source-only distribution

Include only authored `bridge.cpp`, `build.sh`, `test_bridge.py`,
`test_contract.cpp`, `test_full_frame.py`, `run_full_frame.sh`, `README.md`, and
parent `build_model.py` / `test_build_model.py`. Do not include generated ONNX
models, weights, context binaries, SDK headers, shared objects or test executables.
JSON/log evidence may be included separately after checking its local paths.
