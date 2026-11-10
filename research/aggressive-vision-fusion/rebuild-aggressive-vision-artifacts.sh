#!/usr/bin/env bash
# Build local research artifacts from the published source and locally licensed tools.
# Does not contact a device, run full-model inference, or upload any files.
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; exit 1; }
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || fail 'Requires Linux x86_64.'

: "${REPO_ROOT:?Set REPO_ROOT to the jetlink-sm8550-research checkout}"
: "${ORIGINAL_MODEL:?Set ORIGINAL_MODEL to the original embedded-weight Light Speed ONNX}"
: "${WORK_DIR:?Set WORK_DIR to a NEW absolute output directory}"
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to licensed QAIRT 2.42.0.251225}"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT to licensed Hexagon SDK 6.4.0.2}"
: "${HEXAGON_TOOLS_ROOT:?Set HEXAGON_TOOLS_ROOT to licensed Tools 19.0.04/Tools}"
: "${NATIVE_TOOLS:?Set NATIVE_TOOLS to licensed native21.0.04/Tools}"
: "${HOST_TOOLS_ROOT:?Set HOST_TOOLS_ROOT to LLVM19 installation prefix; use / for Debian system layout}"

[[ "$WORK_DIR" = /* ]] || fail 'WORK_DIR must be an absolute path.'
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || fail "Refusing existing WORK_DIR: $WORK_DIR"
REPO_ROOT=$(realpath -e "$REPO_ROOT")
ORIGINAL_MODEL=$(realpath -e "$ORIGINAL_MODEL")
WORK_DIR=$(realpath -m "$WORK_DIR")
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || fail "Refusing existing WORK_DIR: $WORK_DIR"
for variable in QNN_SDK_ROOT HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT NATIVE_TOOLS HOST_TOOLS_ROOT; do
  resolved=$(realpath -e "${!variable}")
  printf -v "$variable" '%s' "$resolved"
  export "$variable"
done
case "${RUN_PACKAGE_TESTS:-1}" in 0|1) ;; *) fail 'RUN_PACKAGE_TESTS must be 0 or 1.' ;; esac
F="$REPO_ROOT/research/aggressive-vision-fusion"
P="$REPO_ROOT/experiments/gelu-polynomial"
QBIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
export HOST_CXX=${HOST_CXX:-$HOST_TOOLS_ROOT/usr/lib/llvm-19/bin/clang++}
export PATH="$HOST_TOOLS_ROOT/usr/lib/llvm-19/bin:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$HOST_TOOLS_ROOT/usr/lib/x86_64-linux-gnu:$HOST_TOOLS_ROOT/usr/lib/llvm-19/lib:$HEXAGON_TOOLS_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for file in "$F/preparation/convert_preserve_arithmetic.py" "$F/preparation/map_model.py" \
  "$F/preparation/promote_images.py" "$F/preparation/repair_preserved_bias.py" \
  "$F/preparation/Gelu.xml" "$F/extract_region.py" \
  "$P/qnn-package/build_v73.sh" "$P/qnn-package/build_host.sh" \
  "$QNN_SDK_ROOT/examples/QNN/OpPackage/HTP/ExampleOpPackageInterface.cpp" \
  "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so" \
  "$NATIVE_TOOLS/libnative/include/hexagon_types.h" \
  "$NATIVE_TOOLS/libnative/lib/libnative.a"; do
  test -r "$file" || { echo "Missing required input: $file" >&2; exit 1; }
done
for file in "$HEXAGON_TOOLS_ROOT/bin/clang-19" "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++" \
  "$HOST_CXX" "$QBIN/qnn-context-binary-generator"; do
  test -x "$file" || { echo "Missing executable: $file. Tools8.7.06 is not a drop-in substitute." >&2; exit 1; }
done
test -r "$QBIN/qnn-model-lib-generator"
test -r "$QBIN/qnn-onnx-converter"
command -v python >/dev/null
command -v gcc >/dev/null
command -v readelf >/dev/null
python - <<'PY'
import sys, onnx, numpy
from qti.aisw.converters.common.converter_ir import op_graph_optimizations as opt
assert hasattr(opt, 'squash_node_into_nn_node'), 'Wrong converter API; do not remove the arithmetic-preservation hook.'
print('Python:',sys.version.split()[0], 'ONNX:',onnx.__version__, 'NumPy:',numpy.__version__)
PY
EXPECTED=6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21
ACTUAL=$(sha256sum "$ORIGINAL_MODEL" | cut -d' ' -f1)
test "$ACTUAL" = "$EXPECTED" || { echo "Original model SHA256 mismatch: $ACTUAL" >&2; exit 1; }

# Create the top-level directory without -p: an existing output is never reused.
mkdir "$WORK_DIR"
mkdir "$WORK_DIR/models" "$WORK_DIR/preparation"
cp --reflink=auto "$ORIGINAL_MODEL" "$WORK_DIR/models/light-speed.onnx"
G="$WORK_DIR/gelu-polynomial"
mkdir -p "$G/qnn-package"
# Copy only build inputs, not stale binaries or a previously generated licensed
# interface. The interface is regenerated from this QNN installation below.
PACKAGE_FILES=(
  build_reference.sh build_test.sh reference.c manifest.json test_core.cpp
  core.cpp core.h vector_core.h coefficients.h
  qnn-package/GeluFp16.cpp qnn-package/build_host.sh qnn-package/build_v73.sh
  qnn-package/coefficients.h qnn-package/compact_gelu.h qnn-package/core.cpp
  qnn-package/core.h qnn-package/gelu_tables.h qnn-package/prepare_interface.py
  qnn-package/run_tests.sh qnn-package/test_layout_matrix.py
  qnn-package/test_qnn_gelu.c qnn-package/vector_core.h
)
for file in "${PACKAGE_FILES[@]}"; do
  [[ -f "$P/$file" && ! -L "$P/$file" ]] || fail "Missing or symlinked package input: $P/$file"
  cp "$P/$file" "$G/$file"
done
(cd "$G" && sha256sum "${PACKAGE_FILES[@]}") > "$WORK_DIR/package-source-SHA256SUMS"

# The target script compiles but does not execute the DSP library.
sh "$G/qnn-package/build_v73.sh" >"$WORK_DIR/build-v73.log" 2>&1
# The host build performs host dlopen checks. It is needed for offline context preparation.
sh "$G/qnn-package/build_host.sh" >"$WORK_DIR/build-host.log" 2>&1

# Recommended correctness gates for this local compiler/libm combination.
# These do not execute on a phone and are not performance measurements.
if [ "${RUN_PACKAGE_TESTS:-1}" = 1 ]; then
  sh "$G/build_reference.sh" >"$WORK_DIR/reference-test.log" 2>&1
  export GELU_REFERENCE="$G/build/reference.bin"
  sh "$G/build_test.sh" >"$WORK_DIR/core-test.log" 2>&1
  sh "$G/qnn-package/run_tests.sh"
  python "$G/qnn-package/test_layout_matrix.py" >"$WORK_DIR/layout-tests.log" 2>&1
fi

python "$F/preparation/map_model.py" "$WORK_DIR/models/light-speed.onnx" \
  "$WORK_DIR/models/mapped.onnx" --report "$WORK_DIR/map.json"
python "$F/preparation/promote_images.py" "$WORK_DIR/models/mapped.onnx" \
  "$WORK_DIR/models/images16.onnx" --report "$WORK_DIR/images.json"
python "$F/preparation/convert_preserve_arithmetic.py" \
  --input_network "$WORK_DIR/models/images16.onnx" \
  --output_path "$WORK_DIR/preparation/preserve.cpp" \
  --op_package_config "$F/preparation/Gelu.xml" \
  --float_bitwidth 16 --float_bias_bitwidth 16 >"$WORK_DIR/convert.log" 2>&1
python "$F/preparation/repair_preserved_bias.py" --directory "$WORK_DIR/preparation"
python "$F/extract_region.py" --source "$WORK_DIR/preparation" \
  --output "$WORK_DIR/vision" --inputs cat_2 --outputs view_39 --name vision_merged

python - "$WORK_DIR/vision" "$QNN_SDK_ROOT" <<'PY'
import json,sys
from pathlib import Path
p,q=map(Path,sys.argv[1:])
(p/'config.json').write_text(json.dumps({'graphs':[{'graph_names':['vision_merged'],'O':3}],
  'devices':[{'soc_model':43,'dsp_arch':'v73'}]}))
(p/'extensions.json').write_text(json.dumps({'backend_extensions':{
  'shared_library_path':str(q/'lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so'),
  'config_file_path':str(p/'config.json')}}))
PY
python "$QBIN/qnn-model-lib-generator" -c "$WORK_DIR/vision/region.cpp" \
  -b "$WORK_DIR/vision/region.bin" -o "$WORK_DIR/vision/libs" \
  -t x86_64-linux-clang >"$WORK_DIR/model-build.log" 2>&1
"$QBIN/qnn-context-binary-generator" \
  --model "$WORK_DIR/vision/libs/x86_64-linux-clang/libregion.so" \
  --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  --op_packages "$G/qnn-package/build/libJetlinkGeluHost.so:exampleInterfaceProvider:CPU" \
  --config_file "$WORK_DIR/vision/extensions.json" \
  --binary_file vision --output_dir "$WORK_DIR/vision/context" \
  --log_level info >"$WORK_DIR/context-build.log" 2>&1

CONTEXT="$WORK_DIR/vision/context/vision.bin"
TARGET="$G/qnn-package/build/libJetlinkGeluV73.so"
test -s "$CONTEXT" && test -s "$TARGET"
sha256sum "$CONTEXT" "$TARGET" | tee "$WORK_DIR/SHA256SUMS"
stat -c '%n %s bytes' "$CONTEXT" "$TARGET"
readelf -h "$TARGET" | grep -E 'Class:|Machine:|Flags:'
printf '\nCreated local artifacts. No Android loading, recurrent acceptance, or target performance was tested.\n'
printf 'New hashes need their own provenance; byte-identical rebuilds are not promised.\n'
