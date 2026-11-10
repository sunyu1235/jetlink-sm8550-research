#!/usr/bin/env bash
# Compile two compatible compact-interface contexts; never execute on a device.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: compile_contexts.sh UPSTREAM_ONNX TAILS_HTP_ONNX NEW_OUTPUT_DIR'; exit 2; }
: "${PRESERVE_CONVERTER:?Set preserve-arithmetic converter launcher}"
: "${QNN_LAUNCHER:?Set working QNN tool launcher}"
: "${QNN_SDK_ROOT:?Set licensed SDK root}"
: "${GELU_XML:?Set custom GELU op XML}"
: "${GELU_HOST_SO:?Set matching custom GELU host library}"
up=$(realpath "$1"); tails=$(realpath "$2"); out=$3
[[ ! -e "$out" ]] || { echo 'Refusing to overwrite output'; exit 2; }; mkdir -p "$out"; out=$(realpath "$out")
for name in upstream tails; do
 model=$up; args=(); packages=()
 if [[ $name == tails ]]; then model=$tails; args=(--op_package_config "$GELU_XML"); packages=(--op_packages "$GELU_HOST_SO:exampleInterfaceProvider:CPU"); fi
 bash "$PRESERVE_CONVERTER" --input_network "$model" --output_path "$out/$name.cpp" "${args[@]}" --float_bitwidth 16 --float_bias_bitwidth 16 --no_simplification --preserve_io layout 2>&1 | tee "$out/$name-convert.log"
 bash "$QNN_LAUNCHER" qnn-model-lib-generator -c "$out/$name.cpp" -b "$out/$name.bin" -o "$out/libs-$name" -t x86_64-linux-clang 2>&1 | tee "$out/$name-build.log"
 python - "$out" "$name" "$QNN_SDK_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);n=sys.argv[2];s=Path(sys.argv[3]);(p/(n+'-htp.json')).write_text(json.dumps({'graphs':[{'graph_names':[n],'O':3}],'devices':[{'soc_model':43,'dsp_arch':'v73'}]}));(p/(n+'-ext.json')).write_text(json.dumps({'backend_extensions':{'shared_library_path':str(s/'lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so'),'config_file_path':str(p/(n+'-htp.json'))}}))
PY
 bash "$QNN_LAUNCHER" qnn-context-binary-generator --model "$out/libs-$name/x86_64-linux-clang/lib$name.so" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" "${packages[@]}" --config_file "$out/$name-ext.json" --binary_file "$name" --output_dir "$out/context-$name" --log_level verbose 2>&1 | tee "$out/$name-prepare.log"
done
echo 'Offline preparation only. Validate native ABI and numerical behavior before deployment.'
