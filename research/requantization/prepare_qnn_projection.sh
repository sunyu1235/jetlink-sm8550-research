#!/usr/bin/env bash
# Requires licensed SDK, working launcher, QDQ projection ONNX, and FLOAT32 raw inputs.
# No network, phone execution, or vendor redistribution is performed.
set -euo pipefail
if [[ $# != 5 ]]; then echo 'Usage: prepare_qnn_projection.sh LAUNCHER QNN_SDK PROJECTION_ONNX INPUT_LIST NEW_OUTPUT_DIR' >&2; exit 2; fi
launcher=$(realpath "$1"); sdk=$(realpath "$2"); projection=$(realpath "$3"); inputs=$(realpath "$4"); out=$5
[[ ! -e "$out" ]] || { echo 'Output must be new' >&2; exit 2; }
mkdir -p "$out"; out=$(realpath "$out")
bash "$launcher" qnn-onnx-converter --input_network "$projection" --output_path "$out/qnn-first-projection-native.cpp" --input_list "$inputs" --act_bitwidth 16 --weights_bitwidth 8 --input_layout layer_norm_49 NONTRIVIAL --preserve_io layout --keep_weights_quantized --no_simplification 2>&1 | tee "$out/converter.log"
bash "$launcher" qnn-model-lib-generator -c "$out/qnn-first-projection-native.cpp" -b "$out/qnn-first-projection-native.bin" -o "$out/model-libs" -t x86_64-linux-clang 2>&1 | tee "$out/model-build.log"
python - "$out" "$sdk" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]);s=Path(sys.argv[2]);(o/'htp.json').write_text(json.dumps({'graphs':[{'graph_names':['qnn_first_projection_native'],'O':3}],'devices':[{'soc_model':43,'dsp_arch':'v73'}]}));(o/'extensions.json').write_text(json.dumps({'backend_extensions':{'shared_library_path':str(s/'lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so'),'config_file_path':str(o/'htp.json')}}))
PY
bash "$launcher" qnn-context-binary-generator --model "$out/model-libs/x86_64-linux-clang/libqnn-first-projection-native.so" --backend "$sdk/lib/x86_64-linux-clang/libQnnHtp.so" --config_file "$out/extensions.json" --binary_file first-projection-sm8550 --output_dir "$out/context" --log_level verbose 2>&1 | tee "$out/context-build.log"
echo 'Offline single-projection preparation only. No whole-model accuracy or device latency claim.'
