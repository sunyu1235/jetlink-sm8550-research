#!/usr/bin/env bash
# Explicitly host-functional. Does not contact a phone or install software.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 SOURCE_QNN_DIR NEW_OUTPUT_DIR INPUT_LIST QNN_TOOL_LAUNCHER GELU_HOST_PACKAGE" >&2
  exit 2
fi
SOURCE=$(realpath "$1")
OUT=$(realpath -m "$2")
INPUTS=$(realpath "$3")
TOOL=$(realpath "$4")
PACKAGE=$(realpath "$5")
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the licensed QAIRT/QNN 2.42 SDK}"
Q=$(realpath "$QNN_SDK_ROOT")
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
test ! -e "$OUT" || { echo "Refusing existing output directory" >&2; exit 1; }
test -r "$PACKAGE" && test -r "$INPUTS" && test -r "$TOOL"
"$PYTHON" "$HERE/extract_region.py" --source "$SOURCE" --output "$OUT" \
  --inputs cat_2 --outputs view_39 --name vision_merged
"$PYTHON" - "$OUT" "$Q" <<'PY'
from pathlib import Path
import json,sys
p,q=map(Path,sys.argv[1:])
(p/'config.json').write_text(json.dumps({'graphs':[{'graph_names':['vision_merged'],'O':3}],
    'devices':[{'soc_model':43,'dsp_arch':'v73'}]}))
(p/'extensions.json').write_text(json.dumps({'backend_extensions':{
    'shared_library_path':str(q/'lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so'),
    'config_file_path':str(p/'config.json')}}))
PY
bash "$TOOL" qnn-model-lib-generator -c "$OUT/region.cpp" -b "$OUT/region.bin" \
  -o "$OUT/libs" -t x86_64-linux-clang > "$OUT/build.log" 2>&1
bash "$TOOL" qnn-context-binary-generator --model "$OUT/libs/x86_64-linux-clang/libregion.so" \
  --backend "$Q/lib/x86_64-linux-clang/libQnnHtp.so" \
  --op_packages "$PACKAGE:exampleInterfaceProvider:CPU" \
  --config_file "$OUT/extensions.json" --binary_file vision --output_dir "$OUT/context" \
  --log_level info > "$OUT/prepare.log" 2>&1
bash "$TOOL" qnn-net-run --retrieve_context "$OUT/context/vision.bin" \
  --backend "$Q/lib/x86_64-linux-clang/libQnnHtp.so" \
  --op_packages "$PACKAGE:exampleInterfaceProvider:CPU" \
  --input_list "$INPUTS" --use_native_input_files --use_native_output_files \
  --output_dir "$OUT/outputs" --log_level info > "$OUT/run.log" 2>&1
echo "Host context execution completed; no target latency or numerical acceptance inferred."
