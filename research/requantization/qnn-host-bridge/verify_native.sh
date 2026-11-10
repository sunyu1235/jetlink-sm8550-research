#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
LIMIT=1 RESULTS=nearest-single.json JETLINK_DUMP_INPUT="$PWD/input-nearest.raw" JETLINK_DUMP_OUTPUT="$PWD/output-nearest.raw" bash run.sh test_bridge.py
printf '%s\n' "$PWD/input-nearest.raw" > input_nearest_list.txt
bash /workspace/shared/jetlink-exact-integration/run_qnn_tool.sh qnn-net-run --model /workspace/shared/jetlink-exact-integration/model-libs-native/x86_64-linux-clang/libqnn-first-projection-native.so --backend libQnnHtp.so --config_file /workspace/shared/jetlink-exact-integration/htp-backend-extensions.json --input_list "$PWD/input_nearest_list.txt" --use_native_input_files --use_native_output_files --output_dir "$PWD/netrun-nearest"
PYTHONPATH=/workspace/shared/jetlink-python python - <<'PY'
import numpy as np,json
from pathlib import Path
p=Path.cwd()
a=np.fromfile(p/'output-nearest.raw',np.uint16);b=np.fromfile(p/'netrun-nearest/Result_0/val_646_DequantizeLinear_Output_native.raw',np.uint16)
r={'elements':a.size,'native_u16_mismatches':int(np.count_nonzero(a!=b))};print(r);json.dump(r,open(p/'native-parity.json','w'),indent=2);assert r['native_u16_mismatches']==0
PY
