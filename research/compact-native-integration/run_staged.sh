#!/bin/bash
# Run only after bridge unit and official-native parity tests pass.
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
B=/workspace/shared/jetlink-compact-native-bridge
R=/workspace/scratch/a607d96f1cc1/htp-work/research/requantization
J=/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink
T=/workspace/scratch/a607d96f1cc1/htp-work/reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare
MODEL=/workspace/shared/jetlink-distillation/real-block/teacher-native-compact.onnx
: "${BRIDGE_LIBRARY:?Set verified bridge shared library}"
: "${BRIDGE_LAUNCHER:?Set verified bridge Python launcher}"
exec 9>"$P/replay.lock"
flock -n 9 || { echo 'Replay already active'; exit 1; }
[[ ! -e "$P/synthetic160" ]] || { echo 'Existing run: inspect checkpoint and outcome before resuming'; exit 1; }
export OPENBLAS_NUM_THREADS=1
bash "$BRIDGE_LAUNCHER" "$R/replay_checkpoint.py" --model "$MODEL" --inputs /workspace/shared/jetlink-history-regenerated --out "$P/synthetic160" --prefix out_link --frames 4 --library "$BRIDGE_LIBRARY" > "$P/synthetic4.log" 2>&1
PYTHONPATH=/workspace/shared/jetlink-python python "$R/score_recorded_replay.py" --teacher "$T" --student "$P/synthetic160" --out "$P/synthetic4-score" --jetlink "$J" --frames 4 --warmup 2
python - "$P/synthetic4-score/summary.json" <<'PYGATE'
import json,sys
s=json.load(open(sys.argv[1]));assert all(v['checker_exit']==0 for v in s['views'].values()),'STOP: native smoke numerical gate failed'
PYGATE
bash "$BRIDGE_LAUNCHER" "$R/replay_checkpoint.py" --model "$MODEL" --inputs /workspace/shared/jetlink-history-regenerated --out "$P/synthetic160" --prefix out_link --frames 32 --library "$BRIDGE_LIBRARY" --resume > "$P/synthetic32.log" 2>&1
PYTHONPATH=/workspace/shared/jetlink-python python "$R/score_recorded_replay.py" --teacher "$T" --student "$P/synthetic160" --out "$P/synthetic32-score" --jetlink "$J" --frames 32 --warmup 16
python - "$P/synthetic32-score/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));assert all(v['checker_exit']==0 for v in s['views'].values()),'STOP: native prefix numerical gate failed'
PY
bash "$BRIDGE_LAUNCHER" "$R/replay_checkpoint.py" --model "$MODEL" --inputs /workspace/shared/jetlink-history-regenerated --out "$P/synthetic160" --prefix out_link --frames 160 --library "$BRIDGE_LIBRARY" --resume > "$P/synthetic160.log" 2>&1
PYTHONPATH=/workspace/shared/jetlink-python python "$R/score_recorded_replay.py" --teacher "$T" --student "$P/synthetic160" --out "$P/synthetic160-score" --jetlink "$J" --frames 160 --warmup 80
