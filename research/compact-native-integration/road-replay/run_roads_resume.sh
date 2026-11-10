#!/bin/bash
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
B=/workspace/shared/jetlink-compact-native-bridge
R=/workspace/scratch/a607d96f1cc1/htp-work/research/requantization
J=/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink
M=/workspace/shared/jetlink-distillation/real-block/teacher-native-compact.onnx
L=$B/libjetlink_compact_bridge.so
exec 9>"$P/replay.lock"; flock -n 9 || { echo 'Another replay active'; exit 1; }
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/workspace/shared/jetlink-python:$R"
echo "RECOVERY START $(date -u +%FT%TZ)"
python - "$B/artifact-hashes.json" "$P/qualification.json" "$M" <<'PY'
import hashlib,json,sys
for p,v in json.load(open(sys.argv[1])).items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==v['sha256'],p
q=json.load(open(sys.argv[2]));assert q['status']=='PASS_NATIVE_COMPACT_CONTEXTS_CPU_REMAINDER_SYNTHETIC160'
assert hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()==q['model_sha256']
print('Artifact hashes verified',flush=True)
PY
for frames in 32 320; do
 for road in night day; do
  if [[ $road == night ]]; then I=/workspace/shared/jetlink-public-road/inputs320; T=/workspace/shared/jetlink-distillation/real-block/road320-teacher-capture; else I=/workspace/shared/jetlink-day-road/inputs320; T=/workspace/shared/jetlink-distillation/policy12/day320-teacher; fi
  O="$P/native-$road-320"; S="$P/native-$road-$frames-score"
  if [[ -e "$S/summary.json" ]]; then
   python - "$S/summary.json" <<'PY'
import json,sys
q=json.load(open(sys.argv[1]));assert all(v['checker_exit']==0 for v in q['views'].values()),'Existing gate failed; inspect before rerun'
print('Existing passed gate:',sys.argv[1],flush=True)
PY
   continue
  fi
  START=$(python - "$O" <<'PY'
import pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1]); c=p/'out_link-checkpoint.npz'
if c.exists():
 with np.load(c,allow_pickle=False) as z: print(int(z['frame'])+1)
else:
 assert not any(p.glob('out_link_*.npy')),'Output files without checkpoint; explicit recovery required'
 print(0)
PY
)
  echo "$road gate=$frames resume_frame=$START $(date -u +%FT%TZ)"
  if (( START < frames )); then
   ARGS=(); (( START == 0 )) || ARGS+=(--resume)
   bash "$B/run.sh" "$P/replay_checkpoint4.py" --model "$M" --inputs "$I" --out "$O" --prefix out_link --frames "$frames" --library "$L" "${ARGS[@]}" >> "$P/native-$road-$frames-recovery.log" 2>&1
  fi
  if [[ $frames == 32 ]]; then W=16; else W=160; fi
  PYTHONPATH=/workspace/shared/jetlink-python python "$R/score_recorded_replay.py" --teacher "$T" --student "$O" --out "$S" --jetlink "$J" --frames "$frames" --warmup "$W"
  python - "$S/summary.json" <<'PY'
import json,sys
assert all(v['checker_exit']==0 for v in json.load(open(sys.argv[1]))['views'].values()),'STOP: native road gate failed'
print('PASS',sys.argv[1],flush=True)
PY
 done
done
echo "ALL ROAD GATES COMPLETE $(date -u +%FT%TZ)"
