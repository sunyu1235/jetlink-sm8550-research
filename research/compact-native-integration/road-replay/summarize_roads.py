"""Summarize native QNN compact contexts inside the full CPU-remainder model."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
P=Path(__file__).parent
B=Path('/workspace/shared/jetlink-compact-native-bridge')
M=Path('/workspace/shared/jetlink-distillation/real-block/teacher-native-compact.onnx')
q=json.loads((P/'qualification.json').read_text())
assert hashlib.sha256(M.read_bytes()).hexdigest()==q['model_sha256']
for p,h in json.loads((B/'artifact-hashes.json').read_text()).items():
 assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h['sha256'],p
r={'scope':'Actual saved native QNN compact upstream and three-tail contexts inside full recurrent model; CPU remainder. Offline recorded-road numerical validation, not driving safety or phone speed.','status':'PENDING_ROAD_GATES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':q['model_sha256'],'independent_recurrence':True,'native_contexts':2,'compact_shared_fp16_boundary_bytes':9216,'all_runtime_artifact_hashes_unchanged':True,'device_execution':False,'device_latency_measured':False,'road_gates':{},'outputs':{}}
for road in ['night','day']:
 for n in [32,320]:
  path=P/f'native-{road}-{n}-score/summary.json'
  if path.exists():
   s=json.loads(path.read_text());r['road_gates'][f'{road}{n}']=s
 for c in [P/f'native-{road}-320/out_link-checkpoint.npz']:
  if c.exists():
   with np.load(c,allow_pickle=False) as z:frames=int(z['frame'])+1
   h=hashlib.sha256()
   for i in range(frames):
    p=c.parent/f'out_link_{i}.npy';v=np.load(p,allow_pickle=False)
    assert v.shape==(18452,) and np.isfinite(v).all()
    h.update(p.read_bytes())
   r['outputs'][road]={'durable_frames':frames,'ordered_outputs_sha256':h.hexdigest()}
failed=any(v['checker_exit']!=0 for s in r['road_gates'].values() for v in s['views'].values())
if failed:r['status']='FAIL_NATIVE_COMPACT_ROAD_GATE'
elif len(r['road_gates'])==4:r['status']='PASS_NATIVE_COMPACT_CONTEXTS_CPU_REMAINDER_NIGHT320_DAY320'
elif all(k in r['road_gates'] for k in ['night32','day32']):r['status']='PASS_NATIVE_COMPACT_PREFIX32_BOTH_ROADS_FULL320_PENDING'
(P/'road-qualification.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['status','outputs','device_execution','device_latency_measured']},indent=2))
