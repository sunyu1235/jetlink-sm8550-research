"""Replay one candidate against archived references, with its own recurrent state."""
import argparse,json
from pathlib import Path
import numpy as np
from replay_qdq_whole import build,replay,ROOT,parity,check
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/workspace/shared/jetlink-models/light-speed.onnx'));p.add_argument('--inputs',type=Path,default=Path('/workspace/shared/jetlink-history-regenerated'));p.add_argument('--reference-dir',type=Path,default=ROOT/'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare');p.add_argument('--qdq',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--frames',default=16,type=int);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
path=a.out/'candidate.onnx';build(a.source,a.qdq,path)
spec=parity.ModelSpec.load(a.inputs/'spec.json')
y,t=replay(path,a.inputs,spec,a.frames,a.out/'outputs')
refdir=a.reference_dir
refs=[np.load(refdir/f'out_ref_{i}.npy').reshape(-1) for i in range(a.frames)]
code,log=check(spec,y,refs);(a.out/'parity.log').write_text(log)
report={'status':'CPU_RECURRENT_SYNTHETIC_REPLAY_NOT_DEVICE_SPEED','frames':a.frames,'checker_exit':code,'verdict':log.strip().splitlines()[-1],'seconds':t}
(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
