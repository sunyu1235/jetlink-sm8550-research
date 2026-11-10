"""Full-model numerical replay with the exact CPU GELU library registered."""
import argparse,json
from pathlib import Path
import numpy as np
from replay_qdq_whole import replay,ROOT,parity,check
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--library',type=Path,required=True,action='append');p.add_argument('--inputs',type=Path,default=Path('/workspace/shared/jetlink-history-regenerated'));p.add_argument('--reference-dir',type=Path,default=ROOT/'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare');p.add_argument('--frames',type=int,default=160);p.add_argument('--warmup',type=int,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args();
if not 0<=a.warmup<a.frames:raise ValueError('warmup must be below frames')
a.out.mkdir(exist_ok=False,parents=True)
spec=parity.ModelSpec.load(a.inputs/'spec.json');y,t=replay(a.model,a.inputs,spec,a.frames,a.out/'outputs',custom_library=a.library)
refs=[np.load(a.reference_dir/f'out_ref_{i}.npy').reshape(-1) for i in range(a.frames)]
c,l=check(spec,y[a.warmup:],refs[a.warmup:]);(a.out/'parity.log').write_text(l);r={'scope':'RECURRENT_HOST_NUMERICAL_TEST_ONLY_NOT_DEVICE_SPEED','libraries':[str(v) for v in a.library],'model':str(a.model),'inputs':str(a.inputs),'reference_dir':str(a.reference_dir),'frames':a.frames,'warmup':a.warmup,'scored_frames':a.frames-a.warmup,'checker_exit':c,'verdict':l.strip().splitlines()[-1],'seconds':t};(a.out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
