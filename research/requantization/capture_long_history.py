#!/usr/bin/env python3
"""Capture long-history teacher block activations on independent synthetic seeds.
No road-representativeness claim. Full model uses the recorded layout-only lowering.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from boundary_sensitivity import parity
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,default=Path('/workspace/shared/jetlink-distillation/real-block/teacher-capture.onnx'));p.add_argument('--spec',type=Path,default=Path('/workspace/shared/jetlink-history-regenerated/spec.json'));p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=160);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
model=a.model
o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;o.log_severity_level=3
s=ort.InferenceSession(str(model),o,providers=['CPUExecutionProvider'])
spec=parity.ModelSpec.load(a.spec)
dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};inputs={x.name:(x.shape,dt[x.type]) for x in s.get_inputs()};names=[v.name for v in s.get_outputs()]
result={};timings=[]
for split,seed in [('train',713),('validation',1777)]:
 state={n:np.zeros(inputs[n][0],inputs[n][1]) for n in spec.state_pairs};xs=[];ys=[];outputs=[]
 for i,(warped,packed) in enumerate(parity.make_inputs(spec,a.frames,seed=seed)):
  feed={'new_img':warped,**state,**{k:packed[at] for k,(at,_) in spec.packed_layout.items()}}
  feed={k:np.asarray(v,dtype=inputs[k][1]).reshape(inputs[k][0]) for k,v in feed.items()}
  t=time.monotonic();r=dict(zip(names,s.run(None,feed),strict=True));elapsed=time.monotonic()-t
  if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite teacher activation')
  state={k:r[nxt] for k,nxt in spec.state_pairs.items()};xs.append(r['layer_norm_49']);ys.append(r['linear_99']);outputs.append(r['outputs'])
  timings.append({'split':split,'frame':i,'seconds':elapsed});print(split,i,round(elapsed,3),flush=True)
 result['x_'+split]=np.stack(xs);result['y_'+split]=np.stack(ys)
 np.save(a.out/f'{split}-teacher-outputs.npy',np.stack(outputs))
 np.savez(a.out/f'{split}.npz',x=result['x_'+split],y=result['y_'+split])
np.savez(a.out/'sequences.npz',**result)
report={'status':'REAL_MODEL_LONG_HISTORY_SYNTHETIC_CAPTURE_NOT_ROAD_CALIBRATION','seeds':{'train':713,'validation':1777},'frames_each':a.frames,'history_feature_capacity':128,'independent_reset':True,'original_acceptance_seed_used_for_training':False,'timings':timings,'ort':ort.__version__,'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
(a.out/'capture.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)
