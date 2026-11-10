"""Replay saved input arrays with independent student state; no target speed claim."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
import numpy as np,onnxruntime as ort
from jetlink.spec import ModelSpec
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,type=Path);ap.add_argument('--inputs',required=True,type=Path);ap.add_argument('--references',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--frames',type=int,default=160);a=ap.parse_args()
if a.out.exists():raise FileExistsError('refuse to overwrite replay evidence')
a.out.mkdir(parents=True);spec=ModelSpec.load(a.inputs/'spec.json');shutil.copyfile(a.inputs/'spec.json',a.out/'spec.json')
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(str(a.model),opt,providers=['CPUExecutionProvider']);shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};dtypes={i.name:dt[i.type] for i in sess.get_inputs()};names=[o.name for o in sess.get_outputs()]
state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs};timings=[]
for i in range(a.frames):
 warped=np.load(a.inputs/f'in_warped_{i}.npy');packed=np.load(a.inputs/f'in_packed_{i}.npy')
 feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}};feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()}
 t=time.monotonic();r=dict(zip(names,sess.run(None,feed)));elapsed=time.monotonic()-t
 if not all(np.isfinite(v).all() for v in r.values()):raise ValueError(f'nonfinite output frame {i}')
 state={n:r[nxt] for n,nxt in spec.state_pairs.items()};np.save(a.out/f'out_link_{i}.npy',r['outputs'].reshape(-1));shutil.copyfile(a.references/f'out_ref_{i}.npy',a.out/f'out_ref_{i}.npy');timings.append(elapsed)
 print(i,elapsed,flush=True)
(a.out/'replay.json').write_text(json.dumps({'status':'CPU_REPLAY_NOT_TARGET_LATENCY','frames':a.frames,'independent_student_recurrence':True,'model':str(a.model),'inputs':str(a.inputs),'references':str(a.references),'seconds_per_frame':timings},indent=2))
