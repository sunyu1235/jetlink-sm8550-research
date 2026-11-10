"""Offline recorded-image inference. Independent recurrent state; no vehicle control."""
import argparse,json,time,hashlib,os
from pathlib import Path
import numpy as np,onnxruntime as ort
from jetlink.spec import ModelSpec
ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',choices=['out_ref','out_link'],required=True);ap.add_argument('--frames',type=int,default=320);ap.add_argument('--capture-block',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
if not a.resume and any(a.out.glob(a.prefix+'_*.npy')):raise FileExistsError('refuse to overwrite existing outputs')
spec=ModelSpec.load(a.inputs/'spec.json');(a.out/'spec.json').write_bytes((a.inputs/'spec.json').read_bytes())
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(str(a.model),opt,providers=['CPUExecutionProvider']);shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};dtypes={i.name:dt[i.type] for i in sess.get_inputs()};names=[o.name for o in sess.get_outputs()]
state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs};timings=[];start=0
fingerprint=hashlib.sha256(a.model.read_bytes()+(a.inputs/'spec.json').read_bytes()).hexdigest()
checkpoint=a.out/(a.prefix+'-checkpoint.npz')
if a.resume:
 with np.load(checkpoint,allow_pickle=False) as cp:
  if str(cp['fingerprint'])!=fingerprint:raise ValueError('checkpoint model/spec mismatch')
  state={n:cp[n] for n in spec.state_pairs};start=int(cp['frame'])+1;timings=cp['timings'].tolist()
  for n in state:
   if list(state[n].shape)!=shapes[n] or state[n].dtype!=dtypes[n]:raise ValueError('checkpoint state schema mismatch')
 print('RESUME',start,flush=True)
for i in range(start,a.frames):
 warped=np.load(a.inputs/f'in_warped_{i}.npy');packed=np.load(a.inputs/f'in_packed_{i}.npy')
 feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}};feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()}
 t=time.monotonic();r=dict(zip(names,sess.run(None,feed)));elapsed=time.monotonic()-t
 if not all(np.isfinite(v).all() for v in r.values()):raise ValueError(f'nonfinite output frame {i}')
 state={n:r[nxt] for n,nxt in spec.state_pairs.items()};np.save(a.out/f'{a.prefix}_{i}.npy',r['outputs'].reshape(-1));timings.append(elapsed)
 if a.capture_block:
  np.save(a.out/f'block_x_{i}.npy',r['layer_norm_49']);np.save(a.out/f'block_y_{i}.npy',r['linear_99'])
 if (i+1)%32==0 or i+1==a.frames:
  temp=checkpoint.with_suffix('.tmp')
  with temp.open('wb') as f:np.savez(f,**state,frame=np.array(i),timings=np.array(timings),fingerprint=np.array(fingerprint))
  os.replace(temp,checkpoint)
 print(i,elapsed,flush=True)
(a.out/(a.prefix+'-run.json')).write_text(json.dumps({'status':'RECORDED_IMAGE_CPU_REPLAY_NOT_DRIVING_SAFETY_OR_TARGET_SPEED','frames':a.frames,'reset_at_first_frame':True,'independent_recurrence':True,'model':str(a.model),'inputs':str(a.inputs),'seconds_per_frame':timings},indent=2))
