"""Stream all12 teacher MLP pairs from independent long synthetic sequences.
Train samples96 temporal tokens/frame, validation retains all288. No acceptance data.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np,onnxruntime as ort
from jetlink.spec import ModelSpec
sys.path.insert(0,'/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink/scripts')
from verify_parity import make_inputs
ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
out=Path(__file__).resolve().parent;blocks=json.loads((out/'capture-blocks.json').read_text());model=out.parent/'real-block/teacher-policy8-cse-capture.onnx';spec=ModelSpec.load('/workspace/shared/jetlink-history-regenerated/spec.json')
checkpoint=out/'capture-checkpoint.npz';fingerprint=hashlib.sha256(model.read_bytes()).hexdigest();start_split=0;start_frame=0;resumed_state=None
if args.resume:
 with np.load(checkpoint,allow_pickle=False) as cp:
  if str(cp['fingerprint'])!=fingerprint:raise ValueError('checkpoint model changed')
  start_split=int(cp['split']);start_frame=int(cp['frame'])+1;resumed_state={n:cp[n] for n in spec.state_pairs}
elif checkpoint.exists() or (out/'capture.json').exists():raise FileExistsError('refuse to repeat capture evidence')
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(str(model),opt,providers=['CPUExecutionProvider']);shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};dtypes={i.name:dt[i.type] for i in sess.get_inputs()};names=[o.name for o in sess.get_outputs()]
train_tokens=np.linspace(0,287,96,dtype=np.int64);assert len(set(train_tokens))==96 and train_tokens[-1]==287
stats=[]
for si,(split,seed) in enumerate([('train',713),('validation',1777)]):
 if si<start_split:continue
 state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs};first=0
 if si==start_split and resumed_state is not None:state=resumed_state;first=start_frame
 reference=np.load('/workspace/shared/jetlink-long-calibration/'+split+'-teacher-outputs.npy',mmap_mode='r')
 for i,(warped,packed) in enumerate(make_inputs(spec,160,seed)):
  if i<first:continue
  feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}};feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()}
  t=time.monotonic();r=dict(zip(names,sess.run(None,feed)));elapsed=time.monotonic()-t
  if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite teacher')
  state={n:r[nxt] for n,nxt in spec.state_pairs.items()}
  for b in blocks:
   for key in ['x','y']:
    v=r[b['capture_'+key]].reshape(288,512);v=v[train_tokens] if split=='train' else v
    np.save(out/b['name']/f'{split}_{i:03d}_{key}.npy',v)
  pred=r['outputs'].reshape(-1);error=float(np.max(abs(pred-reference[i].reshape(-1))))
  np.save(out/f'{split}_out_{i:03d}.npy',pred)
  stats.append({'split':split,'frame':i,'seconds':elapsed,'prior_teacher_max_abs':error})
  if (i+1)%16==0:
   temp=checkpoint.with_suffix('.tmp')
   with temp.open('wb') as f:np.savez(f,**state,split=np.array(si),frame=np.array(i),fingerprint=np.array(fingerprint))
   os.replace(temp,checkpoint)
  print(split,i,elapsed,'prior_teacher_max_abs',error,flush=True)
(out/'capture.json').write_text(json.dumps({'status':'EIGHT_UNIQUE_REAL_BLOCKS_SYNTHETIC_LONG_HISTORY','train_seed':713,'validation_seed':1777,'frames_each':160,'train_tokens':train_tokens.tolist(),'validation_tokens':'all288','independent_reset':True,'original_acceptance_or_road_data_used':False,'resumed':args.resume,'new_frames_this_run':stats},indent=2))
