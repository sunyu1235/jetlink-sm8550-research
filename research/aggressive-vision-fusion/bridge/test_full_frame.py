"""Exactly one full-model native invocation; compare every output to split control."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import onnxruntime as ort
p=Path(__file__).resolve().parent
root=p.parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=p/'full-frame-check')
parser.add_argument('--road',type=Path,default=os.environ.get('JETLINK_ROAD'))
parser.add_argument('--vendor',type=Path,default=os.environ.get('JETLINK_VENDOR'))
parser.add_argument('--split-control',type=Path,default=os.environ.get('JETLINK_SPLIT_CONTROL'))
parser.add_argument('--model',type=Path,default=p/'model/vision-native.onnx')
parser.add_argument('--bridge-library',type=Path,default=p/'libjetlink_vision_bridge.so')
parser.add_argument('--context',type=Path,default=os.environ.get('JETLINK_VISION_CONTEXT'))
parser.add_argument('--weights',type=Path)
args=parser.parse_args()
out=args.output.resolve()
if out.exists():raise FileExistsError(out)
for field in ['road','vendor','split_control','context']:
    if getattr(args,field) is None:parser.error('--'+field.replace('_','-')+' or matching environment setting is required')
for field in ['road','vendor','split_control','model','bridge_library','context']:
    value=getattr(args,field).resolve()
    if not value.exists():parser.error('Missing '+field+': '+str(value))
    setattr(args,field,value)
args.weights=(args.weights or args.model.parent/'teacher-capture.weights').resolve()
if not args.weights.is_file():parser.error('Missing external weights: '+str(args.weights))
for name in ['JETLINK_GELU_PACKAGE','JETLINK_QNN_BACKEND','JETLINK_QNN_SYSTEM']:
    if not os.environ.get(name) or not Path(os.environ[name]).is_file():parser.error('Set '+name+' to an existing library')
os.environ['JETLINK_VISION_CONTEXT']=str(args.context)
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(args.vendor))
from jetlink.spec import ModelSpec
road=args.road.resolve()
spec=ModelSpec.load(road/'spec.json')
opts=ort.SessionOptions()
opts.register_custom_ops_library(str(args.bridge_library))
opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
start=time.monotonic()
sess=ort.InferenceSession(str(args.model),opts,providers=['CPUExecutionProvider'])
print('session loaded; preparing independent zero-state inputs',flush=True)
packed=np.load(road/'in_packed_0.npy')
feed={n:packed[at] for n,(at,_) in spec.packed_layout.items()}
feed['new_img']=np.load(road/'in_warped_0.npy')
dtypes={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8}
for v in sess.get_inputs():
    if v.name in spec.state_pairs:feed[v.name]=np.zeros(v.shape,dtypes[v.type])
    else:feed[v.name]=np.ascontiguousarray(feed[v.name],dtype=dtypes[v.type]).reshape(v.shape)
print('running exactly one native whole-model inference',flush=True)
ys=sess.run(None,feed)
print('native inference finished',flush=True)
rows={}
for v,y in zip(sess.get_outputs(),ys):
    refpath=args.split_control/('out_link_0.npy' if v.name=='outputs' else v.name+'.npy')
    ref=np.load(refpath).reshape(y.shape)
    np.save(out/(v.name+'.npy'),y)
    rows[v.name]=dict(shape=list(y.shape),dtype=str(y.dtype),exact_bytes=y.tobytes()==ref.tobytes(),max_abs=float(np.max(np.abs(y.astype(np.float64)-ref.astype(np.float64)))),nonfinite=int(np.count_nonzero(~np.isfinite(y))))
paths=[p/'bridge.cpp',args.bridge_library,args.model,args.context,road/'in_packed_0.npy',road/'in_warped_0.npy',road/'spec.json',Path(os.environ['JETLINK_GELU_PACKAGE']),Path(os.environ['JETLINK_QNN_BACKEND']),Path(os.environ['JETLINK_QNN_SYSTEM']),args.weights]+list(args.split_control.glob('*.npy'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
report=dict(scope='One recorded-night frame, independent zero states, full model against split native vision control. No phone or latency qualification.',native_invocations=1,gelu_package=os.environ['JETLINK_GELU_PACKAGE'],elapsed_host_seconds=time.monotonic()-start,outputs=rows,hashes={str(x):sha(x) for x in paths})
(out/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
assert len(rows)==4 and all(x['exact_bytes'] and not x['nonfinite'] for x in rows.values())
