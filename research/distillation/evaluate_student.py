"""Four-frame synthetic rejection screen; NOT acceptance of a driving model."""
import gc,json,sys,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
sys.path.insert(0,'/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink/scripts')
from verify_parity import make_inputs
from jetlink.spec import ModelSpec
p=Path('/workspace/shared/jetlink-distillation/real-block');s=np.load(p/'student-1024.npz')
m=onnx.load(p/'teacher-capture.onnx',load_external_data=False)
changes={'val_645':s['w1'],'model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_fc.bias':s['b1'],'val_647':s['w2'],'model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_proj.bias':s['b2']}
for i,w in enumerate(m.graph.initializer):
 if w.name in changes:m.graph.initializer[i].CopyFrom(numpy_helper.from_array(changes[w.name],w.name))
for n in m.graph.node:
 if n.name=='node_gelu_41':n.op_type='Relu';del n.attribute[:]
onnx.save(m,p/'student-full-1024.onnx');del m;gc.collect()
spec_path=Path('/workspace/shared/jetlink-history-regenerated/spec.json');spec=ModelSpec.load(spec_path)
compare=p/'student-1024-compare';compare.mkdir(exist_ok=True);(compare/'spec.json').write_bytes(spec_path.read_bytes())
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
for variant,path,prefix in [('teacher',p/'teacher-capture.onnx','out_ref'),('student',p/'student-full-1024.onnx','out_link')]:
 sess=ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider'])
 shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};dtypes={i.name:dt[i.type] for i in sess.get_inputs()};names=[o.name for o in sess.get_outputs()]
 state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs}
 for i,(warped,packed) in enumerate(make_inputs(spec,4,991)):
  feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}}
  feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()}
  t=time.monotonic();r=dict(zip(names,sess.run(None,feed)));print(variant,i,time.monotonic()-t,flush=True)
  if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite output')
  state={n:r[nxt] for n,nxt in spec.state_pairs.items()}
  np.save(compare/f'{prefix}_{i}.npy',r['outputs'].reshape(-1))
 del sess;gc.collect()
