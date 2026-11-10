"""Capture separate synthetic sequences from a real teacher; never road validation."""
import copy,gc,json,time,sys
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnx import helper,numpy_helper
sys.path.insert(0,'/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink/scripts')
from verify_parity import make_inputs
from jetlink.spec import ModelSpec
out=Path('/workspace/shared/jetlink-distillation/real-block');out.mkdir(exist_ok=True)
m=onnx.load('/workspace/shared/jetlink-models/light-speed.onnx')
vals={v.name:v for v in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output)}
weights={v.name:v for v in m.graph.initializer}
names=['node_MatMul_546','node_linear_98','node_gelu_41','node_MatMul_548','node_linear_99']
ns=[copy.deepcopy(n) for n in m.graph.node if n.name in names]
assert len(ns)==5
used={i for n in ns for i in n.input if i in weights}
block=helper.make_model(helper.make_graph(ns,'original_policy_layer0_mlp',[copy.deepcopy(vals['layer_norm_49'])],[copy.deepcopy(vals['linear_99'])],[copy.deepcopy(weights[w]) for w in sorted(used)]),opset_imports=[helper.make_opsetid('',20)])
block.ir_version=10;onnx.checker.check_model(block);onnx.save(block,out/'teacher-block.onnx')
np.savez(out/'weights.npz',w1=numpy_helper.to_array(weights['val_645']),b1=numpy_helper.to_array(weights['model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_fc.bias']))
for n in m.graph.node:
 if n.domain=='org.tinygrad':
  assert n.op_type=='Contiguous' and len(n.input)==len(n.output)==1
  n.op_type='Identity';n.domain='';del n.attribute[:]
# Contiguous is a storage-layout hint; ONNX tensors carry values, not strides.
for name in ['layer_norm_49','linear_99']:m.graph.output.append(copy.deepcopy(vals[name]))
onnx.save_model(m,out/'teacher-capture.onnx',save_as_external_data=True,all_tensors_to_one_file=True,location='teacher-capture.weights',size_threshold=1024)
del m,vals,weights,block,ns;gc.collect()
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
print('SESSION_START',flush=True);sess=ort.InferenceSession(str(out/'teacher-capture.onnx'),opt,providers=['CPUExecutionProvider']);print('SESSION_READY',flush=True)
spec=ModelSpec.load('/workspace/shared/jetlink-history-regenerated/spec.json')
shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8}
dtypes={i.name:dt[i.type] for i in sess.get_inputs()};outputs=[v.name for v in sess.get_outputs()]
result={};timings=[]
for split,seed,nframes in [('train',713,8),('validation',991,4)]:
 state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs};xs=[];ys=[]
 for i,(warped,packed) in enumerate(make_inputs(spec,nframes,seed)):
  feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}}
  feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()}
  t=time.monotonic();r=dict(zip(outputs,sess.run(None,feed)));elapsed=time.monotonic()-t
  if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite teacher output')
  state={n:r[nxt] for n,nxt in spec.state_pairs.items()};xs.append(r['layer_norm_49']);ys.append(r['linear_99'])
  timings.append({'split':split,'frame':i,'seconds':elapsed});print(split,i,elapsed,flush=True)
 result['x_'+split]=np.stack(xs);result['y_'+split]=np.stack(ys)
np.savez(out/'sequences.npz',**result)
w=np.load(out/'weights.npz');np.savez(out/'pairs.npz',**{k:v.reshape(-1,v.shape[-1]) for k,v in result.items()},w1=w['w1'],b1=w['b1'])
(out/'capture.json').write_text(json.dumps({'status':'REAL_WEIGHTS_SYNTHETIC_INPUTS_NOT_ROAD_VALIDATION','seeds':[713,991],'frames':[8,4],'contiguous_lowering':'org.tinygrad::Contiguous replaced by Identity; layout-only assumption','x':'layer_norm_49','y':'linear_99','block':'off_policy temporal transformer layer 0 MLP','timings':timings},indent=2))
