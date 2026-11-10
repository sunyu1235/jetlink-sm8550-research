"""Actual final MLP tail CPU microbenchmark; not device or pipeline latency."""
import json,time,statistics,copy
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
root=Path(__file__).resolve().parent;base=root.parent/'real-block';report={}
rng=np.random.default_rng(2861);feed={'linear_110':rng.normal(size=(1,288,1024)).astype(np.float16),'add_52':rng.normal(size=(1,288,512)).astype(np.float16)};outputs={}
for name,src in [('full','teacher-capture.onnx'),('compact','teacher-last-query-tail.onnx')]:
 m=onnx.load(base/src,load_external_data=False);prod={o:n for n in m.graph.node for o in n.output};weights={w.name:w for w in m.graph.initializer};needed=set();nodes=set()
 def visit(v):
  if v in feed:return
  if v in weights:needed.add(v);return
  n=prod[v]
  if n.name in nodes:return
  nodes.add(n.name)
  for i in n.input:visit(i)
 visit('select_4');ns=[copy.deepcopy(n) for n in m.graph.node if n.name in nodes];ws=[numpy_helper.from_array(numpy_helper.to_array(weights[n],base_dir=str(base)),n) for n in needed]
 g=helper.make_graph(ns,name,[helper.make_tensor_value_info(k,TensorProto.FLOAT16,list(v.shape)) for k,v in feed.items()],[helper.make_tensor_value_info('select_4',TensorProto.FLOAT16,[1,512])],ws)
 small=helper.make_model(g,opset_imports=[helper.make_opsetid('',20)]);small.ir_version=10;onnx.checker.check_model(small);onnx.save(small,root/(name+'-tail.onnx'))
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
 sess=ort.InferenceSession(small.SerializeToString(),opt,providers=['CPUExecutionProvider']);timings=[]
 for i in range(60):
  t=time.perf_counter();out=sess.run(None,feed)[0];dt=time.perf_counter()-t
  if i>=10:timings.append(dt*1000)
 outputs[name]=out;report[name]={'median_ms':statistics.median(timings),'min_ms':min(timings),'runs':len(timings),'nodes':[(n.name,n.op_type) for n in ns]}
 del sess
report['comparison']={'max_abs':float(np.max(np.abs(outputs['full'].astype(np.float32)-outputs['compact']))),'bit_identical':bool(np.array_equal(outputs['full'],outputs['compact']))};report['status']='CPU_MICROBENCHMARK_NOT_TARGET_DEVICE_OR_END_TO_END_LATENCY';report['random_input_seed']=2861
(root/'cpu-microbenchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
