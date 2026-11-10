from pathlib import Path
import json,numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,TensorProto as T
p=Path(__file__).parent;s=Path('/workspace/shared/jetlink-compact-tail-bindings')
inputs=['gelu_43','gelu_54','gelu_61','add_50','add_70','add_86'];outputs=['select_4','select_7','select_10']
m=h.make_model(h.make_graph([h.make_node('QnnCompactPipeline',inputs,outputs,domain='jetlink.qnn')],'native_compact',[h.make_tensor_value_info(n,T.FLOAT16,[1,288,1024 if i<3 else 512]) for i,n in enumerate(inputs)],[h.make_tensor_value_info(n,T.FLOAT16,[1,512]) for n in outputs]),opset_imports=[h.make_opsetid('',18),h.make_opsetid('jetlink.qnn',1)]);m.ir_version=10;onnx.save(m,p/'single_node.onnx')
opt=ort.SessionOptions();opt.register_custom_ops_library(str(p/'libjetlink_compact_bridge.so'));opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(str(p/'single_node.onnx'),opt,providers=['CPUExecutionProvider']);rows=[]
for i in range(8):
 feed={n:np.fromfile(s/f'pipeline-inputs/{i}-{n}.raw',np.float16).reshape(1,288,1024 if j<3 else 512) for j,n in enumerate(inputs)}
 ys=sess.run(None,feed)
 for name,y in zip(outputs,ys):
  ref=np.fromfile(s/f'tails-host-output/Result_{i}/{name}_native.raw',np.float16).reshape(1,512)
  assert np.isfinite(y).all();diff=int(np.count_nonzero(y.view(np.uint16)!=ref.view(np.uint16)));rows.append(dict(sample=i,output=name,bit_mismatches=diff,max_abs=float(np.abs(y.astype(np.float32)-ref.astype(np.float32)).max())))
  assert diff==0,rows[-1]
 print('sample',i,'bit_exact',flush=True)
neg=[]
for kind in ['shape','dtype','nonfinite']:
 bad=dict(feed)
 if kind=='shape':bad[inputs[0]]=np.zeros((1,287,1024),np.float16)
 elif kind=='dtype':bad[inputs[0]]=bad[inputs[0]].astype(np.float32)
 else:bad[inputs[0]]=bad[inputs[0]].copy();bad[inputs[0]][0,0,0]=np.nan
 try:sess.run(None,bad)
 except Exception as e:neg.append(dict(case=kind,rejected=True,reason=str(e)));continue
 raise AssertionError(kind+' not rejected')
# A rejected invocation must not poison later execution.
ys=sess.run(None,feed)
for name,y in zip(outputs,ys):assert np.array_equal(y.view(np.uint16),np.fromfile(s/f'tails-host-output/Result_7/{name}_native.raw',np.uint16).reshape(1,512))
r=dict(scope='saved-context native host bridge versus official net-run; not phone accuracy or speed',samples=8,values=12288,bit_mismatches=sum(v['bit_mismatches'] for v in rows),rows=rows,negative_tests=neg,recovery_after_rejection=True)
(p/'parity.json').write_text(json.dumps(r,indent=2));print('PASS',flush=True)
