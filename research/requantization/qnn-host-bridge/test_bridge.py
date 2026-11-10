from pathlib import Path
import os, json, numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as T
P=Path(__file__).resolve().parent
model=h.make_model(h.make_graph([h.make_node('QnnFirstProjection',['x'],['y'],domain='jetlink.qnn')],'qnn_bridge',[h.make_tensor_value_info('x',T.FLOAT16,[1,288,512])],[h.make_tensor_value_info('y',T.FLOAT16,[1,288,1024])]),opset_imports=[h.make_opsetid('',18),h.make_opsetid('jetlink.qnn',1)],ir_version=10)
onnx.save(model,P/'single_node.onnx')
o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;o.register_custom_ops_library(str(P/'libjetlink_qnn_bridge.so'))
s=ort.InferenceSession(str(P/'single_node.onnx'),o,providers=['CPUExecutionProvider'])
x=np.load('/workspace/shared/jetlink-long-calibration/train.npz')['x']; rows=[]
for i,xi in enumerate(x[:int(os.getenv("LIMIT","160"))]):
 y=s.run(None,{'x':xi})[0]
 ref=np.fromfile(f'/workspace/shared/jetlink-exact-integration/qnn-htp-host-outputs/Result_{i}/val_646_DequantizeLinear_Output.raw',np.float32).reshape(y.shape)
 r16=ref.astype(np.float16);delta=y.astype(np.float32)-r16.astype(np.float32)
 rows.append({'i':i,'fp16_bit_mismatches':int(np.count_nonzero(y.view(np.uint16)!=r16.view(np.uint16))),'max_abs_vs_ref16':float(abs(delta).max()),'rmse_vs_ref16':float(np.sqrt(np.mean(delta**2)))})
 if i%20==0:print(rows[-1],flush=True)
json.dump({'rows':rows,'total_fp16_bit_mismatches':sum(r['fp16_bit_mismatches'] for r in rows),'elements':int(len(rows)*y.size)},open(P/os.getenv('RESULTS','test_results.json'),'w'),indent=2)
print('TOTAL',sum(r['fp16_bit_mismatches'] for r in rows),flush=True)
