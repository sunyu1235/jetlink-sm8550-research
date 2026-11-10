from pathlib import Path
import os,json,numpy as np,onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent
o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;o.register_custom_ops_library(str(P/'libjetlink_qnn_bridge.so'))
def session():return ort.InferenceSession(str(P/'single_node.onnx'),o,providers=['CPUExecutionProvider'])
s=session();x=np.load('/workspace/shared/jetlink-long-calibration/train.npz')['x'][0]
os.environ['JETLINK_DUMP_INPUT']=str(P/'input-contract.raw')
bits=np.arange(65536,dtype=np.uint16);bits=bits[(bits&0x7c00)!=0x7c00]
allfinite=np.resize(bits,147456).view(np.float16).reshape(1,288,512)
s.run(None,{'x':allfinite});actual=np.fromfile(P/'input-contract.raw',np.uint16)
expected=np.clip(np.rint(allfinite.astype(np.float32)/np.float32(7.662294228794053e-05))+np.float32(38745),0,65535).astype(np.uint16).ravel()
assert np.array_equal(actual,expected)
del os.environ['JETLINK_DUMP_INPUT']
rejected=[]
for label,inp in [('shape',x[:,:,:511]),('type',x.astype(np.float32)),('nan',np.full_like(x,np.nan)),('inf',np.full_like(x,np.inf))]:
 try:s.run(None,{'x':inp})
 except Exception:rejected.append(label)
 else:raise AssertionError(label)
with ThreadPoolExecutor(2) as pool:ys=list(pool.map(lambda _:s.run(None,{'x':x})[0],range(2)))
assert np.array_equal(ys[0].view('u2'),ys[1].view('u2'))
del s
s=session();y=s.run(None,{'x':x})[0];assert np.array_equal(y.view('u2'),ys[0].view('u2'))
report={'all_finite_fp16_patterns':len(bits),'quantized_elements':actual.size,'quantization_mismatches':int(np.count_nonzero(actual!=expected)),'rejected':rejected,'concurrent_same_session_native_repeat':True,'destroy_recreate_same_result':True}
json.dump(report,open(P/'contract-results.json','w'),indent=2);print(report,flush=True)
