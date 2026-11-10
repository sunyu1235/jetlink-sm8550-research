#!/usr/bin/env python3
"""Independent finite-input checks of the repository's CPU ExactGelu plugin."""
import argparse,hashlib,json,math,tempfile
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,TensorProto
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
report={'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'ort':ort.__version__,'scope':'Linux CPU custom-op host equations, not Android or HTP','tests':[]}
for name,dt,onnxtype in [('ExactGelu',np.float16,TensorProto.FLOAT16),('ExactGeluTanh',np.float16,TensorProto.FLOAT16),('ExactGelu32',np.float32,TensorProto.FLOAT),('ExactGeluTanh32',np.float32,TensorProto.FLOAT)]:
 if dt==np.float16:
  bits=np.arange(65536,dtype=np.uint16);x=bits.view(np.float16);x=x[np.isfinite(x)]
 else:x=np.concatenate([np.linspace(-40,40,100001,dtype=np.float32),np.array([-0.,0.,1e-30,-1e-30],np.float32)])
 f=math.tanh if 'Tanh' in name else math.erf
 def scalar(v):
  z=float(v);arg=.79788456080286535588*(z+.044715*z*z*z) if 'Tanh' in name else z*.70710678118654752440
  return .5*z*(1+f(arg))
 ref=np.array([scalar(v) for v in x],dtype=dt)
 g=helper.make_graph([helper.make_node(name,['x'],['y'],domain='jetlink.precision')],name,[helper.make_tensor_value_info('x',onnxtype,list(x.shape))],[helper.make_tensor_value_info('y',onnxtype,list(x.shape))]);m=helper.make_model(g,opset_imports=[helper.make_opsetid('',20),helper.make_opsetid('jetlink.precision',1)]);m.ir_version=10
 o=ort.SessionOptions();o.register_custom_ops_library(str(a.library));o.intra_op_num_threads=2
 s=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']);y=s.run(None,{'x':x})[0]
 uint=np.uint16 if dt==np.float16 else np.uint32;mismatches=int(np.count_nonzero(y.view(uint)!=ref.view(uint)))
 case={'op':name,'finite_inputs':len(x),'bit_mismatches':mismatches,'all_outputs_finite':bool(np.isfinite(y).all()),'max_abs':float(np.abs(y.astype(np.float64)-ref.astype(np.float64)).max())};report['tests'].append(case);print(case,flush=True)
 if mismatches:raise RuntimeError(case)
a.output.write_text(json.dumps(report,indent=2))
