"""Negative gates never initialize QNN. --valid enables native execution explicitly."""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T

p=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--valid', action='store_true')
parser.add_argument('--input',type=Path)
parser.add_argument('--reference',type=Path)
a=parser.parse_args()
opt=ort.SessionOptions()
opt.register_custom_ops_library(str(p/'libjetlink_vision_bridge.so'))
opt.intra_op_num_threads=2
opt.inter_op_num_threads=1
opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL

def session(shape):
    m=h.make_model(h.make_graph([h.make_node('QnnVisionPipeline',['x'],['y'],domain='jetlink.qnn')],'vision_bridge',[h.make_tensor_value_info('x',T.FLOAT16,shape)],[h.make_tensor_value_info('y',T.FLOAT16,[1,32,512])]),opset_imports=[h.make_opsetid('',18),h.make_opsetid('jetlink.qnn',1)])
    m.ir_version=10
    return ort.InferenceSession(m.SerializeToString(),opt,providers=['CPUExecutionProvider'])

sess=session([1,24,128,256])
rows=[]
for kind,data,target in [('dtype',np.zeros((1,24,128,256),np.float32),sess),('shape',np.zeros((1,23,128,256),np.float16),session([1,23,128,256])),('rank',np.zeros((24,128,256),np.float16),session([24,128,256])),('nonfinite',np.full((1,24,128,256),np.nan,np.float16),sess)]:
    try:target.run(None,{'x':data})
    except Exception as e:
        reason=str(e)
        assert 'dlopen' not in reason and 'libQnn' not in reason,reason
        rows.append(dict(case=kind,rejected=True,reason=reason))
    else:raise AssertionError('Not rejected: '+kind)
report={'negative_tests':rows,'native_validation':'not run'}
if a.valid:
    if not a.input or not a.reference:parser.error('--valid requires --input and --reference')
    x=np.fromfile(a.input,np.float16).reshape(1,24,128,256)
    ref=np.fromfile(a.reference,np.float16).reshape(1,32,512)
    y=sess.run(None,{'x':x})[0]
    assert np.isfinite(y).all()
    mismatch=int(np.count_nonzero(y.view(np.uint16)!=ref.view(np.uint16)))
    report.update(native_validation='executed',bit_mismatches=mismatch,max_abs=float(np.max(np.abs(y.astype(np.float32)-ref.astype(np.float32)))))
    assert mismatch==0,report
    np.save(p/'native-output.npy',y)
(p/('parity.json' if a.valid else 'negative-tests.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
