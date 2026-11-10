"""Mirror pinned Jetlink Patches.patchUint8Inputs; verify CPU image boundary."""
import argparse,copy,json,hashlib
from pathlib import Path
import onnx
from onnx import TensorProto as T
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
m=onnx.load(a.source,load_external_data=False);weights=[t.SerializeToString() for t in m.graph.initializer]
sources=[x.name for x in m.graph.input if x.type.tensor_type.elem_type==T.UINT8];assert set(sources)=={'new_img','state_img_q'}
layout={'Concat','Slice','Gather','Reshape','Unsqueeze','Squeeze','Transpose','Flatten','Identity','Expand'}
retyped=set(sources);casts=[]
for idx,n in enumerate(m.graph.node):
 if not retyped.intersection(n.input):continue
 if n.op_type=='Cast':casts.append(idx)
 elif n.op_type in layout:retyped.update(n.output)
 else:raise ValueError((n.name,n.op_type))
assert casts
outputs={v.name for v in m.graph.output};boundaries=[]
for idx in casts:
 n=m.graph.node[idx];assert next(x.i for x in n.attribute if x.name=='to')==T.FLOAT16 and n.output[0] not in outputs
 boundaries.append((n.input[0],n.output[0]))
 for other in m.graph.node:
  for i,x in enumerate(other.input):
   if x==n.output[0]:other.input[i]=n.input[0]
for idx in reversed(casts):del m.graph.node[idx]
for v in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output):
 if v.name in retyped:v.type.tensor_type.elem_type=T.FLOAT16
assert weights==[t.SerializeToString() for t in m.graph.initializer]
onnx.save(m,a.output);onnx.checker.check_model(str(a.output))
r=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),sources=sources,retyped=sorted(retyped),removed_cast_boundaries=boundaries,weights_unchanged=True,semantic_scope='FP16 callers supply exactly original uint8 values0..255; image recurrent state returnedFP16')
a.report.write_text(json.dumps(r,indent=2));print(r)
