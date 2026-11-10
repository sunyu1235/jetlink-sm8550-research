"""Instrument the twelve real policy transformer MLPs without changing arithmetic."""
import copy,json,hashlib
from pathlib import Path
import numpy as np,onnx
from onnx import numpy_helper
base=Path('/workspace/shared/jetlink-distillation/real-block');out=Path(__file__).resolve().parent
source=base/'teacher-capture.onnx';target=base/'teacher-policy12-capture.onnx'
if target.exists() or (out/'blocks.json').exists():raise FileExistsError('preserve existing evidence')
m=onnx.load(source,load_external_data=False);vals={v.name:v for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)};weights={w.name:w for w in m.graph.initializer};prod={o:n for n in m.graph.node for o in n.output}
cons={}
for n in m.graph.node:
 for i in n.input:cons.setdefault(i,[]).append(n)
blocks=[]
for gelu in m.graph.node:
 if gelu.op_type!='Gelu':continue
 bias1=prod[gelu.input[0]]
 if bias1.op_type!='Add' or '.transformer.layers.' not in bias1.input[1] or not bias1.input[1].endswith('.mlp.c_fc.bias'):continue
 fc=prod[bias1.input[0]];assert fc.op_type=='MatMul'
 nexts=cons[gelu.output[0]];assert len(nexts)==1;proj=nexts[0];assert proj.op_type=='MatMul'
 nexts=cons[proj.output[0]];assert len(nexts)==1;bias2=nexts[0];assert bias2.op_type=='Add' and bias2.input[1].endswith('.mlp.c_proj.bias')
 name=gelu.name.removeprefix('node_');d=out/name;d.mkdir()
 weight_names=dict(w1=fc.input[1],b1=bias1.input[1],w2=proj.input[1],b2=bias2.input[1])
 arrays={k:numpy_helper.to_array(copy.deepcopy(weights[v]),base_dir=str(base)) for k,v in weight_names.items()}
 assert arrays['w1'].shape==(512,1024) and arrays['w2'].shape==(1024,512)
 np.savez(d/'teacher-weights.npz',**arrays)
 blocks.append(dict(name=name,gelu_node=gelu.name,x=fc.input[0],y=bias2.output[0],weights=weight_names,source_bias=bias1.input[1],dtype='float16'))
assert len(blocks)==12
original_outputs=list(m.graph.output)[:4];del m.graph.output[:];m.graph.output.extend(original_outputs)
for b in blocks:
 for key in ['x','y']:
  vi=copy.deepcopy(vals[b[key]]);assert [v.dim_value for v in vi.type.tensor_type.shape.dim]==[1,288,512];m.graph.output.append(vi)
onnx.save(m,target);onnx.checker.check_model(str(target))
(out/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
(out/'prepare.json').write_text(json.dumps({'source_metadata_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'capture_model':str(target),'blocks':12,'teacher_arithmetic_changed':False,'added_outputs':24,'teacher_weights_external_file_unchanged':True},indent=2)+'\n')
print([(b['name'],b['x'],b['y']) for b in blocks])
