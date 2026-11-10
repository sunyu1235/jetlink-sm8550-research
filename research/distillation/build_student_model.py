"""Replace one actual teacher MLP with stored student weights; preserve other tensors."""
import argparse
from pathlib import Path
import numpy as np,onnx
from onnx import numpy_helper,helper
ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--activation',choices=['relu','silu'],required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
p=Path('/workspace/shared/jetlink-distillation/real-block');target=p/(a.name+'.onnx')
if target.exists():raise FileExistsError(target)
s=np.load(a.weights,allow_pickle=False);m=onnx.load(p/'teacher-capture.onnx',load_external_data=False)
changes={'val_645':s['w1'],'model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_fc.bias':s['b1'],'val_647':s['w2'],'model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_proj.bias':s['b2']}
for i,w in enumerate(m.graph.initializer):
 if w.name in changes:m.graph.initializer[i].CopyFrom(numpy_helper.from_array(changes[w.name],w.name))
ns=[]
for n in m.graph.node:
 if n.name!='node_gelu_41':ns.append(n);continue
 if a.activation=='relu':
  n.op_type='Relu';del n.attribute[:];ns.append(n)
 else:
  ns.extend([helper.make_node('Mul',[n.input[0],'student_silu_beta'],['student_silu_scaled']),helper.make_node('Sigmoid',['student_silu_scaled'],['student_silu_sigmoid']),helper.make_node('Mul',[n.input[0],'student_silu_sigmoid'],list(n.output))])
if a.activation=='silu':m.graph.initializer.append(numpy_helper.from_array(np.array(1.702,dtype=np.float16),'student_silu_beta'))
del m.graph.node[:];m.graph.node.extend(ns);onnx.save(m,target);onnx.checker.check_model(str(target));print(target)
