"""Quantize only the first projection and retain original FP16 remainder + exact GELU."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
import onnx
from onnx import helper,TensorProto
p=argparse.ArgumentParser();p.add_argument('--source-capture',type=Path,required=True);p.add_argument('--qdq-block',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
if a.source_capture.parent.resolve()!=a.out.parent.resolve() or a.out.exists():raise ValueError('new output must share external-weight directory')
m=onnx.load(a.source_capture,load_external_data=False);q=onnx.load(a.qdq_block)
outs=[x for x in m.graph.output if x.name not in ('layer_norm_49','linear_99')];del m.graph.output[:];m.graph.output.extend(outs)
sub=[]
for n in q.graph.node:
 sub.append(copy.deepcopy(n))
 if list(n.output)==['val_646_DequantizeLinear_Output']:break
if len(sub)!=6 or sub[-1].op_type!='DequantizeLinear':raise ValueError('unexpected QDQ first-projection pattern')
used={x for n in sub for x in n.input};inits=[copy.deepcopy(t) for t in q.graph.initializer if t.name in used];prefix='rq_first/'
for n in sub:
 n.name=prefix+n.name
 for i,x in enumerate(n.input):
  if x:n.input[i]=prefix+x
 for i,x in enumerate(n.output):n.output[i]=prefix+x
for t in inits:t.name=prefix+t.name
replacement=[helper.make_node('Cast',['layer_norm_49'],[prefix+'layer_norm_49'],to=TensorProto.FLOAT,name=prefix+'input_float'),*sub,helper.make_node('Cast',[prefix+'val_646_DequantizeLinear_Output'],['val_646'],to=TensorProto.FLOAT16,name=prefix+'output_half')]
nodes=[];count=0
for n in m.graph.node:
 if n.name=='node_MatMul_546':nodes.extend(replacement);count+=1
 else:nodes.append(n)
if count!=1:raise ValueError('source projection not unique')
del m.graph.node[:];m.graph.node.extend(nodes);m.graph.initializer.extend(inits)
used={x for n in m.graph.node for x in n.input};inits=[t for t in m.graph.initializer if t.name in used];del m.graph.initializer[:];m.graph.initializer.extend(inits)
counts=collections.Counter()
for n in m.graph.node:
 if n.op_type=='Gelu':
  mode=next((a.s.decode() for a in n.attribute if a.name=='approximate'),'none')
  if mode not in ('none','tanh'):raise ValueError(mode)
  n.op_type='ExactGeluTanh' if mode=='tanh' else 'ExactGelu';n.domain='jetlink.precision';del n.attribute[:];counts[n.op_type]+=1
if dict(counts)!={'ExactGelu':36,'ExactGeluTanh':27}:raise ValueError(counts)
existing={d.domain for d in m.opset_import}
for imp in q.opset_import:
 if imp.domain not in existing:m.opset_import.append(copy.deepcopy(imp));existing.add(imp.domain)
m.opset_import.append(helper.make_opsetid('jetlink.precision',1));onnx.save(m,a.out);onnx.checker.check_model(str(a.out))
r={'scope':'one A16W8 MatMul with FP16 cast before original bias/GELU/second-projection, all 63 exact CPU GELUs','custom_counts':dict(counts),'graph_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'qdq_block_sha256':hashlib.sha256(a.qdq_block.read_bytes()).hexdigest(),'remaining_original_weights_preserved':True,'device_compiled':False};a.report.write_text(json.dumps(r,indent=2));print(json.dumps(r))
