"""Select only observable last row before final GELU/second projection.
Does not repeat existing attention/first-projection pruning.
"""
import copy,json
from pathlib import Path
import onnx
from onnx import helper
root=Path(__file__).resolve().parent;base=root.parent/'real-block'
m=onnx.load(base/'teacher-capture.onnx',load_external_data=False)
prod={o:n for n in m.graph.node for o in n.output};cons={}
for n in m.graph.node:
 for v in n.input:cons.setdefault(v,[]).append(n)
insert={};proof=[]
for gelu_name in ['gelu_44','gelu_55','gelu_62']:
 gelu=prod[gelu_name];assert gelu.op_type=='Gelu'
 chain=[];v=gelu.output[0]
 for op in ['MatMul','Add','Add','Gather']:
  users=cons[v];assert len(users)==1,(v,[n.name for n in users]);n=users[0];assert n.op_type==op;chain.append(n);v=n.output[0]
 proj,bias,residual,gather=chain
 assert helper.get_attribute_value(next(a for a in gather.attribute if a.name=='axis'))==1
 index=gather.input[1]
 # Original scalar Gather drops token axis: [1,288,C] -> [1,C].
 # GELU, MatMul, bias and residual addition operate independently per token.
 selected=gelu.input[0]+'__tail_last';residual_source=residual.input[0]
 assert residual.input[1]==bias.output[0]
 residual_selected=residual_source+'__tail_last'
 insert[gelu.name]=[
  helper.make_node('Gather',[gelu.input[0],index],[selected],axis=1,name=gelu.name+'__select_input'),
  helper.make_node('Gather',[residual_source,index],[residual_selected],axis=1,name=gelu.name+'__select_residual')]
 proof.append(dict(gelu=gelu.name,source=gelu.input[0],projection=proj.name,tail=[n.name for n in chain],index_initializer=index,residual_source=residual_source,sole_consumer_chain=True))
 gelu.input[0]=selected;residual.input[0]=residual_selected
 gather.op_type='Identity';del gather.input[1:];del gather.attribute[:]
nodes=[]
for n in m.graph.node:nodes.extend(insert.get(n.name,[]));nodes.append(n)
del m.graph.node[:];m.graph.node.extend(nodes);outs=list(m.graph.output)[:4];del m.graph.output[:];m.graph.output.extend(outs)
# Intermediate shapes are invalidated by intentional rank reduction.
del m.graph.value_info[:]
target=base/'teacher-last-query-tail.onnx';assert not target.exists();onnx.save(m,target);onnx.checker.check_model(str(target))
(root/'proof.json').write_text(json.dumps(dict(model=str(target),source='teacher-capture.onnx',branches=proof,weights_modified=False,activation_modified=False,first_projection_pruned=False,attention_pruned=False,per_branch_second_projection_macs_before=288*1024*512,per_branch_second_projection_macs_after=1024*512,status='STRUCTURAL_PROOF_PENDING_NUMERICAL_PARITY'),indent=2))
print(target)
