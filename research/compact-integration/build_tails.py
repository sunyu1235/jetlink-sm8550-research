"""Build all three compact downstream tails with explicit no-padding I/O contracts."""
from pathlib import Path
import copy,json,hashlib,argparse
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=Path('/workspace/shared/jetlink-distillation/real-block/teacher-capture.onnx'));parser.add_argument('--tail-proof',type=Path,default=Path('/workspace/shared/jetlink-distillation/last-query-tail/proof.json'));parser.add_argument('--out',type=Path,default=Path(__file__).parent);args=parser.parse_args();P=args.out;P.mkdir(parents=True,exist_ok=True);src=args.source
for filename in ['tails-cpu.onnx','tails-htp.onnx','tails-full-cpu.onnx','contract.json']:
 if (P/filename).exists():raise FileExistsError(P/filename)
m=onnx.load(src,load_external_data=False);prod={v:n for n in m.graph.node for v in n.output};inits={v.name:v for v in m.graph.initializer};branches=json.load(open(args.tail_proof))['branches']
nodes=[];used=set();inputs=[];outputs=[];bindings=[]
for b in branches:
 g=next(n for n in m.graph.node if n.name==b['gelu']);assert g.op_type=='Gelu'
 assert next(a.s for a in g.attribute if a.name=='approximate')==b'tanh'
 source=b['source'];res=b['residual_source'];inputs += [h.make_tensor_value_info(source,T.FLOAT16,[1,1,1024]),h.make_tensor_value_info(res,T.FLOAT16,[1,1,512])]
 chain=[copy.deepcopy(g)]+[copy.deepcopy(next(n for n in m.graph.node if n.name==nname)) for nname in b['tail'][:-1]]
 gather=next(n for n in m.graph.node if n.name==b['tail'][-1]);assert gather.op_type=='Gather';assert list(gather.input)==[chain[-1].output[0],b['index_initializer']]
 chain.append(h.make_node('Squeeze',[gather.input[0],'compact_token_axis'],list(gather.output),name=gather.name+'__compact_squeeze'))
 for n in chain:used.update(n.input)
 nodes.extend(chain);outputs.append(h.make_tensor_value_info(gather.output[0],T.FLOAT16,[1,512]));bindings.append(dict(preactivation=source,residual=res,output=gather.output[0],preactivation_shape=[1,1,1024],residual_shape=[1,1,512],output_shape=[1,512]))
weights=[]
for name in sorted(used.intersection(inits)):
 t=copy.deepcopy(inits[name]);onnx.external_data_helper.load_external_data_for_tensor(t,str(src.parent)) if onnx.external_data_helper.uses_external_data(t) else None;t.ClearField('external_data');t.data_location=T.DEFAULT;weights.append(t)
weights.append(nh.from_array(np.array([1],np.int64),'compact_token_axis'))
# Upstream canonical ordering: residual outputs first, preactivations next.
order=['add_52','add_72','add_88','linear_110','linear_151','linear_180'];by={x.name:x for x in inputs};inputs=[by[n] for n in order]
g=h.make_graph(nodes,'compact_three_tail_bindings',inputs,outputs,weights);model=h.make_model(g,opset_imports=list(m.opset_import));model.ir_version=10;model=onnx.shape_inference.infer_shapes(model,strict_mode=True);onnx.checker.check_model(model,full_check=True);onnx.save(model,P/'tails-cpu.onnx')
htp=copy.deepcopy(model);ns=[]
for n in htp.graph.node:
 if n.op_type!='Gelu':ns.append(n);continue
 pre=n.name+'__rank4_in';out=n.name+'__rank4_out'
 ns.extend([h.make_node('Reshape',list(n.input)+['compact_gelu_shape4'],[pre],name=n.name+'__rank4'),h.make_node('GeluTanhFp16',[pre],[out],name=n.name,domain='jetlink_gelu'),h.make_node('Reshape',[out,'compact_gelu_shape3'],list(n.output),name=n.name+'__rank3')]);htp.graph.value_info.extend([h.make_tensor_value_info(x,T.FLOAT16,[1,1,1,1024]) for x in (pre,out)])
del htp.graph.node[:];htp.graph.node.extend(ns);htp.graph.initializer.extend([nh.from_array(np.array([1,1,1,1024],np.int64),'compact_gelu_shape4'),nh.from_array(np.array([1,1,1024],np.int64),'compact_gelu_shape3')]);htp.opset_import.append(h.make_opsetid('jetlink_gelu',1));onnx.checker.check_model(htp);onnx.save(htp,P/'tails-htp.onnx')
r=dict(scope='reconstructed three-tail compact ABI, not cached phone contexts',inputs=[dict(name=x.name,dtype='float16',shape=[d.dim_value for d in x.type.tensor_type.shape.dim],bytes=2*np.prod([d.dim_value for d in x.type.tensor_type.shape.dim]).item()) for x in inputs],outputs=[dict(name=x.name,dtype='float16',shape=[1,512],bytes=1024) for x in outputs],branches=bindings,original_boundary_bytes=2654208,compact_boundary_bytes=9216,zero_prefix_restoration=False,weights_changed=False,gelu_changed=False);(P/'contract.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
# Independent unpruned reference: retain original tokenwise nodes and final Gather.
full_nodes=[];full_used=set()
for b in branches:
 for name in [b['gelu']]+b['tail']:
  node=copy.deepcopy(next(n for n in m.graph.node if n.name==name));full_nodes.append(node);full_used.update(node.input)
full_weights=[]
for name in sorted(full_used.intersection(inits)):
 t=copy.deepcopy(inits[name]);onnx.external_data_helper.load_external_data_for_tensor(t,str(src.parent)) if onnx.external_data_helper.uses_external_data(t) else None;t.ClearField('external_data');t.data_location=T.DEFAULT;full_weights.append(t)
full_inputs=[h.make_tensor_value_info(n,T.FLOAT16,[1,288,512 if n.startswith('add_') else 1024]) for n in order]
full=h.make_model(h.make_graph(full_nodes,'reference_three_full_tails',full_inputs,outputs,full_weights),opset_imports=list(m.opset_import));full.ir_version=10;full=onnx.shape_inference.infer_shapes(full,strict_mode=True);onnx.checker.check_model(full,full_check=True);onnx.save(full,P/'tails-full-cpu.onnx')
