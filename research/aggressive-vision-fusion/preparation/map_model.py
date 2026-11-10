"""Map static FP16 GELUs to QNN custom package, preserving serialized weight metadata."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
import onnx
from onnx import helper as h, TensorProto as T
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
assert a.source.parent.resolve()==a.output.parent.resolve(),'keep external weights alongside graph'
assert not a.output.exists(),'refuse overwrite'
m=onnx.load(a.source,load_external_data=False);before=[t.SerializeToString() for t in m.graph.initializer]
v={x.name:x for x in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output)}
existing={x.name for x in m.graph.initializer}|set(v);nodes=[];changes=[]
for n in m.graph.node:
 if n.op_type!='Gelu' or n.domain not in ('','ai.onnx'):nodes.append(n);continue
 t=v[n.input[0]].type.tensor_type;shape=[d.dim_value for d in t.shape.dim]
 assert t.elem_type==T.FLOAT16 and 1<=len(shape)<=4 and all(x>0 for x in shape),(n.name,shape)
 mode=next((x.s.decode() for x in n.attribute if x.name=='approximate'),'none');assert mode in ('none','tanh')
 op='GeluErfFp16' if mode=='none' else 'GeluTanhFp16'; nn=copy.deepcopy(n); nn.domain='jetlink_gelu'; nn.op_type=op; del nn.attribute[:]
 padded=[1]*(4-len(shape))+shape
 if len(shape)!=4:
  prefix='htp_gelu/'+str(len(changes));names=[prefix+s for s in ('/shape4','/shapeOriginal','/input4','/output4')];assert not existing.intersection(names);existing.update(names)
  s4,so,i4,o4=names;m.graph.initializer.extend([h.make_tensor(s4,T.INT64,[4],padded),h.make_tensor(so,T.INT64,[len(shape)],shape)])
  nodes.append(h.make_node('Reshape',[n.input[0],s4],[i4],name=prefix+'/pre'));nn.input[0]=i4;nn.output[0]=o4;nodes.append(nn);nodes.append(h.make_node('Reshape',[o4,so],list(n.output),name=prefix+'/post'))
  m.graph.value_info.extend([h.make_tensor_value_info(i4,T.FLOAT16,padded),h.make_tensor_value_info(o4,T.FLOAT16,padded)])
 else:nodes.append(nn)
 changes.append(dict(name=n.name,mode=mode,shape=shape,padded=padded))
assert len(changes)==63
assert before==[t.SerializeToString() for t in list(m.graph.initializer)[:len(before)]]
del m.graph.node[:];m.graph.node.extend(nodes);m.opset_import.append(h.make_opsetid('jetlink_gelu',1));onnx.save(m,a.output);onnx.checker.check_model(str(a.output))
r=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),counts=dict(collections.Counter(x['mode'] for x in changes)),rank_adapted=sum(len(x['shape'])!=4 for x in changes),original_weights_metadata_unchanged=True,changes=changes,validated='ONNX checker only, no model execution')
a.report.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='changes'})
