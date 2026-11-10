"""Compose fitted MLPs. Fresh per-block constants avoid changing CSE-shared weights."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,onnx
from onnx import helper,numpy_helper
ap=argparse.ArgumentParser();ap.add_argument('blocks',nargs='+');ap.add_argument('--name',required=True);ap.add_argument('--cse',action='store_true');args=ap.parse_args();root=Path(__file__).resolve().parent;base=root.parent/'real-block';target=base/(args.name+'.onnx')
if target.exists():raise FileExistsError(target)
manifest={b['name']:b for b in json.loads((root/'blocks.json').read_text())};assert len(set(args.blocks))==len(args.blocks)
source=base/('teacher-exact-cse.onnx' if args.cse else 'teacher-capture.onnx');m=onnx.load(source,load_external_data=False);prod={v:n for n in m.graph.node for v in n.output};cons={}
for n in m.graph.node:
 for v in n.input:cons.setdefault(v,[]).append(n)
byname={n.name:n for n in m.graph.node};replacements={};selected=[];new_weights=[]
for name in args.blocks:
 b=manifest[name]
 if b['gelu_node'] not in byname:raise ValueError('block absent/shared in source: '+name)
 p=(root.parent/'long-silu-student/student-best.npz') if name in ['gelu_41','gelu_52'] else root/name/'student-best.npz'
 gelu=byname[b['gelu_node']];bias1=prod[gelu.input[0]];fc=prod[bias1.input[0]];projs=cons[gelu.output[0]];assert len(projs)==1;proj=projs[0];tails=cons[proj.output[0]];assert len(tails)==1;bias2=tails[0]
 assert [fc.op_type,bias1.op_type,proj.op_type,bias2.op_type]==['MatMul','Add','MatMul','Add']
 with np.load(p,allow_pickle=False) as w:
  for key,node,shape in [('w1',fc,(512,1024)),('b1',bias1,(1024,)),('w2',proj,(1024,512)),('b2',bias2,(512,))]:
   a=w[key];assert a.dtype==np.float16 and a.shape==shape and np.isfinite(a).all();new_name='student_'+name+'_'+key;new_weights.append(numpy_helper.from_array(a,new_name));node.input[1]=new_name
 replacements[gelu.name]=name;selected.append(dict(block=name,weights_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reuse_original_passing_single_block=(name in ['gelu_41','gelu_52'])))
ns=[]
for n in m.graph.node:
 if n.name not in replacements:ns.append(n);continue
 name=replacements[n.name];scaled='student_'+name+'_scaled';sig='student_'+name+'_sig'
 ns.extend([helper.make_node('Mul',[n.input[0],'student_policy12_beta'],[scaled]),helper.make_node('Sigmoid',[scaled],[sig]),helper.make_node('Mul',[n.input[0],sig],list(n.output))])
new_weights.append(numpy_helper.from_array(np.array(1.702,np.float16),'student_policy12_beta'));all_weights=list(m.graph.initializer)+new_weights;used={v for n in ns for v in n.input};inits=[w for w in all_weights if w.name in used]
del m.graph.initializer[:];m.graph.initializer.extend(inits);del m.graph.node[:];m.graph.node.extend(ns);original_outputs=list(m.graph.output)[:4];del m.graph.output[:];m.graph.output.extend(original_outputs)
onnx.save(m,target);onnx.checker.check_model(str(target));(root/(args.name+'.json')).write_text(json.dumps({'status':'COMPOSED_NOT_ACCEPTED','model':str(target),'cse_source':args.cse,'replacements':selected,'remaining_gelu_nodes':sum(n.op_type=='Gelu' for n in m.graph.node),'original_output_signature_preserved':True,'new_constants_are_private_to_each_student_block':True},indent=2));print(target)
