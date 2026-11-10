"""Replace exactly the selected QDQ projection by actual x86 QNN host execution.
This host diagnostic does not deploy a QNN EP graph or measure device speed.
"""
import argparse,hashlib,json
from pathlib import Path
import onnx
from onnx import helper
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.source.parent.resolve()!=a.out.parent.resolve() or a.out.exists():raise ValueError('new output must be beside original external weights')
m=onnx.load(a.source,load_external_data=False);selected=[n for n in m.graph.node if n.name.startswith('rq_first/')]
if len(selected)!=8 or selected[0].input[0]!='layer_norm_49' or selected[-1].output[0]!='val_646':raise ValueError('unexpected first QDQ island')
replacement=helper.make_node('QnnFirstProjection',['layer_norm_49'],['val_646'],name='qnn_host_first_projection',domain='jetlink.qnn')
nodes=[];done=False
for n in m.graph.node:
 if n.name.startswith('rq_first/'):
  if not done:nodes.append(replacement);done=True
 else:nodes.append(n)
del m.graph.node[:];m.graph.node.extend(nodes)
used={i for n in nodes for i in n.input};inits=[i for i in m.graph.initializer if i.name in used];del m.graph.initializer[:];m.graph.initializer.extend(inits)
m.opset_import.append(helper.make_opsetid('jetlink.qnn',1));onnx.save(m,a.out);onnx.checker.check_model(str(a.out))
print(json.dumps(dict(scope='x86 QNN host bridge numerical experiment; not hardware deployment',graph=str(a.out),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),removed_nodes=len(selected))))
