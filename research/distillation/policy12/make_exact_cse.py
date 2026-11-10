"""Common-subexpression elimination on deterministic pure ONNX value signatures.
No algebraic reassociation, approximate operator substitution, or training.
"""
import copy,json,collections
from pathlib import Path
import onnx
from onnx import helper
from audit_sharing import m,sig,base
out=base/'teacher-exact-cse.onnx'
if out.exists():raise FileExistsError(out)
original_nodes=len(m.graph.node);original_initializers=len(m.graph.initializer)
outputs=[copy.deepcopy(v) for v in list(m.graph.output)[:4]];seen={};aliases={};kept=[]
def canonical(name):
 while name in aliases:name=aliases[name]
 return name
for w in m.graph.initializer:
 k=sig(w.name)
 if k in seen:aliases[w.name]=seen[k]
 else:seen[k]=w.name
for n in m.graph.node:
 if n.domain not in ('','ai.onnx'):raise ValueError('custom operation requires separate purity proof')
 duplicate=[]
 for name in n.output:
  k=sig(name)
  if k in seen:aliases[name]=seen[k];duplicate.append(True)
  else:seen[k]=name;duplicate.append(False)
 if all(duplicate):continue
 c=copy.deepcopy(n)
 for i,name in enumerate(c.input):c.input[i]=canonical(name)
 kept.append(c)
produced={name for n in kept for name in n.output}
for v in outputs:
 if v.name not in produced and v.name in aliases:
  kept.append(helper.make_node('Identity',[canonical(v.name)],[v.name],name='cse_preserve_'+v.name))
producer={name:i for i,n in enumerate(kept) for name in n.output};weights={w.name:w for w in m.graph.initializer};inputs={x.name for x in m.graph.input};live=set();needed_weights=set()
def visit(name):
 if not name or name in inputs:return
 if name in weights:needed_weights.add(name);return
 if name not in producer:raise ValueError('missing producer '+name)
 i=producer[name]
 if i in live:return
 live.add(i)
 for v in kept[i].input:visit(v)
for v in outputs:visit(v.name)
nodes=[n for i,n in enumerate(kept) if i in live];inits=[w for w in m.graph.initializer if w.name in needed_weights]
old_gelu=sum(n.op_type=='Gelu' for n in m.graph.node);old_counts=collections.Counter(n.op_type for n in m.graph.node)
del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.initializer[:];m.graph.initializer.extend(inits);del m.graph.output[:];m.graph.output.extend(outputs)
values={name for n in nodes for name in n.output}|inputs|needed_weights;infos=[v for v in m.graph.value_info if v.name in values];del m.graph.value_info[:];m.graph.value_info.extend(infos)
onnx.save(m,out);onnx.checker.check_model(str(out));new_counts=collections.Counter(n.op_type for n in nodes)
report={'status':'EXACT_VALUE_GRAPH_REWRITE_NOT_NUMERICALLY_ACCEPTED','model':str(out),'nodes_before':original_nodes,'nodes_after':len(nodes),'initializers_before':original_initializers,'initializers_after':len(inits),'gelu_before':old_gelu,'gelu_after':new_counts['Gelu'],'ops_removed':dict(old_counts-new_counts),'aliases':aliases,'original_four_outputs_preserved':True,'arithmetic_reassociation':False,'weights_quantized_or_retrained':False,'external_file_not_physically_compacted':True}
Path(__file__).with_name('exact-cse.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='aliases'})
