"""Replace only proven 102-node compact cut with a six-input native host pipeline."""
from pathlib import Path
import hashlib,json
import onnx
from onnx import helper as h,TensorProto as T
P=Path(__file__).parent
S=Path('/workspace/shared/jetlink-distillation/real-block')
src=S/'teacher-capture.onnx'; out=S/'teacher-native-compact.onnx'
if out.exists():raise FileExistsError(out)
m=onnx.load(src,load_external_data=False)
cut=onnx.load('/workspace/shared/jetlink-compact-boundaries/artifacts/reconstructed-baseline-131-41.onnx',load_external_data=False)
removed={n.name for n in cut.graph.node}
for b in json.loads(Path('/workspace/shared/jetlink-distillation/last-query-tail/proof.json').read_text())['branches']:
 removed.update([b['gelu']]+b['tail'])
assert len(removed)==102
inputs=['gelu_43','gelu_54','gelu_61','add_50','add_70','add_86'];outputs=['select_4','select_7','select_10']
selected=[n for n in m.graph.node if n.name in removed]
assert len(selected)==102
produced={o for n in selected for o in n.output};init={t.name for t in m.graph.initializer}
assert {v for n in selected for v in n.input if v and v not in produced and v not in init}==set(inputs)
rest=[n for n in m.graph.node if n.name not in removed]
assert {v for n in rest for v in n.input if v in produced}==set(outputs)
assert not ({v.name for v in m.graph.output} & produced)
original_bytes={n.name:n.SerializeToString() for n in rest}
native=h.make_node('QnnCompactPipeline',inputs,outputs,name='jetlink_native_compact_pipeline',domain='jetlink.qnn')
pending=rest+[native];known={v.name for v in m.graph.input}|init;ordered=[]
while pending:
 ready=[n for n in pending if all(not v or v in known for v in n.input)]
 if not ready:raise ValueError('cyclic or unresolved cut')
 for n in ready:
  assert not set(n.output)&known
  ordered.append(n);known.update(n.output)
 ids={id(n) for n in ready};pending=[n for n in pending if id(n) not in ids]
del m.graph.node[:];m.graph.node.extend(ordered)
assert all(n.SerializeToString()==original_bytes[n.name] for n in m.graph.node if n.name in original_bytes)
original_outputs=list(m.graph.output)[:4];del m.graph.output[:];m.graph.output.extend(original_outputs)
used={i for n in m.graph.node for i in n.input};weights=[t for t in m.graph.initializer if t.name in used];del m.graph.initializer[:];m.graph.initializer.extend(weights)
del m.graph.value_info[:]
for i,n in enumerate(inputs):m.graph.value_info.append(h.make_tensor_value_info(n,T.FLOAT16,[1,288,1024 if i<3 else 512]))
for n in outputs:m.graph.value_info.append(h.make_tensor_value_info(n,T.FLOAT16,[1,512]))
m.opset_import.append(h.make_opsetid('jetlink.qnn',1));onnx.save(m,out);onnx.checker.check_model(str(out))
r=dict(scope='Native compact partition and tails via host bridge; CPU remainder',model=str(out),model_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),removed_nodes=sorted(removed),inputs=inputs,outputs=outputs,unrelated_nodes_byte_identical=True,backend_acceptance='pending')
(P/'integration.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
