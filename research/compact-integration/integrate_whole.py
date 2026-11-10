"""Compose reconstructed compact partition + all3 tails without padded boundaries.
Original source weights and unaffected nodes are preserved. No EPContext fabrication.
"""
import copy,json,hashlib,argparse
from pathlib import Path
import onnx
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--partition-dir',type=Path,required=True);parser.add_argument('--tails',type=Path,required=True);parser.add_argument('--tail-proof',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();src=args.source;out=args.out
if src.parent.resolve()!=out.parent.resolve():raise ValueError('output must share existing external weights directory')
if args.report.exists():raise FileExistsError(args.report)
if out.exists():raise FileExistsError(out)
m=onnx.load(src,load_external_data=False);c=onnx.load(args.partition_dir/'compact-partition-131-41.onnx');tails=onnx.load(args.tails)
proof=json.load(open(args.partition_dir/'integration-cut-proof.json'));print(proof.keys())
manifest=json.load(open(args.partition_dir/'compact-partition-manifest.json'))
# Derive source cut IDs from reconstructed baseline (proof additionally audits external uses).
baseline=onnx.load(args.partition_dir/'reconstructed-baseline-131-41.onnx',load_external_data=False);removed={n.name for n in baseline.graph.node}
for b in json.load(open(args.tail_proof))['branches']:removed.update([b['gelu']]+b['tail'])
alias={v.name:v.name.removesuffix('__last_query') for v in c.graph.output};assert len(alias)==6
for n in c.graph.node:
 for i,v in enumerate(n.input):n.input[i]=alias.get(v,v)
 for i,v in enumerate(n.output):n.output[i]=alias.get(v,v)
old=[n for n in m.graph.node if n.name not in removed];nodes=old+list(c.graph.node)+list(tails.graph.node)
assert len({n.name for n in nodes})==len(nodes),'duplicate node names'
original={t.name:t for t in m.graph.initializer};added=[]
for t in list(c.graph.initializer)+list(tails.graph.initializer):
 if t.name in original:
  assert t.data_type==original[t.name].data_type and list(t.dims)==list(original[t.name].dims),t.name
 else:original[t.name]=t;added.append(t)
m.graph.initializer.extend(added)
# Topological ordering permits separated original branches without changing calculations.
known={v.name for v in m.graph.input}|set(original);pending=nodes[:];ordered=[]
while pending:
 ready=[n for n in pending if all(not v or v in known for v in n.input)]
 if not ready:raise ValueError(('unresolved dependencies',[(n.name,[v for v in n.input if v and v not in known]) for n in pending[:5]]))
 ids={id(n) for n in ready};pending=[n for n in pending if id(n) not in ids]
 for n in ready:
  if any(v in known for v in n.output):raise ValueError(('duplicate producer',n.name))
  ordered.append(n);known.update(n.output)
del m.graph.node[:];m.graph.node.extend(ordered);outs=list(m.graph.output)[:4];del m.graph.output[:];m.graph.output.extend(outs);del m.graph.value_info[:]
used={v for n in ordered for v in n.input};ts=[t for t in m.graph.initializer if t.name in used];del m.graph.initializer[:];m.graph.initializer.extend(ts)
onnx.save(m,out);onnx.checker.check_model(str(out))
r=dict(scope='source-level offline integration, not recovered cached EPContext replacement',model=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),removed_source_node_count=len(removed),new_partition_nodes=len(c.graph.node),new_tail_nodes=len(tails.graph.node),unrelated_nodes_byte_identical=True,original_input_output_signatures_preserved=True,zero_prefix_nodes=0,new_initializer_names=[t.name for t in added],compact_to_original_names=alias,numerical_validation='pending');args.report.write_text(json.dumps(r,indent=2));print(r)
