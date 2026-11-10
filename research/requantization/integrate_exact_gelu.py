#!/usr/bin/env python3
"""Add the existing exact CPU GELU repair to a numerically screened QDQ model.
Keep external weights in the same directory, byte-unmodified. Never claim HTP placement.
"""
import argparse,collections,copy,hashlib,json
from pathlib import Path
import onnx
from onnx import TensorProto
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--qdq-block',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
if a.source.parent.resolve()!=a.output.parent.resolve():raise ValueError('preserve external-weight directory; output alongside source')
if a.output.exists():raise ValueError('refusing overwrite')
m=onnx.load(a.source,load_external_data=False);q=onnx.load(a.qdq_block,load_external_data=False)
# Restore exact typed metadata of the inserted QDQ block, derived from its graph.
existing={v.name for v in m.graph.value_info}
for vi in list(q.graph.input)+list(q.graph.value_info)+list(q.graph.output):
 v=copy.deepcopy(vi);v.name='rq_pilot/'+v.name
 if v.name not in existing:m.graph.value_info.append(v);existing.add(v.name)
types={v.name:v.type.tensor_type.elem_type for v in list(m.graph.input)+list(m.graph.output)+list(m.graph.value_info)}
changes=[];weights_before=[t.SerializeToString() for t in m.graph.initializer]
for n in m.graph.node:
 if n.op_type!='Gelu':continue
 dt=types.get(n.input[0]);approx=next((x.s.decode() for x in n.attribute if x.name=='approximate'),'none')
 if dt not in (TensorProto.FLOAT16,TensorProto.FLOAT) or approx not in ('none','tanh'):raise ValueError(f'unsupported {n.name} dtype {dt} mode {approx}')
 op='ExactGelu'+('Tanh' if approx=='tanh' else '')+('32' if dt==TensorProto.FLOAT else '')
 changes.append({'node':n.name,'input':n.input[0],'dtype':int(dt),'formula':approx,'replacement':op})
 n.op_type=op;n.domain='jetlink.precision';attrs=[x for x in n.attribute if x.name!='approximate'];del n.attribute[:];n.attribute.extend(attrs)
if len(changes)!=63:raise ValueError('unexpected GELU replacement count')
counts=dict(collections.Counter(c['replacement'] for c in changes))
if counts!={'ExactGelu':36,'ExactGeluTanh':26,'ExactGeluTanh32':1}:raise ValueError(counts)
if any(x.domain=='jetlink.precision' for x in m.opset_import):raise ValueError('already repaired')
m.opset_import.append(onnx.helper.make_opsetid('jetlink.precision',1))
assert weights_before==[t.SerializeToString() for t in m.graph.initializer]
onnx.save(m,a.output);onnx.checker.check_model(str(a.output))
r={'source_graph_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'output_graph_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'counts':counts,'weights_metadata_unchanged':True,'changes':changes,'scope':'CPU custom operators; no HTP package or placement claim'};a.report.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='changes'}))
