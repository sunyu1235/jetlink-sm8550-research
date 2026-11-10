#!/usr/bin/env python3
"""Splice a QDQ block into source model and replay independently recurrent on CPU.
This is model-level numerical screening, not HTP compilation/speed validation.
"""
import argparse,copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
import onnxruntime as ort
from boundary_sensitivity import ROOT,parity,check

def build(source,qdq,out):
    m=onnx.load(source);q=onnx.load(qdq)
    selected={'node_MatMul_546','node_linear_98','node_gelu_41','node_MatMul_548','node_linear_99'}
    if sum(n.name in selected for n in m.graph.node)!=5:raise ValueError('unexpected original block')
    prefix='rq_pilot/'
    for n in q.graph.node:
        n.name=prefix+n.name
        for j,k in enumerate(n.input):
            if k:n.input[j]=prefix+k
        for j,k in enumerate(n.output):n.output[j]=prefix+k
    for t in q.graph.initializer:t.name=prefix+t.name
    inserted=[helper.make_node('Cast',['layer_norm_49'],[prefix+'layer_norm_49'],name=prefix+'input_fp32',to=TensorProto.FLOAT),*q.graph.node,
              helper.make_node('Cast',[prefix+'linear_99'],['linear_99'],name=prefix+'output_fp16',to=TensorProto.FLOAT16)]
    nodes=[];done=False
    for n in m.graph.node:
        if n.domain=='org.tinygrad':
            if n.op_type!='Contiguous' or len(n.input)!=1 or len(n.output)!=1:raise ValueError('unknown custom op')
            n.op_type='Identity';n.domain='';del n.attribute[:]
        if n.name in selected:
            if not done:nodes.extend(inserted);done=True
        else:nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes)
    m.graph.initializer.extend(q.graph.initializer)
    used={x for n in m.graph.node for x in n.input}
    retained=[t for t in m.graph.initializer if t.name in used];del m.graph.initializer[:];m.graph.initializer.extend(retained)
    # Old FP16 metadata for removed intermediate nodes must not constrain replacement internals.
    removed={'val_646','linear_98','gelu_41','val_648'}
    vis=[v for v in m.graph.value_info if v.name not in removed];del m.graph.value_info[:];m.graph.value_info.extend(vis)
    domains={x.domain:x.version for x in m.opset_import}
    for imp in q.opset_import:
        if imp.domain not in domains:m.opset_import.append(copy.deepcopy(imp))
        elif imp.version>domains[imp.domain]:raise ValueError('unreviewed opset upgrade')
    onnx.checker.check_model(m)
    onnx.save_model(m,out,save_as_external_data=True,all_tensors_to_one_file=True,location=out.name+'.weights',size_threshold=1024)
    del m,q,nodes,retained;gc.collect()

def replay(path,inputs,spec,nframes,dest,custom_library=None):
    dest.mkdir(parents=True,exist_ok=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;options.log_severity_level=3
    if custom_library is not None:
        libraries=custom_library if isinstance(custom_library,(list,tuple)) else [custom_library]
        for library in libraries:options.register_custom_ops_library(str(library))
    s=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8}
    ins={x.name:(tuple(x.shape),dt[x.type]) for x in s.get_inputs()};names=[x.name for x in s.get_outputs()]
    state={k:np.zeros(ins[k][0],ins[k][1]) for k in spec.state_pairs};ys=[];timings=[]
    for i in range(nframes):
        warped=np.load(inputs/f'in_warped_{i}.npy');packed=np.load(inputs/f'in_packed_{i}.npy')
        feed={'new_img':warped,**state,**{k:packed[at] for k,(at,_) in spec.packed_layout.items()}}
        feed={k:np.asarray(v,dtype=ins[k][1]).reshape(ins[k][0]) for k,v in feed.items()}
        start=time.monotonic();r=dict(zip(names,s.run(None,feed),strict=True));timings.append(time.monotonic()-start)
        if not all(np.isfinite(v).all() for v in r.values()):raise ValueError('nonfinite output/state')
        state={k:r[nxt] for k,nxt in spec.state_pairs.items()};ys.append(r['outputs'].reshape(-1));np.save(dest/f'out_{i}.npy',ys[-1])
        print(dest.name,i,round(timings[-1],3),flush=True)
    del s;gc.collect()
    return ys,timings

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=16);p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    source=Path('/workspace/shared/jetlink-models/light-speed.onnx');qdq=Path('/workspace/shared/jetlink-qdq-real-block-v2/QUInt16-w8-pc0.onnx')
    candidate=a.out/'candidate.onnx';build(source,qdq,candidate)
    inputs=Path('/workspace/shared/jetlink-history-regenerated');spec=parity.ModelSpec.load(inputs/'spec.json')
    saved=ROOT/'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare'
    refs=[np.load(saved/f'out_ref_{i}.npy').reshape(-1) for i in range(a.frames)]
    baseline,bt=replay(a.baseline,inputs,spec,a.frames,a.out/'baseline')
    basecode,log=check(spec,baseline,refs);(a.out/'baseline-vs-archived.log').write_text(log)
    actual,ct=replay(candidate,inputs,spec,a.frames,a.out/'candidate')
    code,log=check(spec,actual,refs);(a.out/'candidate-vs-archived.log').write_text(log)
    owncode,log=check(spec,actual,baseline);(a.out/'candidate-vs-own-baseline.log').write_text(log)
    report={'kind':'CPU_WHOLE_MODEL_RECURRENT_SYNTHETIC_REPLAY_NOT_HTP_SPEED_VALIDATION','frames':a.frames,
            'candidate':'one policy MLP A16/W8, other original weights/operators unchanged except layout-only Contiguous->Identity',
            'ort_version':ort.__version__,'baseline_vs_archived_exit':basecode,'candidate_vs_archived_exit':code,'candidate_vs_own_baseline_exit':owncode,
            'baseline_seconds':bt,'candidate_seconds':ct,'qdq_sha256':hashlib.sha256(qdq.read_bytes()).hexdigest(),
            'note':'CPU timings only; original baseline graph has extra capture outputs. Do not compare these CPU timings as a speed result.'}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
