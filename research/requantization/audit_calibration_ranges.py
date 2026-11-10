"""Measure long-history teacher activation coverage under short-calibration ranges."""
import argparse,json
from pathlib import Path
import numpy as np,onnx
from onnx import helper,TensorProto
from quantize_block import session
p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
d=np.load(a.data,allow_pickle=False);x=d['x'] if 'x' in d else d['x_train']
ranges=json.loads((Path(__file__).parent/'real-block-results/short-calibration-ranges.json').read_text())
m=onnx.load('/workspace/shared/jetlink-qdq-real-block-v2/block-fp32.onnx')
for n,width in [('val_646',1024),('gelu_41',1024),('val_648',512)]:m.graph.output.append(helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,288,width]))
tmp=a.output.with_suffix('.probe.onnx');onnx.save(m,tmp);s=session(tmp);names=[v.name for v in s.get_outputs()]
report={n:{'min':float('inf'),'max':float('-inf'),'out_of_range':0,'values':0,'frames_outside':[]} for n in ranges}
for i,xx in enumerate(x):
 vals=dict(zip(names,s.run(None,{'layer_norm_49':xx.astype(np.float32)}),strict=True));vals['layer_norm_49']=xx
 for name,r in ranges.items():
  v=vals[name].astype(np.float64);record=report[name];bad=int(np.count_nonzero((v<r['min_representable'])|(v>r['max_representable'])))
  record['min']=min(record['min'],float(v.min()));record['max']=max(record['max'],float(v.max()));record['out_of_range']+=bad;record['values']+=v.size
  if bad:record['frames_outside'].append(i)
a.output.write_text(json.dumps({'scope':'teacher-block range coverage, not candidate head attribution','frames':len(x),'tensors':report},indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='frames_outside'} for k,v in report.items()},indent=2))
