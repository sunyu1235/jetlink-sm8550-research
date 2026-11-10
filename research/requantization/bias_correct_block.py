"""PTQ bias correction using only calibration-sequence teacher targets.
No acceptance-set target fitting. Writes a new QDQ graph, retaining original.
"""
import argparse,json
from pathlib import Path
import numpy as np,onnx
from onnx import numpy_helper
from quantize_block import session,run,metrics
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(exist_ok=False,parents=True)
d=np.load(a.data,allow_pickle=False);s=session(a.candidate);pred=run(s,d['x_train']);preval=run(s,d['x_validation']);del s
residual=d['y_train'].astype(np.float64)-pred.astype(np.float64)
correction=np.mean(residual,axis=tuple(range(residual.ndim-1))).astype(np.float32)
m=onnx.load(a.candidate);target='model.off_policy.temporal_summarizer.transformer.layers.0.mlp.c_proj.bias'
weights=[t for t in m.graph.initializer if t.name==target]
if len(weights)!=1:raise ValueError('missing or duplicate final bias')
w=weights[0];v=numpy_helper.to_array(w)
if v.shape!=correction.shape:raise ValueError('unexpected correction shape')
w.CopyFrom(numpy_helper.from_array(v+correction,target));onnx.checker.check_model(m);path=a.output/'bias-corrected.onnx';onnx.save(m,path)
s=session(path);postval=run(s,d['x_validation']);del s
np.save(a.output/'correction.npy',correction)
report={'method':'training-sequence mean output residual added to final bias','calibration_seed':713,'validation_seed':991,
        'used_acceptance_set_targets':False,'correction_max_abs':float(abs(correction).max()),
        'before_validation':metrics(preval,d['y_validation']),'after_validation':metrics(postval,d['y_validation']),
        'scope':'real block on synthetic calibration, no full-model or HTP validation'}
(a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
