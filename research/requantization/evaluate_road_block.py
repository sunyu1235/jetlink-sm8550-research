"""Fixed candidate held-out road block test; no training or full-model claim."""
import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=320);a=p.parse_args()
o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;s=ort.InferenceSession(str(a.candidate),o,providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name;dtype=np.float32 if s.get_inputs()[0].type=='tensor(float)' else np.float16
rows=[];sumerr=sumenergy=0.;count=0
for i in range(a.frames):
 x=np.load(a.capture/f'block_x_{i}.npy').astype(dtype); y=np.load(a.capture/f'block_y_{i}.npy').astype(np.float64); c=s.run(None,{name:x})[0].astype(np.float64);d=c-y
 if not np.isfinite(c).all():raise ValueError('nonfinite candidate')
 e=float(np.sum(d*d));en=float(np.sum(y*y));sumerr+=e;sumenergy+=en;count+=d.size
 rows.append(dict(frame=i,rmse=float(np.sqrt(e/d.size)),normalized_rmse=float(np.sqrt(e/en)),max_abs=float(np.max(abs(d)))))
r=dict(scope='fixed CPU block only on held-out recorded road activations; not recurrent model parity or device speed',frames=a.frames,rmse=float(np.sqrt(sumerr/count)),normalized_rmse=float(np.sqrt(sumerr/sumenergy)),max_abs=max(x['max_abs'] for x in rows),frames_detail=rows);a.out.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='frames_detail'})
