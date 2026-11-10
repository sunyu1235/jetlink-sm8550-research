import numpy as np, onnxruntime as ort,json
from pathlib import Path
p=Path('/workspace/shared/jetlink-exact-integration'); z=np.load('/workspace/shared/jetlink-long-calibration/sequences.npz'); x=z['x_validation']
o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1
s=ort.InferenceSession('/workspace/shared/jetlink-distillation/long-silu-student/student-block.onnx',o,providers=['CPUExecutionProvider'])
indices=[0,1,2,3,32,48,64,80,96,112,128,144,156,157,158,159]; details=[]; ss=0; n=0
for j,i in enumerate(indices):
 a=s.run(None,{'x':x[i].astype(np.float16)})[0].astype(np.float64); b=np.fromfile(p/f'student-htp-host-outputs/Result_{j}/y.raw',np.float32).reshape(a.shape).astype(np.float64); d=b-a; ss+=np.sum(d*d); n+=d.size
 details.append(dict(frame=i,rmse=float(np.sqrt(np.mean(d*d))),max_abs=float(np.max(abs(d))),correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]),finite=bool(np.isfinite(b).all())))
r=dict(scope='QNN 2.42 x86 HTP host functional standalone FP16 SiLU student block vs ORT CPU; not hardware or whole-model validation',frames=len(indices),rmse=float(np.sqrt(ss/n)),max_abs=max(v['max_abs'] for v in details),frames_detail=details)
(p/'student-htp-host16-comparison.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
