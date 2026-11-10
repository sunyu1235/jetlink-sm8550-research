"""Real teacher weights, separate synthetic sequences. Not route validation."""
from pathlib import Path
import json,time
import numpy as np
from fit_relu_student import fit,predict,errors
p=Path('/workspace/shared/jetlink-distillation/real-block');data=np.load(p/'pairs.npz');results=[]
for width in [1024,768,512]:
 t=time.monotonic();s=fit(data['x_train'],data['y_train'],data['w1'],data['b1'],width,1e-4)
 elapsed=time.monotonic()-t
 stored={k:v.astype(np.float16) if k!='keep' else v for k,v in s.items()}
 if not all(np.isfinite(v).all() for v in stored.values()):raise ValueError('FP16 overflow')
 np.savez(p/f'student-{width}.npz',**stored)
 reload={k:v.astype(np.float64) if k!='keep' else v for k,v in stored.items()}
 report={'width':width,'fit_seconds':elapsed,'train':errors(predict(reload,data['x_train']),data['y_train']),'validation':errors(predict(reload,data['x_validation']),data['y_validation'])}
 results.append(report);print(width,report['validation']['rmse'],report['validation']['max_abs'],flush=True)
(p/'distillation-results.json').write_text(json.dumps({'status':'REAL_BLOCK_SYNTHETIC_SEQUENCES_ONLY','computation':'FP16 stored weights; FP64 matmul not device arithmetic','train_frames':8,'validation_frames':4,'train_seed':713,'validation_seed':991,'results':results},indent=2))
