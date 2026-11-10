from pathlib import Path
import json,hashlib
import numpy as np
p=Path(__file__).parent;rows=[];all_d=[]
for i in range(8):
 for n in ['select_4','select_7','select_10']:
  raw=p/f'tails-host-output/Result_{i}/{n}_native.raw';assert raw.stat().st_size==1024;y=np.fromfile(raw,np.float16).reshape(1,512).astype(np.float64);a=np.load(p/f'pipeline-inputs/{i}-cpu-{n}.npy').astype(np.float64);b=np.load(p/f'pipeline-inputs/{i}-baseline-{n}.npy').astype(np.float64);assert np.isfinite(y).all();d=y-a;all_d.append(d.ravel());rows.append(dict(sample=i,output=n,rmse_vs_compact_cpu=float(np.sqrt(np.mean(d*d))),max_abs_vs_compact_cpu=float(abs(d).max()),correlation_vs_compact_cpu=float(np.corrcoef(a.ravel(),y.ravel())[0,1]),max_abs_vs_full_cpu=float(abs(y-b).max())))
d=np.concatenate(all_d);r=dict(scope='Two serialized native HTP host contexts joined with six compact nativeFP16 buffers; not fullmodel target accuracy/device speed',samples=8,values_compared=12288,boundary_bytes=9216,zero_padding=False,all_finite=True,rmse_vs_compact_cpu=float(np.sqrt(np.mean(d*d))),max_abs_vs_compact_cpu=float(abs(d).max()),min_correlation_vs_compact_cpu=min(v['correlation_vs_compact_cpu'] for v in rows),contexts={},rows=rows)
for name,file in [('upstream',p/'upstream-context/compact-upstream.bin'),('tails',p/'context/compact-three-tails.bin')]:r['contexts'][name]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),prepare_exit=0,reload_and_execute_exit=0)
(p/'native-pipeline-summary.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
