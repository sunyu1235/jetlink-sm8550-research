"""Read-only saturation audit of fixed trained encodings on held-out road activations."""
import argparse,json
from pathlib import Path
import numpy as np,onnx
from onnx import numpy_helper
p=argparse.ArgumentParser();p.add_argument('--qdq',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=320);a=p.parse_args()
m=onnx.load(a.qdq);params={i.name:numpy_helper.to_array(i) for i in m.graph.initializer};s=float(params['layer_norm_49_scale']);z=int(params['layer_norm_49_zero_point']);lo=-z*s;hi=(65535-z)*s
rows=[]
for i in range(a.frames):
 x=np.load(a.capture/f'block_x_{i}.npy').astype(np.float32)
 if not np.isfinite(x).all():raise ValueError('nonfinite captured input')
 rows.append(dict(frame=i,minimum=float(x.min()),maximum=float(x.max()),below=int((x<lo).sum()),above=int((x>hi).sum()),values=x.size))
r=dict(scope='held-out recorded-road first projection input only; fixed original synthetic-trained encoding, no fitting or acceptance claim',range=[lo,hi],frames=a.frames,clipped=sum(x['below']+x['above'] for x in rows),values=sum(x['values'] for x in rows),frames_detail=rows);a.out.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='frames_detail'})
