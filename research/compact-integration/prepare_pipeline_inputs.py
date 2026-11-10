from pathlib import Path
import json
import numpy as np,onnxruntime as ort
from compact_bindings import bind,execute
p=Path(__file__).parent;q=Path('/workspace/shared/jetlink-compact-boundaries/artifacts');d=p/'pipeline-inputs';d.mkdir(exist_ok=True)
o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
up=ort.InferenceSession(str(q/'compact-partition-131-41.onnx'),o,providers=['CPUExecutionProvider']);tail=ort.InferenceSession(str(p/'tails-cpu.onnx'),o,providers=['CPUExecutionProvider']);base=ort.InferenceSession(str(q/'reconstructed-baseline-131-41.onnx'),o,providers=['CPUExecutionProvider']);fulltail=ort.InferenceSession(str(p/'tails-full-cpu.onnx'),o,providers=['CPUExecutionProvider']);rng=np.random.default_rng(7147);lines=[];rows=[]
for i in range(8):
 x={v.name:rng.normal(0,.5,v.shape).astype(np.float16) for v in up.get_inputs()};parts=[]
 for n,a in x.items():f=d/f'{i}-{n}.raw';a.tofile(f);parts.append(n+':='+str(f))
 lines.append(' '.join(parts));a=execute(up,tail,x);full=fulltail.run(None,dict(zip([v.name for v in base.get_outputs()],base.run(None,x),strict=True)))
 delta=[]
 for (n,y),ref in zip(a.items(),full):np.save(d/f'{i}-cpu-{n}.npy',y);np.save(d/f'{i}-baseline-{n}.npy',ref);delta.append((y.astype(np.float64)-ref.astype(np.float64)).ravel())
 diff=np.concatenate(delta);rows.append(dict(sample=i,max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff**2)))))
(d/'inputs.txt').write_text('\n'.join(lines)+'\n');(p/'pipeline-cpu-comparison.json').write_text(json.dumps(dict(scope='CPU compact upstream + strict compact bindings + 3tail vs reconstructed full-size partition and tails; diagnostic only',rows=rows),indent=2));print(rows)
