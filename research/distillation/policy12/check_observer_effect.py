"""Eight-frame matched logical-observer control, not an acceptance-suite substitute."""
import json,sys,time
from pathlib import Path
import numpy as np,onnxruntime as ort
from jetlink.spec import ModelSpec
sys.path.insert(0,'/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink/scripts')
from verify_parity import make_inputs
root=Path(__file__).resolve().parent;base=root.parent/'real-block';spec=ModelSpec.load('/workspace/shared/jetlink-history-regenerated/spec.json');opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess=ort.InferenceSession(str(base/'teacher-policy12-capture.onnx'),opt,providers=['CPUExecutionProvider']);shapes={i.name:i.shape for i in sess.get_inputs()};dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8};dtypes={i.name:dt[i.type] for i in sess.get_inputs()};names=[o.name for o in sess.get_outputs()];state={n:np.zeros(shapes[n],dtypes[n]) for n in spec.state_pairs};prior=np.load('/workspace/shared/jetlink-long-calibration/train-teacher-outputs.npy');rows=[]
for i,(warped,packed) in enumerate(make_inputs(spec,8,713)):
 feed={'new_img':warped,**state,**{n:packed[at] for n,(at,_) in spec.packed_layout.items()}};feed={n:np.ascontiguousarray(v,dtype=dtypes[n]).reshape(shapes[n]) for n,v in feed.items()};r=dict(zip(names,sess.run(None,feed)));state={n:r[nxt] for n,nxt in spec.state_pairs.items()};full=r['outputs'].reshape(-1);cse=np.load(root/f'train_out_{i:03d}.npy');row={'frame':i,'uncse_all12_observed_vs_cse_unique8_max_abs':float(abs(full-cse).max()),'uncse_all12_observed_vs_old_single_observed_max_abs':float(abs(full-prior[i].reshape(-1)).max()),'duplicate_block_x_equal':bool(np.array_equal(r['layer_norm_49'],r['layer_norm_72'])),'duplicate_block_y_equal':bool(np.array_equal(r['linear_99'],r['linear_140']))};rows.append(row);print(row,flush=True)
(root/'observer-control.json').write_text(json.dumps({'frames':8,'train_seed':713,'limitation':'Short control only; CPU required FP16 transformations may depend on exposed intermediate outputs','rows':rows},indent=2))
