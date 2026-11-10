"""One-frame full-head impact check for a native vision boundary.

All recurrent state starts at zero. It is intentionally not a recurrent pass.
The unchanged original checker is run by the caller on the saved head arrays.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T
from prepare_reference import submodel, options
from extract_region import sha


def metrics(a, b):
    x,y=a.astype(np.float64).ravel(),b.astype(np.float64).ravel()
    return dict(shape=list(a.shape), nonfinite=int((~np.isfinite(y)).sum()),
        exact_bytes=a.tobytes()==b.tobytes(), max_abs=float(np.max(np.abs(x-y))),
        rmse=float(np.sqrt(np.mean((x-y)**2))),
        correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--road',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--native',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    source=onnx.load(a.source,load_external_data=False)
    for loc in {e.value for t in source.graph.initializer for e in t.external_data if e.key=='location'}:
        if Path(loc).is_absolute() or '..' in Path(loc).parts:
            raise ValueError('Unsafe external data path')
        dest=a.output/loc;dest.parent.mkdir(parents=True,exist_ok=True)
        os.symlink(a.source.parent/loc,dest)
    inputs={v.name:v for v in source.graph.input}
    inputs['view_39']=h.make_tensor_value_info('view_39',T.FLOAT16,[1,32,512])
    remainder=submodel(source,inputs,{v.name:v for v in source.graph.output})
    onnx.save(remainder,a.output/'remainder.onnx')
    onnx.checker.check_model(str(a.output/'remainder.onnx'))
    whole=ort.InferenceSession(str(a.source),options(),providers=['CPUExecutionProvider'])
    rest=ort.InferenceSession(str(a.output/'remainder.onnx'),options(),providers=['CPUExecutionProvider'])
    # ModelSpec is the unchanged repository parser; no acceptance thresholds are reimplemented.
    from jetlink.spec import ModelSpec
    spec=ModelSpec.load(a.road/'spec.json')
    packed=np.load(a.road/'in_packed_0.npy')
    dt={'tensor(float)':np.float32,'tensor(float16)':np.float16,'tensor(uint8)':np.uint8}
    feed={n:packed[at] for n,(at,_) in spec.packed_layout.items()}
    feed['new_img']=np.load(a.road/'in_warped_0.npy')
    for v in whole.get_inputs():
        if v.name in spec.state_pairs:
            feed[v.name]=np.zeros(v.shape,dt[v.type])
        else:
            feed[v.name]=np.ascontiguousarray(feed[v.name],dtype=dt[v.type]).reshape(v.shape)
    names=[v.name for v in whole.get_outputs()]
    baseline=dict(zip(names,whole.run(None,feed)))
    feature=np.load(a.reference/'teacher.npy')
    control=dict(zip(names,rest.run(None,{**feed,'view_39':feature})))
    report=dict(scope='Zero-state recorded-night frame 0 only. No recurrent, phone, or speed acceptance.',
                control={n:metrics(baseline[n],control[n]) for n in names})
    cases={'teacher-cut':control}
    if a.native:
        if a.native.stat().st_size!=32768:
            raise ValueError('Native feature file byte count mismatch')
        native=np.fromfile(a.native,dtype='<f2').reshape(1,32,512)
        if not np.isfinite(native).all():
            raise ValueError('Nonfinite native feature')
        report['feature']=metrics(feature,native)
        candidate=dict(zip(names,rest.run(None,{**feed,'view_39':native})))
        cases['native']=candidate
        report['native']={n:metrics(baseline[n],candidate[n]) for n in names}
    for label,values in cases.items():
        dest=a.output/label;dest.mkdir()
        shutil.copyfile(a.road/'spec.json',dest/'spec.json')
        np.save(dest/'out_ref_0.npy',baseline['outputs'].reshape(-1))
        np.save(dest/'out_link_0.npy',values['outputs'].reshape(-1))
        for name in spec.state_pairs.values():
            np.save(dest/(name+'.npy'),values[name])
    report['source_sha256']=sha(a.source)
    report['native_sha256']=sha(a.native) if a.native else None
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
