#!/usr/bin/env python3
"""Recreate archived synthetic replay inputs and fail unless every hash matches."""
import argparse, hashlib, io, json
from pathlib import Path
import numpy as np
from boundary_sensitivity import ROOT, parity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'reports/jetlink-gelu-history-20261101/input-manifest.json').read_text())
    spec_path=ROOT/'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare/spec.json'
    spec=parity.ModelSpec.load(spec_path)
    checks=[]
    for i,(warped,packed) in enumerate(parity.make_inputs(spec,manifest['n'],seed=manifest['seed'])):
        frame=manifest['hashes'][i]
        blobs={}
        for key,value in [('in_warped_npy',warped),('in_packed_npy',packed)]:
            f=io.BytesIO();np.save(f,value,allow_pickle=False)
            blobs[key]=(f.getvalue(),frame[key])
        blobs['new_img']=(np.asarray(warped,dtype='<f2').tobytes(),frame['phone_inputs']['new_img'])
        for name in ('desire','traffic_convention','action_t'):
            at,shape=spec.packed_layout[name]
            blobs[name]=(np.asarray(packed[at].reshape(shape),dtype='<f4').tobytes(),frame['phone_inputs'][name])
        for kind,(blob,expected) in blobs.items():
            digest=hashlib.sha256(blob).hexdigest()
            if len(blob)!=expected['bytes'] or digest!=expected['sha256']:
                raise ValueError(f'frame {i} {kind}: mismatch; stop, do not claim original replay')
            rel=Path(expected['path'].replace('\\','/'))
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe manifest path')
            target=a.output/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
            checks.append({'frame':i,'kind':kind,'sha256':digest})
    (a.output/'spec.json').write_text(spec_path.read_text())
    report={'kind':'exactly-regenerated-synthetic-inputs-not-real-road-calibration','frames':manifest['n'],
            'checked_files':len(checks),'all_hashes_match':True,'numpy_version':np.__version__,
            'source_manifest_sha256':hashlib.sha256((ROOT/'reports/jetlink-gelu-history-20261101/input-manifest.json').read_bytes()).hexdigest(),
            'checks':checks}
    (a.output/'regeneration-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
