#!/usr/bin/env python3
"""Extract only replay-relevant metadata from the public route, without model inference."""
import argparse
import collections
import json
from pathlib import Path
import capnp
import zstandard


CAMERA = {'narrowRoadCameraState','wideRoadCameraState','narrowRoadEncodeIdx','wideRoadEncodeIdx',
          'extrinsicsCalibration','deviceState'}
CONTEXT = {'initData','driverMonitoringStateDEPRECATED','lateralDelay','carParams','modelV2','carControl','carState'}
FIELDS = {'initData': ['version','gitCommit','gitBranch','deviceType','wallTimeNanos'], 'carState': ['vEgo','leftBlinker','rightBlinker','steeringPressed','steeringTorque','leftBlindspot','rightBlindspot'],
          'carControl': ['enabled','latActive','longActive'], 'driverMonitoringStateDEPRECATED': ['isRHD'],
          'carParams': ['longitudinalActuatorDelay','steerActuatorDelay','carFingerprint','brand'],
          'modelV2': ['frameId','frameIdExtra','timestampEof','modelExecutionTime','meta']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).parent)
    a = p.parse_args()
    root = a.root
    schema = capnp.load(str(root/'schema/log.capnp'), imports=[str(root/'schema')])
    raw = zstandard.ZstdDecompressor().decompress((root/'rlog.zst').read_bytes(), max_output_size=250*1024*1024)
    cameras, contexts, counts = [], [], collections.Counter()
    for m in schema.Event.read_multiple_bytes(raw):
        k = m.which()
        counts[k] += 1
        if k in CAMERA | CONTEXT:
            d = getattr(m,k).to_dict()
            if k in FIELDS:
                d = {name:d[name] for name in FIELDS[k] if name in d}
            item = {'type':k,'logMonoTime':m.logMonoTime,'valid':m.valid,'data':d}
            (cameras if k in CAMERA else contexts).append(item)
    (root/'camera_metadata.json').write_text(json.dumps(cameras,indent=2))
    (root/'context_metadata.json').write_text(json.dumps(contexts,indent=2))
    (root/'event_counts.json').write_text(json.dumps(dict(counts),indent=2))
    print(json.dumps({'camera_metadata_events':len(cameras),'context_metadata_events':len(contexts),'counts':dict(counts)}))


if __name__ == '__main__':
    main()
