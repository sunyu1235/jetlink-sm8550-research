#!/usr/bin/env python3
"""Read-only checks of local raw-source joins, scalar evidence and exported inputs.

Writes only audit.json; never runs a model or changes the input pack.
"""
import collections
import json
from pathlib import Path
import numpy as np
from preprocess_road import by_type, latest, hash_file


def main():
    root = Path(__file__).parent
    meta = by_type(json.loads((root/'camera_metadata.json').read_text()))
    context = by_type(json.loads((root/'context_metadata.json').read_text()))
    source = json.loads((root/'source_manifest.json').read_text())
    for name, record in source.items():
        assert (root/name).stat().st_size == record['bytes']
        assert hash_file(root/name) == record['sha256']
    encs, cams = [], []
    for camera in ['narrow', 'wide']:
        enc, cam = meta[camera+'RoadEncodeIdx'], meta[camera+'RoadCameraState']
        assert len(enc) == 1200 and len(cam) == 1203
        assert all(x['valid'] for x in enc + cam)
        ed = {x['data']['segmentId']: x for x in enc}
        cd = {x['data']['frameId']: x for x in cam}
        assert len(ed) == len(enc) and len(cd) == len(cam)
        assert sorted(ed) == list(range(1200))
        assert sorted(cd) == list(range(1203))
        for sid, event in ed.items():
            data = event['data']
            assert data['frameId'] == sid+2 and data['segmentNum'] == 0
            assert cd[sid+2]['data']['sensor'] == 'os04c10'
            for k in ['timestampSof', 'timestampEof']:
                assert data[k] == cd[sid+2]['data'][k]
        encs.append(ed)
        cams.append(cd)
    full_delta = [encs[0][i]['data']['timestampSof']-encs[1][i]['data']['timestampSof'] for i in range(1200)]
    assert max(map(abs, full_delta)) < 1_000_000
    for name in ['fcamera','ecamera']:
        probe = json.loads((root/(name+'_probe.json')).read_text())['streams'][0]
        assert probe['nb_read_frames'] == '1200'
        assert (probe['width'],probe['height'],probe['pix_fmt']) == (1344,760,'yuv420p')
    out = root/'inputs320'
    rows = json.loads((out/'frame_index.json').read_text())
    assert len(rows) == 320
    image = np.load(out/'new_img.npy', mmap_mode='r', allow_pickle=False)
    assert image.shape == (320,2,6,128,256) and image.dtype == np.uint8
    parts = [np.load(out/(n+'.npy'), allow_pickle=False).reshape(320,-1)
             for n in ['desire','traffic_convention','action_t']]
    packed = np.concatenate(parts, axis=1)
    assert packed.shape == (320,12) and packed.dtype == np.float32
    assert np.isfinite(packed).all()
    assert (packed[:,:8] == 0).all()
    np.testing.assert_array_equal(packed[:,8:10], np.tile([1,0],(320,1)))
    source_times = []
    for i, row in enumerate(rows):
        assert row['sequence_index'] == i and row['segmentId'] == i+200 and row['frameId'] == i+202
        np.testing.assert_array_equal(image[i], np.load(out/f'in_warped_{i}.npy',allow_pickle=False))
        np.testing.assert_array_equal(packed[i], np.load(out/f'in_packed_{i}.npy',allow_pickle=False))
        cutoff = row['context_cutoff_ns']
        expected_cutoff = max(cams[cam][row['frameId']]['logMonoTime'] for cam in [0,1])
        assert cutoff == expected_cutoff
        for name, timestamp in row['context_logMonoTime'].items():
            events = meta[name] if name in meta else context[name]
            chosen = latest(events,cutoff)
            assert chosen['valid'] and chosen['logMonoTime'] == timestamp <= cutoff
        source_times.append(cutoff)
    first, last = source_times[0],source_times[-1]
    within = [x for x in context['carState'] if first <= x['logMonoTime'] <= last]
    assert len(within) == 1594
    assert all(x['valid'] and not x['data']['leftBlinker'] and not x['data']['rightBlinker'] for x in within)
    assert not rows[0]['latActive']
    assert (out/'spec.json').read_bytes() == Path('/workspace/shared/jetlink-history-regenerated/spec.json').read_bytes()
    # Verify, without touching, the original night pack's 641 frozen contract files.
    night = Path('/workspace/shared/jetlink-public-road')
    frozen_night = json.loads((night/'frozen_manifest.json').read_text())['files']
    for name, record in frozen_night.items():
        assert hash_file(night/'inputs320'/name) == record['sha256']
    record = {
        'status': 'PASS', 'source_total_bytes': sum(x['bytes'] for x in source.values()),
        'source_files_sha256_verified': True, 'full_decode_frames_per_video': 1200,
        'full_paired_encode_indices': 1200, 'full_max_abs_sof_delta_ns': max(map(abs,full_delta)),
        'camera_frameIds': [0,1202], 'encode_segmentIds': [0,1199], 'encode_frameIds': [2,1201],
        'calibration_validity_counts': dict(collections.Counter(str(x['valid']) for x in meta['extrinsicsCalibration'])),
        'first_valid_calibration_logMonoTime': next(x['logMonoTime'] for x in meta['extrinsicsCalibration'] if x['valid']),
        'selected_segmentIds': [200,519], 'selected_frameIds': [202,521],
        'selected_all_metadata_causal_valid': True, 'selected_flat_arrays_match_batched_arrays': True,
        'selected_high_rate_carState_events': len(within),
        'selected_max_context_age_ms': {name: max((row['context_cutoff_ns']-row['context_logMonoTime'][name])/1e6 for row in rows) for name in rows[0]['context_logMonoTime']},
        'recording_init': context['initData'][0],
        'full_carControl_latActive_counts': dict(collections.Counter(str(x['data']['latActive']) for x in context['carControl'])),
        'full_recorded_lane_change_counts': dict(collections.Counter(x['data']['meta']['laneChangeState'] for x in context['modelV2'])),
        'carParams_records': context['carParams'],
        'spec_sha256': hash_file(out/'spec.json'),
        'night_frozen_files_verified_unchanged': len(frozen_night),
        'limits': ['Offline preprocessing only. No inference, training, vehicle access or redistribution.',
                   'Target scalar formula is reconstructed, not exact captured invocation or historical hidden state.',
                   'Selected route is held out from this local student experiment, not proven absent from the original teacher training data.']}
    (root/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
