#!/usr/bin/env python3
"""Compare source-derived NumPy warp to the pinned official tinygrad function."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

from preprocess_road import WIDTH, HEIGHT, decode, prepare_i420, maps_for


def padded_nv12(raw):
    stride = ((WIDTH + 127) // 128) * 128
    yh = ((HEIGHT + 31) // 32) * 32
    uvh = ((HEIGHT // 2 + 15) // 16) * 16
    frame = np.zeros((yh + uvh, stride), dtype=np.uint8)
    a = np.frombuffer(raw, np.uint8)
    n = WIDTH * HEIGHT
    frame[:HEIGHT, :WIDTH] = a[:n].reshape(HEIGHT, WIDTH)
    frame[yh:yh+HEIGHT//2, :WIDTH:2] = a[n:n+n//4].reshape(HEIGHT//2, WIDTH//2)
    frame[yh:yh+HEIGHT//2, 1:WIDTH:2] = a[n+n//4:].reshape(HEIGHT//2, WIDTH//2)
    return frame.ravel(), (WIDTH, HEIGHT, stride, yh, uvh, frame.size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tinygrad', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).parent)
    a = p.parse_args()
    sys.path.insert(0, str(a.tinygrad))
    from tinygrad import Tensor
    from examples.openpilot.compile_warp import NV12Frame, make_frame_prepare
    root = a.root
    transforms = np.load(root/'inputs320/transforms.npy')
    frame_index = json.loads((root/'inputs320/frame_index.json').read_text())
    samples = []
    # Cover every distinct calibration value and both road cameras.
    selected = []
    seen = set()
    for i, tfm in enumerate(transforms):
        if tfm.tobytes() not in seen:
            selected.append(i)
            seen.add(tfm.tobytes())
    selected = sorted(set(selected + [len(transforms)//2 - 1, len(transforms)-1]))
    saved = np.load(root/'inputs320/new_img.npy', mmap_mode='r', allow_pickle=False)
    for i in selected:
        for cam, fn in enumerate(['fcamera.hevc', 'ecamera.hevc']):
            raw = list(decode(root/fn, frame_index[i]['segmentId'], 1))[0]
            packed, dims = padded_nv12(raw)
            warp = make_frame_prepare(NV12Frame(*dims), 512, 256)
            actual = warp(Tensor(packed), Tensor(transforms[i,cam])).numpy()
            expected = prepare_i420(raw, maps_for(transforms[i,cam]))
            np.testing.assert_array_equal(expected, saved[i,cam])
            d = actual.astype(np.int16) - expected.astype(np.int16)
            sample = {'sequence_index': i, 'camera': fn, 'shape': list(actual.shape), 'dtype': str(actual.dtype),
                      'different_pixels': int(np.count_nonzero(d)), 'total_pixels': d.size,
                      'max_abs_difference': int(np.max(np.abs(d))), 'mean_abs_difference': float(np.mean(np.abs(d)))}
            sample['saved_input_matches_numpy'] = True
            sample['difference_examples_ch_y_x'] = [{'index': loc.tolist(), 'tinygrad': int(actual[tuple(loc)]),
                                                     'numpy': int(expected[tuple(loc)])} for loc in np.argwhere(d)[:8]]
            samples.append(sample)
            print(json.dumps(sample), flush=True)
    report = {'tinygrad_ref': 'd39227beebd0c3a340dc42b2cdc1e3db1c36aed9',
              'implementation': 'examples/openpilot/compile_warp.py make_frame_prepare',
              'runtime': os.environ.get('DEV', 'unspecified'), 'samples': samples,
              'cachelevel': os.environ.get('CACHELEVEL', 'unspecified'),
              'all_exact': all(x['different_pixels'] == 0 for x in samples),
              'limitation': 'CPU source-function comparison; not QCOM GPU runtime/ISP parity.'}
    (root/'warp_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
