#!/usr/bin/env python3
"""Offline public CI road sequence -> single-frame Light Speed inputs.

No vehicle connection, controls, model inference, or network activity.
Requires NumPy, ffmpeg and extracted camera/context metadata JSON alongside this file.
The algorithm is a NumPy port of the original Light Speed-era upstream warp.
See PREPROCESSING.md and provenance.json for exact source references and limits.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 1344, 760
MODEL_W, MODEL_H = 512, 256
VIEW_FROM_DEVICE = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
UV_SCALE = np.array([[1., 1., .5], [1., 1., .5], [2., 2., 1.]], dtype=np.float32)


def intrinsics(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[focal, 0., cx], [0., focal, cy], [0., 0., 1.]])


def rotation(rpy: list[float]) -> np.ndarray:
    # Exact matrix multiplication order of upstream euler2rot_single.
    phi, theta, psi = np.asarray(rpy, dtype=np.float32)
    cx, sx = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(psi), np.sin(psi)
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    return rz @ ry @ rx


def transforms(rpy: list[float]) -> np.ndarray:
    source = [intrinsics(1522. * 3 / 4, WIDTH / 2, HEIGHT / 2),
              intrinsics(567. * 3 / 4, WIDTH / 2, HEIGHT / 2)]
    target = [intrinsics(910., 256., 47.6), intrinsics(455., 256., 151.8)]
    return np.stack([(k @ VIEW_FROM_DEVICE @ rotation(rpy)
                      @ np.linalg.inv(t @ VIEW_FROM_DEVICE)).astype(np.float32)
                     for k, t in zip(source, target)])


def sampling_indices(matrix: np.ndarray, dst_shape: tuple[int, int],
                     src_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor, ties-to-even rounding; clamp to source image border.

    Use explicit float32 elementwise arithmetic, matching compile_warp.py.
    CPU float32 operations are not a claim of QCOM fused-operation bit parity.
    """
    dw, dh = dst_shape
    sh, sw = src_shape
    x = np.broadcast_to(np.arange(dw, dtype=np.float32)[None, :], (dh, dw))
    y = np.broadcast_to(np.arange(dh, dtype=np.float32)[:, None], (dh, dw))
    sx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    sy = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    z = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    sx, sy = sx / z, sy / z
    assert np.isfinite(sx).all() and np.isfinite(sy).all()
    return (np.clip(np.rint(sy), 0, sh - 1).astype(np.int32),
            np.clip(np.rint(sx), 0, sw - 1).astype(np.int32))


def maps_for(matrix: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    return (sampling_indices(matrix, (MODEL_W, MODEL_H), (HEIGHT, WIDTH)),
            sampling_indices(matrix * UV_SCALE, (MODEL_W // 2, MODEL_H // 2), (HEIGHT // 2, WIDTH // 2)))


def prepare_i420(raw: bytes, maps) -> np.ndarray:
    # Decode straight into planar YUV420, not RGB and not a resize approximation.
    a = np.frombuffer(raw, dtype=np.uint8)
    n = WIDTH * HEIGHT
    assert a.size == n * 3 // 2
    y = a[:n].reshape(HEIGHT, WIDTH)
    u = a[n:n + n // 4].reshape(HEIGHT // 2, WIDTH // 2)
    v = a[n + n // 4:].reshape(HEIGHT // 2, WIDTH // 2)
    my, muv = maps
    y, u, v = y[my], u[muv], v[muv]
    return np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v])


def by_type(items):
    result = {}
    for item in items:
        result.setdefault(item['type'], []).append(item)
    for values in result.values():
        values.sort(key=lambda x: x['logMonoTime'])
    return result


def latest(items, time):
    times = [item['logMonoTime'] for item in items]
    idx = bisect.bisect_right(times, time) - 1
    if idx < 0:
        raise ValueError('Requested frame predates required metadata')
    item = items[idx]
    assert item['valid']
    return item


def decode(path: Path, start: int, count: int):
    # HEVC element streams have no trustworthy wall-clock PTS. Count decoder frames;
    # the EncodeIdx table provides their road frame IDs and timestamps.
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-vf',
           f'trim=start_frame={start}:end_frame={start+count}', '-vsync', '0',
           '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-threads', '1', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    size = WIDTH * HEIGHT * 3 // 2
    for _ in range(count):
        buf = proc.stdout.read(size)
        if len(buf) != size:
            raise RuntimeError(f'Incomplete decoded frame from {path}: {len(buf)} of {size}')
        yield buf
    extra = proc.stdout.read(1)
    assert extra == b''
    stderr = proc.stderr.read().decode() if proc.stderr else ''
    status = proc.wait()
    if status:
        raise RuntimeError(f'ffmpeg exit {status}: {stderr}')


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def build(root: Path, start: int, count: int):
    meta = by_type(json.loads((root / 'camera_metadata.json').read_text()))
    context = by_type(json.loads((root / 'context_metadata.json').read_text()))
    enc = [{e['data']['segmentId']: e for e in meta[k]} for k in ['narrowRoadEncodeIdx', 'wideRoadEncodeIdx']]
    cams = [{e['data']['frameId']: e for e in meta[k]} for k in ['narrowRoadCameraState', 'wideRoadCameraState']]
    models = {e['data']['frameId']: e for e in context['modelV2']}
    cp = context['carParams'][0]
    assert len(context['carParams']) == 1 and cp['valid']
    assert all(not e['data']['latActive'] for e in context['carControl'])
    assert all(e['data']['meta']['laneChangeState'] == 'off' for e in context['modelV2'])
    assert all(not e['data']['isRHD'] for e in context['driverMonitoringStateDEPRECATED'])
    rows, matrices = [], []
    for sid in range(start, start + count):
        f, w = [e[sid] for e in enc]
        fid = f['data']['frameId']
        assert fid == w['data']['frameId'] and f['valid'] and w['valid']
        cs = [c[fid] for c in cams]
        assert all(c['valid'] and c['data']['sensor'] == 'os04c10' for c in cs)
        sof = [e['data']['timestampSof'] for e in [f, w]]
        assert abs(sof[0] - sof[1]) < 1_000_000  # stricter than modeld's 10 ms warning
        ts = max(e['logMonoTime'] for e in cs)
        cal = latest(meta['extrinsicsCalibration'], ts)
        device = latest(meta['deviceState'], ts)
        assert cal['data']['calStatus'] == 'calibrated' and device['data']['deviceType'] == 'mici'
        lat = latest(context['lateralDelay'], ts)
        dm = latest(context['driverMonitoringStateDEPRECATED'], ts)
        cc = latest(context['carControl'], ts)
        cs_vehicle = latest(context['carState'], ts)
        assert not cc['data']['latActive'] and not dm['data']['isRHD']
        action = [lat['data']['lateralDelay'] + 0.0 + .05 + .025,
                  cp['data']['longitudinalActuatorDelay'] + .3 + .05 + .025]
        rows.append({'sequence_index': sid - start, 'segmentId': sid, 'frameId': fid,
                     'narrow_timestampSof': sof[0], 'wide_timestampSof': sof[1],
                     'sof_delta_ns': sof[0] - sof[1], 'context_cutoff_ns': ts,
                     'calibration_logMonoTime': cal['logMonoTime'], 'rpyCalib': cal['data']['rpyCalib'],
                     'wideFromDeviceEuler_logged_but_not_used_by_upstream': cal['data']['wideFromDeviceEuler'],
                     'desire': [0.] * 8, 'traffic_convention': [1., 0.], 'action_t': action,
                     'latActive': cc['data']['latActive'], 'vEgo': cs_vehicle['data']['vEgo'],
                     'isRHD': dm['data']['isRHD'], 'recorded_lane_change_state': models[fid]['data']['meta']['laneChangeState'],
                     'scalar_fidelity': 'log-derived, Light Speed-era formula; not captured exact modeld invocation'})
        matrices.append(transforms(cal['data']['rpyCalib']))
    assert all(b['frameId'] == a['frameId'] + 1 for a, b in zip(rows, rows[1:]))
    out = root / f'inputs{count}'
    out.mkdir(exist_ok=True)
    images = np.lib.format.open_memmap(out / 'new_img.npy', mode='w+', dtype=np.uint8,
                                      shape=(count, 2, 6, MODEL_H // 2, MODEL_W // 2))
    maps = {}
    for i, (narrow, wide) in enumerate(zip(decode(root/'fcamera.hevc', start, count), decode(root/'ecamera.hevc', start, count))):
        for cam, raw in enumerate([narrow, wide]):
            key = matrices[i][cam].tobytes()
            if key not in maps:
                maps[key] = maps_for(matrices[i][cam])
            images[i, cam] = prepare_i420(raw, maps[key])
        if i % 40 == 0:
            print(f'prepared {i + 1}/{count}', flush=True)
    images.flush()
    np.save(out/'transforms.npy', np.stack(matrices))
    np.save(out/'desire.npy', np.array([r['desire'] for r in rows], dtype=np.float32))
    np.save(out/'traffic_convention.npy', np.array([r['traffic_convention'] for r in rows], dtype=np.float32)[:,None,:])
    np.save(out/'action_t.npy', np.array([r['action_t'] for r in rows], dtype=np.float32)[:,None,:])
    (out/'frame_index.json').write_text(json.dumps(rows, indent=2))
    summary = {'status': 'REAL_PUBLIC_ROAD_IMAGES_RECONSTRUCTED_CONTEXT_NOT_DRIVING_VALIDATION',
               'frames': count, 'camera_order': ['narrow', 'wide'], 'new_img_shape': list(images.shape),
               'new_img_dtype': str(images.dtype), 'pixel_min': int(images.min()), 'pixel_max': int(images.max()),
               'first_frameId': rows[0]['frameId'], 'last_frameId': rows[-1]['frameId'],
               'video_segmentId_first': start, 'video_segmentId_last': start + count - 1,
               'max_abs_sof_delta_ns': max(abs(r['sof_delta_ns']) for r in rows),
               'unique_calibrations': len(set(tuple(r['rpyCalib']) for r in rows)),
               'unique_action_t': np.unique(np.array([r['action_t'] for r in rows],dtype=np.float32),axis=0).tolist(),
               'source_total_bytes': 86550420,
               'reset': 'Zero all model state_* queues before frame 0. Carry next_state_* forward after each inference. No frame skipping.',
               'scoring': 'Suggested first 160 frames warm up, score last 160; not the original on-vehicle hidden state.',
               'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hash_file(p)} for p in out.glob('*') if p.is_file()}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--start', type=int, default=20)
    parser.add_argument('--frames', type=int, default=320)
    args = parser.parse_args()
    build(args.root, args.start, args.frames)


if __name__ == '__main__':
    main()
