#!/usr/bin/env python3
"""One zero-state CPU frame only. Not a latency result or numerical acceptance."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def run(model, inputs, spec):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    dtypes = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(uint8)': np.uint8}
    packed = np.load(inputs/'in_packed_0.npy').reshape(-1)
    # Exact ModelSpec.packed_shapes order for this stateful model.
    packed_names = ['desire', 'traffic_convention', 'action_t']
    assert set(packed_names) == {n for n in spec['input_shapes'] if n != 'new_img' and not n.startswith('state_')}
    at, named = 0, {}
    for name in packed_names:
        count = int(np.prod(spec['input_shapes'][name]))
        named[name] = packed[at:at+count]
        at += count
    assert at == packed.size
    feed = {}
    for info in session.get_inputs():
        dtype = dtypes[info.type]
        if info.name == 'new_img':
            value = np.load(inputs/'in_warped_0.npy')
        elif info.name.startswith('state_'):
            value = np.zeros(info.shape, dtype=dtype)
        else:
            value = named[info.name]
        feed[info.name] = np.ascontiguousarray(value, dtype=dtype).reshape(info.shape)
    results = dict(zip([v.name for v in session.get_outputs()], session.run(None, feed)))
    del session
    gc.collect()
    return results


def smoke_passes(rows):
    return bool(rows) and all(row['finite'] and row['bit_identical'] for row in rows.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    if args.report.exists():
        p.error('Refusing to overwrite report')
    spec = json.loads((args.inputs/'spec.json').read_text())
    baseline = run(args.baseline, args.inputs, spec)
    candidate = run(args.candidate, args.inputs, spec)
    assert baseline.keys() == candidate.keys()
    rows = {}
    for name, reference in baseline.items():
        actual = candidate[name]
        assert reference.shape == actual.shape and reference.dtype == actual.dtype
        rows[name] = {'shape': list(actual.shape), 'dtype': str(actual.dtype),
                      'finite': bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
                      'bit_identical': actual.tobytes() == reference.tobytes(),
                      'max_abs': float(np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64))))}
    report = {'scope': 'One zero-state CPU frame. No phone timing or 160-frame gate.',
              'frames': 1, 'provider': 'CPUExecutionProvider', 'ort_version': ort.__version__,
              'graph_optimization': 'ORT_DISABLE_ALL', 'all_outputs': rows,
              'input_warped_sha256': hashlib.sha256((args.inputs/'in_warped_0.npy').read_bytes()).hexdigest(),
              'input_packed_sha256': hashlib.sha256((args.inputs/'in_packed_0.npy').read_bytes()).hexdigest(),
              'baseline': str(args.baseline), 'candidate': str(args.candidate)}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not smoke_passes(rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
