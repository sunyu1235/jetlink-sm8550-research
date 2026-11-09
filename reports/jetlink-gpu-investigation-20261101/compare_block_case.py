"""Report same-input block errors; intentionally no full-model PASS label."""
import argparse
import json
from pathlib import Path

import numpy as np


def errors(actual, reference):
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f'output shape mismatch: {a.shape} != {b.shape}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('non-finite block output')
    d = np.abs(a - b)
    return {'mae': float(d.mean()), 'max_abs': float(d.max()),
            'rmse': float(np.sqrt(np.mean(d * d))),
            'correlation': float(np.corrcoef(a, b)[0, 1]),
            'exact_fraction': float(np.mean(a == b)), 'elements': int(a.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    refs = {t: np.load(args.case / f'ref-{t}.npy') for t in ('fp16', 'fp32')}
    records = {'purpose': 'one captured input, one block; not full-model accuracy', 'results': {}}
    for backend, dtype in [('gpu32', np.float32), ('gpu16', np.float16), ('htp16', np.float16)]:
        raw = args.case / f'block-{backend}-diagnostic' / 'out_0.bin'
        a = np.fromfile(raw, dtype=dtype)
        records['results'][backend] = {f'vs_cpu_{t}': errors(a, ref) for t, ref in refs.items()}
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
