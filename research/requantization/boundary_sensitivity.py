#!/usr/bin/env python3
"""Output-boundary QDQ screening, NOT model quantization or recurrent replay.

Quantize only saved accepted outputs; calibrate on an earlier, disjoint prefix.
Evaluate with the unmodified upstream checker. Do not infer HTP latency or
internal layer tolerances from these results. Column grouping is a diagnostic,
not a claim that a backend supports per-column activation quantization.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, io, json, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor/jetlink'))
spec = importlib.util.spec_from_file_location('upstream_parity', ROOT / 'vendor/jetlink/scripts/verify_parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def calibrate(x, bits, margin=1.10):
    """Asymmetric unsigned zero-inclusive min/max, round-to-nearest-even.

    Endpoints expanded about zero by margin to reserve held-out range. Rounded
    zero-point yields exact zero. All-zero tensors use scale 1 and zp 0.
    """
    x = np.asarray(x, np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('calibration requires non-empty finite data')
    if bits not in (8, 12, 16) or not np.isfinite(margin) or margin < 1:
        raise ValueError('unsupported bit width or range margin')
    lo, hi = min(float(x.min()), 0.0) * margin, max(float(x.max()), 0.0) * margin
    scale = (hi - lo) / ((1 << bits) - 1) if hi != lo else 1.0
    zp = int(np.clip(np.rint(-lo / scale), 0, (1 << bits) - 1))
    return {'scale': scale, 'zero_point': zp, 'bits': bits}


def qdq(x, encoding):
    x = np.asarray(x, np.float64)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite evaluation input')
    raw = np.rint(x / encoding['scale']) + encoding['zero_point']
    maxq = (1 << encoding['bits']) - 1
    clipped = int(np.count_nonzero((raw < 0) | (raw > maxq)))
    y = (np.clip(raw, 0, maxq) - encoding['zero_point']) * encoding['scale']
    return y.astype(np.float32), clipped


def groups(model_spec, size, grouping):
    if grouping == 'tensor':
        return {'outputs': np.arange(size)}
    result = {}
    for name, sl in model_spec.output_slices.items():
        idx = np.arange(size)[sl]
        # Column parser uses integer indices exactly as it parses model data.
        cols = parity.columns(name, idx) if grouping == 'column' else None
        if cols:
            for col, indices in cols.items():
                result[f'{name}/{col}'] = indices.astype(np.int64)
        else:
            result[name] = idx
    indices = np.concatenate(list(result.values()))
    if not np.array_equal(np.sort(indices), np.arange(size)):
        raise ValueError('spec groups overlap or fail to cover outputs')
    return result


def load_frames(folder):
    paths = sorted(folder.glob('out_link_*.npy'), key=lambda p: int(p.stem.split('_')[-1]))
    links, refs, hashes = [], [], {}
    for i, p in enumerate(paths):
        if p.name != f'out_link_{i}.npy':
            raise ValueError('non-contiguous source frame indices')
        r = folder / f'out_ref_{i}.npy'
        a, b = np.load(p, allow_pickle=False).reshape(-1), np.load(r, allow_pickle=False).reshape(-1)
        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError('source shape mismatch or nonfinite data')
        links.append(a); refs.append(b)
        for file in (p, r):
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if not links:
        raise ValueError('no outputs')
    return np.stack(links), np.stack(refs), hashes


def check(model_spec, links, refs):
    # Invoke original compare, including every per-frame and pooled check.
    with tempfile.TemporaryDirectory(prefix='jetlink-boundary-') as tmp:
        d = Path(tmp)
        (d / 'spec.json').write_text(json.dumps(model_spec.to_dict()))
        for i, (a, b) in enumerate(zip(links, refs, strict=True)):
            np.save(d / f'out_link_{i}.npy', a)
            np.save(d / f'out_ref_{i}.npy', b)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = parity.compare(argparse.Namespace(dir=str(d), spec=None))
        return code, stream.getvalue()


def run(source, output, split=80, margin=1.10):
    output.mkdir(parents=True, exist_ok=False)
    links, refs, hashes = load_frames(source)
    if split < 1 or len(links) - split < parity.MIN_SAMPLES:
        raise ValueError('need non-empty calibration prefix and at least 16 held-out frames')
    model_spec = parity.ModelSpec.load(source / 'spec.json')
    baseline, log = check(model_spec, links[split:], refs[split:])
    (output / 'baseline.log').write_text(log)
    if baseline:
        raise ValueError('source baseline does not pass held-out checker')
    report = {'kind': 'saved-output-boundary-only', 'source': str(source.relative_to(ROOT)),
              'calibration_frames': [0, split - 1], 'heldout_frames': [split, len(links)-1],
              'margin': margin, 'baseline_exit_code': baseline,
              'limits': ['No model weights, inputs, internal activations, or HTP execution used.',
                         'Temporal split is not an independent drive; output quantization is not fed back.',
                         '12-bit and column encodings are sensitivity diagnostics, not deployment claims.',
                         'No speed or full-model accuracy prediction is justified.'], 'cases': []}
    for grouping in ('tensor', 'head', 'column'):
        partition = groups(model_spec, links.shape[1], grouping)
        for bits in (8, 12, 16):
            name = f'{grouping}-u{bits}'
            candidate = links[split:].copy()
            encodings, clipped = {}, 0
            for key, indices in partition.items():
                enc = calibrate(links[:split, indices], bits, margin)
                candidate[:, indices], count = qdq(links[split:, indices], enc)
                encodings[key] = enc
                clipped += count
            code, log = check(model_spec, candidate, refs[split:])
            (output / f'{name}.log').write_text(log)
            (output / f'{name}.encodings.json').write_text(json.dumps(encodings, indent=2))
            case = {'name': name, 'exit_code': code, 'clipped_values': clipped,
                    'evaluated_values': candidate.size,
                    'max_abs_added_error': float(np.abs(candidate - links[split:]).max()),
                    'verdict': log.strip().splitlines()[-1]}
            report['cases'].append(case)
            print(name, 'PASS' if code == 0 else 'FAIL', 'clipped', clipped)
    hashes['spec.json'] = hashlib.sha256((source / 'spec.json').read_bytes()).hexdigest()
    hashes['upstream_checker'] = hashlib.sha256((ROOT / 'vendor/jetlink/scripts/verify_parity.py').read_bytes()).hexdigest()
    (output / 'input-sha256.json').write_text(json.dumps(hashes, indent=2))
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--split', type=int, default=80)
    p.add_argument('--margin', type=float, default=1.10)
    args = p.parse_args()
    run(args.source.resolve(), args.output, args.split, args.margin)
