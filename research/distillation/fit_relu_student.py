"""Offline block-distillation initializer; NOT a trained/validated driving model.

NPZ contract (no pickle): x_train[N,D], y_train[N,O],
x_validation[M,D], y_validation[M,O], w1[D,H], b1[H].
Teacher y must be post-second-linear, before residual, from the accurate model.
Train/validation must be disjoint driving routes, not randomly split tokens.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def matrix(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return a


def fit(x, y, w1, b1, width, ridge=1e-4):
    x, y, w1 = (matrix(a, n) for a, n in [(x, 'x'), (y, 'y'), (w1, 'w1')])
    b1 = np.asarray(b1, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[1] != w1.shape[0] or b1.shape != (w1.shape[1],):
        raise ValueError('incompatible input, target, or first-layer shapes')
    if not np.isfinite(b1).all() or not 1 <= width <= w1.shape[1] or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('finite bias, valid width, and positive finite ridge required')
    h = np.maximum(x @ w1 + b1, 0)
    # Deterministic inexpensive screening, not optimal neuron selection.
    hc = h - h.mean(0)
    yc = y - y.mean(0)
    scales = np.maximum(np.sqrt(np.mean(hc * hc, axis=0)), 1e-12)
    score = np.linalg.norm((hc / scales).T @ yc / len(x), axis=1)
    keep = np.sort(np.argsort(-score, kind='stable')[:width])
    h = h[:, keep]
    mean = h.mean(0)
    scale = np.maximum(h.std(0), 1e-12)
    z = (h - mean) / scale
    # Solve normalized mean-square loss + ridge*||coefficient||^2.
    gram = z.T @ z / len(x)
    gram.flat[::width + 1] += ridge
    coef = np.linalg.solve(gram, z.T @ yc / len(x))
    w2 = coef / scale[:, None]
    b2 = y.mean(0) - mean @ w2
    return dict(w1=w1[:, keep], b1=b1[keep], w2=w2, b2=b2, keep=keep)


def predict(student, x):
    return np.maximum(x @ student['w1'] + student['b1'], 0) @ student['w2'] + student['b2']


def errors(actual, target):
    a, t = matrix(actual, 'actual'), matrix(target, 'target')
    if a.shape != t.shape:
        raise ValueError('prediction/target shapes differ')
    delta = a - t
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), max_abs=float(np.max(abs(delta))),
                per_channel_rmse=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                per_channel_max_abs=np.max(abs(delta), axis=0).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pairs', type=Path)
    ap.add_argument('--width', required=True, type=int)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--route-split-confirmed', required=True, action='store_true',
                    help='Confirm train and validation were separated by whole route before capture')
    args = ap.parse_args()
    report_path = args.output.with_suffix('.json')
    if args.output.suffix != '.npz' or args.output.exists() or report_path.exists():
        raise ValueError('use a new .npz output and unused matching .json path')
    with np.load(args.pairs, allow_pickle=False) as data:
        student = fit(data['x_train'], data['y_train'], data['w1'], data['b1'], args.width, args.ridge)
        xv, yv = matrix(data['x_validation'], 'x_validation'), matrix(data['y_validation'], 'y_validation')
        if xv.shape[1] != student['w1'].shape[0] or yv.shape != (len(xv), student['w2'].shape[1]):
            raise ValueError('validation shapes are inconsistent')
        report = dict(status='OFFLINE_INITIALIZER_ONLY_NOT_MODEL_ACCEPTANCE',
                      width=args.width, original_width=int(data['w1'].shape[1]), ridge=args.ridge,
                      train=errors(predict(student, data['x_train']), data['y_train']),
                      validation_fp64=errors(predict(student, xv), yv))
        # Quantized WEIGHT storage only. Float64 arithmetic here is NOT FP16 backend validation.
        stored = {k: v.astype(np.float16) if k != 'keep' else v for k, v in student.items()}
        if not all(np.isfinite(v).all() for v in stored.values()):
            raise ValueError('weights overflow FP16; reject candidate')
        reloaded = {k: v.astype(np.float64) if k != 'keep' else v for k, v in stored.items()}
        report['validation_fp16_weights_fp64_compute'] = errors(predict(reloaded, xv), yv)
    digest = hashlib.sha256()
    with args.pairs.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    report['input_sha256'] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as out:
        np.savez(out, **stored)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'width', 'original_width']}))


if __name__ == '__main__':
    main()
