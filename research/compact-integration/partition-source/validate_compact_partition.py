#!/usr/bin/env python3
"""Compare matched synthetic cut inputs; not full-model/road/QNN acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
from prepare_compact_partition import sha


def synthetic_cases(inputs):
    cases = []
    for seed, scale in [(0, 0.0), (1, .01), (2, .25), (3, 1.), (4, 4.), (1729, 1.), (20261109, 1.), (713, 1.)]:
        rng = np.random.default_rng(seed)
        feed = {value.name: (rng.normal(size=value.shape) * scale).astype(np.float16) for value in inputs}
        cases.append((f'normal_seed{seed}_scale{scale}', feed))
    for mode in ['constant_tokens', 'last_token_only', 'prefix_only', 'alternating_sign']:
        rng = np.random.default_rng(937)
        feed = {}
        for value in inputs:
            tensor = rng.normal(size=value.shape).astype(np.float16)
            if mode == 'constant_tokens':
                tensor[:] = tensor[:, -1:, :]
            elif mode == 'last_token_only':
                tensor[:, :-1, :] = 0
            elif mode == 'prefix_only':
                tensor[:, -1:, :] = 0
            else:
                tensor = np.broadcast_to(np.where(np.arange(288)[None, :, None] % 2, -1, 1), value.shape).astype(np.float16).copy()
            feed[value.name] = tensor
        cases.append((mode, feed))
    return cases


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / 'artifacts'
    parser.add_argument('--baseline', type=Path, default=root / 'reconstructed-baseline-131-41.onnx')
    parser.add_argument('--compact', type=Path, default=root / 'compact-partition-131-41.onnx')
    parser.add_argument('--report', type=Path, default=root / 'numeric-validation.json')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--benchmark-runs', type=int, default=20)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-3)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {args.report}')
    if args.threads < 1 or args.benchmark_runs < 0 or args.atol < 0 or args.rtol < 0:
        raise ValueError('Invalid thread/run/tolerance argument')
    baseline, compact = session(args.baseline, args.threads), session(args.compact, args.threads)
    assert [(v.name, v.shape, v.type) for v in baseline.get_inputs()] == [(v.name, v.shape, v.type) for v in compact.get_inputs()]
    assert [v.name + '__last_query' for v in baseline.get_outputs()] == [v.name for v in compact.get_outputs()]
    cases = synthetic_cases(baseline.get_inputs())
    records, total_changed, total_count, max_abs = [], 0, 0, 0.
    all_pass = True
    for label, feed in cases:
        actuals, expected_full = compact.run(None, feed), baseline.run(None, feed)
        outputs = []
        for output, actual, full in zip(compact.get_outputs(), actuals, expected_full):
            expected = full[:, -1:, :]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float16
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            a, b = actual.astype(np.float64), expected.astype(np.float64)
            diff = np.abs(a - b)
            changed = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
            passed = finite and bool(np.allclose(a, b, atol=args.atol, rtol=args.rtol))
            result = {'name': output.name, 'shape': list(actual.shape), 'finite': finite,
                      'bit_different': changed, 'count': int(actual.size), 'max_abs': float(diff.max()),
                      'rmse': float(np.sqrt(np.mean(diff ** 2))), 'passes_local_tolerance': passed}
            total_changed += changed
            total_count += actual.size
            max_abs = max(max_abs, result['max_abs'])
            all_pass &= passed
            outputs.append(result)
        records.append({'case': label, 'outputs': outputs})
        print(label, 'max_abs', max(v['max_abs'] for v in outputs), 'changed', sum(v['bit_different'] for v in outputs), flush=True)
    # Functional non-regression: retaining the last token alone is insufficient.
    # Perturb only the history prefix while preserving each external last row.
    feed = cases[3][1]
    prefix_perturbed = {name: value.copy() for name, value in feed.items()}
    for value in prefix_perturbed.values():
        value[:, :-1, :] *= np.float16(-1)
    outputs_a = compact.run(None, feed)
    outputs_b = compact.run(None, prefix_perturbed)
    sensitivity = [{'name': output.name, 'max_abs': float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())}
                   for output, a, b in zip(compact.get_outputs(), outputs_a, outputs_b)]
    history_sensitive = all(record['max_abs'] > 0 for record in sensitivity)
    all_pass &= history_sensitive
    timings = {'scope': 'Isolated CPU cut diagnostic only; shared host, no HTP or pipeline speed claim',
               'intra_op_threads': args.threads, 'runs': args.benchmark_runs, 'warmups': 3, 'order': 'alternating AB/BA'}
    if args.benchmark_runs:
        for _ in range(3):
            baseline.run(None, feed)
            compact.run(None, feed)
        samples = {'baseline': [], 'compact': []}
        for iteration in range(args.benchmark_runs):
            order = [('baseline', baseline), ('compact', compact)]
            if iteration % 2:
                order.reverse()
            for name, sess in order:
                start = time.perf_counter()
                sess.run(None, feed)
                samples[name].append((time.perf_counter() - start) * 1000)
        for name, elapsed in samples.items():
            timings[name] = {'median_ms': float(np.median(elapsed)), 'p90_ms': float(np.percentile(elapsed, 90)),
                             'min_ms': min(elapsed), 'samples_ms': elapsed}
        timings['median_speedup'] = timings['baseline']['median_ms'] / timings['compact']['median_ms']
    report = {
        'status': 'PASS_LOCAL_SYNTHETIC_SCREEN_NOT_FULL_MODEL_ACCEPTANCE' if all_pass else 'FAIL_LOCAL_SYNTHETIC_SCREEN',
        'baseline_sha256': sha(args.baseline), 'compact_sha256': sha(args.compact),
        'runtime': {'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                    'python': platform.python_version(), 'providers': compact.get_providers(),
                    'graph_optimization': 'ORT_DISABLE_ALL', 'note': 'ORT CPU may internally promote FP16 computation; this is not HTP FP16 execution'},
        'comparison': 'Same synthetic FP16 cut inputs; six compact outputs versus reconstructed-baseline row287 only',
        'tolerance': {'atol': args.atol, 'rtol': args.rtol, 'meaning': 'Local numerical smoke threshold, not upstream acceptance thresholds'},
        'summary': {'cases': len(records), 'values_compared': total_count, 'bit_different': total_changed,
                    'bit_identical': total_changed == 0, 'max_abs': max_abs, 'all_finite': all(v['finite'] for r in records for v in r['outputs'])},
        'cases': records, 'history_prefix_sensitivity': {'passed': history_sensitive, 'outputs': sensitivity},
        'cpu_microbenchmark': timings,
        'not_tested': ['Accepted cached QNN partition', 'Full recurrent driving model', 'Real-road model accuracy', 'HTP backend', 'Phone latency']}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'summary', 'cpu_microbenchmark']}, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
