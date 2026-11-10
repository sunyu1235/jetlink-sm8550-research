#!/usr/bin/env python3
"""Analyze a short phone A/B/A screen. Never invokes a 160-frame numerical run.

Only compare existing whole-model Android execute_ms CSVs, with a matching
thermal start, identical input/configuration hashes and no nonfinite outputs.
These are conservative screening heuristics, not statistical certification.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def timing(path, expected_frames=16):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_frames or [int(r['frame']) for r in rows] != list(range(expected_frames)):
        raise ValueError('Expected exactly the declared consecutive frames, starting at zero')
    values = [float(r['execute_ms']) for r in rows]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Timing contains invalid execute_ms')
    if any(int(r['nonfinite']) != 0 for r in rows):
        raise ValueError('Nonfinite model outputs in short screen')
    return {'frames': len(rows), 'mean_excluding_first_ms': statistics.mean(values[1:]),
            'median_excluding_first_ms': statistics.median(values[1:]), 'first_ms': values[0]}


def summarize(a, b, a2, before_temperatures):
    if len(before_temperatures) != 3 or not all(math.isfinite(t) for t in before_temperatures):
        raise ValueError('Need three actual pre-run temperature readings')
    thermal = max(before_temperatures) < 45 and max(before_temperatures)-min(before_temperatures) <= 2
    means = [x['mean_excluding_first_ms'] for x in (a, b, a2)]
    baseline_drift = abs(means[0]-means[2])
    gain = min(means[0], means[2])-means[1]
    minimum_gain = max(0.5, min(means[0], means[2])*0.01)
    promising = thermal and gain >= minimum_gain and gain > baseline_drift
    return {'screen': 'PROMISING_REQUIRE_REVIEW_BEFORE_160' if promising else 'NOT_PROMISING_OR_CONFOUNDED',
            'baseline_a': a, 'candidate_b': b, 'baseline_a2': a2,
            'start_gpu_c': before_temperatures, 'matched_start_temperature': thermal,
            'gain_vs_faster_baseline_ms': gain, 'baseline_drift_ms': baseline_drift,
            'minimum_screen_gain_ms': minimum_gain,
            'scope': 'Short whole-model phone screen. GPU sensor is not an NPU temperature measurement.',
            'numerical_160_run': False,
            'acceptance': 'Not accepted. No automated 160-frame action is taken.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-a', type=Path, required=True)
    p.add_argument('--candidate-b', type=Path, required=True)
    p.add_argument('--baseline-a2', type=Path, required=True)
    p.add_argument('--frames', type=int, choices=range(2, 33), default=16)
    p.add_argument('--start-gpu-c', type=float, nargs=3, required=True)
    p.add_argument('--manifest', type=Path, required=True,
                   help='Reviewed run identities: runtime/input/config hashes, phone model, placement check')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    if a.report.exists():
        p.error('Refusing to overwrite report')
    manifest = json.loads(a.manifest.read_text())
    required = ('device_serial', 'device_model', 'input_sha256', 'runner_sha256', 'gelu_library_sha256',
                'baseline_artifact_sha256', 'candidate_artifact_sha256', 'configuration', 'placement_verified')
    if any(not manifest.get(key) for key in required) or manifest['placement_verified'] is not True:
        p.error('Incomplete run identity or placement proof; speed result is not qualified')
    result = summarize(*(timing(path, a.frames) for path in (a.baseline_a, a.candidate_b, a.baseline_a2)), a.start_gpu_c)
    result['run_identity'] = manifest
    result['csvs'] = [str(a.baseline_a), str(a.candidate_b), str(a.baseline_a2)]
    a.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
