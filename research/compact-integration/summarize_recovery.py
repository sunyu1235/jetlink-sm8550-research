"""Validate completed replay evidence and record compact integration qualification.

Does not fit models, change checker thresholds, or infer phone performance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CHECKER = Path('/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink/scripts/verify_parity.py')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize():
    integrated = json.loads((ROOT / 'whole-integration.json').read_text())
    model = Path(integrated['model'])
    assert sha(model) == integrated['sha256'], 'integrated model changed'
    roads = {}
    for road in ('night', 'day'):
        output = ROOT / f'{road}-320'
        run = json.loads((output / 'out_link-run.json').read_text())
        assert run['frames'] == 320 and run['independent_recurrence']
        assert Path(run['model']).resolve() == model.resolve()
        digest = hashlib.sha256()
        for i in range(320):
            file = output / f'out_link_{i}.npy'
            values = np.load(file, allow_pickle=False)
            assert values.shape == (18452,) and np.isfinite(values).all(), file
            digest.update(file.read_bytes())
        score = json.loads((ROOT / f'{road}-score/summary.json').read_text())
        assert score['checker_modified'] is False
        assert score['views']['all_frames']['frames'] == 320
        assert score['views']['after_warmup']['frames'] == 160
        roads[road] = dict(
            ordered_outputs_sha256=digest.hexdigest(),
            views=score['views'],
            all_views_pass=all(v['checker_exit'] == 0 for v in score['views'].values()),
        )
    result = dict(
        scope='Source-level CPU whole-model compact-boundary integration; not complete HTP execution or driving safety',
        model_sha256=sha(model), checker_sha256=sha(CHECKER),
        roads=roads,
        road_numerical_gate_pass=all(r['all_views_pass'] for r in roads.values()),
        synthetic160=json.loads((ROOT / 'synthetic160-summary.json').read_text()),
        compact_boundary_bytes=9216, baseline_boundary_bytes=2654208,
        boundary_byte_reduction_factor=288,
        device_execution=False, device_latency_measured=False,
        context='Reconstructed source partition, not byte-identical recovery of accepted phone cached context',
        recovery='Night outputs through 306 existed; recurrent checkpoint resumed at 288, then replayed through 319. Day replay ran fresh.',
    )
    target = ROOT / 'recovery-qualification.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    summarize()
