#!/usr/bin/env python3
"""Repair FP32-stored biases of otherwise FP16 FCs only after bitwise roundtrip proof."""
import argparse, copy, hashlib, io, json, math, re, tarfile, tempfile
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repair(net_json, cpp_path, bin_path, out_prefix, expect_count=None):
    prefix = Path(out_prefix)
    outputs = [Path(str(prefix) + suffix) for suffix in ('.cpp', '.bin', '_net.json', '.repair.json')]
    require(not any(p.exists() for p in outputs), 'Refusing to overwrite output artifacts')
    document = json.loads(Path(net_json).read_text())
    if 'model.cpp' in document:
        document['model.cpp'] = str(outputs[0])
    if 'model.bin' in document:
        document['model.bin'] = str(outputs[1])
    graph = document['graph']
    names = set()
    for node in graph['nodes'].values():
        inputs, outputs_ = node['input_names'], node['output_names']
        if node['type'] == 'FullyConnected' and len(inputs) == 3 and len(outputs_) == 1:
            if [graph['tensors'][x]['data_type'] for x in inputs + outputs_] == [534, 534, 562, 534]:
                names.add(inputs[2])
    if expect_count is not None:
        require(len(names) == expect_count, f'Expected {expect_count} distinct biases, found {len(names)}')
    for node in graph['nodes'].values():
        for i, name in enumerate(node['input_names']):
            if name in names:
                require(node['type'] == 'FullyConnected' and len(node['input_names']) == 3 and i == 2,
                        f'Bias {name} has another kind of consumer')
                require([graph['tensors'][x]['data_type'] for x in node['input_names'] + node['output_names']]
                        == [534, 534, 562, 534], f'Bias {name} shared with incompatible FC')
    cpp = Path(cpp_path).read_text()
    chunks = {}
    for name in sorted(names):
        pattern = r'(static ModelError_t addTensor_' + re.escape(name) + r'\(QnnModel& model\)\{.*?)(?=\nstatic |\Z)'
        matches = list(re.finditer(pattern, cpp, re.S))
        require(len(matches) == 1, f'Expected one C++ tensor function for {name}')
        match = matches[0]
        require(match.group().count('QNN_DATATYPE_FLOAT_32') == 1, f'Ambiguous datatype change: {name}')
        chunks[name] = match.group().replace('QNN_DATATYPE_FLOAT_32', 'QNN_DATATYPE_FLOAT_16')
        cpp = cpp[:match.start()] + chunks[name] + cpp[match.end():]
    changes, payloads = [], {}
    with tarfile.open(bin_path, 'r') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate archive member')
        for name in sorted(names):
            candidates = [m for m in members if m.name == name + '.raw']
            require(len(candidates) == 1 and candidates[0].isfile(), f'Missing/invalid raw bias: {name}')
            original = archive.extractfile(candidates[0]).read()
            dims = graph['tensors'][name].get('dims')
            require(dims is not None and len(original) == 4 * math.prod(dims), f'Bias size/dims mismatch: {name}')
            values = np.frombuffer(original, dtype='<f4')
            require(np.isfinite(values).all(), f'Nonfinite bias: {name}')
            with np.errstate(over='ignore', invalid='ignore'):
                half = values.astype('<f2')
                back = half.astype('<f4')
            require(np.array_equal(values.view('<u4'), back.view('<u4')),
                    f'Bias {name} is not bit-exactly FP16 representable')
            payloads[name + '.raw'] = half.tobytes()
            changes.append({'tensor': name, 'elements': values.size,
                            'exact_float32_bit_roundtrip': True,
                            'original_sha256': hashlib.sha256(original).hexdigest(),
                            'repaired_sha256': hashlib.sha256(payloads[name + '.raw']).hexdigest()})
            graph['tensors'][name]['data_type'] = 534
    # No final output is written until all values, consumers, metadata and C++ checks passed.
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {'changes': changes, 'changed_numeric_values': 0, 'distinct_bias_count': len(changes),
              'source_net_json': str(net_json), 'warning': 'Post-converter storage/metadata repair only; full-model parity still required'}
    with tempfile.TemporaryDirectory(prefix='bias-repair-', dir=prefix.parent) as directory:
        temp = Path(directory)
        with tarfile.open(bin_path, 'r') as source, tarfile.open(temp / 'repaired.bin', 'w') as target:
            for member in source:
                stream = source.extractfile(member) if member.isfile() else None
                if member.name in payloads:
                    data = payloads[member.name]
                    member = copy.copy(member)
                    member.size = len(data)
                    stream = io.BytesIO(data)
                target.addfile(member, stream)
        (temp / 'repaired.cpp').write_text(cpp)
        (temp / 'repaired_net.json').write_text(json.dumps(document, indent=2))
        (temp / 'report.json').write_text(json.dumps(report, indent=2))
        for src, dest in zip(['repaired.cpp', 'repaired.bin', 'repaired_net.json', 'report.json'], outputs):
            (temp / src).rename(dest)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--net-json', required=True)
    p.add_argument('--cpp', required=True)
    p.add_argument('--bin', required=True)
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--expect-count', type=int, help='Optional assertion; no fixed count is assumed')
    a = p.parse_args()
    print(json.dumps(repair(a.net_json, a.cpp, a.bin, a.out_prefix, a.expect_count), indent=2))


if __name__ == '__main__':
    main()
