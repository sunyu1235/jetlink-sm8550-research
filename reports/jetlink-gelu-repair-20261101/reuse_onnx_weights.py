"""Use standard ONNX external-data offsets into a previously copied model.

Every weight's bytes must match the base model exactly. Only graph metadata
is then transferred; the phone retains its verified, unchanged weight file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import onnx
from onnx.external_data_helper import set_external_data


def varint(data, pos):
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError('invalid protobuf varint')


def fields(data, start, end):
    pos = start
    while pos < end:
        tag, pos = varint(data, pos)
        wire, number = tag & 7, tag >> 3
        if wire == 2:
            length, pos = varint(data, pos)
            payload, pos = pos, pos + length
            if pos > end:
                raise ValueError('protobuf field exceeds parent')
            yield number, wire, payload, pos
        elif wire == 0:
            _, pos = varint(data, pos)
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite output graph')
    graph_field = onnx.ModelProto.DESCRIPTOR.fields_by_name['graph'].number
    init_field = onnx.GraphProto.DESCRIPTOR.fields_by_name['initializer'].number
    name_field = onnx.TensorProto.DESCRIPTOR.fields_by_name['name'].number
    raw_field = onnx.TensorProto.DESCRIPTOR.fields_by_name['raw_data'].number
    base = args.base.read_bytes()
    view = memoryview(base)
    offsets = {}
    for num, wire, begin, end in fields(view, 0, len(base)):
        if num != graph_field:
            continue
        for num, wire, begin, end in fields(view, begin, end):
            if num != init_field:
                continue
            name, raw = None, None
            for num, wire, a, b in fields(view, begin, end):
                if num == name_field:
                    name = bytes(view[a:b]).decode('utf-8')
                if num == raw_field:
                    raw = a, b - a
            if name and raw:
                if name in offsets:
                    raise ValueError(f'duplicate base initializer {name}')
                offsets[name] = raw
    model = onnx.load(args.model)
    total = 0
    external = []
    for tensor in model.graph.initializer:
        if not tensor.HasField('raw_data') or not tensor.raw_data:
            continue
        # ORT shape inference reads small shape/index constants from raw_data.
        # Keep them inline, and externalize only substantial floating weights.
        if tensor.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16) or len(tensor.raw_data) < 65536:
            continue
        offset, length = offsets[tensor.name]
        actual = hashlib.sha256(tensor.raw_data).hexdigest()
        expected = hashlib.sha256(view[offset:offset + length]).hexdigest()
        if len(tensor.raw_data) != length or actual != expected:
            raise ValueError(f'weight bytes changed: {tensor.name}')
        set_external_data(tensor, args.base.name, offset=offset, length=length)
        tensor.ClearField('raw_data')
        external.append({'name': tensor.name, 'offset': offset, 'length': length, 'sha256': actual})
        total += length
    # Relative external-data paths must resolve to the same verified base file.
    if args.output.parent.resolve() != args.base.parent.resolve():
        raise ValueError('output must be beside the base weight model')
    onnx.save(model, args.output)
    onnx.checker.check_model(str(args.output))
    report = {'base': str(args.base.resolve()), 'base_sha256': hashlib.sha256(base).hexdigest(),
              'graph': str(args.output.resolve()), 'graph_bytes': args.output.stat().st_size,
              'graph_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'external_initializer_count': len(external), 'weight_bytes_reused': total,
              'external_tensors': external, 'onnx_checker': 'passed'}
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'external_tensors'}, indent=2))


if __name__ == '__main__':
    main()
