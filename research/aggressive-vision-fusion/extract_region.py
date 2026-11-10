"""Extract a closed region from QNN converter output without changing its nodes.

The generated C++/weights stay local. This authored extractor is redistributable;
vendor-generated C++ and model weights are not part of the source deliverable.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def select_region(graph, inputs, outputs):
    if not outputs or len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs)):
        raise ValueError('Cut names must be unique and outputs nonempty')
    for name in inputs + outputs:
        if name not in graph['tensors']:
            raise ValueError('Missing cut tensor descriptor: ' + name)
    nodes = list(graph['nodes'].items())
    producer = {}
    for name, node in nodes:
        for out in node['output_names']:
            if out in producer:
                raise ValueError('Multiple tensor producers: ' + out)
            producer[out] = name
    by_name = dict(nodes)
    seen, used_inputs, constants = set(), set(), set()
    visiting = set()

    def visit(tensor):
        if tensor in inputs:
            used_inputs.add(tensor)
            return
        if tensor not in graph['tensors']:
            raise ValueError('Unknown tensor: ' + tensor)
        if graph['tensors'][tensor]['type'] == 4:
            constants.add(tensor)
            return
        if tensor not in producer:
            raise ValueError('Unclosed region, missing input: ' + tensor)
        name = producer[tensor]
        if name in seen:
            return
        if name in visiting:
            raise ValueError('Cycle: ' + name)
        visiting.add(name)
        for inp in by_name[name]['input_names']:
            visit(inp)
        visiting.remove(name)
        seen.add(name)

    for output in outputs:
        if output in inputs:
            raise ValueError('Outputs must be computed, not aliased inputs')
        if output not in producer:
            raise ValueError('Outputs must be computed, not constants: ' + output)
        visit(output)
    if used_inputs != set(inputs):
        raise ValueError('Unused cut inputs: ' + repr(set(inputs) - used_inputs))
    selected = [(name, node) for name, node in nodes if name in seen]
    # Verify original topological order and no unaccounted dependency.
    ready = set(inputs) | constants
    for name, node in selected:
        if not set(node['input_names']) <= ready:
            raise ValueError('Not topological: ' + name)
        ready.update(node['output_names'])
    produced = {t for _, node in selected for t in node['output_names']}
    consumers_outside = {t for name, node in nodes if name not in seen for t in node['input_names']}
    graph_outputs = {t for t, desc in graph['tensors'].items() if desc['type'] in (1, 2)}
    undeclared = (produced & (consumers_outside | graph_outputs)) - set(outputs)
    if undeclared:
        raise ValueError('Undeclared live side outputs: ' + repr(sorted(undeclared)))
    return selected, sorted(constants)


def validate_cpp_boundary(funcs, graph, names):
    """Fail closed on source descriptor drift before constructing new boundaries.

    This recognizes the pinned converter's literal descriptor syntax, not general
    C++. Dense storage format is checked; permute_order_to_src has no corresponding
    C++ descriptor field and must not be interpreted as a physical stride layout.
    """
    kinds = {0: 'APP_WRITE', 1: 'APP_READ', 2: 'APP_READWRITE',
             3: 'NATIVE', 4: 'STATIC'}
    for name in names:
        desc = graph['tensors'][name]
        marker = r'\.name\s*=\s*"' + re.escape(name) + r'"\s*,'
        owners = [(body, match) for body in funcs.values()
                  for match in re.finditer(marker, body)]
        if len(owners) != 1:
            raise ValueError('Missing or ambiguous C++ boundary descriptor: ' + name)
        body, match = owners[0]
        tail = body[match.end():]
        end = re.search(r'\.isProduced\s*=', tail)
        if end is None or re.search(r'\.name\s*=', tail[:end.start()]):
            raise ValueError('Malformed C++ boundary descriptor: ' + name)
        fields = tail[:end.start()]

        def field(key):
            matches = re.findall(r'\.' + key + r'\s*=\s*([A-Za-z_0-9]+)\s*,', fields)
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous C++ boundary ' + key + ': ' + name)
            return matches[0]

        expected = dict(type='QNN_TENSOR_TYPE_' + kinds[desc['type']],
                        dataType='QNN_DATATYPE_FLOAT_16',
                        dataFormat='QNN_TENSOR_DATA_FORMAT_DENSE',
                        rank=str(len(desc['dims'])), isDynamicDimensions='nullptr')
        for key, value in expected.items():
            if field(key) != value:
                raise ValueError('C++/JSON boundary mismatch (' + key + '): ' + name)
        dimension_name = field('dimensions')
        arrays = re.findall(r'uint32_t\s+' + re.escape(dimension_name) +
                            r'\s*\[\s*\]\s*=\s*\{([^}]*)\}', body)
        if len(arrays) != 1 or not re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*', arrays[0]):
            raise ValueError('Unsupported C++ boundary dimensions: ' + name)
        if [int(x.strip()) for x in arrays[0].split(',')] != desc['dims']:
            raise ValueError('C++/JSON boundary mismatch (dimensions): ' + name)


def extract(source, destination, inputs, outputs, graph_name):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    graph = json.loads((source / 'preserve_net.json').read_text())['graph']
    chosen, constants = select_region(graph, inputs, outputs)
    for name in inputs + outputs:
        tensor = graph['tensors'][name]
        if tensor['data_type'] != 534:
            raise ValueError('Only fixed FP16 cut tensors are supported: ' + name)
        if tensor.get('dataFormat', 0) != 0 or any(tensor.get('is_dynamic_dims', [])):
            raise ValueError('Only dense static boundary tensors are supported: ' + name)
        if not tensor['dims'] or any(type(d) is not int or d <= 0 for d in tensor['dims']):
            raise ValueError('Boundary dimensions must be fixed positive integers: ' + name)
    cpp = (source / 'preserve_fixed.cpp').read_text()
    funcs = {m.group(1): m.group() for m in re.finditer(
        r'^static ModelError_t (add(?:Tensor|Node)_\w+)\(QnnModel& model\)\{.*?^\}',
        cpp, re.M | re.S)}
    validate_cpp_boundary(funcs, graph, inputs + outputs)
    head = cpp[:cpp.index('static ModelError_t ')]
    pos = cpp.index('ModelError_t QnnModel_composeGraphs')
    foot = cpp[cpp.rfind('QNN_API', 0, pos):]
    start = foot.index('  VALIDATE(addTensor_')
    end = foot.index('  // Add all models', start)
    parts, calls, node_hashes = [], [], {}
    template = funcs['addTensor_new_img']
    for name in inputs:
        tensor = graph['tensors'][name]
        text = template.replace('new_img', name)
        text, count = re.subn(r'uint32_t dimensions_\w+\[\] = \{[^}]+\}',
            'uint32_t dimensions_' + name + '[] = {' + ', '.join(map(str, tensor['dims'])) + '}', text)
        assert count == 1
        text = text.replace('.rank= 4', '.rank= ' + str(len(tensor['dims'])))
        parts.append(text)
        calls.append('  VALIDATE(addTensor_' + name + '(preserve), err);')
    for name in constants:
        parts.append(funcs['addTensor_' + name])
        calls.append('  VALIDATE(addTensor_' + name + '(preserve), err);')
    for name, node in chosen:
        text = funcs['addNode_' + name]
        node_hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        for output in outputs:
            if output in node['output_names']:
                text, count = re.subn(r'(\.name= "' + re.escape(output) + r'",\s*\.type= )QNN_TENSOR_TYPE_(?:NATIVE|APP_READ)',
                                     r'\1QNN_TENSOR_TYPE_APP_READ', text)
                assert count == 1
        parts.append(text)
        calls.append('  VALIDATE(addNode_' + name + '(preserve), err);')
    foot = foot[:start] + '\n'.join(calls) + '\n\n' + foot[end:]
    foot = foot.replace('"preserve"', '"' + graph_name + '"')
    raw_needed = set(re.findall(r'BINVARSTART\((\w+)\)', '\n'.join(parts)))
    destination.mkdir(parents=True)
    (destination / 'region.cpp').write_text(head + '\n\n'.join(parts) + '\n\n' + foot)
    with tarfile.open(source / 'preserve_fixed.bin') as inp, tarfile.open(destination / 'region.bin', 'w') as out:
        found = set()
        for member in inp:
            key = member.name.removesuffix('.raw')
            if key in raw_needed:
                if not member.isfile() or key in found:
                    raise ValueError('Invalid constant archive member: ' + member.name)
                out.addfile(member, inp.extractfile(member))
                found.add(key)
        assert found == raw_needed
    manifest = dict(graph_name=graph_name,
        input_tensors={x: graph['tensors'][x] for x in inputs},
        output_tensors={x: graph['tensors'][x] for x in outputs},
        nodes=[dict(name=k, type=v['type']) for k, v in chosen],
        unchanged_node_function_sha256=node_hashes,
        source_files={f: sha(source / f) for f in ['preserve_net.json', 'preserve_fixed.cpp', 'preserve_fixed.bin']},
        generated_files={f: sha(destination / f) for f in ['region.cpp', 'region.bin']},
        constant_bytes=(destination / 'region.bin').stat().st_size,
        semantics='Node and constant bodies preserved. Only graph boundary descriptors changed. Backend may change layouts/fusion; numeric validation is required.',
        target_latency_measured=False)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--outputs', nargs='+', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', args.name):
        parser.error('Graph name must be a simple identifier')
    result = extract(args.source, args.output, args.inputs, args.outputs, args.name)
    print(json.dumps({k: result[k] for k in ['graph_name', 'constant_bytes', 'target_latency_measured']}))
