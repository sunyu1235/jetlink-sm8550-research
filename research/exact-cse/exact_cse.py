#!/usr/bin/env python3
"""Conservative, value-identical ONNX CSE. Never rewrite opaque cached contexts.

No algebraic reassociation, approximate equality, dead-head pruning, training,
precision change, or rewrite of unknown/stateful operators. All graph inputs and
outputs, including recurrent outputs, retain their full protobuf signatures.
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

# Deliberate allowlist: stochastic, control-flow, training and unknown operations
# are barriers. Add a pure operation only after reviewing its ONNX semantics.
PURE = set('Add Cast Concat Constant Conv Div Expand Flatten Gather GatherND Gelu Gemm Identity LayerNormalization MatMul Mul Neg Not Pad Pow ReduceMax ReduceMean ReduceSum Relu Reshape Shape Sigmoid Slice Softmax Split Sqrt Squeeze Sub Tanh Transpose Unsqueeze Where'.split())
EXACT_GELUS = {'ExactGelu', 'ExactGelu32', 'ExactGeluTanh', 'ExactGeluTanh32'}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_signature(tensor, base_dir):
    """Hash actual values, never a filename/offset or tolerance comparison."""
    array = numpy_helper.to_array(tensor, base_dir=str(base_dir))
    if array.dtype.hasobject:
        # No string/object canonicalization is needed by this model.
        return ('unshared', tensor.name)
    return (tensor.data_type, tuple(tensor.dims), hashlib.sha256(array.tobytes()).hexdigest())


def reject_external_attribute_tensors(model):
    """The supported external-data contract is root graph initializers only."""
    def walk(message, path=()):
        if isinstance(message, TensorProto):
            if message.data_location == TensorProto.EXTERNAL and path[:2] != ('graph', 'initializer'):
                raise ValueError('External tensor attributes/functions are unsupported; only graph initializers are supported')
            return
        for descriptor, value in message.ListFields():
            if descriptor.type != descriptor.TYPE_MESSAGE:
                continue
            if descriptor.is_repeated:
                for index, child in enumerate(value):
                    walk(child, path + (descriptor.name, index))
            else:
                walk(value, path + (descriptor.name,))
    walk(model)


def validate_destinations(model, source_dir, output, report):
    targets = [p.resolve() for p in (output, report) if p is not None]
    if len(targets) != len(set(targets)):
        raise ValueError('Output and report paths must be distinct')
    if output is not None:
        locations = {e.value for t in model.graph.initializer for e in t.external_data if e.key == 'location'}
        external_targets = {(output.parent / location).resolve() for location in locations}
        if set(targets) & external_targets:
            raise ValueError('Output/report path collides with an external-data file')
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite an output or report')


def rewrite(model, base_dir, allow_jetlink_exact_gelu=False):
    reject_external_attribute_tensors(model)
    if model.training_info or model.graph.sparse_initializer:
        raise ValueError('Training graphs and sparse initializers require a separate proof')
    function_calls = {(f.domain, f.name, f.overload) for f in model.functions}
    if any(n.op_type == 'EPContext' for n in model.graph.node):
        raise ValueError('EPContext is opaque: use the pre-partition source graph, not cached context ONNX')
    if any(a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for n in model.graph.node for a in n.attribute):
        raise ValueError('Nested graphs require scope-aware CSE and are not supported')
    result = copy.deepcopy(model)
    input_names = {x.name for x in model.graph.input}
    types = {x.name: x.type.SerializeToString(deterministic=True)
             for x in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
    opsets = {x.domain: x.version for x in model.opset_import}
    if allow_jetlink_exact_gelu and opsets.get('jetlink.precision') != 1:
        raise ValueError('Expected jetlink.precision opset 1')
    aliases, weights_seen, operations_seen = {}, {}, {}
    removed, barriers, kept = [], [], []
    known = input_names | {x.name for x in model.graph.initializer}

    def canonical(name):
        while name in aliases:
            name = aliases[name]
        return name

    for tensor in model.graph.initializer:
        # ONNX graph-input initializers may be overridden by the caller.
        if tensor.name in input_names:
            continue
        signature = tensor_signature(tensor, base_dir)
        if signature in weights_seen:
            aliases[tensor.name] = weights_seen[signature]
        else:
            weights_seen[signature] = tensor.name

    for index, original in enumerate(model.graph.node):
        if any(name and name not in known for name in original.input):
            raise ValueError(f'Graph must be topologically sorted: {original.name}')
        if any(name and name in known for name in original.output):
            raise ValueError(f'Duplicate producer: {original.name}')
        node = copy.deepcopy(original)
        for j, name in enumerate(node.input):
            node.input[j] = canonical(name)
        pure_standard = node.domain in ('', 'ai.onnx') and node.op_type in PURE
        pure_custom = (allow_jetlink_exact_gelu and node.domain == 'jetlink.precision'
                       and node.op_type in EXACT_GELUS and not node.attribute
                       and len(node.input) == len(node.output) == 1)
        # Unknown operators remain distinct, including each of their outputs.
        pure = (pure_standard or pure_custom) and (node.domain, node.op_type, node.overload) not in function_calls
        if not pure:
            barriers.append({'index': index, 'name': node.name, 'domain': node.domain, 'op_type': node.op_type})
        key = (node.domain, node.op_type, node.overload,
               opsets.get(node.domain), tuple(node.input),
               tuple(a.SerializeToString(deterministic=True) for a in sorted(node.attribute, key=lambda a: a.name)),
               tuple(bool(name) for name in node.output),
               tuple(types.get(name) for name in node.output))
        prior = operations_seen.get(key) if pure else None
        if prior is not None:
            # Dedup whole multi-output operations atomically, retaining positions.
            for old, new in zip(node.output, prior.output):
                if old:
                    aliases[old] = new
            removed.append({'index': index, 'name': node.name, 'domain': node.domain,
                            'op_type': node.op_type, 'outputs': list(node.output),
                            'replacement_outputs': list(prior.output)})
        else:
            if pure:
                operations_seen[key] = node
            kept.append(node)
        known.update(name for name in original.output if name)

    preserved_outputs = []
    node_names = {n.name for n in kept}
    for output in model.graph.output:
        if output.name in aliases:
            name = 'exact_cse_preserve_' + output.name
            while name in node_names:
                name += '_'
            node_names.add(name)
            kept.append(helper.make_node('Identity', [canonical(output.name)], [output.name], name=name))
            preserved_outputs.append(output.name)
    result.graph.ClearField('node')
    result.graph.node.extend(kept)
    inits = [t for t in model.graph.initializer if t.name not in aliases]
    result.graph.ClearField('initializer')
    result.graph.initializer.extend(inits)
    names = {n for node in kept for n in node.output} | input_names | {t.name for t in inits}
    infos = [v for v in model.graph.value_info if v.name in names]
    result.graph.ClearField('value_info')
    result.graph.value_info.extend(infos)
    for field in ('input', 'output'):
        assert [v.SerializeToString() for v in getattr(result.graph, field)] == [v.SerializeToString() for v in getattr(model.graph, field)]
    before = collections.Counter(n.op_type for n in model.graph.node)
    after = collections.Counter(n.op_type for n in result.graph.node)
    return result, {
        'status': 'SOURCE_GRAPH_ONLY_NOT_PHONE_VALIDATED',
        'nodes_before': len(model.graph.node), 'nodes_after': len(kept),
        'initializers_before': len(model.graph.initializer), 'initializers_after': len(inits),
        'removed_operations': dict(collections.Counter(n['op_type'] for n in removed)),
        'net_operations_removed': dict(before - after),
        'aliases': {k: canonical(v) for k, v in aliases.items()},
        'removed_nodes': removed, 'unknown_operation_barriers': barriers,
        'graph_output_alias_identities': preserved_outputs,
        'all_graph_input_output_protobufs_preserved': True,
        'arithmetic_reassociation': False, 'weights_changed': False,
        'precision_changed': False, 'numerical_160_run': False,
        'phone_speed_measured': False,
    }


def copy_external_files(model, source_dir, output_dir):
    """Keep external offsets valid. Copy only exact files, without repacking."""
    paths = {e.value for t in model.graph.initializer for e in t.external_data if e.key == 'location'}
    for location in paths:
        relative = Path(location)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('External data must stay inside the source directory')
        source = source_dir / relative
        target = output_dir / relative
        if source.resolve() == target.resolve():
            continue
        if target.exists():
            if sha256_file(source) != sha256_file(target):
                raise ValueError(f'Existing external weight differs: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--allow-jetlink-exact-gelu', action='store_true',
                        help='Only for the reviewed existing jetlink.precision opset-1 CPU GELU implementations')
    args = parser.parse_args()
    for target in (args.output, args.report):
        if target and target.exists():
            parser.error(f'Refusing to overwrite {target}')
    source = args.source.resolve()
    model = onnx.load(source, load_external_data=False)
    candidate, report = rewrite(model, source.parent, args.allow_jetlink_exact_gelu)
    validate_destinations(candidate, source.parent, args.output, args.report)
    report['source'] = str(source)
    report['source_sha256'] = sha256_file(source)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        copy_external_files(candidate, source.parent, args.output.parent.resolve())
        onnx.save(candidate, args.output)
        onnx.checker.check_model(str(args.output))
        report.update(output=str(args.output.resolve()), output_sha256=sha256_file(args.output), onnx_checker='passed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('aliases', 'removed_nodes', 'unknown_operation_barriers')}, indent=2))


if __name__ == '__main__':
    main()
