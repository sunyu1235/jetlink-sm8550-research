#!/usr/bin/env python3
"""Reconstruct the 131_41 cut from the public teacher, then expose compact outputs.

This is NOT recovery of the accepted cached QNN partition. The public teacher
retains MatMul/Add and native LayerNormalization, unlike the archived optimized
partition. Only this cut's original external tensor ranges are materialized.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper, shape_inference

INPUT_NAMES = ['gelu_43', 'gelu_54', 'gelu_61', 'add_50', 'add_70', 'add_86']
OUTPUT_NAMES = ['add_52', 'add_72', 'add_88', 'linear_110', 'linear_151', 'linear_180']
QUERY_SOURCES = ['squeeze_9', 'squeeze_21', 'squeeze_33']
KV_SOURCES = [('squeeze_10', 'squeeze_11'), ('squeeze_22', 'squeeze_23'), ('squeeze_34', 'squeeze_35')]
ATTENTION_OUTPUTS = [('matmul_6', 'matmul_7'), ('matmul_14', 'matmul_15'), ('matmul_22', 'matmul_23')]
DEFAULT_SOURCE = Path('/workspace/shared/jetlink-distillation/real-block/teacher-capture.onnx')
DEFAULT_MANIFEST = Path('/workspace/scratch/a607d96f1cc1/htp-work/reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-partition-manifest.json')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: onnx.TensorProto) -> str:
    return hashlib.sha256(numpy_helper.to_array(tensor).tobytes()).hexdigest()


def info(value: onnx.ValueInfoProto) -> dict:
    tensor = value.type.tensor_type
    shape = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in tensor.shape.dim]
    return {'name': value.name, 'elem_type': tensor.elem_type, 'shape': shape}


def value_map(model: onnx.ModelProto) -> dict:
    return {value.name: value for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}


def reachable(nodes, weights, cuts, outputs):
    producers = {}
    for index, node in enumerate(nodes):
        for name in node.output:
            if name in producers:
                raise ValueError(f'Duplicate tensor producer: {name}')
            producers[name] = index
    needed_nodes, needed_weights = set(), set()
    visiting = set()

    def visit(name):
        if not name or name in cuts:
            return
        if name in weights:
            needed_weights.add(name)
            return
        if name not in producers:
            raise ValueError(f'Undeclared source boundary: {name}')
        index = producers[name]
        if index in visiting:
            raise ValueError(f'Graph cycle at {name}')
        if index not in needed_nodes:
            visiting.add(index)
            for dependency in nodes[index].input:
                visit(dependency)
            visiting.remove(index)
            needed_nodes.add(index)

    for name in outputs:
        visit(name)
    return needed_nodes, needed_weights


def checked(model):
    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)
    values = value_map(model)
    missing = []
    for node in model.graph.node:
        for name in node.output:
            if not name:
                continue
            if name not in values or any(not isinstance(dim, int) or dim <= 0 for dim in info(values[name])['shape']):
                missing.append(name)
    if missing:
        raise ValueError(f'Unknown/nonpositive intermediate shapes: {missing}')
    return model


def extract_baseline(source_path: Path, contract: dict):
    if [entry['name'] for entry in contract['inputs']] != INPUT_NAMES:
        raise ValueError('Unexpected input names/order in archived contract')
    if [entry['name'] for entry in contract['outputs']] != OUTPUT_NAMES:
        raise ValueError('Unexpected output names/order in archived contract')
    model = onnx.load(source_path, load_external_data=False)
    values = value_map(model)
    # The supplied teacher already has all boundary shapes. Infer only if absent.
    if any(name not in values for name in INPUT_NAMES + OUTPUT_NAMES):
        model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        values = value_map(model)
    for entry in contract['inputs'] + contract['outputs']:
        if entry['name'] not in values or info(values[entry['name']]) != entry:
            raise ValueError(f'Source boundary disagrees with archived contract: {entry}')
    weights = {tensor.name: tensor for tensor in model.graph.initializer}
    needed_nodes, needed_weights = reachable(model.graph.node, weights, set(INPUT_NAMES), OUTPUT_NAMES)
    nodes = [copy.deepcopy(node) for index, node in enumerate(model.graph.node) if index in needed_nodes]
    if any(node.domain not in ('', 'ai.onnx') for node in nodes):
        raise ValueError('Custom op in source cut; explicit lowering required')
    selected_weights, weight_provenance = [], []
    for name in sorted(needed_weights):
        tensor = copy.deepcopy(weights[name])
        location = {entry.key: entry.value for entry in tensor.external_data}
        if external_data_helper.uses_external_data(tensor):
            external_data_helper.load_external_data_for_tensor(tensor, str(source_path.parent))
        tensor.data_location = TensorProto.DEFAULT
        del tensor.external_data[:]
        selected_weights.append(tensor)
        weight_provenance.append({'name': name, 'shape': list(tensor.dims), 'elem_type': tensor.data_type,
                                  'raw_sha256': tensor_sha(tensor), 'source_external_data': location})
    baseline = helper.make_model(helper.make_graph(
        nodes, 'public_teacher_reconstructed_partition_131_41',
        [copy.deepcopy(values[name]) for name in INPUT_NAMES],
        [copy.deepcopy(values[name]) for name in OUTPUT_NAMES], selected_weights),
        producer_name='jetlink-compact-boundaries',
        opset_imports=[copy.deepcopy(opset) for opset in model.opset_import if opset.domain in ('', 'ai.onnx')])
    baseline.ir_version = model.ir_version
    baseline.doc_string = 'Public teacher reconstruction; not the accepted cached/optimized partition.'
    return checked(baseline), weight_provenance


def compact_partition(baseline: onnx.ModelProto, contract: dict):
    baseline = checked(copy.deepcopy(baseline))
    weights = {tensor.name: tensor for tensor in baseline.graph.initializer}
    producers = {name: node for node in baseline.graph.node for name in node.output}
    consumers = collections.defaultdict(list)
    for node in baseline.graph.node:
        for name in node.input:
            consumers[name].append(node)
    values = value_map(baseline)
    aliases = dict(zip(OUTPUT_NAMES, [name + '__last_query' for name in OUTPUT_NAMES]))
    if [item['name'] for item in contract['compact_outputs']] != list(aliases.values()):
        raise ValueError('Unexpected compact output names/order')
    for source in QUERY_SOURCES:
        users = consumers[source]
        if len(users) != 1 or users[0].op_type != 'LayerNormalization' or users[0].input[0] != source:
            raise ValueError(f'Expected one raw-teacher Q LayerNormalization consumer: {source}')
        axis = next((helper.get_attribute_value(a) for a in users[0].attribute if a.name == 'axis'), -1)
        if axis not in (-1, 3) or info(values[source])['shape'] != [1, 8, 288, 64]:
            raise ValueError(f'Q normalization is not independently token-local: {source}')

    additions, nodes, tainted = [], [], set()
    query_cuts, residual_cuts, mask_proofs, shape_rewrites = [], [], [], []

    def add_ints(name, data):
        additions.append(numpy_helper.from_array(np.asarray(data, dtype=np.int64), name))
        return name

    starts = add_ints('jl_compact_starts', [287])
    ends = add_ints('jl_compact_ends', [288])
    axis1 = add_ints('jl_compact_axis1', [1])
    axis2 = add_ints('jl_compact_axis2', [2])
    allowed_tainted_ops = {'LayerNormalization', 'MatMul', 'Mul', 'Identity', 'Softmax', 'Transpose', 'Reshape', 'Add'}
    for original in baseline.graph.node:
        node = copy.deepcopy(original)
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = aliases.get(name, name)
        if node.input and node.input[0] in QUERY_SOURCES:
            source = node.input[0]
            cut = source + '__last_query'
            nodes.append(helper.make_node('Slice', [source, starts, ends, axis2], [cut], name=source + '__last_query_slice'))
            node.input[0] = cut
            tainted.add(cut)
            query_cuts.append({'source': source, 'output': cut, 'axis': 2, 'start': 287, 'end': 288,
                               'before': original.name, 'raw_source_adapter': 'before token-local Q LayerNormalization'})
        if node.op_type == 'Where':
            mask_node = producers.get(original.input[0])
            if mask_node is None or mask_node.op_type != 'Not' or mask_node.input[0] not in weights:
                raise ValueError(f'Unexpected masked attention source: {node.name}')
            mask = numpy_helper.to_array(weights[mask_node.input[0]])
            if mask.shape != (1, 1, 288, 288) or mask.dtype != np.bool_ or not mask[..., 287, :].all():
                raise ValueError(f'Last query row is not fully unmasked: {node.name}')
            if node.input[2] not in tainted:
                raise ValueError(f'Attention scores were not pruned: {node.name}')
            score = node.input[2]
            node.op_type = 'Identity'
            del node.input[:]
            node.input.append(score)
            del node.attribute[:]
            mask_proofs.append({'node': original.name, 'mask': mask_node.input[0], 'shape': list(mask.shape),
                                'last_row_all_true': True, 'last_row_sha256': hashlib.sha256(mask[..., 287, :].tobytes()).hexdigest()})
        if original.output[0] in OUTPUT_NAMES[:3]:
            if node.op_type != 'Add' or node.input[1] not in tainted or node.input[0] in tainted:
                raise ValueError(f'Unexpected residual branch: {node.name}')
            source = node.input[0]
            cut = source + '__last_query_residual'
            nodes.append(helper.make_node('Slice', [source, starts, ends, axis1], [cut], name=source + '__last_query_residual_slice'))
            node.input[0] = cut
            tainted.add(cut)
            residual_cuts.append({'source': source, 'output': cut, 'axis': 1, 'start': 287, 'end': 288})
        if node.op_type == 'Reshape' and node.input[0] in tainted:
            if node.input[1] not in weights:
                raise ValueError(f'Unknown reshape constant: {node.name}')
            before = numpy_helper.to_array(weights[node.input[1]])
            if before.tolist() != [1, 288, 512]:
                raise ValueError(f'Unexpected compact reshape: {node.name}: {before}')
            after = np.asarray([1, 1, 512], dtype=before.dtype)
            name = node.input[1] + '__' + node.name + '__last_query'
            additions.append(numpy_helper.from_array(after, name))
            node.input[1] = name
            shape_rewrites.append({'node': node.name, 'before': before.tolist(), 'after': after.tolist()})
        if any(name in tainted for name in node.input):
            if node.op_type not in allowed_tainted_ops:
                raise ValueError(f'Unproved operation after query slicing: {node.op_type} {node.name}')
            tainted.update(node.output)
        nodes.append(node)
    if [len(query_cuts), len(residual_cuts), len(mask_proofs), len(shape_rewrites)] != [3, 3, 3, 3]:
        raise ValueError('Did not rewrite exactly three independent final-layer branches')
    if not set(aliases.values()).issubset(tainted):
        raise ValueError('Some output was not reduced to last query')

    optimized = copy.deepcopy(baseline)
    optimized.graph.name = 'public_teacher_compact_partition_131_41'
    del optimized.graph.value_info[:]
    del optimized.graph.output[:]
    for entry in contract['compact_outputs']:
        optimized.graph.output.append(helper.make_tensor_value_info(entry['name'], entry['elem_type'], entry['shape']))
    all_weights = [*baseline.graph.initializer, *additions]
    live_nodes, live_weights = reachable(nodes, {tensor.name: tensor for tensor in all_weights}, set(INPUT_NAMES), aliases.values())
    del optimized.graph.node[:]
    optimized.graph.node.extend(node for index, node in enumerate(nodes) if index in live_nodes)
    del optimized.graph.initializer[:]
    optimized.graph.initializer.extend(tensor for tensor in all_weights if tensor.name in live_weights)
    optimized.doc_string = 'Compact last-query public teacher cut. K/V retains 288 tokens. Requires new downstream ABI.'
    optimized = checked(optimized)
    if [entry.SerializeToString() for entry in optimized.graph.input] != [entry.SerializeToString() for entry in baseline.graph.input]:
        raise ValueError('Input metadata changed')
    if [info(value) for value in optimized.graph.output] != contract['compact_outputs']:
        raise ValueError('Compact output contract changed')
    if any(node.op_type == 'Concat' or '__restore_boundary_shape' in node.name for node in optimized.graph.node):
        raise ValueError('Boundary restoration must not remain')
    if any('zero_prefix' in tensor.name for tensor in optimized.graph.initializer):
        raise ValueError('Zero-prefix storage must not remain')
    optimized_values = value_map(optimized)
    optimized_producers = {name: node for node in optimized.graph.node for name in node.output}
    kv_proofs = []
    for key, value in KV_SOURCES:
        ancestor_ids, _ = reachable(baseline.graph.node, weights, set(INPUT_NAMES), [key, value])
        for index in ancestor_ids:
            original = baseline.graph.node[index]
            after = optimized_producers.get(original.output[0])
            if after is None or after.SerializeToString() != original.SerializeToString():
                raise ValueError(f'K/V ancestor changed: {original.name}')
        for name in (key, value):
            if info(optimized_values[name])['shape'] != [1, 8, 288, 64]:
                raise ValueError(f'K/V sequence was pruned: {name}')
        kv_proofs.append({'key_source': key, 'value_source': value, 'shape': [1, 8, 288, 64],
                          'unchanged_ancestor_nodes': len(ancestor_ids)})
    attention_proofs = []
    for score_name, context_name in ATTENTION_OUTPUTS:
        score_node, context_node = optimized_producers[score_name], optimized_producers[context_name]
        if score_node.op_type != 'MatMul' or context_node.op_type != 'MatMul':
            raise ValueError('Attention MatMul contract changed')
        expected = {score_node.input[0]: [1, 8, 1, 64], score_node.input[1]: [1, 8, 64, 288],
                    score_name: [1, 8, 1, 288], context_node.input[0]: [1, 8, 1, 288],
                    context_node.input[1]: [1, 8, 288, 64], context_name: [1, 8, 1, 64]}
        for name, shape in expected.items():
            if info(optimized_values[name])['shape'] != shape:
                raise ValueError(f'Unexpected attention shape: {name}')
        ancestor_ids, _ = reachable(baseline.graph.node, weights, set(INPUT_NAMES),
                                    [score_node.input[1], context_node.input[1]])
        for index in ancestor_ids:
            original = baseline.graph.node[index]
            if optimized_producers[original.output[0]].SerializeToString() != original.SerializeToString():
                raise ValueError(f'Final K/V ancestor changed: {original.name}')
        attention_proofs.append({'score': score_name, 'context': context_name, 'verified_shapes': expected,
                                 'unchanged_final_kv_ancestor_nodes': len(ancestor_ids)})
    retained = []
    for tensor in optimized.graph.initializer:
        if tensor.name in weights:
            if tensor.SerializeToString() != weights[tensor.name].SerializeToString():
                raise ValueError(f'Retained original weight changed: {tensor.name}')
            retained.append(tensor.name)
    return optimized, {'query_slices': query_cuts, 'residual_slices': residual_cuts, 'mask_proofs': mask_proofs,
                       'reshape_changes': shape_rewrites, 'kv_proofs': kv_proofs, 'attention_matmul_proofs': attention_proofs, 'retained_weight_names': retained,
                       'output_aliases_compact_to_downstream': {compact: original for original, compact in aliases.items()}}


def integration_cut_proof(source_path, baseline):
    source = onnx.load(source_path, load_external_data=False)
    removed = {node.name for node in baseline.graph.node}
    if len(removed) != len(baseline.graph.node):
        raise ValueError('Source cut node names are not unique')
    outputs = {value.name for value in baseline.graph.output}
    values = {name for node in baseline.graph.node for name in node.output}
    crossings = [{'tensor': name, 'consumer': node.name, 'is_declared_boundary': name in outputs}
                 for node in source.graph.node if node.name not in removed for name in node.input if name in values]
    observed = [value.name for value in source.graph.output if value.name in values]
    unexpected = [edge for edge in crossings if not edge['is_declared_boundary']]
    if unexpected or any(name not in outputs for name in observed):
        raise ValueError(f'Undeclared external consumer/observer of removed cut: {unexpected}, {observed}')
    return {'removed_node_names': [node.name for node in baseline.graph.node], 'external_consumers': crossings,
            'selected_source_graph_outputs': observed, 'undeclared_crossings': unexpected, 'status': 'PASS'}


def summary(path, model):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'nodes': len(model.graph.node), 'initializers': len(model.graph.initializer),
            'op_counts': dict(collections.Counter(node.op_type for node in model.graph.node))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--contract', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: args.output_dir / filename for key, filename in {
        'baseline': 'reconstructed-baseline-131-41.onnx', 'compact': 'compact-partition-131-41.onnx',
        'manifest': 'compact-partition-manifest.json', 'integration_proof': 'integration-cut-proof.json'}.items()}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite evidence: {path}')
    contract = json.loads(args.contract.read_text())
    baseline, provenance = extract_baseline(args.source, contract)
    compact, proof = compact_partition(baseline, contract)
    integration = integration_cut_proof(args.source, baseline)
    for name, model in [('baseline', baseline), ('compact', compact)]:
        onnx.save_model(model, paths[name])
        onnx.checker.check_model(str(paths[name]), full_check=True)
    external_files = sorted({entry['source_external_data']['location'] for entry in provenance if entry['source_external_data']})
    report = {
        'status': 'STRUCTURAL_FULL_SHAPE_PASS_NUMERICAL_TEST_PENDING',
        'provenance_caveat': 'Reconstructed original public-teacher cut, NOT byte-identical recovery of the accepted optimized/cached QNN partition. No accepted-context files were available.',
        'source': {'path': str(args.source.resolve()), 'sha256': sha(args.source),
                   'external_files': [{'path': str((args.source.parent / name).resolve()), 'sha256': sha(args.source.parent / name)} for name in external_files]},
        'source_capture_notes': 'teacher-capture.onnx was generated from published light-speed.onnx with Contiguous lowered to Identity and two capture outputs added. The selected cut contains neither custom nodes nor those capture outputs.',
        'archived_contract': {'path': str(args.contract.resolve()), 'sha256': sha(args.contract),
                              'accepted_baseline_sha256': contract['original_partition']['sha256']},
        'baseline': summary(paths['baseline'], baseline), 'compact': summary(paths['compact'], compact),
        'inputs': [info(v) for v in baseline.graph.input], 'baseline_outputs': [info(v) for v in baseline.graph.output],
        'compact_outputs': [info(v) for v in compact.graph.output], 'source_weights': provenance, **proof,
        'output_payload': {'baseline_fp16_bytes': 2654208, 'compact_fp16_bytes': 9216, 'bytes_removed': 2644992, 'ratio': 288},
        'checks': {'onnx_checker_full_check': 'PASS', 'strict_shape_inference_check_type': 'PASS',
                   'all_intermediate_shapes_concrete_positive': 'PASS', 'unchanged_kv_lineage': 'PASS',
                   'original_retained_weights_byte_identical': 'PASS', 'no_zero_padding_or_concat': 'PASS'},
        'acceptance_limits': ['Not accepted cached partition parity', 'Not a complete recurrent model replay',
                              'Not measured HTP/phone latency', 'Downstream must bind compact names/shapes; old cached contexts are incompatible']}
    paths['manifest'].write_text(json.dumps(report, indent=2) + '\n')
    paths['integration_proof'].write_text(json.dumps(integration, indent=2) + '\n')
    print(json.dumps({name: report[name] for name in ['status', 'baseline', 'compact', 'output_payload', 'output_aliases_compact_to_downstream']}, indent=2))


if __name__ == '__main__':
    main()
