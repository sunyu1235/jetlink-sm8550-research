#!/usr/bin/env python3
"""Splice only the selected QDQ projection; run only after a full-model slot is granted."""
import argparse, copy, hashlib, json
from pathlib import Path
import onnx
from onnx import TensorProto as T, helper
from complete_qdq_metadata import complete_metadata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def splice(base_path, qdq_path, outdir):
    outdir = Path(outdir)
    require(not outdir.exists(), f"Output directory already exists: {outdir}")
    base = onnx.load(base_path, load_external_data=True)
    qdq = onnx.load(qdq_path, load_external_data=True)
    targets = [(i, n) for i, n in enumerate(base.graph.node) if n.name == 'node_MatMul_546']
    require(len(targets) == 1, 'Expected exactly one node_MatMul_546')
    index, original = targets[0]
    require(original.op_type == 'MatMul' and list(original.input) == ['layer_norm_49', 'val_645']
            and list(original.output) == ['val_646'], 'Unexpected original projection boundary')
    nodes = list(qdq.graph.node[:6])
    require([n.op_type for n in nodes] == ['QuantizeLinear', 'DequantizeLinear',
            'DequantizeLinear', 'MatMul', 'QuantizeLinear', 'DequantizeLinear'], 'Unexpected QDQ first-six topology')
    require(nodes[0].input[0] == 'layer_norm_49' and nodes[3].name == original.name
            and nodes[-1].output[0] == 'val_646_DequantizeLinear_Output', 'Unexpected QDQ boundary')
    prefix = 'first_projection_qdq__'
    existing = {s for n in base.graph.node for s in list(n.input) + list(n.output)}
    existing.update(t.name for t in base.graph.initializer)
    existing.update(n.name for n in base.graph.node)
    require(not any(n.startswith(prefix) for n in existing), 'Splice prefix collision')
    initializers = {t.name: t for t in qdq.graph.initializer}
    produced = {s for n in nodes for s in n.output}
    required = {s for n in nodes for s in n.input if s and s not in produced and s != 'layer_norm_49'}
    require(required <= initializers.keys(), 'QDQ has undeclared external inputs')
    # Import only selected projection encodings/quantized weight, never later MLP tensors.
    require(initializers['layer_norm_49_zero_point'].data_type == T.UINT16
            and initializers['val_646_zero_point'].data_type == T.UINT16
            and initializers['val_645_quantized'].data_type == T.UINT8, 'Expected A16/W8 encodings')
    require(list(initializers['val_645_quantized'].dims) == [512, 1024], 'Unexpected weight shape')
    mapping = {s: prefix + s for s in produced | required}
    mapping['layer_norm_49'] = prefix + 'input_f32'
    imported = []
    for source in nodes:
        node = copy.deepcopy(source)
        node.name = prefix + source.name
        for i, name in enumerate(node.input):
            if name:
                node.input[i] = mapping[name]
        for i, name in enumerate(node.output):
            node.output[i] = mapping[name]
        imported.append(node)
    replacement = [helper.make_node('Cast', ['layer_norm_49'], [mapping['layer_norm_49']],
                    name=prefix + 'CastInputToFloat32', to=T.FLOAT)] + imported + [
                    helper.make_node('Cast', [mapping[nodes[-1].output[0]]], ['val_646'],
                    name=prefix + 'CastOutputToFloat16', to=T.FLOAT16)]
    preserved = [n.SerializeToString() for i, n in enumerate(base.graph.node) if i != index]
    all_nodes = list(base.graph.node)
    del base.graph.node[:]
    base.graph.node.extend(all_nodes[:index] + replacement + all_nodes[index + 1:])
    require([n.SerializeToString() for n in base.graph.node if not n.name.startswith(prefix)] == preserved,
            'An unrelated node changed')
    hashes = {}
    for name in sorted(required):
        t = copy.deepcopy(initializers[name])
        hashes[name] = hashlib.sha256(t.SerializeToString()).hexdigest()
        t.name = mapping[name]
        base.graph.initializer.append(t)
    # Remove the old FP16 weight only if no node still consumes it.
    used = {s for n in base.graph.node for s in n.input}
    removed = [t.name for t in base.graph.initializer if t.name == original.input[1] and t.name not in used]
    kept = [t for t in base.graph.initializer if t.name not in removed]
    del base.graph.initializer[:]
    base.graph.initializer.extend(kept)
    existing_opsets = {o.domain: o.version for o in base.opset_import}
    for domain in {n.domain for n in nodes}:
        version = next(o.version for o in qdq.opset_import if o.domain == domain)
        if domain not in existing_opsets:
            base.opset_import.append(helper.make_opsetid(domain, version))
        elif domain:
            require(existing_opsets[domain] == version, f'Custom-domain opset conflict: {domain}')
    metadata_report = complete_metadata(base)
    outdir.mkdir(parents=True)
    path = outdir / 'mixed-first-projection.onnx'
    onnx.save_model(base, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location='mixed-first-projection.data', size_threshold=0, convert_attribute=False)
    del base, qdq
    # Path-based checker handles external data without serializing a >2 GiB protobuf.
    onnx.checker.check_model(str(path), full_check=False)
    manifest = {'base': str(base_path), 'qdq_source': str(qdq_path), 'output': str(path),
                'replaced_node': original.name, 'inserted_nodes': len(replacement),
                'all_other_nodes_byte_identical': True, 'original_image_promotion_and_gelu_nodes_unchanged': True,
                'removed_unconsumed_initializers': removed, 'imported_initializer_sha256_before_rename': hashes,
                'metadata_completion': metadata_report,
                'checker_passed': True, 'numerical_parity_tested': False, 'converter_run': False}
    (outdir / 'splice-report.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='/workspace/shared/jetlink-models/light-speed-htp-gelu-images16.onnx')
    p.add_argument('--qdq', default='/workspace/shared/jetlink-qdq-first-only-final-data/QUInt16-w8-pc0.onnx')
    p.add_argument('--outdir', required=True, help='New, nonexistent directory')
    args = p.parse_args()
    print(json.dumps(splice(args.base, args.qdq, args.outdir), indent=2))


if __name__ == '__main__':
    main()
