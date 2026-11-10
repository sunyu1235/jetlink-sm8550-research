#!/usr/bin/env python3
"""Complete truthful static metadata without loading external weights or running inference."""
import argparse, json
from pathlib import Path
import onnx
from onnx import helper, TensorProto as T


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    require(value.type.HasField('tensor_type'), f'Not tensor metadata: {value.name}')
    tensor = value.type.tensor_type
    require(tensor.elem_type != T.UNDEFINED and tensor.HasField('shape'), f'Missing type/shape: {value.name}')
    require(all(d.HasField('dim_value') and d.dim_value >= 0 for d in tensor.shape.dim),
            f'Nonstatic/invalid dimension: {value.name}')
    return tensor.elem_type, tuple(d.dim_value for d in tensor.shape.dim)


def complete_metadata(model):
    infos = {}
    added = []
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        sig = signature(value)
        require(value.name not in infos or infos[value.name] == sig, f'Conflicting metadata: {value.name}')
        infos[value.name] = sig

    def put(name, dtype, shape, evidence):
        sig = dtype, tuple(shape)
        if name in infos:
            require(infos[name] == sig, f'Metadata contradicts {evidence}: {name}: {infos[name]} != {sig}')
        else:
            model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, list(shape)))
            infos[name] = sig
            added.append({'name': name, 'dtype': dtype, 'shape': list(shape), 'evidence': evidence})

    for tensor in model.graph.initializer:
        put(tensor.name, tensor.data_type, tensor.dims, 'initializer TensorProto dtype/dims')
    prefix = 'first_projection_qdq__'
    nodes = [n for n in model.graph.node if n.name.startswith(prefix)]
    require([n.op_type for n in nodes] == ['Cast', 'QuantizeLinear', 'DequantizeLinear',
            'DequantizeLinear', 'MatMul', 'QuantizeLinear', 'DequantizeLinear', 'Cast'],
            'Unexpected first-projection splice topology')
    require(infos.get('layer_norm_49') == (T.FLOAT16, (1, 288, 512)), 'Unexpected input boundary metadata')
    for node in nodes:
        require(len(node.output) == 1 and all(name in infos for name in node.input if name),
                f'Missing input metadata: {node.name}')
        shape = infos[node.input[0]][1]
        if node.op_type == 'Cast':
            attr = [a for a in node.attribute if a.name == 'to']
            require(len(attr) == 1, 'Missing/ambiguous Cast type')
            dtype = attr[0].i
        elif node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
            require(len(node.input) == 3, 'Expected explicit scale and zero point')
            scale_type, scale_shape = infos[node.input[1]]
            zero_type, zero_shape = infos[node.input[2]]
            require(scale_type == T.FLOAT and scale_shape == zero_shape == (), 'Expected per-tensor scalar encodings')
            expected_input_type = scale_type if node.op_type == 'QuantizeLinear' else zero_type
            require(infos[node.input[0]][0] == expected_input_type, f'QDQ input dtype mismatch: {node.name}')
            dtype = zero_type if node.op_type == 'QuantizeLinear' else scale_type
        else:
            left_type, left = infos[node.input[0]]
            right_type, right = infos[node.input[1]]
            require(left_type == right_type == T.FLOAT and left == (1, 288, 512) and right == (512, 1024),
                    'Unexpected selected MatMul dtype/shape')
            dtype, shape = T.FLOAT, (1, 288, 1024)
        put(node.output[0], dtype, shape, f'{node.op_type} static semantics with declared input shapes')
    require(infos.get('val_646') == (T.FLOAT16, (1, 288, 1024)), 'Unexpected output boundary metadata')
    # Match SDK get_all_tensors_in_model exactly, including optional empty slots.
    names = {name for node in model.graph.node for name in list(node.input) + list(node.output)}
    require('' not in names, 'Optional empty tensor slot encountered; never fabricate a declaration')
    missing = sorted(names - infos.keys())
    require(not missing, f'Referenced tensors without truthful metadata: {missing}')
    count = len(model.graph.value_info) + len(model.graph.input) + len(model.graph.output)
    require(bool(model.graph.value_info) and count >= len(names), 'SDK would still invoke shape inference')
    return {'added_metadata': added, 'added_count': len(added), 'referenced_tensor_names': len(names),
            'unique_declared_names': len(infos), 'declaration_count_for_sdk_predicate': count,
            'missing_referenced_names': missing, 'all_referenced_types_and_shapes_static': True,
            'sdk_up_convert_infer_shapes_would_skip_inference': True,
            'shape_inference_executed': False}


def patch(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(source.parent == output.parent, 'Use the same directory so external-data references stay unchanged')
    require(not output.exists(), 'Refusing to overwrite destination')
    model = onnx.load(source, load_external_data=False)
    nodes_before = [n.SerializeToString() for n in model.graph.node]
    tensors_before = [t.SerializeToString() for t in model.graph.initializer]
    report = complete_metadata(model)
    require(nodes_before == [n.SerializeToString() for n in model.graph.node], 'A node changed')
    require(tensors_before == [t.SerializeToString() for t in model.graph.initializer], 'Initializer metadata/payload changed')
    # Direct protobuf write leaves all external locations/offsets/lengths byte-identical.
    output.write_bytes(model.SerializeToString())
    onnx.checker.check_model(str(output), full_check=False)
    report.update({'source': str(source), 'output': str(output), 'nodes_byte_identical': True,
                   'initializers_byte_identical': True, 'external_data_shared_unchanged': True,
                   'checker_passed': True, 'converter_retested': False, 'external_tensor_payloads_loaded': False})
    output.with_suffix('.metadata-report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True, help='New filename beside source, sharing its external data')
    a = p.parse_args()
    print(json.dumps(patch(a.source, a.output), indent=2))


if __name__ == '__main__':
    main()
