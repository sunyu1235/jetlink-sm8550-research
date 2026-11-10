#!/usr/bin/env python3
"""Tiny synthetic fixtures only. Never opens a real full driving model."""
import io, json, tarfile, tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from splice_first_projection import splice
from repair_preserved_bias import repair


def test_splice(p):
    inits = [nh.from_array(np.zeros((512, 1024), np.float16), 'val_645'),
             nh.from_array(np.zeros(1024, np.float16), 'bias')]
    nodes = [h.make_node('MatMul', ['layer_norm_49', 'val_645'], ['val_646'], name='node_MatMul_546'),
             h.make_node('Add', ['val_646', 'bias'], ['biased'], name='original_bias'),
             h.make_node('GeluErfFp16', ['biased'], ['output'], name='exact_gelu', domain='jetlink_gelu')]
    model = h.make_model(h.make_graph(nodes, 'toy', [h.make_tensor_value_info('layer_norm_49', T.FLOAT16, [1, 288, 512])],
                         [h.make_tensor_value_info('output', T.FLOAT16, [1, 288, 1024])], inits),
                         opset_imports=[h.make_opsetid('', 20), h.make_opsetid('jetlink_gelu', 1)])
    model.graph.value_info.extend([h.make_tensor_value_info('biased', T.FLOAT16, [1, 288, 1024]),
                                   h.make_tensor_value_info('val_646', T.FLOAT16, [1, 288, 1024])])
    onnx.save(model, p / 'base.onnx')
    qnodes = [h.make_node('QuantizeLinear', ['layer_norm_49', 'layer_norm_49_scale', 'layer_norm_49_zero_point'], ['qx'], domain='com.microsoft', name='qx'),
              h.make_node('DequantizeLinear', ['val_645_quantized', 'ws', 'wz'], ['dw'], domain='com.microsoft', name='dw'),
              h.make_node('DequantizeLinear', ['qx', 'layer_norm_49_scale', 'layer_norm_49_zero_point'], ['dx'], domain='com.microsoft', name='dx'),
              h.make_node('MatMul', ['dx', 'dw'], ['z'], name='node_MatMul_546'),
              h.make_node('QuantizeLinear', ['z', 'ys', 'val_646_zero_point'], ['qy'], domain='com.microsoft', name='qy'),
              h.make_node('DequantizeLinear', ['qy', 'ys', 'val_646_zero_point'], ['val_646_DequantizeLinear_Output'], domain='com.microsoft', name='dy')]
    qi = [nh.from_array(np.array(.1, np.float32), n) for n in ['layer_norm_49_scale', 'ws', 'ys']]
    qi += [nh.from_array(np.array(32768, np.uint16), n) for n in ['layer_norm_49_zero_point', 'val_646_zero_point']]
    qi += [nh.from_array(np.array(128, np.uint8), 'wz'), nh.from_array(np.zeros((512, 1024), np.uint8), 'val_645_quantized')]
    q = h.make_model(h.make_graph(qnodes, 'qdq', [h.make_tensor_value_info('layer_norm_49', T.FLOAT, [1, 288, 512])],
                    [h.make_tensor_value_info('val_646_DequantizeLinear_Output', T.FLOAT, [1, 288, 1024])], qi),
                    opset_imports=[h.make_opsetid('', 20), h.make_opsetid('com.microsoft', 1)])
    onnx.save(q, p / 'qdq.onnx')
    report = splice(p / 'base.onnx', p / 'qdq.onnx', p / 'out')
    m = onnx.load(p / 'out/mixed-first-projection.onnx')
    assert [n.SerializeToString() for n in m.graph.node[-2:]] == [n.SerializeToString() for n in model.graph.node[-2:]]
    assert m.graph.node[7].op_type == 'Cast' and list(m.graph.node[7].output) == ['val_646']
    assert 'val_645' not in {t.name for t in m.graph.initializer}
    assert report['metadata_completion']['missing_referenced_names'] == []
    assert report['metadata_completion']['sdk_up_convert_infer_shapes_would_skip_inference']
    assert report['checker_passed'] and (p / 'out/mixed-first-projection.data').exists()
    return True


def fixture(p, count, bad_value=False, bad_consumer=False):
    tensors, nodes, chunks = {}, {}, []
    with tarfile.open(p / 'input.bin', 'w') as archive:
        for i in range(count):
            a, w, b, y = [f'{s}{i}' for s in 'awby']
            for name, dtype in [(a, 534), (w, 534), (b, 562), (y, 534)]:
                tensors[name] = {'data_type': dtype, 'dims': [2]}
            nodes[f'fc{i}'] = {'type': 'FullyConnected', 'input_names': [a, w, b], 'output_names': [y]}
            values = np.array([.1 if bad_value else .5, -0.0], '<f4').tobytes()
            member = tarfile.TarInfo(b + '.raw'); member.size = len(values)
            archive.addfile(member, io.BytesIO(values))
            chunks.append(f'static ModelError_t addTensor_{b}(QnnModel& model){{\n QNN_DATATYPE_FLOAT_32;\n}}\n')
    if bad_consumer:
        nodes['bad'] = {'type': 'Add', 'input_names': ['b0', 'a0'], 'output_names': ['y0']}
    (p / 'input_net.json').write_text(json.dumps({'graph': {'nodes': nodes, 'tensors': tensors}}))
    (p / 'input.cpp').write_text('\n'.join(chunks))


def main():
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        p = root / 'splice'; p.mkdir(); results['synthetic_splice_checker_and_preservation'] = test_splice(p)
        for count in [0, 1, 2]:
            p = root / f'repair{count}'; p.mkdir(); fixture(p, count)
            r = repair(p / 'input_net.json', p / 'input.cpp', p / 'input.bin', p / 'fixed')
            assert r['distinct_bias_count'] == count
            with tarfile.open(p / 'fixed.bin') as archive:
                for member in archive:
                    v = np.frombuffer(archive.extractfile(member).read(), '<f2').astype('<f4')
                    assert np.array_equal(v.view('<u4'), np.array([.5, -0.0], '<f4').view('<u4'))
            results[f'repair_count_{count}'] = True
        for label, kwargs in [('nonrepresentable', {'bad_value': True}), ('shared_consumer', {'bad_consumer': True})]:
            p = root / label; p.mkdir(); fixture(p, 1, **kwargs)
            try:
                repair(p / 'input_net.json', p / 'input.cpp', p / 'input.bin', p / 'fixed')
            except ValueError:
                assert not list(p.glob('fixed*')); results[f'reject_{label}_before_writes'] = True
            else:
                raise AssertionError(label)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
