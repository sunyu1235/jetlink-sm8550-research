import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from exact_cse import rewrite, copy_external_files, validate_destinations
from smoke_one_frame import smoke_passes


def value(name, shape=(2, 3)):
    return h.make_tensor_value_info(name, T.FLOAT, list(shape))


def model(nodes, inputs, outputs, weights=()):
    result = h.make_model(h.make_graph(nodes, 'cse_test', inputs, outputs, weights),
                          opset_imports=[h.make_opsetid('', 20)])
    result.ir_version = 9
    return result


class ExactCSETest(unittest.TestCase):
    def test_finite_but_different_smoke_fails(self):
        self.assertFalse(smoke_passes({'outputs': {'finite': True, 'bit_identical': False}}))
        self.assertTrue(smoke_passes({'outputs': {'finite': True, 'bit_identical': True}}))

    def test_artifact_path_collisions_rejected(self):
        m = model([], [], [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                validate_destinations(m, root, root/'same', root/'same')
            t = nh.from_array(np.ones((2, 3), dtype=np.float32), 'w')
            onnx.external_data_helper.set_external_data(t, location='weights.bin')
            m.graph.initializer.append(t)
            with self.assertRaisesRegex(ValueError, 'collides'):
                validate_destinations(m, root, root/'model.onnx', root/'weights.bin')

    def test_external_constant_attribute_rejected(self):
        t = nh.from_array(np.ones((2, 3), dtype=np.float32), 'w')
        onnx.external_data_helper.set_external_data(t, location='constant.bin')
        m = model([h.make_node('Constant', [], ['y'], value=t)], [], [value('y')])
        with self.assertRaisesRegex(ValueError, 'External tensor attributes'):
            rewrite(m, '.')

    def test_duplicates_keep_outputs_bitwise_and_state(self):
        m = model([h.make_node('Add', ['x', 'w1'], ['a']), h.make_node('Add', ['x', 'w2'], ['b']),
                   h.make_node('Mul', ['a', 'a'], ['y']), h.make_node('Mul', ['b', 'b'], ['next_state'])],
                  [value('x')], [value('y'), value('next_state')],
                  [nh.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), 'w1'),
                   nh.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), 'w2')])
        candidate, report = rewrite(m, '.')
        self.assertEqual(report['removed_operations'], {'Add': 1, 'Mul': 1})
        self.assertEqual([x.name for x in candidate.graph.output], ['y', 'next_state'])
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        a = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
        b = ort.InferenceSession(candidate.SerializeToString(), so, providers=['CPUExecutionProvider'])
        x = np.random.default_rng(4).standard_normal((2, 3)).astype(np.float32)
        for i, j in zip(a.run(None, {'x': x}), b.run(None, {'x': x})):
            self.assertEqual(i.tobytes(), j.tobytes())
        onnx.checker.check_model(candidate)

    def test_runtime_initializer_is_not_constant(self):
        m = model([h.make_node('Add', ['x', 'a'], ['y']), h.make_node('Add', ['x', 'b'], ['z'])],
                  [value('x'), value('a')], [value('y'), value('z')],
                  [nh.from_array(np.zeros((2, 3), dtype=np.float32), 'a'), nh.from_array(np.zeros((2, 3), dtype=np.float32), 'b')])
        _, r = rewrite(m, '.')
        self.assertEqual(r['aliases'], {})

    def test_distinct_recurrent_inputs_are_not_deduplicated(self):
        m = model([h.make_node('Tanh', ['state1'], ['next1']), h.make_node('Tanh', ['state2'], ['next2'])],
                  [value('state1'), value('state2')], [value('next1'), value('next2')])
        _, r = rewrite(m, '.')
        self.assertEqual(r['removed_operations'], {})

    def test_no_commutation_or_reassociation(self):
        m = model([h.make_node('Add', ['x', 'z'], ['a']), h.make_node('Add', ['z', 'x'], ['b'])],
                  [value('x'), value('z')], [value('a'), value('b')])
        _, r = rewrite(m, '.')
        self.assertEqual(r['removed_operations'], {})

    def test_custom_is_barrier_without_explicit_opt_in(self):
        m = model([h.make_node('ExactGelu', ['x'], ['a'], domain='jetlink.precision'),
                   h.make_node('ExactGelu', ['x'], ['b'], domain='jetlink.precision')],
                  [value('x')], [value('a'), value('b')])
        m.opset_import.append(h.make_opsetid('jetlink.precision', 1))
        _, r = rewrite(m, '.')
        self.assertEqual(r['removed_operations'], {})
        c, r = rewrite(m, '.', True)
        self.assertEqual(r['removed_operations'], {'ExactGelu': 1})
        onnx.checker.check_model(c)

    def test_different_gelu_formula_never_merged(self):
        m = model([h.make_node('Gelu', ['x'], ['a'], approximate='none'),
                   h.make_node('Gelu', ['x'], ['b'], approximate='tanh')],
                  [value('x')], [value('a'), value('b')])
        _, r = rewrite(m, '.')
        self.assertEqual(r['removed_operations'], {})

    def test_stochastic_and_unknown_are_barriers(self):
        m = model([h.make_node('RandomNormal', [], ['a'], shape=[2, 3]),
                   h.make_node('RandomNormal', [], ['b'], shape=[2, 3])], [], [value('a'), value('b')])
        _, r = rewrite(m, '.')
        self.assertEqual(len(r['unknown_operation_barriers']), 2)
        self.assertEqual(r['removed_operations'], {})

    def test_multioutput_is_atomic(self):
        m = model([h.make_node('Split', ['x'], ['a', 'b'], axis=0, num_outputs=2),
                   h.make_node('Split', ['x'], ['c', 'd'], axis=0, num_outputs=2)],
                  [value('x')], [value(x, (1, 3)) for x in ['a', 'b', 'c', 'd']])
        c, r = rewrite(m, '.')
        self.assertEqual(r['removed_operations'], {'Split': 1})
        self.assertEqual(r['aliases'], {'c': 'a', 'd': 'b'})
        onnx.checker.check_model(c)

    def test_cache_rejected(self):
        m = model([h.make_node('EPContext', ['x'], ['y'], domain='com.microsoft')], [value('x')], [value('y')])
        with self.assertRaisesRegex(ValueError, 'opaque'):
            rewrite(m, '.')

    def test_external_weights_hash_contents_and_keep_offsets(self):
        m = model([h.make_node('Add', ['x', 'a'], ['y']), h.make_node('Add', ['x', 'b'], ['z'])],
                  [value('x')], [value('y'), value('z')],
                  [nh.from_array(np.ones((2, 3), dtype=np.float32), 'a'), nh.from_array(np.ones((2, 3), dtype=np.float32), 'b')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'out').mkdir()
            onnx.save_model(m, root/'model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='weights.bin', size_threshold=0)
            loaded = onnx.load(root/'model.onnx', load_external_data=False)
            c, r = rewrite(loaded, root)
            self.assertEqual(r['aliases']['b'], 'a')
            copy_external_files(c, root, root/'out')
            onnx.save(c, root/'out/model.onnx')
            onnx.checker.check_model(str(root/'out/model.onnx'))
            self.assertEqual((root/'weights.bin').read_bytes(), (root/'out/weights.bin').read_bytes())

    def test_no_tolerance_for_weight_identity(self):
        a = np.ones((2, 3), dtype=np.float32); b = a.copy(); b[0, 0] = np.nextafter(b[0, 0], np.float32(2))
        m = model([h.make_node('Add', ['x', 'a'], ['y']), h.make_node('Add', ['x', 'b'], ['z'])],
                  [value('x')], [value('y'), value('z')], [nh.from_array(a, 'a'), nh.from_array(b, 'b')])
        _, r = rewrite(m, '.')
        self.assertEqual(r['aliases'], {})


if __name__ == '__main__':
    unittest.main()
