"""Focused structural regression tests; closed-cut and boundary-contract checks.
Run: python -m unittest -v test_extract_region.py
No SDK, model weights, ONNX runtime, or long-running processes required.
"""
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from extract_region import select_region, extract


def tensor(kind=3, dims=None):
    return dict(type=kind, data_type=534, dims=[1, 2] if dims is None else dims,
                is_dynamic_dims=[], dataFormat=0)


def node(inputs, outputs):
    return dict(type='Identity', input_names=inputs, output_names=outputs)


def graph():
    return dict(tensors={'x': tensor(0), 'w': tensor(4), 'a': tensor(),
                         'y': tensor(), 'z': tensor()},
                nodes={'first': node(['x', 'w'], ['a']),
                       'second': node(['a'], ['y'])})


def cpp_descriptor(name, desc):
    kinds = {0: 'APP_WRITE', 1: 'APP_READ', 2: 'APP_READWRITE', 3: 'NATIVE', 4: 'STATIC'}
    return (' uint32_t dimensions_' + name + '[] = {' + ', '.join(map(str, desc['dims'])) + '};\n'
            '.name= "' + name + '", .type= QNN_TENSOR_TYPE_' + kinds[desc['type']] + ',\n'
            '.dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE, .dataType= QNN_DATATYPE_FLOAT_16,\n'
            '.rank= ' + str(len(desc['dims'])) + ', .dimensions=dimensions_' + name + ',\n'
            '.isDynamicDimensions= nullptr, .isProduced= 0\n')


def fixture(root, g):
    """Minimal lexical fixture for the generated-C++ extraction contract."""
    root.mkdir()
    (root/'preserve_net.json').write_text(json.dumps({'graph': g}))
    pieces = ['// fixture\n', 'static ModelError_t addTensor_new_img(QnnModel& model){\n',
              cpp_descriptor('new_img', tensor(0, [1, 1, 1, 1])), '}\n',
              'static ModelError_t addTensor_x(QnnModel& model){\n',
              cpp_descriptor('x', g['tensors']['x']), '}\n',
              'static ModelError_t addTensor_w(QnnModel& model){\n // no payload\n}\n']
    for name, n in g['nodes'].items():
        pieces.append('static ModelError_t addNode_'+name+'(QnnModel& model){\n')
        for output in n['output_names']:
            pieces.append(cpp_descriptor(output, g['tensors'][output]))
        pieces.append('}\n')
    pieces.append('QNN_API\nModelError_t QnnModel_composeGraphs(){\n'
                  '  VALIDATE(addTensor_new_img(preserve), err);\n'
                  '  // Add all models\n  // "preserve"\n}\n')
    (root/'preserve_fixed.cpp').write_text(''.join(pieces))
    with tarfile.open(root/'preserve_fixed.bin', 'w'):
        pass


class SelectionTests(unittest.TestCase):
    def test_closed_region_and_constants(self):
        selected, constants = select_region(graph(), ['x'], ['y'])
        self.assertEqual([n for n, _ in selected], ['first', 'second'])
        self.assertEqual(constants, ['w'])

    def test_does_not_mutate_graph(self):
        g = graph(); before = copy.deepcopy(g)
        select_region(g, ['x'], ['y'])
        self.assertEqual(g, before)

    def test_missing_external_dependency_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing input'):
            select_region(graph(), ['z'], ['y'])

    def test_unused_input_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unused cut'):
            select_region(graph(), ['x', 'z'], ['y'])

    def test_aliased_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'aliased'):
            select_region(graph(), ['x'], ['x'])

    def test_cycle_rejected(self):
        g = graph(); g['nodes']['first']['input_names'] = ['y']
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            select_region(g, [], ['y'])

    def test_non_topological_order_rejected(self):
        g = graph(); g['nodes'] = dict(reversed(list(g['nodes'].items())))
        with self.assertRaisesRegex(ValueError, 'Not topological'):
            select_region(g, ['x'], ['y'])

    def test_multiple_producers_rejected(self):
        g = graph(); g['nodes']['other'] = node(['x'], ['a'])
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            select_region(g, ['x'], ['y'])

    def test_missing_cut_input_descriptor_rejected(self):
        g = graph(); del g['tensors']['x']
        with self.assertRaises(ValueError):
            select_region(g, ['x'], ['y'])

    def test_external_side_output_requires_explicit_output(self):
        g = graph(); g['nodes']['outside'] = node(['a'], ['z'])
        with self.assertRaises(ValueError):
            select_region(g, ['x'], ['y'])

    def test_original_graph_output_cannot_be_silently_lost(self):
        g = graph(); g['tensors']['a']['type'] = 1  # APP_READ
        with self.assertRaises(ValueError):
            select_region(g, ['x'], ['y'])

    def test_declared_side_output_is_legal(self):
        g = graph(); g['nodes']['outside'] = node(['a'], ['z'])
        selected, _ = select_region(g, ['x'], ['a', 'y'])
        self.assertEqual([n for n, _ in selected], ['first', 'second'])

    def test_constant_output_is_not_a_computed_output(self):
        with self.assertRaises(ValueError):
            select_region(graph(), [], ['w'])


class ExtractionTests(unittest.TestCase):
    def run_extract(self, g, inputs=None, outputs=None, cpp_change=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root/'source', g)
            if cpp_change:
                cpp = root/'source'/'preserve_fixed.cpp'
                cpp.write_text(cpp_change(cpp.read_text()))
            result = extract(root/'source', root/'out', inputs or ['x'], outputs or ['y'], 'test')
            return result, (root/'out'/'region.cpp').read_text()

    def test_boundary_rank_shape_and_output_promotion(self):
        g = graph(); g['tensors']['x']['dims'] = [2, 3, 5]
        manifest, cpp = self.run_extract(g)
        self.assertIn('dimensions_x[] = {2, 3, 5}', cpp)
        self.assertIn('.rank= 3', cpp)
        self.assertIn('.name= "y", .type= QNN_TENSOR_TYPE_APP_READ', cpp)
        self.assertEqual(manifest['input_tensors']['x']['dims'], [2, 3, 5])

    def test_non_fp16_boundary_rejected(self):
        g = graph(); g['tensors']['x']['data_type'] = 562
        with self.assertRaisesRegex(ValueError, 'FP16'):
            self.run_extract(g)

    def test_dynamic_boundary_rejected(self):
        g = graph(); g['tensors']['x']['is_dynamic_dims'] = [1, 0]
        with self.assertRaises(ValueError):
            self.run_extract(g)

    def test_nonpositive_dimension_rejected(self):
        g = graph(); g['tensors']['x']['dims'] = [0, 2]
        with self.assertRaises(ValueError):
            self.run_extract(g)

    def test_non_dense_input_format_rejected(self):
        g = graph(); g['tensors']['x']['dataFormat'] = 1
        with self.assertRaises(ValueError):
            self.run_extract(g)

    def test_already_app_read_output_supported(self):
        g = graph(); g['tensors']['y']['type'] = 1
        _, cpp = self.run_extract(g)
        self.assertIn('.name= "y", .type= QNN_TENSOR_TYPE_APP_READ', cpp)

    def test_cpp_json_shape_disagreement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace(
                'dimensions_x[] = {1, 2}', 'dimensions_x[] = {2, 1}'))

    def test_cpp_output_shape_disagreement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace(
                'dimensions_y[] = {1, 2}', 'dimensions_y[] = {1, 3}'))

    def test_cpp_dtype_layout_rank_and_dynamic_disagreement_rejected(self):
        for old, new in [('QNN_DATATYPE_FLOAT_16', 'QNN_DATATYPE_FLOAT_32'),
                         ('QNN_TENSOR_DATA_FORMAT_DENSE', 'QNN_TENSOR_DATA_FORMAT_SPARSE'),
                         ('.rank= 2', '.rank= 3'),
                         ('.isDynamicDimensions= nullptr', '.isDynamicDimensions= dyn')]:
            with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'mismatch'):
                self.run_extract(graph(), cpp_change=lambda s: s.replace(old, new))

    def test_cpp_role_disagreement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'type'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace(
                '.name= "y", .type= QNN_TENSOR_TYPE_NATIVE',
                '.name= "y", .type= QNN_TENSOR_TYPE_APP_READ'))

    def test_missing_cpp_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing or ambiguous'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace('.name= "x"', '.name= "lost"'))

    def test_duplicate_cpp_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing or ambiguous'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace('.name= "a"', '.name= "x"'))

    def test_missing_cpp_dimension_array_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.run_extract(graph(), cpp_change=lambda s: s.replace('dimensions_x[]', 'dimensions_missing[]'))


if __name__ == '__main__':
    unittest.main()
