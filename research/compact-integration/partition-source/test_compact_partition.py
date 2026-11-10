"""Actual-source structural regression tests; no complete-model execution."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper
from prepare_compact_partition import (
    DEFAULT_MANIFEST, DEFAULT_SOURCE, INPUT_NAMES, OUTPUT_NAMES,
    checked, compact_partition, extract_baseline, info, integration_cut_proof,
)


class CompactPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(DEFAULT_MANIFEST.read_text())
        cls.baseline, cls.provenance = extract_baseline(DEFAULT_SOURCE, cls.contract)
        cls.compact, cls.proof = compact_partition(cls.baseline, cls.contract)

    def test_exact_external_contract(self):
        self.assertEqual([info(v) for v in self.baseline.graph.input], self.contract['inputs'])
        self.assertEqual([info(v) for v in self.baseline.graph.output], self.contract['outputs'])
        self.assertEqual([info(v) for v in self.compact.graph.output], self.contract['compact_outputs'])
        self.assertEqual(list(self.proof['output_aliases_compact_to_downstream'].values()), OUTPUT_NAMES)

    def test_concrete_shapes_no_restoration(self):
        checked(self.compact)
        self.assertFalse(any(n.op_type == 'Concat' for n in self.compact.graph.node))
        self.assertFalse(any('zero_prefix' in t.name for t in self.compact.graph.initializer))
        self.assertEqual(sum(n.op_type == 'Slice' for n in self.compact.graph.node), 6)

    def test_unchanged_kv_and_weights(self):
        self.assertEqual(len(self.proof['kv_proofs']), 3)
        self.assertEqual(len(self.proof['attention_matmul_proofs']), 3)
        weights = {t.name: t for t in self.baseline.graph.initializer}
        for tensor in self.compact.graph.initializer:
            if tensor.name in weights:
                self.assertEqual(tensor.SerializeToString(), weights[tensor.name].SerializeToString())
        self.assertEqual(len(self.provenance), len(self.baseline.graph.initializer))
        self.assertTrue(all(len(entry['raw_sha256']) == 64 for entry in self.provenance))

    def test_cut_has_only_six_declared_consumer_edges(self):
        proof = integration_cut_proof(DEFAULT_SOURCE, self.baseline)
        self.assertEqual(proof['undeclared_crossings'], [])
        self.assertEqual(proof['selected_source_graph_outputs'], [])
        self.assertEqual(len(proof['external_consumers']), 6)
        self.assertEqual(len(proof['removed_node_names']), 87)

    def test_reject_contract_input_reordering(self):
        contract = copy.deepcopy(self.contract)
        contract['inputs'][0], contract['inputs'][1] = contract['inputs'][1], contract['inputs'][0]
        with self.assertRaisesRegex(ValueError, 'input names/order'):
            extract_baseline(DEFAULT_SOURCE, contract)

    def test_reject_contract_shape_drift(self):
        contract = copy.deepcopy(self.contract)
        contract['outputs'][0]['shape'][1] = 287
        with self.assertRaisesRegex(ValueError, 'boundary disagrees'):
            extract_baseline(DEFAULT_SOURCE, contract)

    def test_reject_masked_last_query(self):
        baseline = copy.deepcopy(self.baseline)
        mask_name = self.proof['mask_proofs'][0]['mask']
        tensor = next(t for t in baseline.graph.initializer if t.name == mask_name)
        data = numpy_helper.to_array(tensor).copy()
        data[..., 287, 0] = False
        tensor.CopyFrom(numpy_helper.from_array(data, mask_name))
        with self.assertRaisesRegex(ValueError, 'not fully unmasked'):
            compact_partition(baseline, self.contract)

    def test_reject_extra_query_consumer(self):
        baseline = copy.deepcopy(self.baseline)
        baseline.graph.node.append(helper.make_node('Identity', ['squeeze_9'], ['test_extra_query'], name='test_extra_query'))
        with self.assertRaisesRegex(ValueError, 'one raw-teacher Q'):
            compact_partition(baseline, self.contract)

    def test_reject_nonlocal_query_normalization(self):
        baseline = copy.deepcopy(self.baseline)
        node = next(n for n in baseline.graph.node if n.name == 'node_layer_norm_59')
        next(a for a in node.attribute if a.name == 'axis').i = 2
        with self.assertRaisesRegex(ValueError, 'not independently token-local'):
            compact_partition(baseline, self.contract)

    def test_deterministic_rewrite(self):
        second, proof = compact_partition(self.baseline, self.contract)
        self.assertEqual(second.SerializeToString(), self.compact.SerializeToString())
        self.assertEqual(proof, self.proof)


if __name__ == '__main__':
    unittest.main(verbosity=2)
