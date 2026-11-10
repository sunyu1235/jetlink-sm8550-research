import copy
import os
from pathlib import Path
import unittest
import onnx
from onnx import helper as h
from build_model import replace, select_cut

SOURCE=Path(os.environ.get('JETLINK_SOURCE_MODEL','/workspace/shared/jetlink-distillation/real-block/teacher-compact-boundaries.onnx'))

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=onnx.load(SOURCE,load_external_data=False)

    def test_exact_remainder(self):
        result, selected=replace(self.model)
        self.assertEqual(len(selected),421)
        expected=[n.SerializeToString() for i,n in enumerate(self.model.graph.node) if i not in selected]
        self.assertEqual(expected,[n.SerializeToString() for n in result.graph.node if n.domain!='jetlink.qnn'])
        self.assertEqual([x.SerializeToString() for x in self.model.graph.output],[x.SerializeToString() for x in result.graph.output])
        self.assertEqual([x.SerializeToString() for x in self.model.graph.input],[x.SerializeToString() for x in result.graph.input])
        self.assertEqual([x.SerializeToString() for x in self.model.graph.initializer],[x.SerializeToString() for x in result.graph.initializer])
        self.assertEqual(sum('view_39' in n.input for n in result.graph.node),3)

    def test_reject_live_side_output(self):
        model=copy.deepcopy(self.model)
        selected=select_cut(model)
        side=next(v for i in selected for v in model.graph.node[i].output if v!='view_39')
        model.graph.node.append(h.make_node('Identity',[side],['unexpected_live_output']))
        with self.assertRaisesRegex(ValueError,'side outputs'):
            select_cut(model)

    def test_reject_unclosed_cut(self):
        model=copy.deepcopy(self.model)
        selected=select_cut(model)
        model.graph.node[min(selected)].input.append('missing_external_dependency')
        with self.assertRaisesRegex(ValueError,'Unclosed'):
            select_cut(model)

if __name__=='__main__':unittest.main()
