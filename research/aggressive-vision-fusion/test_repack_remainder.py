import tempfile
import unittest
from pathlib import Path
import onnx
from onnx import helper, TensorProto as T
from repack_remainder import repack

class RepackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.src=self.root/'source';self.src.mkdir();self.out=self.root/'out'
        self.path=self.src/'model.onnx'
        (self.src/'data.bin').write_bytes(b'\x00\x00\x80?'+b'\x00\x00\x00@'+b'\x00\x00@@')
        self.model=helper.make_model(helper.make_graph([helper.make_node('Add',['x','live'],['y'])],'g',
            [helper.make_tensor_value_info('x',T.FLOAT,[1])],[helper.make_tensor_value_info('y',T.FLOAT,[1])]),opset_imports=[helper.make_opsetid('',20)])
        for name, offset in [('dead',0),('live',4),('override',8)]:
            t=self.model.graph.initializer.add();t.name=name;t.data_type=T.FLOAT;t.dims.append(1);t.data_location=T.EXTERNAL
            for k,v in [('location','data.bin'),('offset',str(offset)),('length','4')]:
                e=t.external_data.add();e.key=k;e.value=v
        self.model.graph.input.append(helper.make_tensor_value_info('override',T.FLOAT,[1]))
    def run_repack(self, **kw):
        self.path.write_bytes(self.model.SerializeToString());return repack(self.path,self.out,**kw)
    def test_prune_and_preserve_overridable(self):
        r=self.run_repack();self.assertEqual((r['kept_initializers'],r['removed_initializers']),(2,1));self.assertEqual((self.out/'remainder.weights').read_bytes(),(self.src/'data.bin').read_bytes()[4:]);self.assertTrue(r['all_live_tensor_hashes_equal'])
    def test_output_is_preserved(self):
        self.out.mkdir();(self.out/'marker').write_text('safe')
        with self.assertRaises(FileExistsError):self.run_repack()
        self.assertEqual((self.out/'marker').read_text(),'safe')
    def test_unsafe_paths(self):
        for location in ['../data.bin','/tmp/data.bin','a\\b','http://host/file','./data.bin']:
            with self.subTest(location=location):
                self.model.graph.initializer[0].external_data[0].value=location
                with self.assertRaises(ValueError):self.run_repack()
    def test_bad_lengths(self):
        self.model.graph.initializer[0].external_data[2].value='3'
        with self.assertRaises(ValueError):self.run_repack()
    def test_truncated_source(self):
        (self.src/'data.bin').write_bytes(b'xx')
        with self.assertRaises(ValueError):self.run_repack()
    def test_negative_offset(self):
        self.model.graph.initializer[0].external_data[1].value='-1'
        with self.assertRaises(ValueError):self.run_repack()
    def test_duplicate_metadata(self):
        t=self.model.graph.initializer[0];t.external_data.append(t.external_data[0])
        with self.assertRaises(ValueError):self.run_repack()
    def test_subgraph(self):
        self.model.graph.node[0].attribute.append(helper.make_attribute('body',helper.make_graph([],'sub',[],[])))
        with self.assertRaises(ValueError):self.run_repack()
    def test_function(self):
        self.model.functions.append(helper.make_function('local','f',[],[],[],[helper.make_opsetid('',20)]))
        with self.assertRaises(ValueError):self.run_repack()
    def identity_function(self):
        return helper.make_function('org.tinygrad','Contiguous',['input'],['output'],[helper.make_node('Identity',['input'],['output'])],[helper.make_opsetid('',20)])
    def test_exact_identity_function(self):
        f=self.identity_function();self.model.functions.append(f)
        r=self.run_repack();m=onnx.load(self.out/'vision-native.onnx',load_external_data=False)
        self.assertEqual(f.SerializeToString(),m.functions[0].SerializeToString());self.assertEqual(r['preserved_functions'][0]['root_call_count'],0)
    def test_identity_function_invalid_variants(self):
        for variant in ['name','op','attribute','mapping','opset']:
            with self.subTest(variant=variant):
                f=self.identity_function()
                if variant=='name':f.name='Other'
                if variant=='op':f.node[0].op_type='Relu'
                if variant=='attribute':f.node[0].attribute.append(helper.make_attribute('x',1))
                if variant=='mapping':f.node[0].input[0]='other'
                if variant=='opset':f.opset_import[0].version=19
                del self.model.functions[:];self.model.functions.append(f)
                with self.assertRaises(ValueError):self.run_repack()
    def test_training(self):
        self.model.training_info.add()
        with self.assertRaises(ValueError):self.run_repack()
    def test_sparse(self):
        self.model.graph.sparse_initializer.add()
        with self.assertRaises(ValueError):self.run_repack()
    def test_symlink_escape_requires_allowlist(self):
        target=self.root/'outside.bin';(self.src/'data.bin').rename(target);(self.src/'data.bin').symlink_to(target)
        with self.assertRaises(ValueError):self.run_repack()
        self.assertTrue(self.run_repack(allow_external_files=[target])['all_live_tensor_hashes_equal'])
    def test_embedded_tensor(self):
        self.model.graph.initializer[1].CopyFrom(helper.make_tensor('live',T.FLOAT,[1],[2.0]))
        r=self.run_repack();self.assertFalse(r['tensors'][0]['external'])
    def test_graph_output_initializer(self):
        self.model.graph.output.append(helper.make_tensor_value_info('dead',T.FLOAT,[1]))
        self.assertEqual(self.run_repack()['kept_initializers'],3)
    def test_duplicate_initializer(self):
        self.model.graph.initializer[0].name='live'
        with self.assertRaises(ValueError):self.run_repack()
    def test_mixed_payload(self):
        self.model.graph.initializer[0].raw_data=b'1234'
        with self.assertRaises(ValueError):self.run_repack()
    def test_complete_size(self):
        c=self.root/'context.bin';c.write_bytes(b'12345');r=self.run_repack(context=c);s=r['sizes']
        self.assertEqual(s['complete_model_data_bytes'],s['onnx_bytes']+s['weights_bytes']+5)

if __name__=='__main__':unittest.main()
