"""Extract the selected six-node QDQ projection for independent QNN compilation."""
import argparse,copy
from pathlib import Path
import onnx
from onnx import helper,TensorProto
p=argparse.ArgumentParser();p.add_argument('--block',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
m=onnx.load(a.block);nodes=[]
for n in m.graph.node:
 nodes.append(copy.deepcopy(n))
 if list(n.output)==['val_646_DequantizeLinear_Output']:break
if len(nodes)!=6 or nodes[-1].op_type!='DequantizeLinear':raise ValueError('unexpected QDQ pattern')
used={s for n in nodes for s in n.input};g=helper.make_graph(nodes,'first_projection',[helper.make_tensor_value_info('layer_norm_49',TensorProto.FLOAT,[1,288,512])],[helper.make_tensor_value_info('val_646_DequantizeLinear_Output',TensorProto.FLOAT,[1,288,1024])],[copy.deepcopy(t) for t in m.graph.initializer if t.name in used]);out=helper.make_model(g,opset_imports=m.opset_import);out.ir_version=m.ir_version;onnx.checker.check_model(out);onnx.save(out,a.out)
