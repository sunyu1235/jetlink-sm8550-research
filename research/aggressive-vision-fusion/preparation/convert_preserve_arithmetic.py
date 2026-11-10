"""Experimental QNN2.42 wrapper: retain separate elementwise arithmetic after NN ops.
SDK bytes are not modified. This prevents FC weight/bias folding across FP16 rounding.
The hook is version-specific and does not itself establish full-model accuracy.
"""
import os,runpy,sys
from qti.aisw.converters.common.converter_ir import op_graph_optimizations as opt
assert hasattr(opt,'squash_node_into_nn_node')
def preserve_arithmetic(graph,matched_node_list):
 print('JETLINK_RETAIN_SEPARATE_ARITHMETIC',len(matched_node_list))
opt.squash_node_into_nn_node=preserve_arithmetic
runpy.run_path(os.environ['QNN_SDK_ROOT']+'/bin/x86_64-linux-clang/qnn-onnx-converter',run_name='__main__')
