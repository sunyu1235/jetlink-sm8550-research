"""Test the mathematical lookup over every finite IEEE FP16 input pattern."""
from pathlib import Path
import argparse
import math
import numpy as np
import onnx
from onnx import helper, TensorProto

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=('erf', 'tanh'), default='erf')
args = p.parse_args()
out = Path(__file__).resolve().parents[2] / 'outputs' / 'jetlink-gelu-repair-20261101' / ('finite-domain' if args.variant == 'erf' else 'finite-domain-tanh')
out.mkdir(parents=True, exist_ok=True)
bits = np.arange(65536, dtype=np.uint16)
bits = bits[(bits & 0x7c00) != 0x7c00]
x = bits.view(np.float16)
(out / 'inputs').mkdir(exist_ok=True)
x.tofile(out / 'inputs' / '0_x.bin')
def reference(t):
    z = float(t)
    f = math.erf(z / math.sqrt(2.0)) if args.variant == 'erf' else math.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z**3))
    return 0.5 * z * (1.0 + f)
ref = np.array([reference(t) for t in x], dtype=np.float16)
np.save(out / 'ref-double-rounded-fp16.npy', ref)
g = helper.make_graph([helper.make_node('ExactGelu' if args.variant == 'erf' else 'ExactGeluTanh', ['x'], ['outputs'], domain='jetlink.precision')],
                      'finite-fp16-domain', [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [len(x)])],
                      [helper.make_tensor_value_info('outputs', TensorProto.FLOAT16, [len(x)])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 20), helper.make_opsetid('jetlink.precision', 1)])
m.ir_version = 10
onnx.checker.check_model(m)
onnx.save(m, out / 'exact-gelu-domain.onnx')
print(f'{len(x)} finite input patterns; all model inputs and references saved to {out}')
