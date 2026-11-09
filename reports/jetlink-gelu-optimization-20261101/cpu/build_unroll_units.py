"""Complete finite-domain lookup checks plus a non-multiple-of-eight tail."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T

root = Path(__file__).resolve().parents[3]
out = root / 'outputs/jetlink-gelu-optimization-20261101/cpu/units'
out.mkdir(parents=True, exist_ok=True)
bits = np.arange(65536, dtype=np.uint16)
bits = bits[(bits & 0x7c00) != 0x7c00]
x = np.concatenate([bits.view(np.float16), np.array([-0.0, -2**-24, 65504], dtype=np.float16)])
(out / 'inputs').mkdir(exist_ok=True)
x.tofile(out / 'inputs/0_x.bin')
for variant, name, base in [('erf', 'ExactGelu', 'finite-domain'), ('tanh', 'ExactGeluTanh', 'finite-domain-tanh')]:
    reference = np.load(root / f'outputs/jetlink-gelu-repair-20261101/{base}/ref-double-rounded-fp16.npy')
    # Every tail value already appears in the complete finite input domain.
    lookup = {int(b): reference[i] for i, b in enumerate(bits)}
    tail = np.array([lookup[int(b)] for b in x[-3:].view(np.uint16)], dtype=np.float16)
    np.save(out / f'reference-{variant}.npy', np.concatenate([reference, tail]))
    g = h.make_graph([h.make_node(name, ['x'], ['outputs'], domain='jetlink.precision')],
                     f'unroll_{variant}_tail', [h.make_tensor_value_info('x', T.FLOAT16, [len(x)])],
                     [h.make_tensor_value_info('outputs', T.FLOAT16, [len(x)])])
    model = h.make_model(g, opset_imports=[h.make_opsetid('', 20), h.make_opsetid('jetlink.precision', 1)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, out / f'unit-{variant}.onnx')
print(json.dumps({'finite_patterns': len(bits), 'tail_elements': 3, 'total_elements': len(x), 'out': str(out)}))
