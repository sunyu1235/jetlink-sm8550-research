"""Exact FP16-domain GELU using standard ONNX operations on the accelerator.

Normalize finite FP16 magnitudes with five exact power-of-two steps, derive
their FP16 bit-pattern indices using integer arithmetic, then gather from the
same mathematical table as the accepted CPU custom operator. No fitted values.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def lookup_nodes(x, output, prefix, variant):
    nodes, initializers = [], []

    def const(name, value, dtype):
        key = prefix + name
        initializers.append(nh.from_array(np.asarray(value, dtype=dtype), key))
        return key

    def op(kind, inputs, suffix, **attributes):
        key = prefix + suffix
        nodes.append(h.make_node(kind, inputs, [key], name=key, **attributes))
        return key

    zero = const('zero', 0, np.float16)
    negzero = const('negative_zero', -0.0, np.float16)
    minimum = const('min_normal', 2**-14, np.float16)
    maximum = const('clip_magnitude', 8, np.float16)
    one = const('one', 1, np.float16)
    scale1024 = const('scale1024', 1024, np.float16)
    scale4096 = const('scale4096', 4096, np.float16)
    izero = const('int_zero', 0, np.int32)
    i1024 = const('int_1024', 1024, np.int32)
    signbit = const('sign_bit', 32768, np.int32)
    magnitude = op('Abs', [x], 'abs')
    clipped = op('Min', [magnitude, maximum], 'clipped')
    v = op('Max', [clipped, minimum], 'normal_magnitude')
    exponent = const('exponent_bias', 15, np.float16)
    # Each chosen multiplication is exact in FP16. The unchosen branch also
    # stays finite; normalizing never introduces a rounding or underflow.
    stages = [('Less', 2**-4, 256, -8), ('Less', 2**-2, 16, -4),
              ('Less', 1, 4, -2), ('GreaterOrEqual', 4, .25, 2),
              ('GreaterOrEqual', 2, .5, 1)]
    for j, (comparison, threshold, factor, delta) in enumerate(stages):
        limit = const(f'limit{j}', threshold, np.float16)
        scale = const(f'factor{j}', factor, np.float16)
        change = const(f'delta{j}', delta, np.float16)
        condition = op(comparison, [v, limit], f'condition{j}')
        scaled = op('Mul', [v, scale], f'scaled{j}')
        v = op('Where', [condition, scaled, v], f'normalized{j}')
        shift = op('Where', [condition, change, zero], f'shift{j}')
        exponent = op('Add', [exponent, shift], f'exponent{j}')
    mantissa = op('Cast', [op('Mul', [v, scale1024], 'mantissa_scaled')], 'mantissa', to=T.INT32)
    exponent = op('Cast', [op('Sub', [exponent, one], 'biased_exponent')], 'exponent_int', to=T.INT32)
    normal_index = op('Add', [mantissa, op('Mul', [exponent, i1024], 'exponent_bits')], 'normal_index')
    sub = op('Min', [clipped, minimum], 'subnormal_magnitude')
    sub = op('Mul', [op('Mul', [sub, scale4096], 'subnormal_scale1'), scale4096], 'subnormal_scale2')
    sub = op('Cast', [sub], 'subnormal_index', to=T.INT32)
    is_sub = op('Less', [clipped, minimum], 'is_subnormal')
    positive_index = op('Where', [is_sub, sub, normal_index], 'positive_index')
    is_negative = op('Less', [x, zero], 'is_negative')
    sign = op('Where', [is_negative, signbit, izero], 'sign_offset')
    index = op('Add', [positive_index, sign], 'index')
    bits = np.arange(65536, dtype=np.uint16)
    values = bits.view(np.float16)
    result = []
    for value in values:
        z = float(value)
        if not math.isfinite(z):
            result.append(float('nan'))
        elif variant == 'erf':
            result.append(.5 * z * (1 + math.erf(z / math.sqrt(2))))
        else:
            result.append(.5 * z * (1 + math.tanh(math.sqrt(2 / math.pi) * (z + .044715 * z**3))))
    table = const('table_' + variant, result, np.float16)
    looked_up = op('Gather', [table, index], 'lookup', axis=0)
    # For |x| >= 8 the FP16 result is x (positive) or -0 (negative).
    large = op('Where', [is_negative, negzero, x], 'large_result')
    small = op('Less', [magnitude, maximum], 'is_small')
    selected = op('Where', [small, looked_up, large], 'selected')
    # Preserve the sign of an actual zero without extracting its sign bit.
    is_zero = op('Equal', [x, zero], 'is_zero')
    nodes.append(h.make_node('Where', [is_zero, x, selected], [output], name=prefix + 'result'))
    return nodes, initializers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=('erf', 'tanh'), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bits = np.arange(65536, dtype=np.uint16)
    x = bits[(bits & 0x7c00) != 0x7c00].view(np.float16)
    n = len(x)
    nodes, constants = lookup_nodes('x', 'outputs', 'lut_', args.variant)
    graph = h.make_graph(nodes, 'native_fp16_lookup', [h.make_tensor_value_info('x', T.FLOAT16, [n])],
                         [h.make_tensor_value_info('outputs', T.FLOAT16, [n])], constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 20)], ir_version=10)
    onnx.checker.check_model(model)
    path = args.out / 'native-lut.onnx'
    onnx.save(model, path)
    inputs = args.out / 'inputs'; inputs.mkdir(exist_ok=True)
    x.tofile(inputs / '0_x.bin')
    table = nh.to_array(next(t for t in constants if t.name == 'lut_table_' + args.variant))
    ref = table[x.view(np.uint16)]
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'x': x})[0]
    mismatches = int(np.count_nonzero(actual.view(np.uint16) != ref.view(np.uint16)))
    different_bits = actual.view(np.uint16) != ref.view(np.uint16)
    zero_sign_only = bool(np.all((actual[different_bits] == 0) & (ref[different_bits] == 0)))
    np.save(args.out / 'reference.npy', ref)
    np.save(args.out / 'cpu-graph.npy', actual)
    summary = {'variant': args.variant, 'finite_inputs': n, 'nodes': len(nodes),
               'cpu_value_mismatches': int(np.count_nonzero(actual != ref)),
               'cpu_bit_mismatches': mismatches, 'model_bytes': path.stat().st_size,
               'bit_mismatches_are_only_zero_sign': zero_sign_only,
               'purpose': 'standard-op NPU candidate; not yet a phone or model parity result'}
    (args.out / 'cpu-proof.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if np.any(actual != ref) or not zero_sign_only:
        raise SystemExit('CPU graph fails finite-domain numerical equivalence')


if __name__ == '__main__':
    main()
