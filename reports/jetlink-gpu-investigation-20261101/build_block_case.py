"""Extract one real ConvNeXt block, with identical input for every backend.

This is an operator diagnostic, not an accuracy test for the driving model.
The source model, layer-norm epsilon, weights and arithmetic are not tuned.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort


def promote(model):
    for t in model.graph.initializer:
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    for n in model.graph.node:
        for a in n.attribute:
            if n.op_type == 'Cast' and a.name == 'to' and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.source)
    start, end = 'add_34', 'add_35'
    extracted = onnx.utils.Extractor(model).extract_model([start], [end])
    # Capture a real intermediate of frame zero, with the image history reset.
    vi = {v.name: v for v in model.graph.value_info}
    model.graph.output.append(copy.deepcopy(vi[start]))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    feed = {}
    for item in session.get_inputs():
        shape = tuple(item.shape)
        dtype = np.float16 if item.type == 'tensor(float16)' else np.float32
        if item.name.startswith('state_'):
            feed[item.name] = np.zeros(shape, dtype=dtype)
        else:
            raw = args.frames / f'0_{item.name}.bin'
            feed[item.name] = np.fromfile(raw, dtype=dtype).reshape(shape)
    activation = session.run([start], feed)[0]
    del session, model
    # Stable output name lets the existing runner dump this one result.
    for node in extracted.graph.node:
        for k, name in enumerate(node.output):
            if name == end:
                node.output[k] = 'outputs'
    extracted.graph.output[0].name = 'outputs'
    manifest = {'source': str(args.source.resolve()), 'boundary': [start, end],
                'input_capture': 'synthetic frame 0, independent zero image history',
                'purpose': 'same-input backend numerical and graph-finalization diagnostic',
                'nodes': [n.op_type for n in extracted.graph.node], 'variants': {}}
    for precision in ('fp16', 'fp32'):
        variant = copy.deepcopy(extracted)
        if precision == 'fp32':
            promote(variant)
        onnx.checker.check_model(variant)
        path = args.out / f'block-{precision}.onnx'
        onnx.save(variant, path)
        inp = activation.astype(np.float16 if precision == 'fp16' else np.float32)
        inputs = args.out / f'input-{precision}'
        inputs.mkdir(exist_ok=True)
        inp.tofile(inputs / f'0_{start}.bin')
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        ref = session.run(None, {start: inp})[0]
        np.save(args.out / f'ref-{precision}.npy', ref)
        manifest['variants'][precision] = {'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'model_bytes': path.stat().st_size, 'input_shape': list(inp.shape),
            'input_dtype': str(inp.dtype), 'input_sha256': hashlib.sha256(inp.tobytes()).hexdigest(),
            'output_shape': list(ref.shape), 'output_dtype': str(ref.dtype)}
        del session
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
