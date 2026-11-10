"""Create a teacher-only vision reference and a matched recorded-road input.

The cut is the existing uint8-to-FP16 cast; all source arithmetic is retained.
These are one-frame boundary diagnostics, not autonomous recurrent acceptance.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T
from extract_region import sha


def submodel(source, inputs, outputs):
    model = copy.deepcopy(source)
    producers = {}
    for i, node in enumerate(model.graph.node):
        for output in node.output:
            if output in producers:
                raise ValueError('Duplicate source producer: ' + output)
            producers[output] = i
    initializers = {t.name: t for t in model.graph.initializer}
    selected, constants, visiting, used_inputs = set(), set(), set(), set()
    def visit(name):
        if name in inputs:
            used_inputs.add(name)
            return
        if name in initializers:
            constants.add(name)
            return
        if name not in producers:
            raise ValueError('Unclosed source cut: ' + name)
        i = producers[name]
        if i in selected:
            return
        if i in visiting:
            raise ValueError('Cycle in source graph')
        visiting.add(i)
        for value in model.graph.node[i].input:
            if value:
                visit(value)
        visiting.remove(i)
        selected.add(i)
    for output in outputs:
        visit(output)
    if set(inputs) != used_inputs:
        raise ValueError('Unused supplied input: ' + repr(sorted(set(inputs) - used_inputs)))
    nodes = [n for i, n in enumerate(model.graph.node) if i in selected]
    ready = set(inputs) | constants
    for node in nodes:
        if any(value and value not in ready for value in node.input):
            raise ValueError('Source graph is not topological')
        ready.update(node.output)
    inits = [t for t in model.graph.initializer if t.name in constants]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    del model.graph.input[:]
    model.graph.input.extend(inputs.values())
    del model.graph.output[:]
    model.graph.output.extend(outputs.values())
    del model.graph.value_info[:]
    return model


def options():
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--road', type=Path, required=True)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = onnx.load(a.source, load_external_data=False)
    casts = [n for n in source.graph.node if '_to_copy_4' in n.output]
    if len(casts) != 1 or casts[0].op_type != 'Cast' or list(casts[0].input) != ['cat_2']:
        raise ValueError('Expected original image FP16 cast was not found')
    attrs = {v.name: h.get_attribute_value(v) for v in casts[0].attribute}
    if attrs.get('to') != T.FLOAT16:
        raise ValueError('Original image cast does not target FP16')
    a.output.mkdir(parents=True)
    locations = {e.value for t in source.graph.initializer for e in t.external_data if e.key == 'location'}
    for loc in locations:
        if Path(loc).is_absolute() or '..' in Path(loc).parts:
            raise ValueError('Unsafe external initializer location')
        target = a.source.parent / loc
        if not target.is_file():
            raise FileNotFoundError(target)
        (a.output / loc).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, a.output / loc)
    source_inputs = {v.name: v for v in source.graph.input}
    pre = submodel(source, {n: source_inputs[n] for n in ['new_img', 'state_img_q']},
        {'cat_2': h.make_tensor_value_info('cat_2', T.UINT8, [1,24,128,256])})
    vision = submodel(source,
        {'_to_copy_4': h.make_tensor_value_info('_to_copy_4', T.FLOAT16, [1,24,128,256])},
        {'view_39': h.make_tensor_value_info('view_39', T.FLOAT16, [1,32,512])})
    for name, m in [('pre', pre), ('teacher_vision', vision)]:
        onnx.save(m, a.output / (name + '.onnx'))
        onnx.checker.check_model(str(a.output / (name + '.onnx')))
    new = np.load(a.road / f'in_warped_{a.frame}.npy').astype(np.uint8)
    state = np.zeros((2,5,6,128,256), np.uint8)
    # Exact recent image queue; no teacher feature state is injected into a candidate.
    input_paths = [a.road / f'in_warped_{a.frame}.npy']
    for offset in range(1,6):
        frame = a.frame - offset
        if frame >= 0:
            path = a.road / f'in_warped_{frame}.npy'
            input_paths.append(path)
            state[:,5-offset] = np.load(path)
    sess = ort.InferenceSession(str(a.output/'pre.onnx'), options(), providers=['CPUExecutionProvider'])
    cat = sess.run(None, {'new_img': new, 'state_img_q': state})[0].astype('<f2')
    assert cat.shape == (1,24,128,256) and np.isfinite(cat).all()
    cat.tofile(a.output/'cat_2.raw')
    (a.output/'inputs.txt').write_text('cat_2:='+str((a.output/'cat_2.raw').resolve())+'\n')
    sess = ort.InferenceSession(str(a.output/'teacher_vision.onnx'), options(), providers=['CPUExecutionProvider'])
    result = sess.run(None, {'_to_copy_4':cat})[0]
    assert result.shape == (1,32,512) and np.isfinite(result).all()
    np.save(a.output/'teacher.npy', result)
    (a.output/'reference.json').write_text(json.dumps(dict(source=str(a.source), road=str(a.road), frame=a.frame,
        vision_nodes=len(vision.graph.node), ort_version=ort.__version__,
        source_sha256=sha(a.source),
        external_weights_sha256={loc:sha(a.source.parent/loc) for loc in sorted(locations)},
        input_sha256={str(path):sha(path) for path in input_paths},
        artifact_sha256={name:sha(a.output/name) for name in ['cat_2.raw','teacher.npy','pre.onnx','teacher_vision.onnx']},
        note='Boundary diagnostic against ORT teacher. Accepted Android hybrid boundary is unavailable. Not recurrent or target timing validation.'),indent=2))


if __name__ == '__main__':
    main()
