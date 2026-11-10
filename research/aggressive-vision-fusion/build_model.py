"""Replace only the closed 421-node vision region; keep the original FP16 cast.

Generated models, weights and vendor context binaries are local-only artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import onnx
from onnx import helper as h, TensorProto as T

INPUT = '_to_copy_4'
OUTPUT = 'view_39'


def select_cut(model):
    nodes = list(model.graph.node)
    producers = {}
    for i, node in enumerate(nodes):
        for name in node.output:
            if name in producers:
                raise ValueError('Duplicate producer: ' + name)
            producers[name] = i
    constants = {t.name for t in model.graph.initializer}
    selected, visiting, used = set(), set(), set()

    def visit(name):
        if not name:
            return
        if name == INPUT:
            used.add(name)
            return
        if name in constants:
            return
        if name not in producers:
            raise ValueError('Unclosed vision cut: ' + name)
        i = producers[name]
        if i in selected:
            return
        if i in visiting:
            raise ValueError('Cycle in vision cut')
        visiting.add(i)
        for inp in nodes[i].input:
            visit(inp)
        visiting.remove(i)
        selected.add(i)
    visit(OUTPUT)
    if used != {INPUT}:
        raise ValueError('Unused vision cut input')
    produced = {v for i in selected for v in nodes[i].output}
    outside = {v for i, n in enumerate(nodes) if i not in selected for v in n.input}
    outside.update(v.name for v in model.graph.output)
    side = (produced & outside) - {OUTPUT}
    if side:
        raise ValueError('Undeclared live side outputs: ' + repr(sorted(side)))
    ready = constants | {INPUT}
    for i in sorted(selected):
        if any(v and v not in ready for v in nodes[i].input):
            raise ValueError('Cut is not topological')
        ready.update(nodes[i].output)
    return selected


def replace(model):
    selected = select_cut(model)
    if len(selected) != 421:
        raise ValueError('Expected pinned 421-node vision cut')
    cast = [n for n in model.graph.node if INPUT in n.output]
    if len(cast) != 1 or cast[0].op_type != 'Cast' or list(cast[0].input) != ['cat_2'] or h.get_attribute_value(next(a for a in cast[0].attribute if a.name == 'to')) != T.FLOAT16:
        raise ValueError('Original exact FP16 cast missing')
    if len(model.graph.output) != 4:
        raise ValueError('Expected original four outputs')
    consumers = [n.SerializeToString() for n in model.graph.node if OUTPUT in n.input]
    if len(consumers) != 3:
        raise ValueError('Expected all three vision consumers')
    result = onnx.ModelProto()
    result.CopyFrom(model)
    kept = []
    for i, node in enumerate(model.graph.node):
        if i == min(selected):
            kept.append(h.make_node('QnnVisionPipeline', [INPUT], [OUTPUT], name='jetlink_native_vision', domain='jetlink.qnn'))
        if i not in selected:
            kept.append(node)
    del result.graph.node[:]
    result.graph.node.extend(kept)
    dead = {v for i in selected for v in model.graph.node[i].output} - {OUTPUT}
    vi = [v for v in result.graph.value_info if v.name not in dead | {INPUT, OUTPUT}]
    del result.graph.value_info[:]
    result.graph.value_info.extend(vi + [h.make_tensor_value_info(INPUT,T.FLOAT16,[1,24,128,256]),h.make_tensor_value_info(OUTPUT,T.FLOAT16,[1,32,512])])
    if any(o.domain == 'jetlink.qnn' for o in result.opset_import):
        raise ValueError('Existing jetlink.qnn opset is not supported')
    result.opset_import.append(h.make_opsetid('jetlink.qnn',1))
    assert consumers == [n.SerializeToString() for n in result.graph.node if OUTPUT in n.input]
    assert [v.SerializeToString() for v in result.graph.output] == [v.SerializeToString() for v in model.graph.output]
    return result, selected


def build(source, output):
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source, load_external_data=False)
    result, selected = replace(model)
    locations = {e.value for t in model.graph.initializer for e in t.external_data if e.key == 'location'}
    for loc in locations:
        if Path(loc).is_absolute() or '..' in Path(loc).parts or not (source.parent / loc).is_file():
            raise ValueError('Unsafe or missing external weights: ' + loc)
    output.mkdir(parents=True)
    for loc in locations:
        dst = output / loc
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source.parent / loc, dst)
    path = output / 'vision-native.onnx'
    onnx.save(result, path)
    onnx.checker.check_model(str(path))
    report = dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),removed_nodes=len(selected),native_nodes=1,preserved_outputs=[v.name for v in model.graph.output],all_three_consumers_preserved=True,native_validation='pending',scope='Host candidate; no device or speed qualification')
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.source,a.output),indent=2))
