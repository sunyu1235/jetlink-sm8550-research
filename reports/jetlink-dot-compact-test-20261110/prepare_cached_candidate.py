"""Apply Dot's compact-boundary idea to the accepted Android cached model.

Only two QNN partitions change. The existing Android exact CPU GELUs and all
other cached NPU partitions remain. No new HTP OpPackage is required.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
REPAIR = ROOT / "outputs/jetlink-gelu-repair-20261101"
OUT = ROOT / "outputs/jetlink-dot-compact-test-20261110"
BASE = OLD / "whole-context-last-query-shared-all.onnx"
COMPACT = {"add_52", "add_72", "add_88", "linear_110", "linear_151", "linear_180",
           "gelu_44", "gelu_55", "gelu_62"}


def save(model, path):
    if path.exists():
        raise FileExistsError(path)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    up = onnx.load(OLD / "last-query-partition-131-41.onnx")
    restore = [n for n in up.graph.node if n.name.endswith("__restore_boundary_shape")]
    assert len(restore) == 6
    alias = {n.input[1]: n.output[0] for n in restore}
    nodes = [copy.deepcopy(n) for n in up.graph.node if n not in restore]
    for n in nodes:
        for values in (n.input, n.output):
            for i, name in enumerate(values):
                values[i] = alias.get(name, name)
    del up.graph.node[:]
    up.graph.node.extend(nodes)
    used = {v for n in nodes for v in n.input}
    weights = [copy.deepcopy(t) for t in up.graph.initializer if t.name in used]
    del up.graph.initializer[:]
    up.graph.initializer.extend(weights)
    for value in up.graph.output:
        assert value.type.tensor_type.shape.dim[1].dim_value == 288
        value.type.tensor_type.shape.dim[1].dim_value = 1
    del up.graph.value_info[:]
    up.graph.name = "android_compact_upstream_20261110"
    up = shape_inference.infer_shapes(up, strict_mode=True)
    save(up, OUT / "upstream.onnx")

    cached = onnx.load(BASE)
    boundary = next(n for n in cached.graph.node if n.name.endswith("_132_42"))
    source_path = REPAIR / "light-speed-gelu-both-inline-shapes.onnx"
    source = onnx.load(source_path, load_external_data=False)
    values = {v.name: v for v in [*source.graph.input, *source.graph.value_info, *source.graph.output]}
    initializers = {t.name: t for t in source.graph.initializer}
    producers = {v: i for i, n in enumerate(source.graph.node) for v in n.output}
    selected, needed_weights = set(), set()

    def visit(name):
        if not name or name in boundary.input:
            return
        if name in initializers:
            needed_weights.add(name)
            return
        i = producers[name]
        if i not in selected:
            selected.add(i)
            for dep in source.graph.node[i].input:
                visit(dep)

    for name in boundary.output:
        visit(name)
    nodes = [copy.deepcopy(n) for i, n in enumerate(source.graph.node) if i in selected]
    weights = []
    for name in sorted(needed_weights):
        t = copy.deepcopy(initializers[name])
        if external_data_helper.uses_external_data(t):
            external_data_helper.load_external_data_for_tensor(t, str(source_path.parent))
        t.data_location = TensorProto.DEFAULT
        del t.external_data[:]
        weights.append(t)
    by_name = {t.name: t for t in weights}
    shrunk, gathers = [], []
    for n in nodes:
        assert n.domain in ("", "ai.onnx")
        if n.op_type == "Reshape":
            shape = numpy_helper.to_array(by_name[n.input[1]])
            if 288 in shape:
                new_shape = shape.copy()
                new_shape[new_shape == 288] = 1
                by_name[n.input[1]].CopyFrom(numpy_helper.from_array(new_shape, n.input[1]))
                shrunk.append(n.name)
        if n.op_type == "Gather" and n.output[0] in ("select_4", "select_7", "select_10"):
            assert numpy_helper.to_array(by_name[n.input[1]]).item() == 287
            assert helper.get_attribute_value(next(a for a in n.attribute if a.name == "axis")) == 1
            n.op_type = "Squeeze"
            n.input[1] = "jl_compact_token_axis"
            del n.attribute[:]
            gathers.append(n.name)
    assert len(shrunk) == len(gathers) == 3
    weights.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "jl_compact_token_axis"))
    inputs = [copy.deepcopy(values[name]) for name in boundary.input]
    for value in inputs:
        assert value.name in COMPACT and value.type.tensor_type.shape.dim[1].dim_value == 288
        value.type.tensor_type.shape.dim[1].dim_value = 1
    tail = helper.make_model(helper.make_graph(nodes, "android_compact_downstream_20261110", inputs,
        [copy.deepcopy(values[name]) for name in boundary.output], weights),
        opset_imports=[copy.deepcopy(o) for o in source.opset_import if o.domain in ("", "ai.onnx")])
    tail.ir_version = source.ir_version
    tail = shape_inference.infer_shapes(tail, strict_mode=True)
    save(tail, OUT / "downstream.onnx")
    report = {"dot_commit": "af713f089848d1f6725a3ebfb7b3bf9f42e07296",
        "scope": "two rebuilt cached partitions; accepted Android CPU exact GELUs; existing NPU remainder",
        "original_boundary_bytes": 2654208, "compact_boundary_bytes": 9216,
        "changed_gathers": gathers, "changed_reshapes": shrunk,
        "upstream_nodes": len(up.graph.node), "downstream_nodes": len(tail.graph.node)}
    (OUT / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


def compose():
    model = onnx.load(BASE)
    targets = [95, 99]
    replacements = []
    for label in ("upstream", "downstream"):
        part = onnx.load(OUT / (label + "-ctx.onnx"))
        assert len(part.graph.node) == 1 and part.graph.node[0].op_type == "EPContext", label
        replacement = copy.deepcopy(part.graph.node[0])
        attr = next(a for a in replacement.attribute if a.name == "ep_cache_context")
        filename = attr.s.decode()
        assert Path(filename).name == filename and filename.endswith(".bin")
        attr.s = ("jetlink-dot-compact-20261110/" + filename).encode()
        replacements.append(replacement)
    for i, replacement in zip(targets, replacements):
        original = model.graph.node[i]
        assert set(original.input) == set(replacement.input)
        assert set(original.output) == set(replacement.output)
        model.graph.node[i].CopyFrom(replacement)
    changed_metadata = []
    for value in [*model.graph.value_info, *model.graph.output]:
        if value.name in COMPACT:
            dims = value.type.tensor_type.shape.dim
            assert dims[1].dim_value == 288
            dims[1].dim_value = 1
            changed_metadata.append(value.name)
    save(model, OUT / "compact-whole.onnx")
    print(json.dumps({"whole_model": str(OUT / "compact-whole.onnx"),
        "ep_contexts": sum(n.op_type == "EPContext" for n in model.graph.node),
        "exact_gelu_nodes": sum(n.domain == "jetlink.precision" for n in model.graph.node),
        "changed_boundary_metadata": sorted(set(changed_metadata))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "compose"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else compose()
