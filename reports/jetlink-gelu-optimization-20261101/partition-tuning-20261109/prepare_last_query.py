"""Prune unused final-layer attention queries in accepted QNN partition 131_41.

K/V retain all 288 tokens. Only Q token 287 and its subsequent residual/MLP
work are computed. The six original boundary shapes are restored with zero
prefixes so the existing downstream cached partition can still be loaded.
Only the last boundary row preserves the original mathematical semantics;
the complete model must be checked before this candidate is accepted.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
REPAIR = ROOT / "outputs/jetlink-gelu-repair-20261101"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def info(value: onnx.ValueInfoProto) -> dict:
    t = value.type.tensor_type
    return {"name": value.name, "elem_type": t.elem_type,
            "shape": [dimension.dim_value for dimension in t.shape.dim]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT / "original-partition-131-41.onnx"
    optimized_path = OUT / "last-query-partition-131-41.onnx"
    manifest_path = REPORT / "last-query-partition-manifest.json"
    for path in (baseline_path, optimized_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    shapes_path = REPAIR / "light-speed-gelu-both-inline-shapes.onnx"
    whole_path = REPAIR / "light-speed-gelu-both.onnx"
    context_path = REPAIR / "light-speed-gelu-both-ctx.onnx"
    shaped = onnx.load(shapes_path, load_external_data=False)
    context = onnx.load(context_path, load_external_data=False)
    matches = [node for node in context.graph.node if node.op_type == "EPContext" and node.name.endswith("_131_41")]
    if len(matches) != 1:
        raise ValueError("expected one accepted _131_41 partition")
    boundary = matches[0]
    input_names, output_names = list(boundary.input), list(boundary.output)
    if output_names != ["add_52", "add_72", "add_88", "linear_110", "linear_151", "linear_180"]:
        raise ValueError(f"unexpected partition outputs: {output_names}")
    values = {value.name: value for value in list(shaped.graph.input) + list(shaped.graph.value_info) + list(shaped.graph.output)}
    initializers = {tensor.name: tensor for tensor in shaped.graph.initializer}
    producer = {name: index for index, node in enumerate(shaped.graph.node) for name in node.output}
    needed_nodes, needed_weights = set(), set()
    cuts = set(input_names)

    def visit(name: str) -> None:
        if not name or name in cuts:
            return
        if name in initializers:
            needed_weights.add(name)
            return
        if name not in producer:
            raise ValueError(f"undeclared source boundary: {name}")
        index = producer[name]
        if index not in needed_nodes:
            needed_nodes.add(index)
            for value in shaped.graph.node[index].input:
                visit(value)

    for name in output_names:
        visit(name)
    nodes = [copy.deepcopy(node) for index, node in enumerate(shaped.graph.node) if index in needed_nodes]
    if any(node.domain not in ("", "ai.onnx") for node in nodes):
        raise ValueError("selected partition contains a custom node")
    whole = onnx.load(whole_path)
    if [n.SerializeToString() for n in whole.graph.node] != [n.SerializeToString() for n in shaped.graph.node]:
        raise ValueError("accepted source graph nodes differ from inline-shapes source")
    source_weights = {tensor.name: tensor for tensor in whole.graph.initializer}
    weights = [copy.deepcopy(source_weights[name]) for name in sorted(needed_weights)]
    baseline = helper.make_model(helper.make_graph(
        nodes, "accepted_original_partition_131_41",
        [copy.deepcopy(values[name]) for name in input_names],
        [copy.deepcopy(values[name]) for name in output_names], weights),
        producer_name="jetlink-last-query-20261109",
        opset_imports=[copy.deepcopy(opset) for opset in whole.opset_import if opset.domain in ("", "ai.onnx")])
    baseline.ir_version = whole.ir_version
    del source_weights, whole
    baseline = shape_inference.infer_shapes(baseline, strict_mode=True)
    onnx.checker.check_model(baseline, full_check=True)
    onnx.save(baseline, baseline_path)

    optimized = copy.deepcopy(baseline)
    optimized.graph.name = "last_query_only_partition_131_41"
    del optimized.graph.value_info[:]
    weights_by_name = {tensor.name: tensor for tensor in optimized.graph.initializer}
    additions = []
    rewritten_nodes = []
    tainted = set()
    compact_names = {name: name + "__last_query" for name in output_names}
    query_sources = {"squeeze_9", "squeeze_21", "squeeze_33"}
    query_cuts, shape_rewrites, residual_cuts, mask_proofs = [], [], [], []

    def add_ints(name: str, data: list[int]) -> str:
        additions.append(numpy_helper.from_array(np.asarray(data, dtype=np.int64), name))
        return name

    starts = add_ints("jl_last_query_starts", [287])
    ends = add_ints("jl_last_query_ends", [288])
    axis_1 = add_ints("jl_last_query_axis1", [1])
    axis_2 = add_ints("jl_last_query_axis2", [2])

    for original in baseline.graph.node:
        node = copy.deepcopy(original)
        for index, name in enumerate(node.input):
            node.input[index] = compact_names.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = compact_names.get(name, name)
        if node.op_type == "Mul" and node.input[0] in query_sources:
            source = node.input[0]
            cut = source + "__last_query"
            rewritten_nodes.append(helper.make_node("Slice", [source, starts, ends, axis_2], [cut],
                                                    name=source + "__last_query_slice"))
            node.input[0] = cut
            tainted.add(cut)
            query_cuts.append({"source": source, "output": cut, "axis": 2, "start": 287, "end": 288})
        if node.op_type == "Where":
            mask_name = original.input[0]
            not_node = next(candidate for candidate in baseline.graph.node if mask_name in candidate.output)
            if not_node.op_type != "Not" or not_node.input[0] not in weights_by_name:
                raise ValueError(f"unexpected masked attention source: {node.name}")
            mask = numpy_helper.to_array(weights_by_name[not_node.input[0]])
            if mask.shape != (1, 1, 288, 288) or not bool(mask[..., 287, :].all()):
                raise ValueError(f"last query row is not fully unmasked: {node.name}")
            score = node.input[2]
            if score not in tainted:
                raise ValueError(f"masked attention score was not pruned: {node.name}")
            node.op_type = "Identity"
            del node.input[:]
            node.input.append(score)
            del node.attribute[:]
            mask_proofs.append({"node": original.name, "mask": not_node.input[0],
                                "shape": list(mask.shape), "last_row_all_true": True})
        if original.output[0] in ("add_52", "add_72", "add_88"):
            if node.op_type != "Add" or node.input[1] not in tainted:
                raise ValueError(f"unexpected final residual node: {node.name}")
            source = node.input[0]
            cut = source + "__last_query_residual"
            rewritten_nodes.append(helper.make_node("Slice", [source, starts, ends, axis_1], [cut],
                                                    name=source + "__last_query_residual_slice"))
            node.input[0] = cut
            tainted.add(cut)
            residual_cuts.append({"source": source, "output": cut, "axis": 1, "start": 287, "end": 288})
        if node.op_type == "Reshape" and node.input[0] in tainted:
            shape_tensor = weights_by_name[node.input[1]]
            before = numpy_helper.to_array(shape_tensor)
            after = before.copy()
            after[after == 288] = 1
            if not np.array_equal(before, after):
                replacement = node.input[1] + "__" + node.name + "__last_query"
                additions.append(numpy_helper.from_array(after, replacement))
                node.input[1] = replacement
                shape_rewrites.append({"node": node.name, "before": before.tolist(), "after": after.tolist()})
        if any(name in tainted for name in node.input):
            tainted.update(node.output)
        rewritten_nodes.append(node)
    if len(query_cuts) != 3 or len(residual_cuts) != 3 or len(mask_proofs) != 3:
        raise ValueError("did not rewrite all three independent final-layer attention branches")
    zero_names = {}
    for name in output_names:
        tensor = values[name].type.tensor_type
        original_shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != TensorProto.FLOAT16 or original_shape[:2] != [1, 288]:
            raise ValueError(f"unexpected output boundary shape/dtype: {name}")
        channels = original_shape[-1]
        if channels not in zero_names:
            zeros_name = f"jl_last_query_zero_prefix_{channels}"
            additions.append(numpy_helper.from_array(np.zeros((1, 287, channels), dtype=np.float16), zeros_name))
            zero_names[channels] = zeros_name
        rewritten_nodes.append(helper.make_node("Concat", [zero_names[channels], compact_names[name]], [name],
                                                name=name + "__restore_boundary_shape", axis=1))
    all_weights = list(optimized.graph.initializer) + additions
    producer_by_name = {name: index for index, node in enumerate(rewritten_nodes) for name in node.output}
    initializer_by_name = {tensor.name: tensor for tensor in all_weights}
    live_nodes, live_weights = set(), set()

    def live(name: str) -> None:
        if name in cuts or not name:
            return
        if name in initializer_by_name:
            live_weights.add(name)
            return
        if name not in producer_by_name:
            raise ValueError(f"rewritten graph dependency missing: {name}")
        index = producer_by_name[name]
        if index not in live_nodes:
            live_nodes.add(index)
            for item in rewritten_nodes[index].input:
                live(item)

    for name in output_names:
        live(name)
    del optimized.graph.node[:]
    optimized.graph.node.extend(node for index, node in enumerate(rewritten_nodes) if index in live_nodes)
    del optimized.graph.initializer[:]
    optimized.graph.initializer.extend(tensor for tensor in all_weights if tensor.name in live_weights)
    optimized = shape_inference.infer_shapes(optimized, strict_mode=True)
    onnx.checker.check_model(optimized, full_check=True)
    inferred = {value.name: value for value in list(optimized.graph.value_info) + list(optimized.graph.output)}
    for name, compact in compact_names.items():
        expected = info(values[name])["shape"]
        expected[1] = 1
        if info(inferred[compact])["shape"] != expected:
            raise ValueError(f"incorrect compact output shape: {compact}: {info(inferred[compact])}")
    if [value.SerializeToString() for value in optimized.graph.input] != [value.SerializeToString() for value in baseline.graph.input]:
        raise ValueError("external input metadata changed")
    if [value.SerializeToString() for value in optimized.graph.output] != [value.SerializeToString() for value in baseline.graph.output]:
        raise ValueError("external output metadata changed")
    onnx.save(optimized, optimized_path)
    onnx.checker.check_model(str(optimized_path), full_check=True)
    manifest = {
        "status": "structural_and_shape_check_passed_not_numerically_accepted",
        "source_partition": boundary.name,
        "source_whole": {"path": str(whole_path.resolve()), "sha256": sha(whole_path)},
        "original_partition": {"path": str(baseline_path.resolve()), "bytes": baseline_path.stat().st_size,
                               "sha256": sha(baseline_path), "nodes": len(baseline.graph.node)},
        "candidate_partition": {"path": str(optimized_path.resolve()), "bytes": optimized_path.stat().st_size,
                                "sha256": sha(optimized_path), "nodes": len(optimized.graph.node),
                                "op_counts": dict(collections.Counter(node.op_type for node in optimized.graph.node))},
        "inputs": [info(value) for value in baseline.graph.input],
        "outputs": [info(value) for value in baseline.graph.output],
        "compact_outputs": [info(inferred[name]) for name in compact_names.values()],
        "query_slices": query_cuts, "residual_slices": residual_cuts,
        "mask_proofs": mask_proofs, "reshape_changes": shape_rewrites,
        "preserved": "Original external I/O names/order/shapes, K/V 288-token path, preceding MLP, retained weights, precision-sensitive GELU implementations outside this partition.",
        "boundary_prefix_semantics": "Rows 0..286 are explicitly zero-filled. Only row287 retains original mathematical semantics; downstream final Gather selects exactly287 and intervening MLP ops are token independent.",
        "acceptance_required": "Run the complete recurrent model through the unchanged upstream per-head parity checker. FP16 backend kernels may choose different reductions for new shapes despite mathematical equivalence.",
        "onnx_checker_full_check": "passed", "onnx_infer_shapes_strict": "passed"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_partition": manifest["candidate_partition"],
                      "original_partition": manifest["original_partition"],
                      "manifest": str(manifest_path.resolve()),
                      "compact_outputs": manifest["compact_outputs"]}, indent=2))


if __name__ == "__main__":
    main()
