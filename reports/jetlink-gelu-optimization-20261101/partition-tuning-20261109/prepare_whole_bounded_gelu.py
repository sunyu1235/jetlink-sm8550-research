"""Assemble the validated last-query graph and bounded rank1-Gather GELUs.

Original model metadata, weights and four public outputs remain intact. The
last-query partition keeps only the last boundary row's semantics, as already
validated in the complete cached model. This graph needs fresh strict-HTP
compilation and full-model recurrent parity before acceptance.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import heapq
import json
from pathlib import Path

import onnx

from prepare_bounded_gelu import rewrite_gelus


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


def topological_sort(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    roots = {tensor.name for tensor in model.graph.initializer} | {value.name for value in model.graph.input}
    producer = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if output in producer or output in roots:
                raise ValueError(f"duplicate tensor producer: {output}")
            producer[output] = index
    consumers = collections.defaultdict(list)
    indegree = []
    for index, node in enumerate(nodes):
        dependencies = set()
        for name in node.input:
            if not name or name in roots:
                continue
            if name not in producer:
                raise ValueError(f"unresolved graph input {name} for {node.name}")
            dependencies.add(producer[name])
        indegree.append(len(dependencies))
        for parent in dependencies:
            consumers[parent].append(index)
    ready = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        index = heapq.heappop(ready)
        order.append(index)
        for consumer in consumers[index]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(order) != len(nodes):
        raise ValueError("graph contains a cycle")
    replacement = onnx.GraphProto()
    replacement.node.extend(nodes[index] for index in order)
    del model.graph.node[:]
    model.graph.node.extend(replacement.node)


def main() -> None:
    source_path = REPAIR / "light-speed-gelu-both.onnx"
    original_partition_path = OUT / "original-partition-131-41.onnx"
    last_query_path = OUT / "last-query-partition-131-41.onnx"
    output_path = OUT / "whole-last-query-bounded-gelu.onnx"
    temporary_path = OUT / "whole-last-query-bounded-gelu.tmp.onnx"
    manifest_path = REPORT / "whole-last-query-bounded-gelu-manifest.json"
    if any(path.exists() for path in (output_path, temporary_path, manifest_path)):
        raise FileExistsError("refusing to overwrite a whole bounded-GELU candidate")
    original_partition = onnx.load(original_partition_path)
    last_query = onnx.load(last_query_path)
    onnx.checker.check_model(last_query, full_check=True)
    if [value.SerializeToString() for value in original_partition.graph.input] != [value.SerializeToString() for value in last_query.graph.input]:
        raise ValueError("last-query partition input interface changed")
    if [value.SerializeToString() for value in original_partition.graph.output] != [value.SerializeToString() for value in last_query.graph.output]:
        raise ValueError("last-query partition output interface changed")
    model = onnx.load(source_path)
    original_inputs = [value.SerializeToString() for value in model.graph.input]
    original_outputs = [value.SerializeToString() for value in model.graph.output]
    original_metadata = [item.SerializeToString() for item in model.metadata_props]
    original_ir_version = model.ir_version
    original_opsets = [item.SerializeToString() for item in model.opset_import]
    original_node_count = len(model.graph.node)
    original_initializer_count = len(model.graph.initializer)
    deleted = {node.name: node for node in original_partition.graph.node}
    if len(deleted) != 111 or "" in deleted:
        raise ValueError("expected 111 uniquely named original partition nodes")
    source_nodes = {node.name: node for node in model.graph.node}
    for name, node in deleted.items():
        if name not in source_nodes or source_nodes[name].SerializeToString() != node.SerializeToString():
            raise ValueError(f"original partition node differs from accepted model: {name}")
    nodes = [copy.deepcopy(node) for node in model.graph.node if node.name not in deleted]
    nodes.extend(copy.deepcopy(node) for node in last_query.graph.node)
    initializer_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    reused, added_initializers = [], []
    for tensor in last_query.graph.initializer:
        if tensor.name in initializer_by_name:
            if initializer_by_name[tensor.name].SerializeToString() != tensor.SerializeToString():
                raise ValueError(f"same-name initializer differs: {tensor.name}")
            reused.append(tensor.name)
        else:
            copied = model.graph.initializer.add()
            copied.CopyFrom(tensor)
            initializer_by_name[tensor.name] = copied
            added_initializers.append({"name": tensor.name, "raw_bytes": len(tensor.raw_data)})
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Original internal shapes for the pruned partition must not constrain its
    # new 1-token tensors. Replace them with the candidate's inferred metadata.
    available = {name for node in model.graph.node for name in node.output}
    available.update(value.name for value in model.graph.input)
    output_names = {value.name for value in model.graph.output}
    value_info_by_name = {value.name: copy.deepcopy(value) for value in model.graph.value_info
                          if value.name in available and value.name not in output_names}
    for value in last_query.graph.value_info:
        value_info_by_name[value.name] = copy.deepcopy(value)
    for value in last_query.graph.output:
        value_info_by_name[value.name] = copy.deepcopy(value)
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info_by_name.values())
    topological_sort(model)
    changes = rewrite_gelus(model, flat_indices=True)
    if changes["rewritten_count"] != 63 or changes["erf_count"] != 36 or changes["tanh_count"] != 27:
        raise ValueError(f"unexpected GELU formula counts: {changes['rewritten_count']}")
    topological_sort(model)
    if any(node.domain not in ("", "ai.onnx") or node.op_type == "EPContext" for node in model.graph.node):
        raise ValueError("whole candidate still contains a custom node or EPContext")
    print(f"assembled {len(model.graph.node)} standard nodes; inferring shapes", flush=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if [value.SerializeToString() for value in model.graph.input] != original_inputs:
        raise ValueError("whole model public inputs changed")
    if [value.SerializeToString() for value in model.graph.output] != original_outputs:
        raise ValueError("whole model public outputs changed")
    if [item.SerializeToString() for item in model.metadata_props] != original_metadata:
        raise ValueError("model metadata changed")
    if model.ir_version != original_ir_version or [item.SerializeToString() for item in model.opset_import] != original_opsets:
        raise ValueError("model IR/opsets changed")
    print("strict shape inference passed; saving complete candidate", flush=True)
    onnx.save(model, temporary_path)
    onnx.checker.check_model(str(temporary_path), full_check=True)
    temporary_path.replace(output_path)
    manifest = {
        "status": "whole_standard_onnx_candidate_shape_checker_passed_pending_full_htp_parity",
        "source_whole": {"path": str(source_path.resolve()), "sha256": sha(source_path)},
        "source_last_query": {"path": str(last_query_path.resolve()), "sha256": sha(last_query_path)},
        "model": {"path": str(output_path.resolve()), "bytes": output_path.stat().st_size,
                  "sha256": sha(output_path)},
        "original_node_count": original_node_count, "candidate_node_count": len(model.graph.node),
        "original_initializer_count": original_initializer_count,
        "candidate_initializer_count": len(model.graph.initializer),
        "last_query_replaced_nodes": sorted(deleted), "last_query_replacement_node_count": len(last_query.graph.node),
        "last_query_reused_initializer_count": len(reused), "last_query_added_initializers": added_initializers,
        "gelu_rewrite": changes, "custom_nodes": 0, "epcontext_nodes": 0,
        "op_counts": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "public_inputs_preserved": True, "public_outputs_preserved": True,
        "public_output_names": [value.name for value in model.graph.output],
        "original_metadata_ir_and_opsets_preserved": True,
        "original_weights_retained": "All source initializers preserved byte-for-byte; same-name last-query constants reused only after exact protobuf equality checks.",
        "topological_sort": "passed", "onnx_shape_inference_strict": "passed", "onnx_checker_full_check": "passed",
        "operator_validation_reference": str((REPORT / "bounded-gelu-flat-manifest.json").resolve()),
        "scope": "The rank1 Gather operator formulation passed the device finite-domain check. Complete recurrent model error/per-head parity and timing remain to be measured; original thresholds are unchanged."
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "nodes": len(model.graph.node),
                      "gelu_replacements": changes["rewritten_count"], "custom_nodes": 0,
                      "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
