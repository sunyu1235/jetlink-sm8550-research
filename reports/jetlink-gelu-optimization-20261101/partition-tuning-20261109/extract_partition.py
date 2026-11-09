"""Extract a measured QNN partition and a capture-only accepted-context wrapper.

The standard ONNX partition reuses exact initializer bytes from the accepted
whole model. Its boundaries come from the saved QNN EPContext node, and their
types/shapes come from the accepted inline-shapes graph. No phone access occurs.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
REPAIR_DIR = ROOT / "outputs/jetlink-gelu-repair-20261101"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": file_hash(path)}


def node_digest(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for node in model.graph.node:
        payload = node.SerializeToString()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def tensor_info(value: onnx.ValueInfoProto) -> dict:
    if not value.type.HasField("tensor_type"):
        raise ValueError(f"non-tensor boundary: {value.name}")
    tensor = value.type.tensor_type
    if tensor.elem_type == TensorProto.UNDEFINED or not tensor.HasField("shape"):
        raise ValueError(f"untyped/unshaped boundary: {value.name}")
    dimensions = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"dynamic or invalid boundary shape: {value.name}")
        dimensions.append(dimension.dim_value)
    return {"name": value.name, "elem_type": tensor.elem_type,
            "dtype": TensorProto.DataType.Name(tensor.elem_type), "shape": dimensions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole", type=Path, default=REPAIR_DIR / "light-speed-gelu-both.onnx")
    parser.add_argument("--shapes", type=Path, default=REPAIR_DIR / "light-speed-gelu-both-inline-shapes.onnx")
    parser.add_argument("--context", type=Path, default=REPAIR_DIR / "light-speed-gelu-both-ctx.onnx")
    parser.add_argument("--suffix", default="_130_40")
    parser.add_argument("--remote-cache", default=None,
                        help="optional relative cache basename; default preserves source attributes")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=REPORT_DIR / "extraction-manifest.json")
    args = parser.parse_args()
    partition_path = args.output_dir / "partition-130-40.onnx"
    capture_path = args.output_dir / "accepted-context-capture-130-40.onnx"
    for path in (partition_path, capture_path, args.manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    if args.remote_cache and ("/" in args.remote_cache or "\\" in args.remote_cache):
        raise ValueError("remote cache must be a relative basename; absolute paths are rejected by this runtime")

    context = onnx.load(args.context, load_external_data=False)
    shaped = onnx.load(args.shapes, load_external_data=False)
    matches = [node for node in context.graph.node
               if node.op_type == "EPContext" and node.name.endswith(args.suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one EPContext ending {args.suffix}, found {len(matches)}")
    selected = matches[0]
    input_names, output_names = list(selected.input), list(selected.output)
    if len(input_names) != 6 or len(output_names) != 6:
        raise ValueError("expected six input and six output tensors for partition _130_40")
    if len(set(input_names + output_names)) != 12:
        raise ValueError("partition boundary tensor names are not distinct")
    values = {value.name: value for value in
              list(shaped.graph.input) + list(shaped.graph.value_info) + list(shaped.graph.output)}
    missing = set(input_names + output_names) - values.keys()
    if missing:
        raise ValueError(f"boundary metadata missing: {sorted(missing)}")
    boundary_inputs = [tensor_info(values[name]) for name in input_names]
    boundary_outputs = [tensor_info(values[name]) for name in output_names]

    initializers = {tensor.name: tensor for tensor in shaped.graph.initializer}
    producer = {}
    for index, node in enumerate(shaped.graph.node):
        for name in node.output:
            if name in producer:
                raise ValueError(f"duplicate producer: {name}")
            producer[name] = index
    cut = set(input_names)
    used_inputs, needed_initializers, needed_nodes = set(), set(), set()

    def visit(name: str) -> None:
        if not name:
            return
        if name in cut:
            used_inputs.add(name)
            return
        if name in initializers:
            needed_initializers.add(name)
            return
        if name not in producer:
            raise ValueError(f"subgraph depends on undeclared boundary: {name}")
        index = producer[name]
        if index in needed_nodes:
            return
        needed_nodes.add(index)
        for input_name in shaped.graph.node[index].input:
            visit(input_name)

    for name in output_names:
        visit(name)
    if used_inputs != cut:
        raise ValueError(f"unused boundary inputs: {sorted(cut - used_inputs)}")
    nodes = [copy.deepcopy(node) for index, node in enumerate(shaped.graph.node)
             if index in needed_nodes]
    if any(node.domain not in ("", "ai.onnx") for node in nodes):
        raise ValueError("partition contains a custom-domain node")
    if any("Gelu" in node.op_type or node.op_type == "EPContext" for node in nodes):
        raise ValueError("partition unexpectedly contains a GELU or cached context node")

    whole = onnx.load(args.whole, load_external_data=True)
    graph_digest = node_digest(shaped)
    if node_digest(whole) != graph_digest:
        raise ValueError("whole-model and inline-shapes graph nodes differ")
    whole_initializers = {tensor.name: tensor for tensor in whole.graph.initializer}
    weights, weight_manifest = [], []
    for name in sorted(needed_initializers):
        tensor = whole_initializers[name]
        shape_tensor = initializers[name]
        if tensor.data_type != shape_tensor.data_type or tensor.dims != shape_tensor.dims:
            raise ValueError(f"initializer metadata differs: {name}")
        if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
            raise ValueError(f"initializer not materialized from accepted whole model: {name}")
        weights.append(copy.deepcopy(tensor))
        weight_manifest.append({"name": name, "elem_type": tensor.data_type,
                                "shape": list(tensor.dims), "raw_bytes": len(tensor.raw_data),
                                "raw_sha256": hashlib.sha256(tensor.raw_data).hexdigest(),
                                "protobuf_sha256": hashlib.sha256(tensor.SerializeToString()).hexdigest()})
    graph = helper.make_graph(nodes, "accepted_qnn_partition_130_40",
                              [copy.deepcopy(values[name]) for name in input_names],
                              [copy.deepcopy(values[name]) for name in output_names],
                              initializer=weights,
                              value_info=[copy.deepcopy(value) for value in shaped.graph.value_info
                                          if value.name in producer and producer[value.name] in needed_nodes
                                          and value.name not in output_names and value.name not in input_names])
    partition = helper.make_model(graph, producer_name="jetlink-partition-tuning-20261109",
                                   opset_imports=[copy.deepcopy(opset) for opset in whole.opset_import
                                                 if opset.domain in ("", "ai.onnx")])
    partition.ir_version = whole.ir_version
    del whole_initializers, whole
    onnx.checker.check_model(partition, full_check=True)

    wrapper = copy.deepcopy(context)
    original_outputs = [value.name for value in wrapper.graph.output]
    for name in input_names + output_names:
        wrapper.graph.output.append(copy.deepcopy(values[name]))
    old_cache_locations = set()
    rewrites = []
    for node in wrapper.graph.node:
        if node.op_type != "EPContext":
            continue
        for attribute in node.attribute:
            if attribute.name == "ep_cache_context" and attribute.s:
                previous = attribute.s.decode("utf-8")
                old_cache_locations.add(previous)
                if args.remote_cache:
                    if Path(previous).name != args.remote_cache:
                        raise ValueError(f"cache binary basename differs from original context: {previous}")
                    attribute.s = args.remote_cache.encode("utf-8")
                    rewrites.append({"node": node.name, "before": previous, "after": args.remote_cache})
    if not old_cache_locations:
        raise ValueError("no non-empty ep_cache_context attribute found")
    onnx.checker.check_model(wrapper)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(partition, partition_path)
    onnx.save(wrapper, capture_path)
    onnx.checker.check_model(str(partition_path), full_check=True)
    onnx.checker.check_model(str(capture_path))
    manifest = {
        "purpose": "isolated standard-ONNX partition and capture-only accepted EPContext wrapper",
        "status": "onnx_checker_passed_no_phone_execution",
        "source_whole": file_info(args.whole), "source_shapes": file_info(args.shapes),
        "source_context": file_info(args.context), "source_node_sequence_sha256": graph_digest,
        "source_node_sequence_identical_between_whole_and_shapes": True,
        "selected_context_node": selected.name, "partition_suffix": args.suffix,
        "partition": {**file_info(partition_path), "node_count": len(nodes),
                      "op_counts": dict(sorted(collections.Counter(node.op_type for node in nodes).items())),
                      "custom_domain_nodes": 0, "gelu_nodes": 0,
                      "inputs": boundary_inputs, "outputs": boundary_outputs,
                      "initializer_count": len(weights),
                      "initializer_raw_bytes": sum(item["raw_bytes"] for item in weight_manifest),
                      "initializers": weight_manifest, "onnx_checker_full_check": "passed"},
        "capture_wrapper": {**file_info(capture_path), "node_count": len(wrapper.graph.node),
                            "original_outputs_preserved_in_order": original_outputs,
                            "appended_outputs": boundary_inputs + boundary_outputs,
                            "old_cache_locations": sorted(old_cache_locations),
                            "cache_path_rewrites": rewrites, "onnx_checker": "passed",
                            "arithmetic_changes": "none; original nodes and graph inputs preserved"},
        "single_partition_context_replacement_assessment": {
            "status": "source_supported_feasibility_only_not_implemented_or_device_validated",
            "mechanism": "Keep the original main EPContext and binary, replace only the selected EPContext with a uniquely named main_context=1 node for the new standalone partition binary. Preserve all six boundary input/output names and their order/dtypes/shapes.",
            "requirements": [
                "Standalone context and original context must use the same installed QNN/ORT runtime and compatible HTP target.",
                "Use the standalone generator's EPContext metadata and a unique node/partition name; avoid collision with any graph deserialized from the original 45-graph binary.",
                "Original main_context=1 node must remain reachable and must keep its original binary reference.",
                "The original binary still contains the unused old partition; two contexts may increase memory and spill/fill overhead.",
                "Only a full recurrent replay with the unchanged upstream per-head parity checker and latency/temperature comparison can establish acceptance."
            ],
            "source": "https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/qnn_execution_provider.cc#L1208",
            "source_evidence": "ORT v1.29.0 enumerates main context nodes, loads each binary, then maps cached graphs to EPContext nodes; the backend explicitly handles multiple contexts."
        },
        "scope": "Extraction/checker only. No latency, device acceptance, or numerical equivalence result claimed."
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest.resolve()),
                      "partition": str(partition_path.resolve()), "partition_bytes": partition_path.stat().st_size,
                      "partition_nodes": len(nodes), "initializer_raw_bytes": manifest["partition"]["initializer_raw_bytes"],
                      "capture_wrapper": str(capture_path.resolve()), "appended_output_count": 12,
                      "cache_locations": sorted(old_cache_locations)}, indent=2))


if __name__ == "__main__":
    main()
