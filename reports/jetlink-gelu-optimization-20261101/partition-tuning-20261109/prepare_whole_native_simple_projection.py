"""Remove the measured-unnecessary tail branch from the six-op thin model.

Only the small ONNX protobuf is loaded; external weight bytes are never loaded.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import onnx

from prepare_whole_bounded_gelu import topological_sort


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def serialized(items) -> list[bytes]:
    return [item.SerializeToString() for item in items]


def main() -> None:
    source_path = OUT / "whole-native-projection-thin.onnx"
    output_path = OUT / "whole-native-simple-projection-thin.onnx"
    temporary_path = OUT / "whole-native-simple-projection-thin.tmp.onnx"
    source_manifest_path = REPORT / "whole-native-projection-manifest.json"
    manifest_path = REPORT / "whole-native-simple-projection-manifest.json"
    if any(path.exists() for path in (output_path, temporary_path, manifest_path)):
        raise FileExistsError("refusing to overwrite simple native projection evidence")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = fingerprint(source_path)
    if source != source_manifest["model"]:
        raise ValueError("six-op source thin model fingerprint differs")
    model = onnx.load(source_path, load_external_data=False)
    inputs, outputs = serialized(model.graph.input), serialized(model.graph.output)
    metadata, opsets, ir_version = serialized(model.metadata_props), serialized(model.opset_import), model.ir_version
    by_name = {node.name: node for node in model.graph.node}
    if len(by_name) != len(model.graph.node) or len(model.graph.node) != 1504:
        raise ValueError("unexpected source graph or duplicate node names")
    before_nodes = {name: node.SerializeToString() for name, node in by_name.items()}
    threshold = source_manifest["gelu_rewrite"]["threshold_initializer"]
    retained_initializers = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer if tensor.name != threshold}
    external_before = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer
                       if tensor.data_location == onnx.TensorProto.EXTERNAL}
    removed, renamed, stale_value_info, replacements = set(), {}, set(), []
    for change in source_manifest["gelu_rewrite"]["replacements"]:
        names = change["new_nodes"]
        nodes = [by_name[name] for name in names]
        if [node.op_type for node in nodes] != ["Gelu", "Relu", "Min", "Abs", "Greater", "Where"]:
            raise ValueError("unexpected six-op projection structure")
        gelu, relu, minimum, absolute, greater, where = nodes
        if (list(gelu.input) != [change["input"]] or list(relu.input) != [change["input"]]
                or list(minimum.input) != [gelu.output[0], relu.output[0]]
                or list(absolute.input) != [change["input"]]
                or list(greater.input) != [absolute.output[0], threshold]
                or list(where.input) != [greater.output[0], relu.output[0], minimum.output[0]]
                or list(where.output) != [change["output"]]):
            raise ValueError(f"projection wiring differs: {change['original_node']}")
        stale_value_info.add(minimum.output[0])
        minimum.output[0] = change["output"]
        renamed[minimum.name] = minimum.SerializeToString()
        for node in (absolute, greater, where):
            removed.add(node.name)
            stale_value_info.update(node.output)
        # The final output metadata remains valid and must stay in the graph.
        stale_value_info.discard(change["output"])
        replacements.append({**change, "new_nodes": names[:3]})
    if len(removed) != 189 or len(renamed) != 63:
        raise ValueError("unexpected native projection rewrite count")
    kept = onnx.GraphProto()
    kept.node.extend(node for node in model.graph.node if node.name not in removed)
    kept.initializer.extend(tensor for tensor in model.graph.initializer if tensor.name != threshold)
    kept.value_info.extend(value for value in model.graph.value_info if value.name not in stale_value_info)
    del model.graph.node[:]
    model.graph.node.extend(kept.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept.initializer)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept.value_info)
    del kept
    topological_sort(model)
    if len(model.graph.node) != 1315:
        raise ValueError("expected 1315 simple-projection whole nodes")
    if any(threshold in node.input or any(name in stale_value_info for name in node.input) for node in model.graph.node):
        raise ValueError("removed tail tensor is still consumed")
    for node in model.graph.node:
        if node.SerializeToString() != renamed.get(node.name, before_nodes[node.name]):
            raise ValueError(f"unintended node modification: {node.name}")
    if {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer} != retained_initializers:
        raise ValueError("retained initializer protobuf changed")
    if any(node.domain not in ("", "ai.onnx") or node.op_type == "EPContext" for node in model.graph.node):
        raise ValueError("custom or EPContext node remains")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if serialized(model.graph.input) != inputs or serialized(model.graph.output) != outputs:
        raise ValueError("public input/output metadata changed")
    if serialized(model.metadata_props) != metadata or serialized(model.opset_import) != opsets or model.ir_version != ir_version:
        raise ValueError("metadata/IR/opsets changed")
    if {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer
            if tensor.data_location == onnx.TensorProto.EXTERNAL} != external_before:
        raise ValueError("external tensor metadata/offset changed")
    onnx.save_model(model, temporary_path)
    onnx.checker.check_model(str(temporary_path), full_check=True)
    temporary_path.replace(output_path)
    manifest = {
        "status": "whole_native_simple_projection_thin_strict_shape_and_checker_passed_pending_full_model_htp_parity",
        "model": fingerprint(output_path), "source_model": source,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_container": source_manifest["source_container"],
        "external_data": source_manifest["external_data"],
        "external_initializer_protobufs_identical": True,
        "large_raw_weights_loaded_by_preparation": False,
        "original_node_count": 1504, "candidate_node_count": len(model.graph.node),
        "removed_tail_node_count": len(removed), "removed_initializer": threshold,
        "candidate_initializer_count": len(model.graph.initializer),
        "gelu_rewrite": {"rewritten_count": 63, "erf_count": 36, "tanh_count": 27,
                         "nodes_per_formula": 3,
                         "definition": "raw=Gelu(x,original approximate); r=Relu(x); y=Min(raw,r)",
                         "replacements": replacements},
        "operator_measurement_basis": {
            "source": "Root agent analysis of previously measured complete 63488 finite-FP16 native NPU GELU raw outputs",
            "both_reference_formulas_max_abs": 0.001953125,
            "erf_rms_approx": 0.000214, "tanh_rms_approx": 0.000212,
            "nonfinite_count": 0,
            "scope": "Operator finite-domain result; full-model precision/timing remain unmeasured for this candidate"},
        "op_counts": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "retained_non_projection_nodes_byte_identical": True,
        "last_query_node_count_preserved": 120,
        "public_inputs_preserved": True, "public_outputs_preserved": True,
        "public_output_names": [value.name for value in model.graph.output],
        "metadata_ir_opsets_preserved": True, "custom_nodes": 0, "epcontext_nodes": 0,
        "topological_sort": "passed", "onnx_shape_inference_strict": "passed", "onnx_checker_full_check": "passed",
        "device_access": "none", "precision_thresholds": "original per-head thresholds unchanged"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "nodes": len(model.graph.node),
                      "manifest": str(manifest_path.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
