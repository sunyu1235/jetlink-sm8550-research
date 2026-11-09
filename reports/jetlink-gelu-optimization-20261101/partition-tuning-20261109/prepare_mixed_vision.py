"""Extract the accepted accurate cached vision prefix for HTP + GPU policy.

All 128 legal prefix boundary outputs remain public for shared prebinding.
The runner bridges view_70 FP16[1,16384] to policy view_39 FLOAT[1,32,512].
No cached EP interface or binary is rewritten and no device is accessed.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def io(value) -> dict:
    tensor_type = value.type.tensor_type
    return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
            "shape": [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
                      for dimension in tensor_type.shape.dim]}


def main() -> None:
    source_path = OUT / "whole-context-last-query-shared-all.onnx"
    output_path = OUT / "mixed-vision-context.onnx"
    manifest_path = REPORT / "mixed-vision-manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite mixed vision candidate/evidence")
    model = onnx.load(source_path, load_external_data=False)
    original_node_count, original_output_count = len(model.graph.node), len(model.graph.output)
    original_inputs = [value.SerializeToString() for value in model.graph.input]
    if original_node_count != 108 or original_output_count != 215 or len(model.graph.initializer):
        raise ValueError("unexpected accepted shared-all cached wrapper")
    prefix = list(model.graph.node[:73])
    prefix_bytes = [node.SerializeToString() for node in prefix]
    counts = collections.Counter(node.op_type for node in prefix)
    if counts != {"EPContext": 37, "ExactGelu": 36}:
        raise ValueError(f"unexpected accurate vision prefix: {counts}")
    if prefix[-1].name != "QNNExecutionProvider_QNN_11262752905814198793_126_36":
        raise ValueError("unexpected last vision EP")
    if not {"view_70", "mean"}.issubset(prefix[-1].output):
        raise ValueError("last vision EP does not expose the established feature seam")
    produced = {name for node in prefix for name in node.output if name}
    roots = {value.name for value in model.graph.input}
    available = set(roots)
    for node in prefix:
        missing = set(node.input) - available - {""}
        if missing:
            raise ValueError(f"prefix depends on removed producer: {node.name}: {sorted(missing)}")
        available.update(node.output)
    # Keep the original order and exact types/shapes of every legal shared boundary.
    retained = onnx.GraphProto()
    retained.output.extend(value for value in model.graph.output if value.name in produced)
    retained.value_info.extend(value for value in model.graph.value_info if value.name in available)
    if {value.name for value in retained.output} != produced or len(retained.output) != 128:
        raise ValueError("not all 128 legal prefix tensors have public output metadata")
    expected_io = {
        "view_70": {"name": "view_70", "dtype": "FLOAT16", "shape": [1, 16384]},
        "next_state_img_q": {"name": "next_state_img_q", "dtype": "FLOAT16", "shape": [2, 5, 6, 128, 256]},
        "next_state_desire_q": {"name": "next_state_desire_q", "dtype": "FLOAT", "shape": [132, 1, 8]},
    }
    info = {value.name: io(value) for value in retained.output}
    if any(info.get(name) != description for name, description in expected_io.items()):
        raise ValueError("feature/state output contract differs")
    del model.graph.node[73:]
    del model.graph.output[:]
    model.graph.output.extend(retained.output)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained.value_info)
    if [node.SerializeToString() for node in model.graph.node] != prefix_bytes:
        raise ValueError("a retained EP/custom node changed")
    if [value.SerializeToString() for value in model.graph.input] != original_inputs:
        raise ValueError("the required seven-input cached EP contract changed")
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, output_path)
    onnx.checker.check_model(str(output_path), full_check=True)
    saved = onnx.load(output_path, load_external_data=False)
    if [node.SerializeToString() for node in saved.graph.node] != prefix_bytes:
        raise ValueError("serialized prefix nodes differ from the accepted source")
    cache_files = [onnx.helper.get_attribute_value(attribute).decode("utf-8")
                   for node in saved.graph.node for attribute in node.attribute
                   if node.op_type == "EPContext" and attribute.name == "ep_cache_context"]
    if cache_files != ["light-speed-gelu-both-ctx_qnn.bin"]:
        raise ValueError("cached binary reference differs")
    manifest = {
        "status": "accurate_cached_vision_prefix_static_checker_passed_pending_mixed_runtime_parity",
        "model": fingerprint(output_path), "source": fingerprint(source_path),
        "source_node_range_inclusive": [0, 72],
        "original_node_count": original_node_count, "candidate_node_count": len(saved.graph.node),
        "op_counts": dict(sorted(counts.items())),
        "cached_ep_and_custom_nodes_byte_identical": True,
        "cached_ep_binary_files_unchanged": cache_files,
        "inputs": [io(value) for value in saved.graph.input],
        "required_outputs": list(expected_io.values()),
        "original_output_count": original_output_count,
        "output_count": len(saved.graph.output),
        "outputs": [io(value) for value in saved.graph.output],
        "all_legal_prefix_outputs_retained_for_shared_prebinding": True,
        "feature_bridge": {"source": "view_70", "source_dtype": "FLOAT16", "source_shape": [1, 16384],
                           "policy_input": "view_39", "policy_dtype": "FLOAT", "policy_shape": [1, 32, 512],
                           "implementation": "runner reshapes and converts; no bridge operators inserted into this graph"},
        "state_ownership": {"image": "use prefix next_state_img_q",
                            "desire_and_feature": "use GPU policy next states; both sessions read the same pre-frame states",
                            "prefix_next_state_desire_q": "retained as a legal shared output; duplicate update is not fed into policy in the same frame"},
        "onnx_checker_in_memory_full_check": "passed", "onnx_checker_path_full_check": "passed",
        "static_topology_check": "passed", "serialization_node_equality_check": "passed",
        "device_access": "none", "new_weight_or_context_binary_created": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "nodes": len(saved.graph.node),
                      "inputs": manifest["inputs"], "required_outputs": manifest["required_outputs"],
                      "output_count": manifest["output_count"], "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
