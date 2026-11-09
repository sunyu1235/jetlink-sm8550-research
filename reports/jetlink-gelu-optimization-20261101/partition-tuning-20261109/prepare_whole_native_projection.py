"""Build a thin, all-standard native-GELU projection whole-model candidate.

Large float weights reference verified raw_data byte ranges in the existing
bounded-GELU ONNX container. No weight file is copied and no device is accessed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import mmap
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from prepare_whole_bounded_gelu import topological_sort


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
SOURCE_SHA256 = "c5681e60028d8ee9f488195809342acacfcd2134b25527b3d5f0a1913eab88f4"
THRESHOLD_NAME = "jl_native_projection_tail_threshold"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def values(items) -> list[bytes]:
    return [item.SerializeToString() for item in items]


def projection(target, index: int) -> tuple[list, dict]:
    formula = "tanh" if target.op_type == "ExactGeluTanh" else "erf"
    prefix = f"jl_native_projection_{index}_" + re.sub(r"[^A-Za-z0-9_]", "_", target.name)
    x, y = target.input[0], target.output[0]
    raw, relu, bounded, absolute, tail = [prefix + suffix for suffix in
                                        ("_raw", "_relu", "_bounded", "_abs", "_tail")]
    nodes = [
        helper.make_node("Gelu", [x], [raw], name=prefix + "_gelu_node",
                         approximate="tanh" if formula == "tanh" else "none"),
        helper.make_node("Relu", [x], [relu], name=prefix + "_relu_node"),
        helper.make_node("Min", [raw, relu], [bounded], name=prefix + "_bound_node"),
        helper.make_node("Abs", [x], [absolute], name=prefix + "_abs_node"),
        helper.make_node("Greater", [absolute, THRESHOLD_NAME], [tail], name=prefix + "_tail_node"),
        helper.make_node("Where", [tail, relu, bounded], [y], name=prefix + "_select_node"),
    ]
    return nodes, {"original_node": target.name, "original_op": target.op_type,
                   "input": x, "output": y, "formula": formula,
                   "new_nodes": [node.name for node in nodes]}


def main() -> None:
    source_path = OUT / "whole-last-query-bounded-gelu.onnx"
    target_path = ROOT / "outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both-inline-shapes.onnx"
    output_path = OUT / "whole-native-projection-thin.onnx"
    temporary_path = OUT / "whole-native-projection-thin.tmp.onnx"
    manifest_path = REPORT / "whole-native-projection-manifest.json"
    if any(path.exists() for path in (output_path, temporary_path, manifest_path)):
        raise FileExistsError("refusing to overwrite native projection candidate/evidence")
    source = fingerprint(source_path)
    if source["sha256"] != SOURCE_SHA256:
        raise ValueError("existing whole-model weight container hash differs")
    source_manifest = json.loads((REPORT / "whole-last-query-bounded-gelu-manifest.json").read_text(encoding="utf-8"))
    target_model = onnx.load(target_path, load_external_data=False)
    targets = [node for node in target_model.graph.node if node.domain == "jetlink.precision"
               and node.op_type in ("ExactGelu", "ExactGeluTanh")]
    if len(targets) != 63 or collections.Counter(node.op_type for node in targets) != {
            "ExactGelu": 36, "ExactGeluTanh": 27}:
        raise ValueError("unexpected exact-GELU source signatures")
    expected = {(item["original_node"], item["input"], item["output"], item["formula"])
                for item in source_manifest["gelu_rewrite"]["replacements"]}
    actual = {(node.name, node.input[0], node.output[0],
               "tanh" if node.op_type == "ExactGeluTanh" else "erf") for node in targets}
    if actual != expected or any(len(node.input) != 1 or len(node.output) != 1 for node in targets):
        raise ValueError("whole LUT boundaries differ from original exact GELUs")
    print("loading existing whole container", flush=True)
    model = onnx.load(source_path, load_external_data=False)
    public_inputs, public_outputs = values(model.graph.input), values(model.graph.output)
    metadata, opsets, ir_version = values(model.metadata_props), values(model.opset_import), model.ir_version
    old_node_count, old_initializer_count = len(model.graph.node), len(model.graph.initializer)
    if any(node.name.startswith("jl_native_projection_") for node in model.graph.node):
        raise ValueError("native projection prefix already in use")
    if THRESHOLD_NAME in {item.name for item in model.graph.initializer}:
        raise ValueError("native projection threshold initializer already exists")
    retained_nodes = {node.name: node.SerializeToString() for node in model.graph.node
                      if not node.name.startswith("jl_bounded_")}
    deleted_nodes = [node.name for node in model.graph.node if node.name.startswith("jl_bounded_")]
    if len(deleted_nodes) != 63 * 9 or len(retained_nodes) != old_node_count - len(deleted_nodes):
        raise ValueError("unexpected bounded-GELU node count or duplicate retained names")
    replacement_graph = onnx.GraphProto()
    replacement_graph.node.extend(node for node in model.graph.node if not node.name.startswith("jl_bounded_"))
    del model.graph.node[:]
    model.graph.node.extend(replacement_graph.node)
    del replacement_graph
    removed_initializers = [tensor.name for tensor in model.graph.initializer if tensor.name.startswith("jl_bounded_")]
    # Copy only retained small protobufs after externalizing the large raw bytes.
    external, inline, search_start = [], [], 0
    with source_path.open("rb") as source_stream, mmap.mmap(source_stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for original_index, tensor in enumerate(model.graph.initializer):
            if tensor.name.startswith("jl_bounded_"):
                continue
            raw = tensor.raw_data
            item = {"name": tensor.name, "shape": list(tensor.dims),
                    "dtype": TensorProto.DataType.Name(tensor.data_type), "data_type": tensor.data_type,
                    "original_initializer_index": original_index}
            if tensor.data_location == TensorProto.EXTERNAL:
                raise ValueError(f"source weight is already external: {tensor.name}")
            if tensor.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16) and len(raw) >= 16 * 1024:
                offset = mapped.find(raw, search_start)
                if offset < 0 or mapped[offset:offset + len(raw)] != raw:
                    raise ValueError(f"raw bytes not found in initializer order: {tensor.name}")
                item.update(location=source_path.name, offset=offset, length=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest())
                external.append(item)
                search_start = offset + len(raw)
                external_data_helper.set_external_data(tensor, location=source_path.name,
                                                       offset=offset, length=len(raw))
                tensor.ClearField("raw_data")
            else:
                item.update(raw_bytes=len(raw), protobuf_sha256=hashlib.sha256(tensor.SerializeToString()).hexdigest())
                inline.append(item)
        retained_initializers = onnx.GraphProto()
        retained_initializers.initializer.extend(tensor for tensor in model.graph.initializer
                                                if not tensor.name.startswith("jl_bounded_"))
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained_initializers.initializer)
        del retained_initializers
        # Recheck every referenced range separately against its recorded digest.
        for item in external:
            if hashlib.sha256(mapped[item["offset"]:item["offset"] + item["length"]]).hexdigest() != item["sha256"]:
                raise ValueError(f"external byte range hash mismatch: {item['name']}")
    kept_info = onnx.GraphProto()
    kept_info.value_info.extend(value for value in model.graph.value_info if not value.name.startswith("jl_bounded_"))
    removed_value_info_count = len(model.graph.value_info) - len(kept_info.value_info)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info.value_info)
    del kept_info
    replacements = []
    for index, target in enumerate(targets):
        nodes, change = projection(target, index)
        model.graph.node.extend(nodes)
        replacements.append(change)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(4, dtype=np.float16), THRESHOLD_NAME))
    topological_sort(model)
    if any(node.domain not in ("", "ai.onnx") or node.op_type == "EPContext" for node in model.graph.node):
        raise ValueError("custom/EPContext node remains")
    if any(node.name.startswith("jl_bounded_") or any(name.startswith("jl_bounded_") for name in list(node.input) + list(node.output))
           for node in model.graph.node):
        raise ValueError("bounded-GELU graph remnant remains")
    final_nodes = {node.name: node.SerializeToString() for node in model.graph.node}
    if any(final_nodes.get(name) != serialized for name, serialized in retained_nodes.items()):
        raise ValueError("a retained non-GELU node changed")
    last_query_path = OUT / "last-query-partition-131-41.onnx"
    last_query = onnx.load(last_query_path, load_external_data=False)
    if len(last_query.graph.node) != 120 or any(final_nodes.get(node.name) != node.SerializeToString() for node in last_query.graph.node):
        raise ValueError("validated 120-node last-query graph changed")
    print(f"strictly inferring {len(model.graph.node)} nodes; {len(external)} external weights", flush=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if values(model.graph.input) != public_inputs or values(model.graph.output) != public_outputs:
        raise ValueError("public input/output metadata changed")
    if values(model.metadata_props) != metadata or values(model.opset_import) != opsets or model.ir_version != ir_version:
        raise ValueError("metadata/IR/opsets changed")
    if any(tensor.raw_data for tensor in model.graph.initializer if tensor.data_location == TensorProto.EXTERNAL):
        raise ValueError("external weight unexpectedly has inline raw_data")
    onnx.save_model(model, temporary_path)
    onnx.checker.check_model(str(temporary_path), full_check=True)
    temporary_path.replace(output_path)
    manifest = {
        "status": "whole_native_projection_thin_strict_shape_and_checker_passed_pending_full_model_htp_parity",
        "model": fingerprint(output_path), "source_container": source, "gelu_signature_source": fingerprint(target_path),
        "external_data": {"location": source_path.name, "existing_container_reused": True,
                          "new_weight_file_created": False, "minimum_raw_bytes": 16 * 1024,
                          "tensor_count": len(external), "total_raw_bytes": sum(item["length"] for item in external),
                          "range_bytes_and_sha256_verified": True, "tensors": external},
        "inline_retained_initializers": inline,
        "original_node_count": old_node_count, "candidate_node_count": len(model.graph.node),
        "original_initializer_count": old_initializer_count, "candidate_initializer_count": len(model.graph.initializer),
        "removed_bounded_node_count": len(deleted_nodes), "removed_bounded_initializers": removed_initializers,
        "removed_bounded_value_info_count": removed_value_info_count,
        "gelu_rewrite": {"rewritten_count": len(replacements), "erf_count": 36, "tanh_count": 27,
                         "nodes_per_formula": 6, "definition": "raw=Gelu(x,original approximate); r=Relu(x); b=Min(raw,r); tail=Abs(x)>FLOAT16(4); y=Where(tail,r,b)",
                         "threshold_initializer": THRESHOLD_NAME, "replacements": replacements},
        "op_counts": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "native_projection_gather_nodes": 0, "custom_nodes": 0, "epcontext_nodes": 0,
        "public_inputs_preserved": True, "public_outputs_preserved": True,
        "public_output_names": [value.name for value in model.graph.output],
        "metadata_ir_opsets_preserved": True, "retained_non_gelu_nodes_byte_identical": True,
        "last_query_node_count_preserved": len(last_query.graph.node),
        "topological_sort": "passed", "onnx_shape_inference_strict": "passed", "onnx_checker_full_check": "passed",
        "device_access": "none", "full_model_precision_and_timing": "pending actual HTP execution; original per-head thresholds unchanged",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "nodes": len(model.graph.node),
                      "external_tensors": len(external), "external_raw_bytes": manifest["external_data"]["total_raw_bytes"],
                      "manifest": str(manifest_path.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
