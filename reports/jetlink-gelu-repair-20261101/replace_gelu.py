#!/usr/bin/env python3
"""Replace selected ONNX GELU nodes with Jetlink precision custom ops.

This only changes operator identity/domain and removes ONNX's `approximate`
attribute. Tensor names, edges, graph inputs/outputs, value_info and weights
are kept as-is. It does not implement the custom operator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto


CUSTOM_DOMAIN = "jetlink.precision"
CUSTOM_OPSET = 1
# Mirrors Jetlink's upstream Patches.imageInputs list. `vision` scope is the
# transitive dependency mask seeded by these graph input values, not a name
# filter on Gelu nodes.
IMAGE_INPUTS = ("img", "big_img", "new_img", "state_img_q")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def attribute(node, key):
    return next((a for a in node.attribute if a.name == key), None)


def attribute_string(attr) -> str | None:
    if attr is None:
        return None
    if attr.type != AttributeProto.STRING:
        return "<non-string>"
    return attr.s.decode("utf-8", errors="strict")


def tensor_metadata(model: onnx.ModelProto) -> tuple[dict[str, int], dict[str, list[int | str]]]:
    """Type metadata from the graph and ONNX shape inference, without mutating source."""
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), check_type=True, strict_mode=False)
    out: dict[str, int] = {}
    shapes: dict[str, list[int | str]] = {}
    for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]:
        if value.type.HasField("tensor_type"):
            tensor_type = value.type.tensor_type
            out[value.name] = tensor_type.elem_type
            shapes[value.name] = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                                  for dim in tensor_type.shape.dim]
    for tensor in inferred.graph.initializer:
        out[tensor.name] = tensor.data_type
        shapes[tensor.name] = list(tensor.dims)
    return out, shapes


def image_dependency_mask(model: onnx.ModelProto) -> tuple[set[int], list[str], list[str]]:
    """Mirror upstream Split.visionMask's all-data-inputs dependency rule."""
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {tensor.name for tensor in model.graph.initializer}
    seeded = sorted(graph_inputs.intersection(IMAGE_INPUTS))
    dependent_values = set(seeded)
    dependent_nodes: set[int] = set()
    # Upstream removes initializer names before testing nonempty data inputs;
    # it selects a node only when it has data inputs and every one is already
    # in visionTensors. Initializer-only nodes are therefore not selected.
    for index, node in enumerate(model.graph.node):
        data_inputs = [name for name in node.input if name and name not in initializers]
        if data_inputs and all(name in dependent_values for name in data_inputs):
            dependent_nodes.add(index)
            dependent_values.update(name for name in node.output if name)
    return dependent_nodes, seeded, sorted(dependent_values)


def ensure_custom_opset(model: onnx.ModelProto) -> dict:
    matches = [op for op in model.opset_import if op.domain == CUSTOM_DOMAIN]
    if len(matches) > 1:
        raise ValueError(f"duplicate opset imports for {CUSTOM_DOMAIN}")
    if matches:
        old = matches[0].version
        if old != CUSTOM_OPSET:
            matches[0].version = CUSTOM_OPSET
        return {"action": "set_existing", "old_version": old, "version": CUSTOM_OPSET}
    model.opset_import.append(onnx.helper.make_opsetid(CUSTOM_DOMAIN, CUSTOM_OPSET))
    return {"action": "added", "version": CUSTOM_OPSET}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="destination ONNX file")
    parser.add_argument("--scope", choices=("all", "vision"), required=True)
    parser.add_argument("--include-tanh", action="store_true",
                        help="also replace approximate='tanh' GELU nodes with ExactGeluTanh variants")
    parser.add_argument("--report", type=Path, help="JSON report path (defaults beside this script)")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    report_path = (args.report or (Path(__file__).resolve().parent / f"{output.stem}.json")).resolve()
    if not source.is_file():
        parser.error(f"source not found: {source}")
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")

    model = onnx.load(str(source), load_external_data=True)
    types, shapes = tensor_metadata(model)
    original_model = copy.deepcopy(model)
    image_nodes, seeded_image_inputs, image_values = image_dependency_mask(model)
    initial_node_count = len(model.graph.node)
    changes = []
    skipped = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "Gelu":
            continue
        input_name = node.input[0] if node.input else ""
        original_attr = attribute(node, "approximate")
        # ONNX Gelu defaults approximate to "none" if the attribute is omitted.
        approximate = attribute_string(original_attr) if original_attr is not None else "none (schema default)"
        exact_gelu = approximate in ("none", "none (schema default)")
        tanh_gelu = approximate == "tanh"
        if not exact_gelu and not (args.include_tanh and tanh_gelu):
            skipped.append({"index": index, "name": node.name, "reason": "approximate is not none",
                            "approximate": approximate})
            continue
        elem_type = types.get(input_name)
        if elem_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
            skipped.append({"index": index, "name": node.name, "reason": "input type is not statically FLOAT16/FLOAT",
                            "input": input_name, "elem_type": elem_type})
            continue
        if args.scope == "vision" and index not in image_nodes:
            skipped.append({"index": index, "name": node.name,
                            "reason": "not transitively dependent on upstream image_inputs",
                            "input": input_name})
            continue

        old_domain, old_op = node.domain, node.op_type
        if exact_gelu:
            new_op = "ExactGelu" if elem_type == TensorProto.FLOAT16 else "ExactGelu32"
        else:
            new_op = "ExactGeluTanh" if elem_type == TensorProto.FLOAT16 else "ExactGeluTanh32"
        kept = [attr for attr in node.attribute if attr.name != "approximate"]
        node.ClearField("attribute")
        node.attribute.extend(kept)
        node.domain = CUSTOM_DOMAIN
        node.op_type = new_op
        changes.append({
            "index": index, "name": node.name, "input": input_name,
            "input_elem_type": TensorProto.DataType.Name(elem_type),
            "input_shape": shapes.get(input_name),
            "outputs": list(node.output), "old_domain": old_domain, "old_op_type": old_op,
            "old_approximate": approximate, "new_domain": CUSTOM_DOMAIN, "new_op_type": new_op,
            "semantic_variant": "exact-erf" if exact_gelu else "tanh-approximation",
            "attribute_removed": original_attr is not None,
            "image_dependent": index in image_nodes,
        })

    if args.scope == "all" and not changes:
        raise SystemExit("no eligible Gelu nodes found for scope=all")
    opset_change = ensure_custom_opset(model) if changes else {"action": "unchanged_no_replacements"}
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))

    checker = {"status": "not_run", "message": None}
    try:
        onnx.checker.check_model(str(output))
        checker = {"status": "passed", "message": "ONNX structural/schema check passed"}
    except Exception as exc:  # custom operators have no ONNX semantic schema
        checker = {"status": "failed", "message": str(exc)}

    # Independently verify untouched graph interfaces and node wiring/metadata.
    saved = onnx.load(str(output), load_external_data=False)
    structural = {
        "graph_inputs_and_metadata_unchanged": [v.SerializeToString() for v in original_model.graph.input]
        == [v.SerializeToString() for v in saved.graph.input],
        "graph_outputs_and_metadata_unchanged": [v.SerializeToString() for v in original_model.graph.output]
        == [v.SerializeToString() for v in saved.graph.output],
        "graph_value_info_unchanged": [v.SerializeToString() for v in original_model.graph.value_info]
        == [v.SerializeToString() for v in saved.graph.value_info],
        "node_count_unchanged": initial_node_count == len(saved.graph.node),
        "replacement_edges_preserved": all(
            list(original_model.graph.node[item["index"]].input) == list(saved.graph.node[item["index"]].input)
            and list(original_model.graph.node[item["index"]].output) == list(saved.graph.node[item["index"]].output)
            and original_model.graph.node[item["index"]].name == saved.graph.node[item["index"]].name
            for item in changes
        ),
        "custom_opset_import_present": any(op.domain == CUSTOM_DOMAIN and op.version == CUSTOM_OPSET
                                            for op in saved.opset_import) if changes else True,
        "no_approximate_attr_on_replacements": all(
            attribute(saved.graph.node[item["index"]], "approximate") is None for item in changes
        ),
    }
    manifest = {
        "source": str(source), "source_sha256": sha256(source),
        "output": str(output), "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
        "scope": args.scope, "include_tanh": args.include_tanh,
        "custom_domain": CUSTOM_DOMAIN, "custom_opset": CUSTOM_OPSET,
        "upstream_image_inputs": list(IMAGE_INPUTS),
        "image_dependency_seeded_inputs_present_in_graph": seeded_image_inputs,
        "image_dependent_value_count": len(image_values), "image_dependent_node_count": len(image_nodes),
        "image_dependency_mask_note": "matches upstream Split.visionMask: remove initializer inputs, require at least one nonempty data input, then require every data input to be in visionTensors; initializer-only nodes are excluded",
        "replacements": changes, "replacement_count": len(changes), "skipped_gelu_nodes": skipped,
        "custom_opset_change": opset_change, "onnx_checker": checker,
        "structural_invariants": structural,
        "semantic_check_note": "ONNX checker cannot validate custom Jetlink operator semantics or the CPU ExactGelu, ExactGelu32, ExactGeluTanh, or ExactGeluTanh32 implementations.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "replacements": len(changes), "checker": checker,
                      "structural_invariants": structural, "report": str(report_path)}, indent=2))
    return 0 if all(structural.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
