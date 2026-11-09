"""Prepare one bounded-error, all-standard-ONNX GELU finite-domain probe.

Gelu(x) = Relu(x) + Gelu(-abs(x)). A shared positive INT32 index uses
floor(min(abs(x), 8) * 4096), with separate erf/tanh correction tables.
The graph performs no INT32 arithmetic beyond the FP16-to-INT32 cast.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/bounded-gelu"
SHAPE = [1, 31, 32, 64]
STEP = 4096
LAST_INDEX = 8 * STEP


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def gelu_double(value: float, tanh: bool) -> float:
    if tanh:
        return 0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) *
                                             (value + 0.044715 * value * value * value)))
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def make_correction_tables() -> dict[str, np.ndarray]:
    """The same table bytes used by the finite-domain probe and whole rewrite."""
    return {name: np.asarray([gelu_double(-float(k) / STEP, use_tanh)
                              for k in range(LAST_INDEX + 1)], dtype=np.float16)
            for name, use_tanh in (("erf", False), ("tanh", True))}


def build_ops(input_name: str, output_name: str, formula: str, prefix: str,
              flat_indices: bool = False, input_shape: list[int] | None = None) -> list[onnx.NodeProto]:
    """Seven standard nodes, or nine with the device-validated rank1 Gather path.

    All floating tensors are FP16. Only the index is INT32. `prefix` must be
    unique in the containing graph; original output names are preserved.
    """
    if formula not in ("erf", "tanh"):
        raise ValueError(f"unknown GELU formula: {formula}")
    absolute, clipped, scaled, index, relu, correction = [prefix + suffix for suffix in
                                                         ("_abs", "_clip", "_scale", "_index", "_relu", "_correction")]
    nodes = [
        helper.make_node("Abs", [input_name], [absolute], name=prefix + "_abs_node"),
        helper.make_node("Clip", [absolute, "jl_bounded_clip_min", "jl_bounded_clip_max"], [clipped], name=prefix + "_clip_node"),
        helper.make_node("Mul", [clipped, "jl_bounded_index_scale"], [scaled], name=prefix + "_scale_node"),
        helper.make_node("Cast", [scaled], [index], name=prefix + "_index_node", to=TensorProto.INT32),
        helper.make_node("Relu", [input_name], [relu], name=prefix + "_relu_node"),
    ]
    if flat_indices:
        if not input_shape or any(dimension <= 0 for dimension in input_shape):
            raise ValueError("flat Gather requires a fixed positive input shape")
        shape_name = "jl_bounded_shape_" + "x".join(str(dimension) for dimension in input_shape)
        flat_index, flat_correction = index + "__rank1", correction + "__rank1"
        nodes.extend([
            helper.make_node("Reshape", [index, "jl_bounded_flat_shape"], [flat_index], name=prefix + "_flatten_index"),
            helper.make_node("Gather", [f"jl_bounded_table_{formula}", flat_index], [flat_correction], name=prefix + "_gather_node", axis=0),
            helper.make_node("Reshape", [flat_correction, shape_name], [correction], name=prefix + "_restore_shape"),
        ])
    else:
        nodes.append(helper.make_node("Gather", [f"jl_bounded_table_{formula}", index], [correction], name=prefix + "_gather_node", axis=0))
    nodes.append(helper.make_node("Add", [relu, correction], [output_name], name=prefix + "_add_node"))
    return nodes


def rewrite_gelus(model: onnx.ModelProto, flat_indices: bool = False) -> dict:
    """Rewrite exact FP16 GELU custom nodes IN PLACE and return a change manifest.

    Call this on the original whole ONNX graph, not on an EPContext wrapper.
    Usage: changes = rewrite_gelus(model); checker.check_model(model); save(...).
    No weights or interfaces are changed. This function itself does not infer
    shapes or execute the graph. The caller owns full-model device validation.
    """
    if any(node.op_type == "EPContext" for node in model.graph.node):
        raise ValueError("rewrite_gelus requires the original graph, not compiled EPContext nodes")
    all_names = {value.name for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    all_names.update(tensor.name for tensor in model.graph.initializer)
    all_names.update(name for node in model.graph.node for name in list(node.input) + list(node.output))
    all_names.update(node.name for node in model.graph.node)
    if any(name.startswith("jl_bounded_") for name in all_names):
        raise ValueError("jl_bounded_ namespace already exists; refusing collisions or a repeated rewrite")
    types = {value.name: value.type.tensor_type.elem_type for value in
             list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
             if value.type.HasField("tensor_type")}
    shapes = {value.name: [dimension.dim_value if dimension.HasField("dim_value") else 0
                          for dimension in value.type.tensor_type.shape.dim]
              for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
              if value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape")}
    targets = [node for node in model.graph.node if node.domain == "jetlink.precision"
               and node.op_type in ("ExactGelu", "ExactGeluTanh")]
    if not targets:
        raise ValueError("no ExactGelu or ExactGeluTanh custom nodes found")
    for node in targets:
        if len(node.input) != 1 or len(node.output) != 1 or types.get(node.input[0]) != TensorProto.FLOAT16:
            raise ValueError(f"unsupported or untyped FP16 GELU signature: {node.name}")
    tables = make_correction_tables()
    new_initializers = [
        numpy_helper.from_array(np.asarray(0, dtype=np.float16), "jl_bounded_clip_min"),
        numpy_helper.from_array(np.asarray(8, dtype=np.float16), "jl_bounded_clip_max"),
        numpy_helper.from_array(np.asarray(STEP, dtype=np.float16), "jl_bounded_index_scale"),
        numpy_helper.from_array(tables["erf"], "jl_bounded_table_erf"),
        numpy_helper.from_array(tables["tanh"], "jl_bounded_table_tanh"),
    ]
    if flat_indices:
        new_initializers.append(numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "jl_bounded_flat_shape"))
        unique_shapes = sorted({tuple(shapes.get(node.input[0], ())) for node in targets})
        for input_shape in unique_shapes:
            if not input_shape or any(dimension <= 0 for dimension in input_shape):
                raise ValueError(f"unavailable fixed shape for flat GELU rewrite: {input_shape}")
            shape_name = "jl_bounded_shape_" + "x".join(str(dimension) for dimension in input_shape)
            new_initializers.append(numpy_helper.from_array(np.asarray(input_shape, dtype=np.int64), shape_name))
    nodes, replacements = [], []
    for index, node in enumerate(model.graph.node):
        if node.domain == "jetlink.precision" and node.op_type in ("ExactGelu", "ExactGeluTanh"):
            formula = "tanh" if node.op_type == "ExactGeluTanh" else "erf"
            prefix = f"jl_bounded_{index}_" + re.sub(r"[^A-Za-z0-9_]", "_", node.name)
            expanded = build_ops(node.input[0], node.output[0], formula, prefix,
                                 flat_indices=flat_indices, input_shape=shapes.get(node.input[0]))
            nodes.extend(expanded)
            replacements.append({"original_node": node.name, "original_op": node.op_type,
                                 "input": node.input[0], "output": node.output[0],
                                 "formula": formula, "new_nodes": [item.name for item in expanded]})
        else:
            # NodeProto references stay alive until copied by repeated-field extend.
            nodes.append(node)
    # Copy the retained nodes before clearing their parent repeated field.
    replacement_graph = onnx.GraphProto()
    replacement_graph.node.extend(nodes)
    del model.graph.node[:]
    model.graph.node.extend(replacement_graph.node)
    model.graph.initializer.extend(new_initializers)
    return {"rewritten_count": len(replacements),
            "erf_count": sum(item["formula"] == "erf" for item in replacements),
            "tanh_count": sum(item["formula"] == "tanh" for item in replacements),
            "new_standard_nodes_per_formula": 9 if flat_indices else 7,
            "flat_indices": flat_indices,
            "shared_initializer_names": [tensor.name for tensor in new_initializers],
            "shared_table_raw_bytes": int(sum(table.nbytes for table in tables.values())),
            "replacements": replacements,
            "acceptance": "Pending strict HTP finite-domain probe and full-model recurrent head parity; thresholds unchanged"}


def prepare_flat_probe() -> None:
    """Repair only the proven rank4 Gather layout path; reuse existing data."""
    source_path = OUT / "bounded-gelu-finite-domain.onnx"
    output_path = OUT / "bounded-gelu-finite-flat.onnx"
    source_manifest_path = REPORT / "bounded-gelu-manifest.json"
    manifest_path = REPORT / "bounded-gelu-flat-manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite bounded GELU flat evidence")
    source = onnx.load(source_path, load_external_data=False)
    model = copy.deepcopy(source)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if fingerprint(source_path)["sha256"] != source_manifest["model"]["sha256"]:
        raise ValueError("source finite-domain graph changed since original manifest")
    flat_shape = "jl_bounded_flat_indices_shape"
    output_shape = "jl_bounded_original_output_shape"
    flat_indices = "jl_bounded_flat_indices"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), flat_shape),
        numpy_helper.from_array(np.asarray(SHAPE, dtype=np.int64), output_shape),
    ])
    nodes, gather_count, flatten_count = [], 0, 0
    for original in source.graph.node:
        node = copy.deepcopy(original)
        if node.op_type == "Gather":
            if list(node.input)[1] != "index":
                raise ValueError(f"unexpected Gather indices: {node.name}")
            if not flatten_count:
                nodes.append(helper.make_node("Reshape", ["index", flat_shape], [flat_indices],
                                               name="jl_bounded_flatten_index_to_rank1"))
                flatten_count += 1
            original_output = node.output[0]
            flat_output = original_output + "__rank1"
            node.input[1] = flat_indices
            node.output[0] = flat_output
            nodes.append(node)
            nodes.append(helper.make_node("Reshape", [flat_output, output_shape], [original_output],
                                           name=node.name + "__restore_rank4"))
            gather_count += 1
        else:
            nodes.append(node)
    if gather_count != 2 or flatten_count != 1:
        raise ValueError("expected two correction gathers sharing one rank1 index")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.name = "bounded_gelu_finite_domain_rank1_gather"
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    if [value.SerializeToString() for value in source.graph.input] != [value.SerializeToString() for value in model.graph.input]:
        raise ValueError("input metadata changed")
    if [value.SerializeToString() for value in source.graph.output] != [value.SerializeToString() for value in model.graph.output]:
        raise ValueError("output metadata changed")
    info_by_name = {value.name: value for value in list(model.graph.value_info) + list(model.graph.output)}
    for node in model.graph.node:
        if node.op_type == "Gather":
            index_value = info_by_name[node.input[1]]
            index_dims = [dimension.dim_value for dimension in index_value.type.tensor_type.shape.dim]
            if index_dims != [math.prod(SHAPE)]:
                raise ValueError(f"Gather indices are not rank1: {index_dims}")
    onnx.save(model, output_path)
    onnx.checker.check_model(str(output_path), full_check=True)
    manifest = {
        "status": "rank1_gather_layout_repair_checker_passed_pending_device_execution",
        "source_model": fingerprint(source_path), "model": fingerprint(output_path),
        "input": source_manifest["input"], "input_dtype": "FLOAT16", "input_shape": SHAPE,
        "outputs": {"erf": {"dtype": "FLOAT16", "shape": SHAPE},
                    "tanh": {"dtype": "FLOAT16", "shape": SHAPE},
                    "index": {"dtype": "INT32", "shape": SHAPE}},
        "change": "Shared Reshape(index,[-1]) -> rank1 Gather -> Reshape to original rank4; floating arithmetic, table bytes and index calculation unchanged.",
        "reason": "Device rank4 probe had all indices correct but correction output errors up to about0.17. Prior rank1 Gather-only finite-domain diagnostic passed. This repair targets the demonstrated multidimensional Gather path.",
        "nodes": len(model.graph.node), "shared_index_reshape_count": 1,
        "gather_indices_shape": [math.prod(SHAPE)], "correction_output_reshape_count": 2,
        "flatten_note": "ONNX Flatten creates a rank2 tensor; this implementation deliberately uses Reshape([-1]) for actual rank1 indices.",
        "tables_unchanged": True, "index_calculation_unchanged": True,
        "input_output_metadata_unchanged": True,
        "original_reference_manifest": str(source_manifest_path.resolve()),
        "host_fp16_simulation": source_manifest["host_fp16_simulation"],
        "double_formula_to_fp16_references": source_manifest["double_formula_to_fp16_references"],
        "index": source_manifest["index"],
        "onnx_checker_full_check": "passed", "shape_inference_strict": "passed",
        "device_execution": "not performed by preparation script"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "nodes": len(model.graph.node),
                      "input": manifest["input"], "gather_indices_shape": manifest["gather_indices_shape"],
                      "manifest": str(manifest_path.resolve())}, indent=2))


def metrics(actual: np.ndarray, expected: np.ndarray, values: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(delta)
    maximum = int(np.argmax(absolute))
    reference_zero = expected == np.float16(0)
    return {"max_abs": float(absolute.max()),
            "rms": float(np.sqrt(np.mean(np.square(delta)))),
            "mean_abs": float(absolute.mean()),
            "nonfinite_count": int((~np.isfinite(actual)).sum()),
            "max_error_input": float(values[maximum]),
            "max_error_input_bits": f"0x{int(values.view(np.uint16)[maximum]):04x}",
            "max_error_actual": float(actual[maximum]),
            "max_error_reference": float(expected[maximum]),
            "value_mismatch_count": int((actual != expected).sum()),
            "signed_zero_bit_mismatches": int(((actual.view(np.uint16) != expected.view(np.uint16)) &
                                                reference_zero & (actual == np.float16(0))).sum())}


def main() -> None:
    model_path = OUT / "bounded-gelu-finite-domain.onnx"
    manifest_path = REPORT / "bounded-gelu-manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing bounded GELU evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / "inputs"
    reference = OUT / "reference"
    simulation = OUT / "host-fp16-simulation"
    for directory in (inputs, reference, simulation):
        directory.mkdir(parents=True, exist_ok=True)
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & np.uint16(0x7c00)) != np.uint16(0x7c00)]
    values = bits.view(np.float16)
    if values.size != math.prod(SHAPE) or not np.isfinite(values).all():
        raise ValueError("finite-domain enumeration does not match rank4 input shape")
    values.tofile(inputs / "0_x.bin")
    references = {}
    tables = make_correction_tables()
    for name, use_tanh in (("erf", False), ("tanh", True)):
        references[name] = np.asarray([gelu_double(float(value), use_tanh)
                                      for value in values], dtype=np.float16)
        references[name].tofile(reference / f"{name}.bin")
        tables[name].tofile(OUT / f"table-{name}.bin")

    # Every arithmetic stage below is explicitly FP16, like the ONNX graph.
    absolute = np.abs(values).astype(np.float16)
    clipped = np.clip(absolute, np.float16(0), np.float16(8)).astype(np.float16)
    scaled = np.multiply(clipped, np.float16(STEP), dtype=np.float16)
    indices = scaled.astype(np.int32)
    ideal_indices = np.floor(np.minimum(np.abs(values.astype(np.float64)), 8.0) * STEP).astype(np.int32)
    if not np.array_equal(indices, ideal_indices):
        raise ValueError("FP16 power-of-two scaling differs from mathematical floor index")
    if indices.min() < 0 or indices.max() > LAST_INDEX:
        raise ValueError("host simulation index exceeded table bounds")
    indices.tofile(reference / "index.bin")
    indices.tofile(simulation / "index.bin")
    relu = np.maximum(values, np.float16(0)).astype(np.float16)
    simulation_metrics = {}
    for name in ("erf", "tanh"):
        approximate = np.add(relu, tables[name][indices], dtype=np.float16)
        approximate.tofile(simulation / f"{name}.bin")
        simulation_metrics[name] = metrics(approximate, references[name], values)
        # The table quantization/truncation error can be distinguished from
        # the final FP16 addition by comparing a double sum to the reference.
        double_sum = relu.astype(np.float64) + tables[name][indices].astype(np.float64)
        simulation_metrics[name]["pre_add_round_max_abs_vs_fp16_reference"] = float(
            np.abs(double_sum - references[name].astype(np.float64)).max())

    initializers = [
        numpy_helper.from_array(np.asarray(0, dtype=np.float16), "clip_min"),
        numpy_helper.from_array(np.asarray(8, dtype=np.float16), "clip_max"),
        numpy_helper.from_array(np.asarray(STEP, dtype=np.float16), "index_scale"),
        numpy_helper.from_array(tables["erf"], "erf_correction_table"),
        numpy_helper.from_array(tables["tanh"], "tanh_correction_table"),
    ]
    nodes = [
        helper.make_node("Abs", ["x"], ["absolute"], name="bounded_abs"),
        helper.make_node("Clip", ["absolute", "clip_min", "clip_max"], ["clipped"], name="bounded_clip"),
        helper.make_node("Mul", ["clipped", "index_scale"], ["scaled"], name="bounded_scale"),
        helper.make_node("Cast", ["scaled"], ["index"], name="bounded_index_cast", to=TensorProto.INT32),
        helper.make_node("Relu", ["x"], ["relu"], name="bounded_relu"),
        helper.make_node("Gather", ["erf_correction_table", "index"], ["erf_correction"], name="bounded_erf_lookup", axis=0),
        helper.make_node("Add", ["relu", "erf_correction"], ["erf"], name="bounded_erf_add"),
        helper.make_node("Gather", ["tanh_correction_table", "index"], ["tanh_correction"], name="bounded_tanh_lookup", axis=0),
        helper.make_node("Add", ["relu", "tanh_correction"], ["tanh"], name="bounded_tanh_add"),
    ]
    graph = helper.make_graph(nodes, "bounded_gelu_finite_domain",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT16, SHAPE)],
                              [helper.make_tensor_value_info("erf", TensorProto.FLOAT16, SHAPE),
                               helper.make_tensor_value_info("tanh", TensorProto.FLOAT16, SHAPE),
                               helper.make_tensor_value_info("index", TensorProto.INT32, SHAPE)],
                              initializers)
    model = helper.make_model(graph, producer_name="jetlink-bounded-gelu-20261109",
                              opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, model_path)
    onnx.checker.check_model(str(model_path), full_check=True)
    manifest = {
        "status": "host_simulation_and_onnx_checker_passed_pending_strict_htp_execution",
        "model": fingerprint(model_path), "input": fingerprint(inputs / "0_x.bin"),
        "input_dtype": "FLOAT16", "input_shape": SHAPE,
        "finite_fp16_count": int(values.size),
        "enumeration": "ascending UINT16 bit patterns with exponent !=31; includes both signed zeros, excludes all NaN/Inf",
        "identity": "Gelu(x) = Relu(x) + Gelu(-abs(x)); separately preserves erf and tanh definitions",
        "table": {"samples_per_formula": LAST_INDEX + 1, "step_reciprocal": STEP,
                  "domain_abs": [0, 8], "dtype": "FLOAT16", "formulas": 2,
                  "total_raw_bytes": int(sum(table.nbytes for table in tables.values())),
                  "erf": fingerprint(OUT / "table-erf.bin"), "tanh": fingerprint(OUT / "table-tanh.bin")},
        "index": {"method": "FP16 Abs -> Clip(0,8) -> Mul(4096) -> Cast(INT32 truncation)",
                  "minimum": int(indices.min()), "maximum": int(indices.max()),
                  "table_last_index": LAST_INDEX, "out_of_bounds_count": 0,
                  "fp16_scaling_matches_double_floor_for_all_finite_inputs": True,
                  "int32_arithmetic_nodes": 0, "gather_indices_rank": 4,
                  "ideal_reference": fingerprint(reference / "index.bin")},
        "implementation": {"nodes": len(model.graph.node), "shared_nodes": ["Abs", "Clip", "Mul", "Cast", "Relu"],
                           "per_formula_nodes": ["Gather", "Add"],
                           "floating_intermediates": "all FLOAT16", "integer_intermediates": "index only, INT32",
                           "gather": "Direct rank4 nonnegative INT32 indices; ORT1.29 standard Gather builder has no rank1-only restriction and delegates final shape validation to QNN.",
                           "gather_builder_source": "https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/builder/opbuilder/gather_op_builder.cc"},
        "approximation_scope": {"grid_truncation_correction_bound_before_fp16_rounding": 0.5 / STEP,
                                "extra_rounding": "FP16 correction-table storage and final FP16 addition; full measured host errors below include both",
                                "tail": "abs(x)>8 uses correction at8; negligible tail correction rounds to signed zero in FP16",
                                "acceptance": "No model precision thresholds changed. Operator error and full-model recurrent error require device validation."},
        "host_fp16_simulation": simulation_metrics,
        "double_formula_to_fp16_references": {name: fingerprint(reference / f"{name}.bin") for name in ("erf", "tanh")},
        "host_simulated_outputs": {name: fingerprint(simulation / f"{name}.bin") for name in ("erf", "tanh", "index")},
        "onnx_checker_full_check": "passed", "shape_inference_strict": "passed",
        "device_execution": "not performed by preparation script"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "input": manifest["input"],
                      "input_shape": SHAPE, "table_raw_bytes": manifest["table"]["total_raw_bytes"],
                      "host_fp16_simulation": simulation_metrics,
                      "index_bounds": [int(indices.min()), int(indices.max())],
                      "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-indices", action="store_true",
                        help="prepare a new rank1 Gather layout repair using existing finite-domain artifacts")
    args = parser.parse_args()
    if args.flat_indices:
        prepare_flat_probe()
    else:
        main()
