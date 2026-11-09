"""Expose accepted cached boundaries for preallocated shared I/O.

Only graph output metadata is appended. Original nodes, inputs, outputs, and
relative QNN context references are preserved. No model compilation or phone
access occurs. Deploy this wrapper in /data/local/tmp beside the existing cache
binary, retaining the exact relative basename recorded in this manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
SOURCE_CONTEXT = ROOT / "outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both-ctx.onnx"


def file_info(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=SOURCE_CONTEXT)
    parser.add_argument("--all-boundaries", action="store_true",
                        help="expose all cached context value_info tensors, including FP32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT_DIR / ("whole-context-shared-all-boundaries.onnx" if args.all_boundaries
                                    else "whole-context-shared-gelu-boundaries.onnx")
    if args.manifest is None:
        args.manifest = REPORT_DIR / ("shared-all-boundaries-manifest.json" if args.all_boundaries
                                      else "shared-boundaries-manifest.json")
    for path in (args.output, args.manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    source = onnx.load(args.context, load_external_data=False)
    model = copy.deepcopy(source)
    values = {value.name: value for value in
              list(source.graph.input) + list(source.graph.value_info) + list(source.graph.output)}
    custom = [node for node in source.graph.node if node.domain == "jetlink.precision"
              and node.op_type in ("ExactGelu", "ExactGeluTanh")]
    if len(custom) != 63:
        raise ValueError(f"expected 63 exact CPU GELU nodes, found {len(custom)}")
    input_names = {value.name for value in source.graph.input}
    original_output_names = [value.name for value in source.graph.output]
    if args.all_boundaries:
        boundaries = list(dict.fromkeys(value.name for value in source.graph.value_info
                                        if value.name not in input_names and value.name not in original_output_names))
    else:
        boundaries = list(dict.fromkeys(name for node in custom for name in list(node.input) + list(node.output)))
    names, skipped = [], []
    roles = {}
    for node in custom:
        for name in node.input:
            roles.setdefault(name, []).append({"node": node.name, "role": "gelu_input", "op_type": node.op_type})
        for name in node.output:
            roles.setdefault(name, []).append({"node": node.name, "role": "gelu_output", "op_type": node.op_type})
    tensor_manifest = []
    for name in boundaries:
        if name not in values:
            skipped.append(name)
            continue
        value = values[name]
        if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
            raise ValueError(f"missing tensor shape: {name}")
        tensor = value.type.tensor_type
        allowed_types = (TensorProto.FLOAT16, TensorProto.FLOAT) if args.all_boundaries else (TensorProto.FLOAT16,)
        if tensor.elem_type not in allowed_types:
            raise ValueError(f"unexpected boundary dtype {tensor.elem_type}: {name}")
        shape = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise ValueError(f"dynamic boundary shape: {name}")
            shape.append(dimension.dim_value)
        if name not in original_output_names:
            model.graph.output.append(copy.deepcopy(value))
            names.append(name)
        element_bytes = 2 if tensor.elem_type == TensorProto.FLOAT16 else 4
        tensor_manifest.append({"name": name, "elem_type": tensor.elem_type,
                                "dtype": TensorProto.DataType.Name(tensor.elem_type), "shape": shape,
                                "bytes": math.prod(shape) * element_bytes,
                                "roles": roles.get(name, [{"role": "cached_context_intermediate"}])})
    if skipped:
        raise ValueError(f"accepted context lacks boundary value_info: {skipped}")
    expected_boundary_count = 211 if args.all_boundaries else 126
    if len(names) != expected_boundary_count:
        raise ValueError(f"expected {expected_boundary_count} distinct appended boundaries, found {len(names)}")
    cache_locations = sorted({attribute.s.decode("utf-8")
                              for node in source.graph.node if node.op_type == "EPContext"
                              for attribute in node.attribute
                              if attribute.name == "ep_cache_context" and attribute.s})
    if not cache_locations or any("/" in name or "\\" in name for name in cache_locations):
        raise ValueError(f"expected relative cache basenames: {cache_locations}")
    if [node.SerializeToString() for node in source.graph.node] != [node.SerializeToString() for node in model.graph.node]:
        raise ValueError("original context node metadata changed")
    if [value.SerializeToString() for value in source.graph.input] != [value.SerializeToString() for value in model.graph.input]:
        raise ValueError("graph inputs changed")
    if [value.SerializeToString() for value in source.graph.output] != [value.SerializeToString() for value in list(model.graph.output)[:len(source.graph.output)]]:
        raise ValueError("original graph outputs changed")
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    onnx.checker.check_model(str(args.output))
    manifest = {
        "purpose": "prebind accepted cached context boundaries to QnnHtpShared allocations through graph output I/O binding",
        "boundary_mode": "all_cached_value_info" if args.all_boundaries else "exact_cpu_gelu_inputs_and_outputs",
        "status": "onnx_checker_passed_no_phone_execution",
        "source_context": file_info(args.context), "wrapper": file_info(args.output),
        "custom_gelu_nodes": len(custom), "original_outputs": original_output_names,
        "appended_output_count": len(names), "output_count_total": len(model.graph.output),
        "appended_output_names": names, "boundary_tensors": tensor_manifest,
        "boundary_total_bytes": sum(item["bytes"] for item in tensor_manifest),
        "boundary_total_mib": sum(item["bytes"] for item in tensor_manifest) / (1024 * 1024),
        "cache_locations_preserved": cache_locations,
        "deployment": "Put the wrapper directly in /data/local/tmp beside the existing QNN cache binary, preserving the relative basename above. The phone rejected hard links and absolute ep_cache_context paths; do not create any links.",
        "original_nodes_byte_identical": True, "original_inputs_byte_identical": True,
        "original_outputs_preserved_in_order": True, "onnx_checker": "passed",
        "formula_or_weight_changes": "none",
        "scope": "Metadata preparation only; zero-copy placement and performance require device verification."
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("wrapper", "appended_output_count", "output_count_total",
                     "boundary_total_bytes", "boundary_total_mib", "cache_locations_preserved")}, indent=2))


if __name__ == "__main__":
    main()
