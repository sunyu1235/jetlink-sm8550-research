#!/usr/bin/env python3
"""Replace QNN-ignored tanh-approximate GELU nodes with explicit FP32 ops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


DEFAULT_SOURCE = Path(r"C:\Users\sunbo\Documents\moto-x40-root\policy-fp32.onnx")
DEFAULT_OUTPUT = Path(
    "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/mixed-policy-fp32-tanh.onnx"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_gelu(model: onnx.ModelProto) -> int:
    gelus = [node for node in model.graph.node if node.op_type == "Gelu"]
    if len(gelus) != 27:
        raise ValueError(f"Expected exactly 27 Gelu nodes, found {len(gelus)}")
    for node in gelus:
        approximate = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "approximate"),
            b"none",
        )
        if approximate != b"tanh":
            raise ValueError(f"Gelu node {node.name!r} has approximate={approximate!r}, expected b'tanh'")

    constants = {
        "mixed_gelu_cubic_coefficient": np.asarray(0.044715, dtype=np.float32),
        "mixed_gelu_sqrt_2_over_pi": np.asarray(math.sqrt(2.0 / math.pi), dtype=np.float32),
        "mixed_gelu_one": np.asarray(1.0, dtype=np.float32),
        "mixed_gelu_half": np.asarray(0.5, dtype=np.float32),
    }
    for name, value in constants.items():
        model.graph.initializer.append(helper.make_tensor(name, TensorProto.FLOAT, [], [float(value)]))

    new_nodes = []
    for node in model.graph.node:
        if node.op_type != "Gelu":
            new_nodes.append(node)
            continue

        x = node.input[0]
        y = node.output[0]
        tag = node.name or y
        names = {suffix: f"{tag}__explicit_tanh_gelu_{suffix}" for suffix in (
            "x2", "x3", "cubic", "inner", "scaled", "tanh", "plus_one", "half_x"
        )}
        new_nodes.extend([
            helper.make_node("Mul", [x, x], [names["x2"]], name=f"{tag}__gelu_x_squared"),
            helper.make_node("Mul", [names["x2"], x], [names["x3"]], name=f"{tag}__gelu_x_cubed"),
            helper.make_node("Mul", [names["x3"], "mixed_gelu_cubic_coefficient"], [names["cubic"]], name=f"{tag}__gelu_cubic_term"),
            helper.make_node("Add", [x, names["cubic"]], [names["inner"]], name=f"{tag}__gelu_inner_sum"),
            helper.make_node("Mul", [names["inner"], "mixed_gelu_sqrt_2_over_pi"], [names["scaled"]], name=f"{tag}__gelu_scale"),
            helper.make_node("Tanh", [names["scaled"]], [names["tanh"]], name=f"{tag}__gelu_tanh"),
            helper.make_node("Add", ["mixed_gelu_one", names["tanh"]], [names["plus_one"]], name=f"{tag}__gelu_add_one"),
            helper.make_node("Mul", [x, names["plus_one"]], [names["half_x"]], name=f"{tag}__gelu_x_times_sum"),
            helper.make_node("Mul", [names["half_x"], "mixed_gelu_half"], [y], name=f"{tag}__gelu_half_scale"),
        ])

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return len(gelus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("mixed-policy-manifest.json"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    model = onnx.load(str(source), load_external_data=True)
    original_node_count = len(model.graph.node)
    original_initializer_count = len(model.graph.initializer)
    graph_inputs = [value.name for value in model.graph.input]
    graph_outputs = [value.name for value in model.graph.output]
    gelu_count = replace_gelu(model)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, str(output))

    # Check the serialized artifact too, including its embedded tensors.
    onnx.checker.check_model(str(output), full_check=True)
    final = onnx.load(str(output), load_external_data=True)
    if any(node.op_type == "Gelu" for node in final.graph.node):
        raise RuntimeError("Serialized model still contains Gelu nodes")
    if [value.name for value in final.graph.input] != graph_inputs or [value.name for value in final.graph.output] != graph_outputs:
        raise RuntimeError("Model I/O changed during conversion")

    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "conversion": "replace 27 Gelu(approximate=tanh) with explicit FP32 scalar arithmetic",
        "formula": "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x*x*x)))",
        "gelu_nodes_replaced": gelu_count,
        "gelu_nodes_remaining": 0,
        "source_graph_nodes": original_node_count,
        "output_graph_nodes": len(final.graph.node),
        "source_initializers": original_initializer_count,
        "output_initializers": len(final.graph.initializer),
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "other_transformations": [],
        "external_weights": False,
        "onnx_checker": "passed (in-memory and serialized model, full_check=True)",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
