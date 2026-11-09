#!/usr/bin/env python3
"""Extract one exact node of each requested type from the FP16 block.

The full block runs once on CPU with selected intermediates exposed. Each
diagnostic model then contains exactly one copied ONNX node and its original
constant initializers. Dynamic input tensors are those captured intermediates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_value_info_map(model):
    return {v.name: v for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}


def select_nodes(model):
    conv, layer_norm, gelu, matmuls = None, None, None, []
    for node in model.graph.node:
        if node.op_type == "Conv" and conv is None:
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            weight = next((t for t in model.graph.initializer if t.name == node.input[1]), None)
            if weight is not None and group == weight.dims[0] and weight.dims[1] == 1:
                conv = node
        elif node.op_type == "LayerNormalization" and layer_norm is None:
            layer_norm = node
        elif node.op_type == "MatMul":
            matmuls.append(node)
        elif node.op_type == "Gelu" and gelu is None:
            gelu = node
    selected = {"depthwise_conv": conv, "layer_normalization": layer_norm,
                "matmul_1": matmuls[0] if matmuls else None,
                "matmul_2": matmuls[1] if len(matmuls) > 1 else None,
                "gelu": gelu}
    missing = [k for k, v in selected.items() if v is None]
    if missing:
        raise SystemExit(f"requested node not found in block: {missing}")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is nonempty; refusing overwrite: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    raw_model = onnx.load(str(args.model), load_external_data=True)
    model = onnx.shape_inference.infer_shapes(raw_model, check_type=True, strict_mode=True)
    onnx.checker.check_model(model)
    meta = tensor_value_info_map(model)
    selected = select_nodes(model)
    init_names = {t.name for t in model.graph.initializer}

    input_vi = next((v for v in model.graph.input if v.name == "add_34"), None)
    if input_vi is None:
        raise SystemExit("block input add_34 not found")
    in_shape = tuple(d.dim_value for d in input_vi.type.tensor_type.shape.dim)
    in_type = input_vi.type.tensor_type.elem_type
    in_dtype = np.float16 if in_type == onnx.TensorProto.FLOAT16 else np.float32
    x = np.fromfile(args.input, dtype=in_dtype)
    if x.size != int(np.prod(in_shape)):
        raise SystemExit(f"block input has {x.size} elements; expected {int(np.prod(in_shape))}")
    x = np.ascontiguousarray(x.reshape(in_shape))

    # Expose all dynamic tensors entering and leaving selected nodes during one
    # CPU run of the unchanged extracted block.
    dynamic_inputs = {}
    desired_outputs = []
    model_input_names = {v.name for v in model.graph.input}
    dynamic_capture_outputs = []
    for node in selected.values():
        for name in node.input:
            if name and name not in init_names and name not in dynamic_inputs:
                if name not in meta:
                    raise SystemExit(f"shape inference did not annotate input tensor {name}")
                dynamic_inputs[name] = copy.deepcopy(meta[name])
                if name not in model_input_names and name not in dynamic_capture_outputs:
                    dynamic_capture_outputs.append(name)
        for name in node.output:
            if name and name not in desired_outputs:
                if name not in meta:
                    raise SystemExit(f"shape inference did not annotate output tensor {name}")
                desired_outputs.append(name)
    for name in [*desired_outputs, *dynamic_capture_outputs]:
        if name not in {v.name for v in model.graph.output}:
            model.graph.output.append(copy.deepcopy(meta[name]))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    full = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    requested_outputs = list(dict.fromkeys(["outputs", *desired_outputs, *dynamic_capture_outputs]))
    fetched = full.run(requested_outputs, {"add_34": x})
    full_values = dict(zip(requested_outputs, fetched, strict=True))
    for name in model_input_names:
        full_values[name] = x if name == "add_34" else None
    del full

    manifest = {
        "source_block": str(args.model.resolve()), "source_sha256": sha(args.model),
        "input_file": str(args.input.resolve()), "input_sha256": sha(args.input),
        "input_shape": list(in_shape), "input_dtype": str(x.dtype),
        "shape_inference": "onnx.shape_inference.infer_shapes(check_type=True, strict_mode=True)",
        "cpu_reference": {"provider": "CPUExecutionProvider", "intra_op_threads": 4},
        "operators": {},
    }
    initializers = {t.name: t for t in model.graph.initializer}
    for label, original_node in selected.items():
        dynamic = [name for name in original_node.input if name and name not in init_names]
        if len(original_node.output) != 1:
            raise SystemExit(f"{label} has {len(original_node.output)} outputs; expected a single-output op")
        missing = [name for name in dynamic if name not in full_values]
        if missing:
            raise SystemExit(f"CPU capture missed {label} inputs: {missing}")
        captured = {name: np.ascontiguousarray(full_values[name]) for name in dynamic}
        full_ref = np.ascontiguousarray(full_values[original_node.output[0]])

        opdir = args.out / label
        opdir.mkdir()
        ins = [copy.deepcopy(meta[name]) for name in dynamic]
        outs = copy.deepcopy(meta[original_node.output[0]])
        outs.name = "outputs"
        node = copy.deepcopy(original_node)
        node.output[0] = "outputs"
        used_initializers = [copy.deepcopy(initializers[name]) for name in original_node.input if name in init_names]
        mini_graph = helper.make_graph([node], f"one_{label}", ins, [outs], initializer=used_initializers)
        mini = helper.make_model(mini_graph, opset_imports=copy.deepcopy(model.opset_import))
        mini.ir_version = model.ir_version
        mini.producer_name = "jetlink-operator-audit"
        onnx.checker.check_model(mini)
        model_path = opdir / "operator.onnx"
        onnx.save(mini, str(model_path))
        inputs_dir = opdir / "inputs"
        inputs_dir.mkdir()
        input_manifest = []
        feeds = {}
        for name, value in captured.items():
            data_path = inputs_dir / f"0_{name}.bin"
            value.tofile(data_path)
            feeds[name] = value
            input_manifest.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                   "bytes": data_path.stat().st_size, "sha256": sha(data_path)})
        cpu_mini = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        mini_ref = np.ascontiguousarray(cpu_mini.run(None, feeds)[0])
        np.save(opdir / "ref.npy", mini_ref)
        mini_ref.tofile(opdir / "ref.bin")
        np.save(opdir / "full_block_node_ref.npy", full_ref)
        mini_minus_full = mini_ref.astype(np.float64) - full_ref.astype(np.float64)
        attrs = {}
        for attr in original_node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = int(attr.i)
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = float(attr.f)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8", errors="replace")
        manifest["operators"][label] = {
            "node_name": original_node.name, "op_type": original_node.op_type,
            "attributes": attrs, "dynamic_inputs": input_manifest,
            "output_shape": list(mini_ref.shape), "output_dtype": str(mini_ref.dtype),
            "model_bytes": model_path.stat().st_size, "model_sha256": sha(model_path),
            "cpu_ref_bytes": (opdir / "ref.bin").stat().st_size, "cpu_ref_sha256": sha(opdir / "ref.bin"),
            "full_block_ref_sha256": sha(opdir / "full_block_node_ref.npy"),
            "mini_cpu_vs_full_node_max_abs": float(np.abs(mini_minus_full).max(initial=0)),
            "mini_cpu_vs_full_node_rmse": float(np.sqrt(np.mean(mini_minus_full ** 2))),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: {"node": v["node_name"], "inputs": [i["name"] for i in v["dynamic_inputs"]],
                         "shape": v["output_shape"], "dtype": v["output_dtype"]}
                      for k, v in manifest["operators"].items()}, indent=2))


if __name__ == "__main__":
    main()
