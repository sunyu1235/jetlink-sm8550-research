#!/usr/bin/env python3
"""Add CPU FP32 math references for the already captured FP16 op inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(a, ref):
    d = a.astype(np.float64).ravel() - ref.astype(np.float64).ravel()
    rmse = float(np.sqrt(np.mean(d * d)))
    spread = float(ref.astype(np.float64).std())
    return {"mae": float(np.mean(np.abs(d))), "max_abs": float(np.max(np.abs(d), initial=0)),
            "rmse": rmse, "normalized_rmse": rmse / spread if spread else None,
            "ref_std": spread, "elements": int(ref.size)}


def main():
    root = Path(__file__).resolve().parents[2] / "outputs" / "jetlink-operator-audit-20261101"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for label, record in manifest["operators"].items():
        opdir = root / label
        model_path = opdir / "operator.onnx"
        model = onnx.load(str(model_path))
        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                vi.type.tensor_type.elem_type = TensorProto.FLOAT
        for init in model.graph.initializer:
            if init.data_type == TensorProto.FLOAT16:
                init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                    attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float32), attr.t.name))
        fp32_model_path = opdir / "operator-fp32-math.onnx"
        onnx.checker.check_model(model)
        onnx.save(model, str(fp32_model_path))
        feeds = {}
        fp32_input_dir = opdir / "inputs-fp32-math"
        fp32_input_dir.mkdir(exist_ok=False)
        for input_info in model.graph.input:
            dims = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
            dtype = np.float32
            src_path = opdir / "inputs" / f"0_{input_info.name}.bin"
            raw = np.fromfile(src_path, dtype=np.float16)
            if raw.size != int(np.prod(dims)):
                raise SystemExit(f"{src_path}: unexpected element count")
            val = np.ascontiguousarray(raw.reshape(dims).astype(dtype))
            dst_path = fp32_input_dir / f"0_{input_info.name}.bin"
            val.tofile(dst_path)
            feeds[input_info.name] = val
        session = ort.InferenceSession(str(fp32_model_path), so, providers=["CPUExecutionProvider"])
        ref32 = np.ascontiguousarray(session.run(None, feeds)[0])
        np.save(opdir / "ref-fp32-math.npy", ref32)
        ref32.tofile(opdir / "ref-fp32-math.bin")
        ref16 = np.load(opdir / "ref.npy")
        record["fp32_math_reference"] = {
            "model": fp32_model_path.name, "model_bytes": fp32_model_path.stat().st_size,
            "model_sha256": sha(fp32_model_path), "input_dtype": "float32 exact cast from captured float16",
            "input_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                            for p in sorted(fp32_input_dir.glob("*.bin"))],
            "output_shape": list(ref32.shape), "output_dtype": str(ref32.dtype),
            "ref_bytes": (opdir / "ref-fp32-math.bin").stat().st_size,
            "ref_sha256": sha(opdir / "ref-fp32-math.bin"),
            "cpu_fp16_vs_fp32_math": metrics(ref16, ref32),
        }
    manifest["fp32_reference_note"] = (
        "Each FP32 math model is the same copied ONNX node with original FP16 constants exactly "
        "cast to FP32 and the captured FP16 dynamic input exactly cast to FP32; this is a diagnostic reference, not the full model."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v["fp32_math_reference"]["cpu_fp16_vs_fp32_math"]
                      for k, v in manifest["operators"].items()}, indent=2))


if __name__ == "__main__":
    main()
