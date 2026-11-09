"""Prepare one native HTP GELU projection probe; no host accuracy claim.

The HTP-native GELU implementation must be measured on-device. For finite
raw output, clipping to Relu(x) cannot increase error against either exact
erf GELU or specified-tanh GELU, both of which are <= Relu(x).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

from prepare_bounded_gelu import OUT, REPORT, SHAPE, fingerprint


def main() -> None:
    model_path = OUT / "native-projection-finite.onnx"
    manifest_path = REPORT / "native-projection-manifest.json"
    source_manifest_path = REPORT / "bounded-gelu-manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite native projection evidence")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    input_path = Path(source["input"]["path"])
    if fingerprint(input_path)["sha256"] != source["input"]["sha256"]:
        raise ValueError("finite-domain input changed")
    nodes = [
        helper.make_node("Gelu", ["x"], ["raw"], name="native_projection_gelu", approximate="none"),
        helper.make_node("Relu", ["x"], ["r"], name="native_projection_relu"),
        helper.make_node("Min", ["raw", "r"], ["b"], name="native_projection_upper_bound"),
        helper.make_node("Abs", ["x"], ["absolute"], name="native_projection_abs"),
        helper.make_node("Greater", ["absolute", "tail_threshold"], ["tail"], name="native_projection_tail"),
        helper.make_node("Where", ["tail", "r", "b"], ["projected"], name="native_projection_select"),
    ]
    graph = helper.make_graph(nodes, "native_gelu_projection_finite_domain",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT16, SHAPE)],
                              [helper.make_tensor_value_info("projected", TensorProto.FLOAT16, SHAPE),
                               helper.make_tensor_value_info("raw", TensorProto.FLOAT16, SHAPE)],
                              [numpy_helper.from_array(np.asarray(4, dtype=np.float16), "tail_threshold")])
    model = helper.make_model(graph, producer_name="jetlink-native-projection-20261109",
                              opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, model_path)
    onnx.checker.check_model(str(model_path), full_check=True)
    erf_tail_at_four = 2.0 * (1.0 + math.erf(-4.0 / math.sqrt(2.0)))
    tanh_tail_at_four = 2.0 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (-4.0 + 0.044715 * -64.0)))
    manifest = {
        "status": "onnx_checker_passed_pending_actual_htp_native_gelu_measurement",
        "model": fingerprint(model_path), "input": fingerprint(input_path),
        "input_shape": SHAPE, "input_dtype": "FLOAT16", "finite_fp16_count": math.prod(SHAPE),
        "outputs": {"projected": {"dtype": "FLOAT16", "shape": SHAPE},
                    "raw": {"dtype": "FLOAT16", "shape": SHAPE, "role": "diagnostic native GELU result"}},
        "nodes": len(model.graph.node),
        "definition": "raw=Gelu(x,approximate=none); r=Relu(x); b=Min(raw,r); tail=Abs(x)>4; projected=Where(tail,r,b)",
        "floating_intermediates": "FLOAT16", "condition_dtype": "BOOL", "gather_nodes": 0,
        "projection_mathematical_fact": "For finite native raw and either reference formula, the true GELU is <=Relu(x), so Min(raw,Relu(x)) cannot increase absolute error. This does not bound the native implementation's remaining error inside [-4,4].",
        "tail_mathematical_fact": {"condition": "strict abs(x)>4", "replacement": "Relu(x)",
                                   "erf_correction_magnitude_at_abs4_double": erf_tail_at_four,
                                   "tanh_correction_magnitude_at_abs4_double": tanh_tail_at_four,
                                   "scope": "Real-arithmetic tail correction at the cutoff, not a measured HTP full-domain or model error; FP16 reference rounding also applies."},
        "comparison_references": source["double_formula_to_fp16_references"],
        "reference_formulas": ["erf", "tanh"],
        "host_accuracy_simulation": "not performed; host ONNX GELU cannot stand in for HTP native GELU",
        "full_model_generated": False, "device_execution": "not performed by preparation script",
        "onnx_checker_full_check": "passed", "shape_inference_strict": "passed"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": manifest["model"], "input": manifest["input"],
                      "outputs": list(manifest["outputs"]), "nodes": len(model.graph.node),
                      "tail_mathematical_fact": manifest["tail_mathematical_fact"],
                      "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
