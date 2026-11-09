#!/usr/bin/env python3
"""Count positive GELU outputs exceeding their input in the saved one-op case."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
BASE = WORKSPACE / "outputs" / "jetlink-operator-audit-20261101" / "gelu"


def main():
    x = np.fromfile(BASE / "inputs" / "0_linear_70.bin", dtype=np.float16).astype(np.float32)
    variants = {
        "htp_fp16": np.fromfile(BASE / "htp-output.bin", dtype=np.float16).astype(np.float32),
        "cpu_fp16": np.fromfile(BASE / "ref.bin", dtype=np.float16).astype(np.float32),
        "cpu_fp32_math": np.fromfile(BASE / "ref-fp32-math.bin", dtype=np.float32),
    }
    positive = x > 0
    results = {}
    for name, y in variants.items():
        if y.shape != x.shape:
            raise SystemExit(f"shape mismatch in {name}: {y.shape} vs {x.shape}")
        over = positive & (y > x)
        indices = np.flatnonzero(over)
        overshoots = y.astype(np.float64) - x.astype(np.float64)
        idx = int(indices[np.argmax(overshoots[indices])]) if indices.size else None
        results[name] = {
            "positive_output_gt_input_count": int(over.sum()),
            "positive_output_gt_input_fraction": float(over.sum() / positive.sum()),
            "max_positive_overshoot": float(overshoots[idx]) if idx is not None else 0.0,
            "max_index_flat": idx,
            "max_case_input": float(x[idx]) if idx is not None else None,
            "max_case_output": float(y[idx]) if idx is not None else None,
        }
    out = {"node_name": "node_gelu_35", "attribute_approximate": "none",
           "input_shape": [1, 4, 8, 8192], "input_dtype": "float16", "total_elements": int(x.size),
           "positive_input_count": int(positive.sum()), "metric": "count/fraction with output > positive input",
           "variants": results}
    dest = Path(__file__).resolve().parent / "gelu-overshoot.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
