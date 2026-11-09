#!/usr/bin/env python3
"""Compare each one-op HTP capture against CPU FP16 and FP32-math references."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2] / "outputs" / "jetlink-operator-audit-20261101"
REPORTS = Path(__file__).resolve().parent


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual: np.ndarray, reference: np.ndarray):
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {reference.shape}")
    delta = actual.astype(np.float64).ravel() - reference.astype(np.float64).ravel()
    rmse = float(np.sqrt(np.mean(delta * delta)))
    spread = float(reference.astype(np.float64).std())
    return {"mae": float(np.mean(np.abs(delta))), "max_abs": float(np.max(np.abs(delta), initial=0)),
            "rmse": rmse, "normalized_rmse": rmse / spread if spread else None,
            "ref_std": spread, "elements": int(reference.size)}


def main():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_path = REPORTS / "htp16-runs.json"
    runs = json.loads(run_path.read_text(encoding="utf-8"))
    run_map = {r["operator"]: r for r in runs["runs"]}
    report = {"device": runs["device"], "backend": runs["backend"],
              "strict_cpu_fallback_disabled": runs["strict_cpu_fallback_disabled"],
              "operators": {}}
    overshoot_path = REPORTS / "gelu-overshoot.json"
    if overshoot_path.is_file():
        report["gelu_positive_output_overshoot"] = json.loads(overshoot_path.read_text(encoding="utf-8"))
    failures = []
    for label, rec in manifest["operators"].items():
        opdir = ROOT / label
        run = run_map[label]
        if run["exit_code"] != 0:
            failures.append(label)
            report["operators"][label] = {"node_name": rec["node_name"], "status": "HTP unsupported or runtime failed",
                                          "exit_code": run["exit_code"], "log": run["log"]}
            continue
        htp_file = opdir / "htp-output.bin"
        if not htp_file.is_file():
            raise SystemExit(f"missing HTP output for successful run: {htp_file}")
        htp = np.fromfile(htp_file, dtype=np.float16).reshape(rec["output_shape"])
        ref16 = np.load(opdir / "ref.npy")
        ref32 = np.load(opdir / "ref-fp32-math.npy")
        if htp.dtype != ref16.dtype:
            raise SystemExit(f"output dtype mismatch for {label}: {htp.dtype} vs {ref16.dtype}")
        logtext = Path(run["log"]).read_text(encoding="utf-8")
        log_match = re.search(r"output_type=(\d+) nonfinite=(\d+)/(\d+).*?mean_ms=([0-9.]+) p50_ms=([0-9.]+) p99_ms=([0-9.]+)", logtext, re.S)
        if not log_match:
            raise SystemExit(f"missing strict probe output summary in {run['log']}")
        output_type, nonfinite, element_count = map(int, log_match.group(1, 2, 3))
        record = {
            "node_name": rec["node_name"], "op_type": rec["op_type"], "attributes": rec["attributes"],
            "input_names_shapes_hashes": rec["dynamic_inputs"],
            "output_shape": list(htp.shape), "output_dtype": str(htp.dtype),
            "qnn_output_type_enum": output_type, "nonfinite": nonfinite, "element_count": element_count,
            "htp_output_sha256": sha(htp_file), "cpu_fp16_ref_sha256": sha(opdir / "ref.bin"),
            "cpu_fp32_math_ref_sha256": sha(opdir / "ref-fp32-math.bin"),
            "htp_runtime_ms_one_run": {"mean": float(log_match.group(4)), "p50": float(log_match.group(5)), "p99": float(log_match.group(6))},
            "htp_fp16_vs_cpu_fp16": metrics(htp, ref16),
            "htp_fp16_vs_cpu_fp32_math": metrics(htp, ref32),
            "cpu_fp16_vs_cpu_fp32_math": rec["fp32_math_reference"]["cpu_fp16_vs_fp32_math"],
            "status": "PASS",
        }
        rec["htp16_capture"] = {"bytes": htp_file.stat().st_size, "sha256": sha(htp_file),
                                "qnn_output_type_enum": output_type, "nonfinite": nonfinite,
                                "strict_probe_log": run["log"]}
        rec["htp_fp16_vs_cpu_fp16"] = record["htp_fp16_vs_cpu_fp16"]
        rec["htp_fp16_vs_cpu_fp32_math"] = record["htp_fp16_vs_cpu_fp32_math"]
        report["operators"][label] = record

    (REPORTS / "operator-errors.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for label, rec in report["operators"].items():
        if rec["status"] == "PASS":
            a, b = rec["htp_fp16_vs_cpu_fp16"], rec["htp_fp16_vs_cpu_fp32_math"]
            print(f"{label}: HTP-vs-CPU16 MAE={a['mae']:.7g} max={a['max_abs']:.7g} nRMSE={a['normalized_rmse']:.7g}; "
                  f"vs-CPU32 MAE={b['mae']:.7g} max={b['max_abs']:.7g} nRMSE={b['normalized_rmse']:.7g}")
        else:
            print(f"{label}: {rec['status']} (exit={rec['exit_code']})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
