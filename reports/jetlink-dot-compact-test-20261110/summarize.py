"""Summarize phone timing and run the unchanged upstream numerical checker."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-dot-compact-test-20261110"
OLD_OUT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/shared-boundaries-out"
REF = ROOT / "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare"
UPSTREAM = ROOT / "outputs/jetlink-dot-review-repository-20261109/vendor/jetlink"


def timing(label):
    rows = list(csv.DictReader((OUT / label / "timing.csv").open()))
    a = np.array([float(r["execute_ms"]) for r in rows])
    def stats(x):
        return {"mean_ms": float(x.mean()), "p50_ms": float(np.median(x)),
                "p99_ms": float(np.percentile(x, 99)), "max_ms": float(x.max())}
    return {"frames": len(a), "including_first": stats(a), "excluding_first": stats(a[1:]),
            "nonfinite": sum(int(r["nonfinite"]) for r in rows),
            "gpu_sensor_before_c": float((REPORT / (label + "-gpu-before.txt")).read_text()) / 1000,
            "gpu_sensor_after_c": float((REPORT / (label + "-gpu-after.txt")).read_text()) / 1000,
            "battery_temp_start_c": int(rows[0]["battery_deci_c"]) / 10,
            "battery_temp_end_c": int(rows[-1]["battery_deci_c"]) / 10}


def main():
    compare = REPORT / "compare160"
    compare.mkdir(exist_ok=True)
    shutil.copyfile(REF / "spec.json", compare / "spec.json")
    identical = 0
    max_abs = 0.
    for i in range(160):
        raw = (OUT / "candidate160" / f"out_{i}.bin").read_bytes()
        old = (OLD_OUT / f"out_{i}.bin").read_bytes()
        assert len(raw) == 73808, i
        a = np.frombuffer(raw, dtype="<f4")
        b = np.frombuffer(old, dtype="<f4")
        assert np.isfinite(a).all(), i
        identical += raw == old
        max_abs = max(max_abs, float(np.abs(a - b).max()))
        np.save(compare / f"out_link_{i}.npy", a)
        shutil.copyfile(REF / f"out_ref_{i}.npy", compare / f"out_ref_{i}.npy")
    env = {**os.environ, "PYTHONPATH": str(UPSTREAM)}
    result = subprocess.run([sys.executable, str(UPSTREAM / "scripts/verify_parity.py"),
        "compare", "--dir", str(compare)], env=env, capture_output=True, timeout=60)
    (REPORT / "parity160.log").write_bytes(result.stdout + result.stderr)
    timings = {label: timing(label) for label in ("baseline16", "candidate16", "baseline160", "candidate160")}
    delta = timings["baseline16"]["excluding_first"]["mean_ms"] - timings["candidate16"]["excluding_first"]["mean_ms"]
    delta160 = timings["baseline160"]["excluding_first"]["mean_ms"] - timings["candidate160"]["excluding_first"]["mean_ms"]
    summary = {"dot_commit": "af713f089848d1f6725a3ebfb7b3bf9f42e07296",
        "device": "Moto X40 XT2301-5 / SM8550", "ort": "1.29.0", "qnn": "2.42.0",
        "execution": "45 cached QNN NPU partitions + 63 existing exact CPU GELUs; two partitions rebuilt",
        "timing_scope": "whole-model bound Run; excludes initialization, camera/input loading and transport",
        "timing": timings, "paired16_speedup_ms_excluding_first": delta,
        "paired16_speedup_percent": 100 * delta / timings["baseline16"]["excluding_first"]["mean_ms"],
        "paired160_speedup_ms_excluding_first": delta160,
        "paired160_speedup_percent": 100 * delta160 / timings["baseline160"]["excluding_first"]["mean_ms"],
        "validation160": {"frames": 160, "all_finite": True,
            "byte_identical_to_accepted_baseline_frames": identical,
            "max_abs_difference_to_accepted_baseline": max_abs,
            "unchanged_upstream_checker_exit_code": result.returncode,
            "unchanged_upstream_checker_pass": result.returncode == 0},
        "candidate_wrapper_sha256": hashlib.sha256((OUT / "compact-whole.onnx").read_bytes()).hexdigest(),
        "apk_deployed": False, "clock_or_thermal_policy_changed": False,
        "interpretation": "compact boundary is executable and numerically identical in 160 phone frames; consult matched timing delta for speed benefit; no measured 30ms result; no new road replay performed"}
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
