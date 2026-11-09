#!/usr/bin/env python3
"""Summarize a real 160-frame shared-boundaries phone run."""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
REPORT_ROOT = ROOT / "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109"
BASELINE = ROOT / "outputs/jetlink-gelu-history-20261101/gelu-history-160-out"
SOURCE_ROOT = OUT_ROOT / "upstream-pinned-source"
PARITY_TOOL = SOURCE_ROOT / "scripts/verify_parity.py"
PARITY_DATA = ROOT / "outputs/jetlink-gelu-history-20261101/parity-160"
UPSTREAM_PIN = "a863c3801da6fd925d14eb1f5b1e250f8efcc649"
FRAME_COUNT = 160
OUT_BYTES = 73808
OUT_VALUES = 18452


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_root_temp(path):
    if not path.is_file():
        return {"path": str(path), "available": False, "value_milli_c": None}
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    match = re.search(r"-?\d+", raw)
    return {"path": str(path), "available": bool(match), "raw": raw,
            "value_milli_c": int(match.group(0)) if match else None}


def summarize(run_id, frame_count=FRAME_COUNT, allow_rounding_error=False):
    raw_dir = OUT_ROOT / run_id / "shared-boundaries-out"
    report_dir = REPORT_ROOT / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "shared-boundaries.csv"
    errors = []
    output_rows = []
    all_size = True
    all_finite = True
    all_byte_match = True

    for i in range(frame_count):
        actual = raw_dir / f"out_{i}.bin"
        accepted = BASELINE / f"out_{i}.bin"
        row = {"frame": i, "actual_path": str(actual), "baseline_path": str(accepted),
               "present": actual.is_file() and accepted.is_file()}
        if not actual.is_file():
            errors.append(f"missing run output out_{i}.bin")
            all_size = all_finite = all_byte_match = False
            output_rows.append(row)
            continue
        row["bytes"] = actual.stat().st_size
        row["sha256"] = sha256(actual)
        row["size_73808"] = row["bytes"] == OUT_BYTES
        all_size &= row["size_73808"]
        if row["size_73808"]:
            vals = np.fromfile(actual, dtype="<f4")
            row["finite"] = bool(np.isfinite(vals).all())
            all_finite &= row["finite"]
        else:
            row["finite"] = False
            all_finite = False
            errors.append(f"frame {i}: expected {OUT_BYTES} bytes, found {row['bytes']}")
        if accepted.is_file():
            row["baseline_bytes"] = accepted.stat().st_size
            row["baseline_sha256"] = sha256(accepted)
            row["byte_identical_to_accepted"] = (
                row["bytes"] == accepted.stat().st_size and actual.read_bytes() == accepted.read_bytes())
        else:
            row["baseline_bytes"] = None
            row["baseline_sha256"] = None
            row["byte_identical_to_accepted"] = False
            errors.append(f"missing accepted baseline out_{i}.bin")
        all_byte_match &= row["byte_identical_to_accepted"]
        output_rows.append(row)

    rows = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        errors.append(f"missing timing CSV: {csv_path}")
    if len(rows) != frame_count:
        errors.append(f"timing CSV row count is {len(rows)}, expected {frame_count}")
    timing = np.array([float(r["execute_ms"]) for r in rows], dtype=np.float64)
    timing_finite = bool(np.isfinite(timing).all())
    if not timing_finite:
        errors.append("timing CSV contains nonfinite execute_ms")

    def stats(v):
        if not len(v):
            return {k: None for k in ("mean", "p50", "p99", "max", "min")}
        return {"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)),
                "p99": float(np.percentile(v, 99)), "max": float(np.max(v)), "min": float(np.min(v))}

    before = read_root_temp(report_dir / "shared-boundaries.gpu-before.txt")
    after = read_root_temp(report_dir / "shared-boundaries.gpu-after.txt")
    parity = {"status": "unknown", "pass": None, "verifier_found": PARITY_TOOL.is_file(),
              "verifier_path": str(PARITY_TOOL),
              "reason": "verify_parity.py was not found; original compare was not run"}
    if PARITY_TOOL.is_file():
        import os
        import shutil
        compare_dir = report_dir / ("compare" if frame_count == FRAME_COUNT else f"compare-{frame_count}")
        compare_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARITY_DATA / "spec.json", compare_dir / "spec.json")
        prep_errors = []
        for i in range(frame_count):
            ref_path = PARITY_DATA / f"out_ref_{i}.npy"
            actual_path = raw_dir / f"out_{i}.bin"
            if not ref_path.is_file() or not actual_path.is_file():
                prep_errors.append(f"frame {i}: missing reference or run output")
                continue
            ref = np.load(ref_path, allow_pickle=False)
            actual = np.fromfile(actual_path, dtype="<f4")
            if ref.size != OUT_VALUES or actual.size != OUT_VALUES:
                prep_errors.append(f"frame {i}: expected {OUT_VALUES} values in ref and run")
                continue
            np.save(compare_dir / f"out_link_{i}.npy", actual.reshape(ref.shape))
            shutil.copy2(ref_path, compare_dir / ref_path.name)
        parity_log = report_dir / "parity.log"
        if prep_errors:
            proc_returncode = 1
            proc_output = "Parity compare inputs could not be prepared:\n" + "\n".join(prep_errors) + "\n"
        else:
            cmd = [sys.executable, str(PARITY_TOOL), "compare", "--dir", str(compare_dir)]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            (compare_dir / "command.txt").write_text(
                " ".join(cmd) + "\nPYTHONPATH=" + env["PYTHONPATH"] + "\n", encoding="utf-8")
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            proc_returncode, proc_output = proc.returncode, proc.stdout
            (compare_dir / "compare.stdout.txt").write_text(proc_output, encoding="utf-8")
            (compare_dir / "compare.exitcode.txt").write_text(str(proc_returncode) + "\n", encoding="ascii")
        parity_log.write_text(proc_output, encoding="utf-8")
        parity_pass = proc_returncode == 0 and "OK: every slice" in proc_output
        parity = {"status": "pass" if parity_pass else "fail",
                  "pass": parity_pass,
                  "exit_code": proc_returncode, "log": str(parity_log), "compare_dir": str(compare_dir), "verifier_found": True,
                  "verifier_path": str(PARITY_TOOL), "source_revision": UPSTREAM_PIN}
    (report_dir / "parity.log").write_text(
        "Parity status: unknown\nverify_parity.py not found at the required path; original parity comparison was not run.\n",
        encoding="utf-8") if parity["status"] == "unknown" else None

    checks = {
        f"all_{frame_count}_outputs_present": all(r.get("present", False) for r in output_rows),
        f"all_{frame_count}_outputs_73808_bytes": all_size,
        f"all_{frame_count}_outputs_finite_float32": all_finite,
        f"all_{frame_count}_outputs_byte_identical_to_accepted_baseline": all_byte_match,
        f"timing_csv_has_{frame_count}_rows": len(rows) == frame_count,
        "timing_values_finite": timing_finite,
        "root_temp_before_after_available": before["available"] and after["available"],
    }
    parity_pass = parity["status"] == "pass"
    required_checks = {k: v for k, v in checks.items()
                       if not k.endswith("_byte_identical_to_accepted_baseline")
                       and (not allow_rounding_error or k != "root_temp_before_after_available")}
    if not allow_rounding_error:
        required_checks.update({k: v for k, v in checks.items()
                                if k.endswith("_byte_identical_to_accepted_baseline")})
    required_checks["original_parity_checker_pass"] = parity_pass
    summary = {
        "run_id": run_id, "frame_count_expected": frame_count,
        "allow_rounding_error": allow_rounding_error,
        "output_validation": {"bytes_expected": OUT_BYTES, "float32_values_expected": OUT_VALUES,
                              "count": sum(r.get("present", False) for r in output_rows),
                              "byte_identical_baseline_count": sum(r.get("byte_identical_to_accepted", False) for r in output_rows),
                              f"all_{frame_count}_outputs_finite": all_finite,
                              "sha256_by_frame": [{"frame": r["frame"], "run": r.get("sha256"),
                                                   "accepted": r.get("baseline_sha256")} for r in output_rows]},
        "latency_ms": {"scope": f"{frame_count} CSV execute_ms rows, including frame 0",
                       "all_frames": stats(timing),
                       "excluding_first_frame": {"scope": f"CSV rows with frame != 0; {max(frame_count - 1, 0)} rows",
                                                  **stats(timing[1:])}},
        "root_temp_milli_c": {"before": before, "after": after},
        "parity": parity, "checks": checks, "errors": errors,
        "required_checks": required_checks,
        "data_status": "PASS" if all(required_checks.values()) else "FAIL",
        "status": ("FAIL" if not all(required_checks.values()) else "PASS"),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="run folder name under outputs and reports")
    parser.add_argument("--frames", type=int, default=FRAME_COUNT, help=f"expected frames to analyze (default: {FRAME_COUNT})")
    parser.add_argument("--allow-rounding-error", action="store_true",
                        help="record baseline byte differences without requiring byte identity for PASS")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id must be a simple folder name")
    if args.frames < 1:
        parser.error("--frames must be positive")
    result = summarize(args.run_id, args.frames, args.allow_rounding_error)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "output_count": result["output_validation"]["count"],
                      "baseline_matches": result["output_validation"]["byte_identical_baseline_count"],
                      "latency_ms": result["latency_ms"], "root_temp_milli_c": result["root_temp_milli_c"],
                      "parity": result["parity"]}, indent=2))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"analyze_shared_run: {exc}", file=sys.stderr)
        sys.exit(2)
