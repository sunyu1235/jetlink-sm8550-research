#!/usr/bin/env python3
"""Analyze captured boundary tensors and standalone partition output dumps."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE.parents[2] / "outputs" / "jetlink-gelu-history-20261101" / "gelu-history-160-out"
DTYPES = {
    "FLOAT": np.dtype("<f4"), "FLOAT16": np.dtype("<f2"),
    "DOUBLE": np.dtype("<f8"), "INT8": np.dtype("i1"),
    "UINT8": np.dtype("u1"), "INT16": np.dtype("<i2"),
    "UINT16": np.dtype("<u2"), "INT32": np.dtype("<i4"),
    "UINT32": np.dtype("<u4"), "INT64": np.dtype("<i8"),
    "UINT64": np.dtype("<u8"), "BOOL": np.dtype("u1"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    path = HERE / "extraction-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["partition"]["inputs"], data["partition"]["outputs"]


def tensor_size(spec):
    if any(not isinstance(d, int) or d < 0 for d in spec["shape"]):
        raise ValueError(f"non-static shape for {spec['name']}: {spec['shape']}")
    return math.prod(spec["shape"]) * DTYPES[spec["dtype"]].itemsize


def inspect(path, spec):
    item = {"path": str(path), "present": path.is_file(), "expected_bytes": tensor_size(spec)}
    if not path.is_file():
        item["error"] = "missing"
        return item, None
    item["bytes"] = path.stat().st_size
    item["sha256"] = sha256(path)
    if item["bytes"] != item["expected_bytes"]:
        item["error"] = "wrong_size"
        return item, None
    raw = path.read_bytes()
    a = np.frombuffer(raw, dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"])
    item["finite"] = bool(np.isfinite(a).all()) if np.issubdtype(a.dtype, np.floating) else True
    return item, a


def compare(ref, result, spec):
    a, b = ref.reshape(-1), result.reshape(-1)
    diff = a.astype(np.float64) - b.astype(np.float64)
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    finite = bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) if finite else float("nan")
    finite_metrics = finite and bool(np.isfinite(diff).all())
    return {
        "bitwise_equal_count": int(np.count_nonzero(np.all(a.view(np.uint8).reshape(-1, a.dtype.itemsize) == b.view(np.uint8).reshape(-1, b.dtype.itemsize), axis=1))),
        "element_count": int(a.size),
        "finite": finite_metrics,
        "max_abs": float(np.max(np.abs(diff))) if finite_metrics and diff.size else (0.0 if finite_metrics else None),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))) if finite_metrics and diff.size else (0.0 if finite_metrics else None),
        "mincos": (float(np.dot(aa, bb) / denom) if denom else (1.0 if not np.any(aa) and not np.any(bb) else None)) if finite_metrics else None,
        "allclose_rtol_1e-3_atol_1e-3": bool(finite_metrics and np.allclose(a, b, rtol=1e-3, atol=1e-3, equal_nan=False)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path, required=True, help="directory containing out_i.bin and i_tensorname.bin")
    p.add_argument("--run-dir", type=Path, help="directory containing i_outputname.bin standalone partition outputs")
    p.add_argument("--frames", type=int, required=True, help="number of capture frames to inspect")
    p.add_argument("--input-modulo", type=int, default=8, help="capture index used by run i is i modulo this value (default: 8)")
    p.add_argument("--output", type=Path, required=True, help="write JSON report here; '-' writes stdout")
    p.add_argument("--capture-only", action="store_true", help="compare capture out_i.bin byte-for-byte with accepted historical baseline")
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE, help=f"accepted output baseline (default: {DEFAULT_BASELINE})")
    args = p.parse_args()
    if args.frames < 1 or args.input_modulo < 1:
        p.error("--frames and --input-modulo must be positive")
    if not args.capture_only and args.run_dir is None:
        p.error("--run-dir is required unless --capture-only is selected")
    return args


def main():
    args = parse_args()
    inputs, outputs = load_manifest()
    report = {
        "purpose": "real capture and partition-output comparison; allclose is diagnostic only, not end-to-end precision",
        "capture_dir": str(args.capture_dir), "run_dir": str(args.run_dir) if args.run_dir else None,
        "frames": args.frames, "input_modulo": args.input_modulo,
        "capture_only": args.capture_only, "all_files_present": True,
        "all_correct_sizes": True, "all_finite": True,
        "all_bits_match": True, "allclose": True, "frames_detail": [],
    }
    for i in range(args.frames):
        frame = {"run_index": i, "capture_index": i if args.capture_only else i % args.input_modulo, "tensors": []}
        ci = frame["capture_index"]
        if args.capture_only:
            cap = args.capture_dir / f"out_{i}.bin"
            base = args.baseline_dir / f"out_{i}.bin"
            citem = {"path": str(cap), "present": cap.is_file()}
            bitem = {"path": str(base), "present": base.is_file()}
            for item, path in ((citem, cap), (bitem, base)):
                if path.is_file():
                    item["bytes"] = path.stat().st_size
                    item["sha256"] = sha256(path)
            present = cap.is_file() and base.is_file()
            equal = present and cap.stat().st_size == base.stat().st_size and cap.read_bytes() == base.read_bytes()
            frame.update({"capture": citem, "accepted_baseline": bitem, "bitwise_equal": bool(equal)})
            report["all_files_present"] &= present
            report["all_correct_sizes"] &= present and cap.stat().st_size == base.stat().st_size
            report["all_bits_match"] &= bool(equal)
            # Main model output is a 18,452-element float32 tensor (73,808 bytes).
            expected_main_bytes = 18452 * np.dtype("<f4").itemsize
            main_finite = False
            if cap.is_file() and cap.stat().st_size == expected_main_bytes:
                main_values = np.fromfile(cap, dtype="<f4")
                main_finite = bool(np.isfinite(main_values).all())
            citem["expected_bytes"] = expected_main_bytes
            citem["finite"] = main_finite
            citem["size_valid"] = bool(cap.is_file() and cap.stat().st_size == expected_main_bytes)
            report["all_correct_sizes"] &= citem["size_valid"]
            report["all_finite"] &= main_finite
            report["allclose"] &= bool(equal)
            for role, specs in (("capture_input", inputs), ("capture_output", outputs)):
                for spec in specs:
                    path = args.capture_dir / f"{i}_{spec['name']}.bin"
                    item, _ = inspect(path, spec)
                    frame["tensors"].append({"role": role, "name": spec["name"], **item})
                    report["all_files_present"] &= item["present"]
                    report["all_correct_sizes"] &= "error" not in item
                    if "finite" in item:
                        report["all_finite"] &= item["finite"]
        else:
            for spec in inputs:
                path = args.capture_dir / f"{ci}_{spec['name']}.bin"
                item, _ = inspect(path, spec)
                frame["tensors"].append({"role": "capture_input", "name": spec["name"], **item})
                report["all_files_present"] &= item["present"]
                report["all_correct_sizes"] &= "error" not in item
                if "finite" in item:
                    report["all_finite"] &= item["finite"]
            for spec in outputs:
                ref_path = args.capture_dir / f"{ci}_{spec['name']}.bin"
                got_path = args.run_dir / f"{i}_{spec['name']}.bin"
                ref_item, ref = inspect(ref_path, spec)
                got_item, got = inspect(got_path, spec)
                tensor = {"role": "partition_output", "name": spec["name"], "capture_reference": ref_item, "run_result": got_item}
                report["all_files_present"] &= ref_item["present"] and got_item["present"]
                report["all_correct_sizes"] &= "error" not in ref_item and "error" not in got_item
                if "finite" in ref_item:
                    report["all_finite"] &= ref_item["finite"]
                if "finite" in got_item:
                    report["all_finite"] &= got_item["finite"]
                if ref is not None and got is not None:
                    metrics = compare(ref, got, spec)
                    tensor["comparison"] = metrics
                    report["all_finite"] &= metrics["finite"]
                    report["all_bits_match"] &= metrics["bitwise_equal_count"] == metrics["element_count"]
                    report["allclose"] &= metrics["allclose_rtol_1e-3_atol_1e-3"]
                else:
                    report["all_bits_match"] = False
                    report["allclose"] = False
                frame["tensors"].append(tensor)
        report["frames_detail"].append(frame)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if str(args.output) == "-":
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("all_files_present", "all_correct_sizes", "all_finite", "all_bits_match", "allclose")}))
    # Numerical mismatches remain a successful analysis and are represented in the report.
    return 0 if report["all_files_present"] and report["all_correct_sizes"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"analyze_partition: {exc}", file=sys.stderr)
        sys.exit(2)
