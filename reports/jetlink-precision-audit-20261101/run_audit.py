#!/usr/bin/env python3
"""Read-only batch audit using the upstream verify_parity.py compare command.

All dataset paths are read-only. The script stores each upstream stdout/stderr,
the exact command, and its exit status under --output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\sunbo\Documents\moto-x40-root")
UPSTREAM = Path(r"C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research\scripts\verify_parity.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent
CASES = {
    # Folder names are the experiment labels present in the supplied data.
    "original_npu": "parity-varied",
    "npu_vision_cpu16_policy": "parity-hybrid",
    "npu_vision_cpu32_policy": "parity-policy-fp32",
    "cpu_vision_npu_policy": "parity-cpu-vision-htp",
    "all_cpu32_pc": "parity-all-fp32",
    "phone_fp32": "parity-phone-fp32",
    "phone_fp32_cpuvision": "parity-phone-fp32-cpuvision",
    "phone_hybrid": "parity-phone-hybrid",
    "phone_cpu32_full": "@phone-full-cpu32-compare",
}


def parse_result(label: str, dataset: Path, code: int, output: str) -> dict:
    failed_heads: dict[str, dict] = {}
    max_abs = None
    # Upstream's per-head rows: name corr ... max abs N ... [FAIL]
    for line in output.splitlines():
        m = re.match(r"\s{4}([\w-]+)\s+corr\s+([0-9.]+).*?max abs\s+([0-9.eE+-]+)(.*)$", line)
        if not m:
            continue
        err = float(m.group(3))
        max_abs = err if max_abs is None else max(max_abs, err)
        if "FAIL" in m.group(4):
            head = m.group(1)
            item = failed_heads.setdefault(head, {"head": head, "worst_corr": 1.0, "max_abs": 0.0})
            item["worst_corr"] = min(item["worst_corr"], float(m.group(2)))
            item["max_abs"] = max(item["max_abs"], err)
    overall = re.search(r"FAIL: (\d+) slice\(s\).*?: (.*)$", output, re.M)
    ok = "\nOK: every slice" in output
    return {
        "label": label,
        "dataset": str(dataset),
        "frames": len(list(dataset.glob("out_link_*.npy"))),
        "exit_code": code,
        "pass": bool(ok and code == 0),
        "max_abs_head_error": max_abs,
        "failed_heads": list(failed_heads.values()),
        "upstream_fail_summary": overall.group(2).strip() if overall else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--upstream", type=Path, default=UPSTREAM)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.upstream.parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for label, dirname in CASES.items():
        dataset = (args.output / dirname[1:]) if dirname.startswith("@") else (args.root / dirname)
        if not dataset.is_dir() or not (dataset / "spec.json").is_file():
            raise SystemExit(f"missing dataset/spec for {label}: {dataset}")
        if not list(dataset.glob("out_link_*.npy")) or not list(dataset.glob("out_ref_*.npy")):
            raise SystemExit(f"missing out_ref/out_link arrays for {label}: {dataset}")
        cmd = [sys.executable, str(args.upstream), "compare", "--dir", str(dataset)]
        proc = subprocess.run(cmd, cwd=str(args.root), env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = args.output / f"{label}.compare.log"
        log.write_text("$ " + subprocess.list2cmdline(cmd) + "\n\n" + proc.stdout,
                       encoding="utf-8")
        result = parse_result(label, dataset, proc.returncode, proc.stdout)
        result["log"] = str(log)
        results.append(result)
        print(f"{label}: exit={proc.returncode}, pass={result['pass']}, "
              f"max_abs={result['max_abs_head_error']}, "
              f"failed={','.join(x['head'] for x in result['failed_heads']) or '-'}")
    summary = {"source_script": str(args.upstream), "thresholds": {
        "MIN_CORR": 0.999, "MIN_SAMPLES": 16, "QUIET_FRACTION": 0.05,
        "TINY_TOLERANCE": 0.1, "CONSTANT_FRACTION": 0.001,
    }, "results": results}
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
