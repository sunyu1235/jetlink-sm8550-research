#!/usr/bin/env python3
"""Regenerate SHA-256 manifest for the Jetlink GELU optimization evidence.

This is a read-only hash inventory. It does not hash arbitrary workspace files;
key model/runtime inputs, experimental sources/proofs, and summaries are listed
explicitly below. Missing paths are recorded instead of treated as errors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
ARTIFACTS = {
  "both63_ctx_model": "outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both-ctx.onnx",
  "both63_inline_shapes_model": "outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both-inline-shapes.onnx",
  "both63_whole_model": "outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both.onnx",
  "history_runner": "outputs/jetlink-gelu-repair-20261101/history_probe",
  "original_cpu_gelu_library": "outputs/jetlink-gelu-repair-20261101/libspecifiedgelu.so",
  "unrolled_cpu_gelu_library": "outputs/jetlink-gelu-optimization-20261101/cpu/libunrolledgelu.so",
  "spin_probe_source": "reports/jetlink-gelu-optimization-20261101/spin/spin_probe.c",
  "spin_probe_binary": "outputs/jetlink-gelu-optimization-20261101/spin/spin_probe",
  "native_lut_probe_source": "reports/jetlink-gelu-optimization-20261101/lut/int32_probe.c",
  "native_lut_probe_binary": "outputs/jetlink-gelu-optimization-20261101/lut/int32_probe",
  "unrolled_cpu_source": "reports/jetlink-gelu-optimization-20261101/cpu/unrolled_gelu.c",
  "unrolled_cpu_unit_proof": "reports/jetlink-gelu-optimization-20261101/cpu/unit-proof.json",
  "spin_smoke_proof": "reports/jetlink-gelu-optimization-20261101/spin/smoke-proof.json",
  "lut_cast_result": "reports/jetlink-gelu-optimization-20261101/lut/cast-errors.json",
  "lut_gather_result": "reports/jetlink-gelu-optimization-20261101/lut/gather-errors.json",
  "lut_index_result": "reports/jetlink-gelu-optimization-20261101/lut/index-errors.json",
  "lut_combined_failure_log": "reports/jetlink-gelu-optimization-20261101/lut/native-erf-domain.log",
  "shared_ab_summary": "reports/jetlink-gelu-optimization-20261101/ab/summary.json",
  "shared_ab_hash_manifest": "reports/jetlink-gelu-optimization-20261101/ab/hash-manifest.json",
  "affinity_ab_summary": "reports/jetlink-gelu-optimization-20261101/affinity/summary.json",
  "affinity_ab_hash_manifest": "reports/jetlink-gelu-optimization-20261101/affinity/hash-manifest.json",
  "unrolled_ab_summary": "reports/jetlink-gelu-optimization-20261101/cpu/ab/summary.json",
  "unrolled_ab_hash_manifest": "reports/jetlink-gelu-optimization-20261101/cpu/ab/hash-manifest.json",
  "thread_runner_source": "reports/jetlink-gelu-optimization-20261101/threads/history_probe_threads.c",
  "thread_runner_binary": "outputs/jetlink-gelu-optimization-20261101/threads/history_probe_threads",
  "thread_build_manifest": "reports/jetlink-gelu-optimization-20261101/threads/build-manifest.json",
  "thread_experiment_summary": "reports/jetlink-gelu-optimization-20261101/threads/summary.json",
  "thread_experiment_checks": "reports/jetlink-gelu-optimization-20261101/threads/checks.json",
  "thread_experiment_report": "reports/jetlink-gelu-optimization-20261101/threads/README.md",
  "input_manifest": "reports/jetlink-gelu-history-20261101/input-manifest.json",
  "sdk_research": "reports/jetlink-gelu-optimization-20261101/sdk/README.md",
  "optimization_summary": "reports/jetlink-gelu-optimization-20261101/README.md",
  "stress_run_summary": "reports/jetlink-gelu-optimization-20261101/stress/best-1000-summary.json",
  "stress_run_csv": "reports/jetlink-gelu-optimization-20261101/stress/best-1000.csv",
  "stress_prefix_parity_summary": "reports/jetlink-gelu-optimization-20261101/stress/prefix160-verification/summary.json",
  "stress_prefix_parity_output": "reports/jetlink-gelu-optimization-20261101/stress/prefix160-verification/verify-parity.stdout.txt",
  "native_lut_report": "reports/jetlink-gelu-optimization-20261101/lut/README.md",
  "accurate_ort_profile_summary": "reports/jetlink-gelu-optimization-20261101/profile-20261102/ort-profile-summary.json",
  "accurate_ort_profile_report": "reports/jetlink-gelu-optimization-20261101/profile-20261102/README.md",
  "accurate_qnn_optrace_attempt_log": "reports/jetlink-gelu-optimization-20261101/profile-20261102/accurate-qnn-optrace.stdout.txt",
  "native_gelu_all_htp_qnn_optrace_csv": "outputs/jetlink-gelu-optimization-20261101/profile-optrace/artifacts/jetlink_qnn_optrace_20261102.csv",
  "runtime_tuning_probe_source": "reports/jetlink-gelu-optimization-20261101/runtime-tuning-20261102/history_probe_runtime_tuning.c",
  "runtime_tuning_probe_build_manifest": "reports/jetlink-gelu-optimization-20261101/runtime-tuning-20261102/build-manifest.json",
  "runtime_tuning_probe_build_script": "reports/jetlink-gelu-optimization-20261101/runtime-tuning-20261102/build_probe.ps1",
  "runtime_tuning_probe_report": "reports/jetlink-gelu-optimization-20261101/runtime-tuning-20261102/README.md",
  "runtime_tuning_probe_binary": "outputs/jetlink-gelu-optimization-20261101/runtime-tuning-20261102/history_probe_runtime_tuning",
  "partition_tuning_extractor": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/extract_partition.py",
  "partition_tuning_manifest": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/extraction-manifest.json",
  "partition_tuning_probe_source": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/partition_probe.c",
  "partition_tuning_probe_binary": "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/partition_probe",
  "partition_tuning_build_manifest": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/build-manifest.json",
  "partition_tuning_build_script": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/build_probe.ps1",
  "partition_tuning_runner": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/run_partition_aba.ps1",
  "partition_tuning_analyzer": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/analyze_partition.py",
  "partition_tuning_report": "reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/README.md",
  "partition_tuning_model": "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/partition-130-40.onnx",
  "partition_tuning_capture_wrapper": "outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109/accepted-context-capture-130-40.onnx",
  "optimization_summary_after_qnn_profile": "reports/jetlink-gelu-optimization-20261101/README.md",
}

def file_hash(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()

def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, default=REPORT_DIR / "version-hashes.json")
  args = parser.parse_args()
  entries = {}
  for key, rel in ARTIFACTS.items():
    path = ROOT / rel
    if path.is_file():
      entries[key] = {"path": rel, "exists": True, "size_bytes": path.stat().st_size,
                      "sha256": file_hash(path)}
    else:
      entries[key] = {"path": rel, "exists": False, "size_bytes": None,
                      "sha256": None}
  manifest = {"purpose": "offline evidence/version inventory; no phone operations",
              "root": str(ROOT), "artifacts": entries}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.output} ({sum(x['exists'] for x in entries.values())}/{len(entries)} present)")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
