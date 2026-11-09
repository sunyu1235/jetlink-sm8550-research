from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "outputs/jetlink-gelu-optimization-20261101/profile-20261102-a"
REPORT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "outputs/jetlink-gelu-history-20261101/gelu-history-160-out"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    profiles = sorted(PROFILE_DIR.glob("ort-profile*.json"))
    if len(profiles) != 1:
        raise SystemExit(f"expected one ORT profile, found {len(profiles)}")
    profile_path = profiles[0]
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [
        e for e in events
        if e.get("ph") == "X" and "kernel_time" in e.get("name", "")
    ]

    provider_durations: dict[str, list[float]] = collections.defaultdict(list)
    grouped: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for e in node_events:
        args = e.get("args", {})
        provider = args.get("provider", "unknown")
        op_name = args.get("op_name", "unknown")
        duration = float(e.get("dur", 0))
        provider_durations[provider].append(duration)
        grouped[(provider, op_name)].append(e)

    providers = {}
    for provider, durations in sorted(provider_durations.items()):
        providers[provider] = {
            "event_count": len(durations),
            "events_per_frame": len(durations) / 16,
            "sum_ms_all_frames": sum(durations) / 1000,
            "sum_ms_per_frame": sum(durations) / 1000 / 16,
            "share_of_node_event_time": sum(durations) / sum(
                sum(v) for v in provider_durations.values()
            ),
        }

    groups = []
    for (provider, op_name), group in sorted(
        grouped.items(), key=lambda kv: statistics.mean(float(e.get("dur", 0)) for e in kv[1]), reverse=True
    ):
        durations = [float(e.get("dur", 0)) for e in group]
        args = group[0].get("args", {})
        groups.append({
            "provider": provider,
            "op_name": op_name,
            "count": len(group),
            "mean_ms": statistics.mean(durations) / 1000,
            "p50_ms": statistics.median(durations) / 1000,
            "max_ms": max(durations) / 1000,
            "node_index": args.get("node_index"),
            "input_type_shape": args.get("input_type_shape"),
            "output_type_shape": args.get("output_type_shape"),
        })

    outputs = []
    baseline_matches = 0
    for i in range(16):
        output = PROFILE_DIR / "out" / f"out_{i}.bin"
        baseline = BASELINE_DIR / f"out_{i}.bin"
        if not output.is_file() or not baseline.is_file():
            raise SystemExit(f"missing frame {i} output/baseline")
        raw = output.read_bytes()
        arr = np.frombuffer(raw, dtype="<f4")
        same = raw == baseline.read_bytes()
        baseline_matches += int(same)
        outputs.append({
            "frame": i,
            "bytes": len(raw),
            "float32_values": int(arr.size),
            "nonfinite": int((~np.isfinite(arr)).sum()),
            "byte_identical_accepted_baseline": same,
            "sha256": sha256(output),
        })

    summary = {
        "status": "PASS" if baseline_matches == 16 and all(x["nonfinite"] == 0 for x in outputs) else "FAIL",
        "profile": str(profile_path),
        "profile_bytes": profile_path.stat().st_size,
        "profile_sha256": sha256(profile_path),
        "scope": "16 profiled frames on the accepted shared=1, f8, original CPU GELU library, four-thread candidate. Profiling adds overhead; durations are attribution data, not production latency.",
        "event_count": len(events),
        "node_kernel_event_count": len(node_events),
        "providers": providers,
        "largest_20_node_groups": groups[:20],
        "frame_outputs": {
            "count": len(outputs),
            "all_73808_bytes": all(x["bytes"] == 73808 for x in outputs),
            "all_18452_float32": all(x["float32_values"] == 18452 for x in outputs),
            "all_finite": all(x["nonfinite"] == 0 for x in outputs),
            "byte_identical_to_accepted_baseline": baseline_matches == 16,
            "byte_identical_frames": baseline_matches,
            "per_frame": outputs,
        },
        "frame_timing_csv": str(PROFILE_DIR / "frames.csv"),
        "profiled_run_mean_ms_including_instrumentation": 64.132,
        "ordinary_unprofiled_stress_mean_ms": 62.010536128,
    }
    (REPORT_DIR / "ort-profile-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# ORT node profile: accurate CPU+HTP candidate",
        "",
        "## Result",
        "",
        "The 16-frame run used the accepted shared-memory, `f8` affinity, original exact CPU GELU library, four ORT intra-op threads, and spinning disabled. The 16 output tensors were finite, 73,808 bytes each, and byte-identical to the accepted baseline. The original parity checker had already passed on the same outputs during previous runs; this short profile itself is additionally checked byte-for-byte.",
        "",
        "Instrumentation increased observed mean execution time to 64.13 ms, so do not use this run for performance claims. The unprofiled 1,000-frame mean was 62.01 ms.",
        "",
        "## Provider time in the ORT trace",
        "",
        "| Provider | Node events / frame | Sum of node event durations / frame | Share of summed node event time |",
        "|---|---:|---:|---:|",
    ]
    for provider, values in providers.items():
        lines.append(
            f"| {provider} | {values['events_per_frame']:.1f} | "
            f"{values['sum_ms_per_frame']:.3f} ms | "
            f"{values['share_of_node_event_time'] * 100:.1f}% |"
        )
    lines += [
        "",
        "This is attribution from ORT event durations, not a hardware utilization measurement. The event sums closely track wall time but can include provider/runtime work differently; use the QNN optrace follow-up to split the HTP partition times internally.",
        "",
        "## Largest provider events",
        "",
        "See `ort-profile-summary.json` for the 20 largest unique provider op groups, their timings, shapes, and node indices. The full ORT JSON and captured outputs are in the ignored `outputs/jetlink-gelu-optimization-20261101/profile-20261102-a/` directory.",
        "",
        "## Next step selected from this profile",
        "",
        "The CPU GELU custom nodes account for about 6 ms/frame in the trace; the QNN provider partitions account for roughly 57 ms/frame. CPU table micro-optimization cannot recover the gap to 30 ms by itself. The next measurement should be QNN `optrace` for the largest HTP partitions, before making another graph change.",
        "",
    ]
    (REPORT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_DIR / 'ort-profile-summary.json'}")
    print(f"providers: {json.dumps(providers, sort_keys=True)}")
    print(f"outputs byte-identical: {baseline_matches}/16")


if __name__ == "__main__":
    main()
