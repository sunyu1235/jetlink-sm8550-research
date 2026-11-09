# Isolated policy partition tuning, 2026-11-09

## Latest results and review of the old 32 ms route, 2026-11-09

The preparation notes below are historical. Phone execution has since resumed.
Re-reading the user's pasted October 31 record and the original
`C:\Users\sunbo\Documents\moto-x40-root\light-speed-baseline-20261031.md`
confirms that **32.475 ms was a measured standalone all-HTP/shared-I/O result**:
1,200 frames at 20 Hz, one repeated synthetic road-like image, recurrent state
feedback, inference time only. It did not establish numerical parity. Subsequent
five-frame and 16-varied-frame comparisons failed. Do not describe that timing
as a usable, accurate 32 ms model or as the installed app's end-to-end latency.

The targeted repair is already implemented. The strongest existing evidence is
native HTP GELU error and a QNN mapping that does not preserve ONNX's GELU
`approximate` selection. The model uses 36 erf-form vision GELUs and 27 tanh-form
policy GELUs. Correcting vision alone still left policy failures; separately
preserving both formulas passed. Those CPU operators split the accelerator graph
into 45 EPContext nodes, adding execution boundaries as well as CPU work.
This establishes an effective repair, not proof that all 63 replacements are
mathematically necessary. No smaller validated replacement subset was found.
See [the causal repair record](../../jetlink-gelu-repair-20261101/README.md) and
[the independent branch audit](../../jetlink-precision-audit-20261101/README.md).

| Existing result | Measured mean | Numerical evidence | Assessment |
| --- | ---: | --- | --- |
| Accurate repair, all boundaries shared, final-query optimization | 52.409 ms | 160 recurrent frames, unchanged upstream checker PASS; all 160 byte-identical to accepted baseline | Best existing accurate native prototype; [result](last-query-full160/summary.json) |
| Six-op native GELU projection | 63.745 ms | 16 recurrent frames, upstream checker PASS | Not a route back to 32 ms; [result](projection-whole16/summary.json) |
| Three-op simplified projection | 50.463 ms | 16 frames finite, upstream checker FAIL in five heads | Rejected; [result](simple-projection-cache16/summary.json) |
| Exact vision prefix + LiteRT GPU policy | 65.284 ms | 160 frames finite and normal cleanup; upstream checker FAIL in two quiet plan columns | Connected mixed prototype, no demonstrated speed advantage; [result](mixed-litert-160-confirm/summary.json) |

The larger bounded-LUT graph also ran but took 258.901 ms (see
`bounded-whole16-a`); isolated finite-domain accuracy did not make that approach
fast. Boundary sharing and final-query pruning have already been applied to the
accepted repair. Repeating these attempts is not a new optimization path.

**Decision:** stop pursuing an accurate approximately 32 ms implementation by
small changes to this runtime/model configuration. Accuracy can be repaired,
but the demonstrated repaired result is 52.4 ms. Keeping the old all-NPU speed
would require a correct, efficient HTP activation implementation or a runtime
that supplies one; no such implementation is currently built or validated here.
The separate [custom-HTP build research](../sdk/README.md) documents that as a
new development effort, not a guaranteed fix. This review performed no new
phone experiments and did not alter the installed APK or system settings.

## Historical preparation notes

**Prepared; no phone execution or speedup result yet.** Wireless ADB at `192.168.1.50:5555` responded when re-enabled through USB, then stopped accepting connections before this optimization turn started any model. A new host ADB service (37.0.1) found neither USB nor mDNS devices, and Fastboot found no device. The precise cause is unconfirmed. Device reconnection is required to run the experiment below.

## Reason for the experiment

The accepted accurate model remains approximately 62 ms/frame. Its instrumented ORT trace attributed 6.000 ms/frame to CPU GELU and 57.482 ms/frame to QNN partitions. Raising performance mode, priority, shared memory, or O3 is not new work: those are already enabled. Moving from the known accurate model to the native-HTP GELU graph would lose the accepted precision result.

The previously prepared `soc_model` and `vtcm_mb` knobs require new graph compilation. ORT v1.29's EPContext path loads an existing binary using `contextCreateFromBinary`; it does not re-finalize the graph. In particular, changing `vtcm_mb` while running the old cached context cannot establish a compilation benefit. `soc_model` also configures device creation, but does not rewrite a cached graph's scheduling. Sources: [ORT QNN EP source](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/providers/qnn/qnn_execution_provider.cc) and [Qualcomm HTP configuration](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html).

Instead of compiling the whole 792 MB hybrid graph again, isolate the existing `_130_40` policy partition. This partition took about 4.529 ms/frame in the accepted ORT trace. It contains 111 standard ONNX nodes and 13.0 MB of model data. No GELU node, custom domain, or EPContext remains inside it. The adjacent exact CPU GELU values become explicit inputs. The first comparison changes only `soc_model` from unset to `43` (SM8550), with fresh compilation for every A/B/A run. An 8 MiB VTCM reservation is deferred until the runtime/device capacity is established.

## Prepared files and host checks

- `extract_partition.py` copies the partition's original nodes and weights and uses original static shapes. It also builds a small wrapper around the accepted EPContext that exposes the partition's six inputs and six outputs. The cache binary reference is absolute: `/data/local/tmp/light-speed-gelu-both-ctx_qnn.bin`.
- `extraction-manifest.json` records source/model hashes, the exact boundary, and successful ONNX checker results. The standalone model is 13,002,409 bytes, SHA-256 `6fd17ee6b41bfd9dc7582ffb9baa55546dcd8c415a939b699bae600f8ffa8a57`.
- `partition_probe.c` retains the accepted runner's settings and adds selected tensor dumps, an eight-frame dump limit, and a 120-second alarm during session creation. The NDK build passed `-Wall -Wextra -Werror`; `build-manifest.json` records its hash. No phone workload was started.
- `run_partition_aba.ps1` passed the PowerShell syntax parser. It checks model transfer hashes, records device/thermal/memory snapshots, and uses unique output paths. It first captures eight real frames and requires the original main outputs to match the accepted baseline byte-for-byte before compiling anything. It then benchmarks default / SoC43 / default for 160 calls each, replaying the eight captured inputs. Each run captures the first eight sets of boundary outputs for numeric comparison. Numeric checks cover those eight sets, not all 160 timed calls. A GPU sensor reading at or above 60 C prevents a new run; a post-run reading of at least 75 C stops the sequence.
- `analyze_partition.py` checks file sizes, finite values, byte identity, hashes, and local numeric errors. Its `allclose` diagnostic (`rtol=1e-3`, `atol=1e-3`) is not a replacement for the upstream full-model per-head parity checker.

Build from the repository root with `& reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/build_probe.ps1`. Once the phone is reachable, run `& reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/run_partition_aba.ps1 -Serial <actual-ADB-serial>`.

## Decision after measurement

If SoC targeting has no repeatable benefit in the hot partition, stop this variable. If it helps while preserving the captured outputs, test a replacement of this single EPContext partition in the full accurate model before considering broader compilation. That replacement must pass the unchanged 160-frame upstream checker and demonstrate a full-model timing benefit; standalone timing is insufficient. The accepted binary would still retain the old unused partition, and an additional context can add memory or switching overhead. No installed APK, system library, clock, or thermal policy was changed.
