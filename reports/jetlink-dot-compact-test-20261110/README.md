# Dot compact-boundary phone test, 2026-11-10

Dot commit: `af713f089848d1f6725a3ebfb7b3bf9f42e07296` on `research/jetlink-progress-20261110`.

The compact-boundary idea was adapted to the existing accurate Android model.
Only its upstream final-attention partition and following MLP/head partition
were rebuilt on the phone. The rest remains accelerated by the original QNN
cache. All 63 existing Android exact CPU GELUs remain, with three now receiving
one token rather than 288. The complete model still has 45 QNN EPContexts.
No new HTP custom package, Linux host bridge, quantization or distillation is
part of this candidate.

Six boundary buffers shrink from 2,654,208 to 9,216 bytes. No zero prefix is
restored. The following token-local tails use the compact shapes and squeeze
the one token instead of gathering token 287 from padded tensors.

## Actual phone results

Moto X40 XT2301-5 / SM8550, ORT 1.29.0, QNN 2.42.0. Same 160 synthetic inputs,
independent zero-initialized recurrent state for each process, CPU affinity f8,
shared buffers enabled, existing exact GELU library, burst mode, no profiling.

| Whole-model Run | Baseline | Compact |
| --- | ---: | ---: |
| All 160 frames, mean | 53.6327 ms | 52.4360 ms |
| Excluding first frame, mean | 53.2008 ms | 51.9981 ms |
| Excluding first frame, median | 53.0399 ms | 51.9548 ms |
| Excluding first frame, P99 | 54.9926 ms | 52.8097 ms |
| GPU sensor before / after | 23.6 / 48.1 C | 23.6 / 46.9 C |

Measured improvement excluding the first frame: **1.2027 ms / 2.2608%**.
The earlier paired 16-frame screen showed 1.3631 ms / 2.5569%; the 160-frame
measurement is the final comparison. Each Run timing includes the whole NPU/CPU
model, while excluding initialization, input file loading, camera and transport.
This short run is not a sustained thermal stress test or an energy measurement.

All 160 candidate outputs are finite and byte-identical to the previously
accepted 160-frame baseline outputs. The unchanged vendored upstream
`verify_parity.py` passes every original head/column/frame/pooled gate. This
validates numerical agreement for the existing synthetic sequence, not new
real-road accuracy.

**Conclusion:** the compact boundary is usable on this phone and gives a small
speed gain. Whole-model inference remains around 52 ms; this modification does
not establish a route to 30 ms. No APK was replaced or deployed, and no clock,
thermal or Android system policy was changed.

## Files and reproduction

- `summary.json`: final machine-readable timing and numerical results.
- `preparation` source: `prepare_cached_candidate.py`; it keeps existing weights
  and replaces only the two relevant cached partitions after recompilation.
- `phone_test.py`: exact native runner, runtime paths, build/cache options and
  execution parameters used here. Compilation uses ordinary buffers; measured
  whole-model runs use the original shared allocator.
- `baseline16.log`, `candidate16.log`, `baseline160.log`, `candidate160.log`:
  actual native execution logs.
- `parity160.log`: unchanged upstream checker output.
- `compare160/`: numerical inputs to that checker.
- Raw outputs, timing CSVs, source ONNX partitions and compiled context
  descriptors: `outputs/jetlink-dot-compact-test-20261110/` in the original
  workspace. Context binaries remain in the corresponding phone tmp directory.

Existing accepted model/weights, Android runtime and probe binary are required.
The source scripts record this workspace's paths and refuse existing model or
run outputs; adapt to a fresh output directory before repeating. The sequence
used was prepare, compile, compose, baseline16, candidate16, candidate160,
baseline160, then summarize. No source-only test suites or additional backend
sweeps were run for this result.
