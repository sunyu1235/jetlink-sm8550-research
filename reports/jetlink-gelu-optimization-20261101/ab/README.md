# Shared input/output memory A/B

- Shared=1 smoke (16 frames) passed; no allocation, symbol or nonfinite errors.
- Primary full run order: A0 shared=0, B1 shared=1, A2R shared=0. Each ran the same 160 frames, context, custom library and probe options; only the shared-memory option changed.
- All primary runs and the supplementary A2 run passed the unchanged upstream parity comparator. All 160 outputs per run were finite, exactly 73,808 bytes each, and byte-for-byte identical across every full run.

| Run | Shared | Mean ms | P50 ms | P99 ms | GPU temp start/end °C* | First CSV battery °C |
|---|---:|---:|---:|---:|---:|---:|
| a0-shared0 | 0 | 73.660 | 73.874 | 77.429 | 26.8/51.3 | 22.0 |
| b1-shared1 | 1 | 72.402 | 72.544 | 75.011 | 28.4/52.8 | 22.0 |
| a2r-shared0 | 0 | 74.198 | 74.516 | 77.951 | 28.4/54.4 | 23.0 |

- Against the average of A0 and A2R, shared=1 improved mean by 1.527 ms (2.07%), P50 by 1.650 ms (2.22%), and P99 by 2.679 ms (3.45%).
- A0 versus A2R baseline differences: mean 0.538 ms, P50 0.642 ms, P99 0.523 ms. The shared=1 result is faster than both baseline brackets. Only one full shared=1 run was performed, so independent B repetition remains unmeasured.
- A2’s first attempt started 2.4°C above A0 and is excluded from the primary speed comparison; its output is retained and passes correctness.
- The runner’s first-frame CSV records battery temperature, while GPU temperature/clock are `-1` because the shell process cannot read KGSL nodes. Separate root-only GPU snapshots are retained as ancillary start/end readings; the table marks them with an asterisk. Battery CSV temperatures were 22.0°C for A0/B1 and 23.0°C for A2R.

## Parity and integrity

- `verify_parity.py` exit code was 0 for A0, B1, A2 and A2R; original thresholds were unchanged.
- Raw outputs were identical byte-for-byte for every pair of full runs; every frame was finite and exactly 18,452 float32 values.
- Raw/CSV logs, commands, exact exits, temperature records and hashes are in this directory.
