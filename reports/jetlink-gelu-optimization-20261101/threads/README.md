# ORT intra-op thread-count A-B-A

## Result

Test: 160 frames from the existing seed-1 history sequence, recursively updated states, same context and original CPU GELU library, shared I/O enabled, `f8` affinity (CPUs 3–7), and ORT spinning disabled. Only `JL_CPU_THREADS` varied: 4 → 1 → 4.

| Run | Intra-op threads | Mean ms | P50 ms | P99 ms | First CSV battery temp | Root GPU temp before/after |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 4 | 61.985 | 61.659 | 64.254 | 21.0 °C | 24.4 / 49.7 °C |
| B1 | 1 | 68.363 | 67.637 | 71.643 | 21.0 °C | 26.8 / 50.5 °C |
| A2 | 4 | 61.966 | 61.606 | 64.394 | 21.0 °C | 28.4 / 53.2 °C |
| Four-thread A average | 4 | 61.975 | 61.633 | 64.324 | — | — |
| One thread minus A average | 1 | **+6.388 (+10.31%)** | **+6.004 (+9.74%)** | **+7.319 (+11.38%)** | — | — |

One intra-op thread was slower in this A-B-A. Keep the existing four-thread setting; do not promote one thread for this workload. The two A runs are close (mean difference 0.019 ms), but this remains one B run on one input sequence.

## Correctness

All three runs logged the intended intra-op count (4/1/4), `Cpus_allowed=f8` / `Cpus_allowed_list=3-7`, `shared=1`, and runner exit 0. Each produced 160 files of 73,808 bytes (18,452 float32 values), all finite. All three runs matched the accepted `gelu-history-160-out` baseline byte-for-byte for every frame and passed the unchanged upstream `verify_parity.py compare` command and thresholds (exit 0).

The runner source emits `intra_op.num_threads` immediately before ORT environment/session setup. `JL_CPU_THREADS` defaults to 4 for the CPU/custom-op path and rejects non-integer values or values outside 1–8. No-model device preflights confirmed `0`, `9`, and `bad` are rejected with exit 2 before ORT initialization. Actual performance runs exercised 1 and 4 threads; the upper endpoint 8 was not benchmarked. A0/A2 explicitly set `JL_CPU_THREADS=4`, so the unset-variable default is established by source logic, not a separate full inference.

## Temperature and constraints

The first CSV battery sample is the consistent in-run temperature sample and was 21.0 °C for each run. The CSV GPU field is `-1` (unreadable to the shell runner); root-only KGSL sensor snapshots are recorded separately in the table. No GPU start reading reached the 60.0 °C stop threshold. No clock/frequency, thermal setting, app, model, system library, or prior runner was changed. The new runner and all outputs use new `/data/local/tmp/.../threads` paths; no prior phone output was deleted or overwritten.

## Build and artifacts

- `history_probe_threads.c` is an isolated copy of the existing probe with only intra-op environment parsing, range validation, and a logged thread count added. The upstream source remains unchanged.
- Built with the existing NDK 27.3.13750724 targeting Android API 30 and `-O2 -Wall -Wextra -Werror -DJL_ORT_DLOPEN`; compilation succeeded. Local and deployed runner SHA-256 match; see `build-manifest.json`.
- `run-threads-aba.ps1` records unique per-run output/CSV/log paths and checks GPU temperature before continuing. `analyze_threads.py` validates outputs, baseline byte identity, and parity.
- `summary.json`, `checks.json`, the original comparator stdout/exit per run, range-rejection logs, and all pulled raw outputs are preserved in this directory and `outputs/jetlink-gelu-optimization-20261101/threads/`.
