# CPU affinity A-B-A: shared=1 GELU-both context

## Result

With the same 160-frame input, context, custom-op library, runner options, and shared-memory setting (`shared=1`), constraining execution to non-little CPUs (`f8`, CPUs 3–7) was faster than allowing all CPUs (`ff`, CPUs 0–7) in this A-B-A sample.

| Run | Affinity | Mean ms | P50 ms | P99 ms | Min–max ms | First CSV battery temp |
|---|---|---:|---:|---:|---:|---:|
| A0 | `ff` (0–7) | 72.448 | 72.613 | 75.989 | 65.316–78.238 | 22.0 °C |
| B1 | `f8` (3–7) | 61.840 | 61.624 | 64.458 | 61.341–69.134 | 22.0 °C |
| A2 | `ff` (0–7) | 72.630 | 73.162 | 75.669 | 68.092–77.949 | 23.0 °C |
| A average | `ff` | 72.539 | 72.888 | 75.829 | — | — |
| B minus A average | `f8` | **−10.699 (−14.75%)** | **−11.264 (−15.45%)** | **−11.372 (−15.00%)** | — | — |

This is one B run bracketed by two A runs. The outputs are byte-for-byte identical across A0, B1, and A2 for all 160 frames, so affinity changed timing but not the result for this input. All three runs also passed the unchanged upstream `verify_parity.py compare` command and thresholds; every output was 73,808 bytes, float32, and finite.

## Device and command verification

Read-only topology values identified CPU0–2 as the little cluster (capacity 280, max 2.016 GHz), CPU3–6 as the middle cluster (capacity 855, max 2.803 GHz), and CPU7 as the big core (capacity 1024, max 3.187 GHz). The tested `f8` mask therefore selects all non-little CPUs, 3–7. `taskset --help` on the device documents a hexadecimal mask followed by a command. The run wrapper recorded `/proc/self/status` `Cpus_allowed` and `Cpus_allowed_list` before `exec` of the same runner, confirming `ff`/`0-7` and `f8`/`3-7` respectively. Toybox `taskset -p` printed a misleading abbreviated mask in this environment, so the report uses `/proc` status as the effective-mask evidence.

The only inference setting changed between A and B was the process CPU affinity mask. Each command used the same runner, `shared=1`, 160 frames, model, custom-op library, input path, profiling option, RPC option, and output type. The exact commands and stdout are retained per run.

## Temperature and measurement limits

The runner CSV cannot read the GPU sensor (`gpu_milli_c=-1` throughout). Following the requested common sampling point, the first-row CSV battery temperatures are used for the run-to-run CSV comparison: A0 22.0 °C, B1 22.0 °C, A2 23.0 °C. Root-only KGSL temperature snapshots are separately retained as ancillary pre/post readings: A0 26.4/50.9 °C, B1 29.2/53.6 °C, A2 30.7/55.2 °C. These are not substituted for the CSV first-frame readings; the A0-to-B1 root snapshot starts differ by 2.8 °C and remain a thermal caveat. KGSL frequency could not be read in this run because the sysfs frequency node denied access.

The observed 14.75% mean reduction is promising, but this is one bracketed affinity comparison on one sequence and is not a broad performance guarantee. The later CPU unrolled-GELU candidate has not been exercised in this test.

## Reproduction and artifacts

- `run-affinity-aba.ps1` records and runs the A-B-A commands. It does not alter model, app, CPU frequency policy, or thermal controls.
- `analyze_affinity.py` validates file sizes, finite values, byte equality, computes latency statistics, stages per-run `out_link_*.npy` with the original references, and runs the original parity comparator.
- `summary.json` contains the complete per-run statistics, parity exit codes, output hashes, and caveats.
- `hash-manifest.json` records SHA-256 hashes for the model, runner, custom-op library, original input manifest, each raw frame output, CSV, command, and logs.
- `compare/<run>/compare.stdout.txt` and `compare.exitcode.txt` preserve the upstream comparison. The first import-only attempt is retained as `.import-failure`; the rerun used the same environment setup as the existing audit (`PYTHONPATH` set to the upstream research package root) and passed.
- Raw outputs and per-run CSV files are under `outputs/jetlink-gelu-optimization-20261101/affinity/` and this report directory.
