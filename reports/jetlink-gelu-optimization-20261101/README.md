# Jetlink GELU optimization: evidence summary

**2026-11-09 latest result:** the accurate native prototype now averages
**52.409 ms** over 160 recurrent synthetic frames; all 160 match the accepted
baseline byte-for-byte and pass the original checker. The old **32.475 ms**
all-NPU shared-memory configuration did not pass numerical parity. Existing
repair and approximation results do not demonstrate a way to preserve that
32 ms speed with acceptable parity. The mixed NPU-vision/GPU-policy prototype
runs but averages 65.284 ms and fails the full 160-frame checker in two quiet
plan columns. See [the current results and decision](partition-tuning-20261109/README.md).
The November 2 summary below is retained as historical evidence.

**State as of 2026-11-02.** This report summarizes completed offline and phone measurements; existing per-test reports and raw artifacts remain the detailed source of record. A sustained 1,000-frame run of the best supported candidate completed, and its first 160 frames passed the original parity checker and matched the accepted baseline byte-for-byte. The timing remains well above the 30 ms target.

## Best supported configuration

Keep the **original CPU GELU library + `shared=1` + `f8` affinity + 4 ORT intra-op threads + `intra_op.allow_spinning=0`** as the best supported candidate. This configuration preserves parity on the tested 160-frame sequence and had the lowest demonstrated full-run latency among the tested conservative configurations. The results below are controlled experiments, not a production guarantee: each A-B-A comparison had only one B run, and thermal sampling has limits.

The prior accurate original-library CPU+NPU candidate averaged about **61.8 ms/frame** in the `f8/shared=1` A-B-A run. The earlier roughly 40 ms path failed accuracy comparison and is **not** treated as a speed baseline or improvement.

## Sustained run

With the configuration above, the 1,000-frame synthetic replay averaged **62.01 ms/frame**, with **63.78 ms P99** and **70.59 ms maximum**. The loop, including writing output files, ran at 15.77 frames/s. The first 160 frames were finite, byte-identical to the accepted 160-frame baseline, and passed the unchanged upstream parity checker. The other 840 recurrent frames were checked for finite outputs and timed, but not numerically compared. Battery temperature moved from 23°C to 24°C; the GPU thermal sensor reached 75.7°C. This is a prototype measurement, not the installed APK's live performance. Full details and scope are in [the sustained-run report](stress/README.md), with machine-readable timing in [best-1000-summary.json](stress/best-1000-summary.json) and parity evidence in [prefix160-verification](stress/prefix160-verification/README.md).

## Full-run measurements

| Variable | A/B/A result | Correctness | Assessment |
|---|---|---|---|
| Shared I/O (`shared=0` vs `1`) | `shared=1`: mean 72.402 ms; A0/A2R `shared=0` mean 73.660/74.198 ms. Against the two A runs’ average, mean improved 2.07%, P50 2.22%, P99 3.45%. | All compared 160-frame outputs finite, 73,808 bytes, byte-identical; original parity checker passed. | Keep `shared=1`. Only one full B run; see [shared A-B-A summary](ab/summary.json). |
| CPU affinity (`ff` vs non-little `f8`), `shared=1` | `ff` A mean 72.448/72.630 ms; `f8` B mean 61.840 ms. Against A average: mean −14.75%, P50 −15.45%, P99 −15.00%. | Three outputs byte-identical across all 160 frames; all parity checks passed. | Keep `f8` for the candidate. Single B run. First CSV battery sample matched A0/B1 (22°C); root GPU start snapshots differed 2.8°C, so thermal comparability is limited. See [affinity summary](affinity/summary.json). |
| ORT intra-op threads (`shared=1`, `f8`) | 4-thread A mean 61.985/61.966 ms; 1-thread B mean 68.363 ms. B was +10.31% mean, +9.74% P50, +11.38% P99 vs A average. | 160 outputs per run matched the accepted baseline byte-for-byte; all parity checks passed. | Keep 4 threads. Single B run; see [thread-count summary](threads/summary.json). |
| GELU CPU loop unrolling (original vs unrolled library), `f8/shared=1` | Original A mean 61.717/61.795 ms; unrolled B 61.392 ms. Against A average: mean −0.59%, P50 −0.59%, P99 −0.80%. | Every output matched the original accepted `shared=1` baseline bit-for-bit; all three parity checks passed. | **Do not promote**: under 1% in a single B run, not a demonstrated repeatable benefit. Keep the original library. See [unroll summary](cpu/ab/summary.json). |
| ORT spinning (`allow_spinning=0` vs `1`) | 16-frame smoke only: 62.285 vs 61.280 ms mean. | All 16 output frames were byte-identical. | Keep spinning off. The timing is a short smoke, may carry a power cost, and was not repeated over 160 frames. See [spin proof](spin/smoke-proof.json). |
| Sustained best-candidate replay | 1,000 frames: 62.011 ms mean, 61.790 ms P50, 63.775 ms P99, 70.594 ms max; 15.77 loop FPS including output writes. | All 1,000 outputs finite; prefix frames 0–159 byte-identical to accepted baseline and passed original parity thresholds. Frames 160–999 were not numerically compared. | Confirms steady operation near 62 ms for this synthetic replay. GPU thermal sensor reached 75.7°C; the installed APK was not changed. See [stress report](stress/README.md). |

The first-frame CSV battery temperature is the consistent within-run sample. The runner's GPU temperature field is unavailable (`-1`); separate root-only GPU snapshots are ancillary and are not substituted for CSV samples. Do not treat the one-B A-B-A point estimates as stable population performance.

## Precision and LUT findings

- The **49-op pure-NPU standard-op LUT attempt failed**: the composed HTP graph returned QNN error 1100 and segfaulted. Do not describe it as a successful NPU implementation. Component Cast and Gather checks passed alone, but the composed index calculation produced 60,317/63,488 mismatches, including 56,689 out-of-bounds indices. The cause remains unidentified; no complete model was converted. See [LUT report](lut/README.md).
- In isolated diagnostics, **Cast-only** and **Gather-only** each matched all 63,488 finite FP16 cases. The **index-only** graph did not: 60,317 mismatches, with 56,689 out-of-bounds indices. The failing combined graph therefore remains unexplained at the individual-op level; no complete model was converted to or validated with this pure-NPU route. Evidence: [`lut/cast-errors.json`](lut/cast-errors.json), [`lut/gather-errors.json`](lut/gather-errors.json), [`lut/index-errors.json`](lut/index-errors.json), and [`lut/native-erf-domain.log`](lut/native-erf-domain.log).
- The separate CPU full-domain GELU lookup proof showed numerically equal values; its only noted bit-level discrepancy was the sign bit of zero. The independent CPU unrolled-kernel unit proof covers 63,491 cases per formula with zero value or bit mismatches ([`cpu/unit-proof.json`](cpu/unit-proof.json)). These are distinct checks and should not be conflated.
- A single-Cast HTP diagnostic and a Gather-only HTP diagnostic each matched 63,488 finite inputs. The isolated index HTP graph was not correct. This is operator-level evidence only, not a successful full-model NPU LUT conversion.

## SDK and scope

The QNN 2.42 / HTP v73 custom-op route remains a fallback research path. The exact 2.42 Hexagon toolchain package was not obtained; the current Qualcomm V73 toolchain table cannot establish historical 2.42 compatibility. No SDK was downloaded or installed. No app, system library, root setup, frequency policy, or thermal control was changed for this offline summary. See [SDK research](sdk/README.md).

## Hashing and reproducibility

Run `python reports/jetlink-gelu-optimization-20261101/hash_versions.py` from the repository root to regenerate `version-hashes.json`. It SHA-256 hashes the key model, runtime, custom-op libraries, proof files, and per-experiment summaries that support this review. Raw frame hashes and exact commands are retained in the individual experiment manifests.

The main A-B-A source files are [`affinity/README.md`](affinity/README.md) and [`cpu/ab/README.md`](cpu/ab/README.md). The sustained replay and its prefix parity evidence are in [`stress/README.md`](stress/README.md) and [`stress/prefix160-verification/README.md`](stress/prefix160-verification/README.md).

## Focused QNN profile follow-up (2026-11-02)

**2026-11-09 update:** The compile-time settings below cannot be validated by loading an old EPContext. A 111-node, 13 MB hot policy partition has now been extracted, checked, and paired with an input-capture wrapper and a compiled probe. The prepared experiment recompiles this small graph for default / SoC43 / default, then requires full-model parity before promotion. VTCM is deferred until device capacity is confirmed. No new phone timing is available because ADB disconnected before a model was started. See [`partition-tuning-20261109/README.md`](partition-tuning-20261109/README.md).

The accurate ORT trace attributes about 6.0 ms/frame to CPU GELU nodes and 57.5 ms/frame to QNN partitions. Four QNN partitions alone account for about 18.8 ms/frame. The completed HTP optrace came from the all-HTP native-GELU graph that fails parity, so it only ranks workload families; it does not identify the exact accurate-path HTP hotspots. The accurate graph's direct optrace could not complete after wireless ADB disconnected. The accepted runner already uses burst performance, high QNN priority, shared memory, and graph optimization mode 3. A narrowly scoped next check is prepared for the two remaining documented QNN configuration variables: explicit SM8550 target model 43 and VTCM size 8 MiB, each tested separately against the accepted accurate model with parity. Details and the compiled runner are in [`runtime-tuning-20261102/README.md`](runtime-tuning-20261102/README.md). This is a measured hypothesis, not an expected speedup claim.

