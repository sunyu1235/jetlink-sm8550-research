# Jetlink precision parity audit

This audit re-runs the upstream `scripts/verify_parity.py compare` command on eight existing 16-frame datasets and one new full-phone CPU32 run. It does not edit source datasets or thresholds. The command output for each run and its exact exit code are retained beside `audit.json`.

Thresholds are the upstream constants: `MIN_CORR=0.999`, `MIN_SAMPLES=16`, `QUIET_FRACTION=0.05`, `TINY_TOLERANCE=0.1`, and `CONSTANT_FRACTION=0.001`.

| Experiment label | Dataset | Exit | Result | Failed heads (worst reported statistic; max absolute error) | Max absolute error across heads |
|---|---|---:|---|---|---:|
| Original NPU whole, 16 varied frames | `parity-varied` | 1 | FAIL | `meta` frame corr 0.998422; `lane_lines` pooled; `lead` pooled; `plan` pooled; `pose` pooled; `road_edges` pooled | 5.1875 |
| NPU vision + PC CPU FP16 policy | `parity-hybrid` | 1 | FAIL | `lane_lines` `mu[1]` corr 0.998772; 0.3594. `road_edges` `mu[1]` corr 0.998910; 0.3516 | 2.1250 |
| NPU vision + CPU32 policy | `parity-policy-fp32` | 1 | FAIL | `lane_lines` `mu[1]` corr 0.998773; 0.3616. `road_edges` `mu[1]` corr 0.998911; 0.3537 | 2.0986 |
| PC CPU vision + NPU policy | `parity-cpu-vision-htp` | 1 | FAIL | `lane_lines` `std[0]` corr 0.998700; 0.3281. `lead` `mu[1]` corr 0.998181 and `std[3]` abs 0.3403 (bound 0.3359). `plan` `std[5]` corr 0.997868; 1.3906 | 3.7500 |
| All CPU32 on PC | `parity-all-fp32` | 0 | PASS | None | 0.2098 |
| Phone FP32 variant | `parity-phone-fp32` | 1 | FAIL | `lane_lines` pooled (`mu[1]` corr 0.998773); `road_edges` pooled (`mu[1]` corr 0.998911) | 2.0985 |
| Phone FP32 with CPU vision | `parity-phone-fp32-cpuvision` | 0 | PASS | None | 0.1081 |
| Phone CPU FP16 policy hybrid | `parity-phone-hybrid` | 1 | FAIL | `meta` frame corr 0.998970; `lane_lines`, `lead`, `plan`, `pose`, `road_edges` pooled | 4.6250 |
| Full phone CPU32, vision then policy | `phone-full-cpu32-compare` | 0 | PASS | None | 0.2101 |

All eight existing `parity-*` arrays in this table use the same 16-frame varied sequence. An earlier five-frame fixed-sequence capture (`parity-light-speed`) is excluded. The `parity-hybrid` case is NPU vision with PC CPU FP16 policy; `parity-phone-hybrid` is a separate phone CPU FP16 policy case. The CPU vision in `parity-cpu-vision-htp` was computed on PC, so it is not a full-phone result. The full-phone CPU32 comparison below is the new two-stage run with both vision and policy executed on the phone. The `parity-policy-fp32` and `parity-phone-fp32` results match closely and retain lane/edge failures. `parity-phone-fp32-cpuvision` passes the upstream checks. The PC CPU-vision + NPU-policy split fails `lane_lines`, `lead`, and `plan`. The all-CPU32 PC run and the new full-phone CPU32 run pass. The maximum absolute errors alone do not determine pass/fail; the upstream per-slice/per-column gates do.

The existing group labels follow the supplied experiment names. Their `out_ref` files are CPU references and `out_link` files are the experiment outputs. The new full-phone result feeds 16 phone CPU vision outputs into 16 phone CPU policy runs and compares its results with `parity-varied/out_ref_*.npy`.

The phone CPU32 run used ORT 1.29.0 with four CPU threads. Vision: mean 608.229 ms, p50 613.225 ms, p99 633.318 ms. Policy: mean 150.245 ms, p50 144.962 ms, p99 166.653 ms. Inputs were checked on device before running: 16/16 `new_img` files at 1,572,864 bytes; 64/64 policy tensors at exact expected sizes. All 16 vision outputs were 65,536 bytes; all 16 policy outputs were 73,808 bytes and had no non-finite values. Device timestamps were 2026-11-01 13:34–13:36 +08:00. Full hashes, commands, sizes, and timing details are in `phone-cpu32-run.json`; transferred outputs are retained in the `phone-*-output` directories.

## Reproduce

Run `python run_audit.py` from any working directory. It invokes the specified upstream script on the eight existing datasets and on the saved full-phone comparison directory, and writes `audit.json` plus nine unmodified upstream comparison logs here. Override `--root`, `--upstream`, or `--output` when those locations differ.

## Probe source review

Read-only review of `C:\Users\sunbo\Documents\moto-x40-root\qnn_context_probe.c`:

- Lines 39–40 set up a coherent ping-pong pair for `state_*` inputs and matching `next_state_*` outputs when runs are sequential. This means state is fed back from the previous probe inference.
- Line 41 checks `fread` against the expected byte count and returns 4 on a short read. The real gap is that `fopen` failure silently skips the tensor, leaving a zero or prior-frame buffer; it also does not check for extra trailing bytes. Before replay, record that every expected input file exists and has exactly `in[j].nbytes` bytes.
- The state input side of the second binding points at the opposite output buffer (`in_alt[i].data = out[j].data`, line 39). For autoregressive inference this ping-pong works. If captured `state_*` files are supplied, line 41 writes them into `in[j].data` only; on alternating runs, the second binding reads the ping-ponged `out[j].data` instead, so every other captured state is ignored. Current `vision-fp32-inputs` contains only per-frame `new_img`, and `policy-fp32-inputs` contains `view_39` plus controls (no `state_*` files), so state feedback is the intended behavior for these specific test inputs. Preserve that distinction.
- Line 40 only performs warmup when `argc <= 7`; directory-driven inference skips warmup. That does not by itself alter deterministic outputs, but latency numbers are not comparable with the warmup path.
- Line 41 writes `out_i.bin` for either `outputs` or `view_39` and uses the same filename. If a graph exposes both names as outputs, the later one overwrites the earlier one. Check output names before relying on those captures.

These issues are probe-replay correctness concerns; they do not change the already captured parity arrays used in this audit.
