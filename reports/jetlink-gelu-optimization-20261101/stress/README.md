# Sustained run of the best supported candidate

## Configuration

- Original, accepted CPU GELU lookup library.
- Shared I/O enabled.
- Linux CPU affinity mask `f8` (online CPUs 3 through 7 on this device).
- ORT intra-op spinning disabled.
- Same compiled CPU+NPU graph and unchanged QNN/runtime libraries.
- No frequency, voltage, or thermal-policy changes.
- 1,000 frames with the same 160 synthetic input frames cycled repeatedly; recurrent states were carried continuously and not reset between cycles.

## Result

The run completed with exit code 0. Its per-frame output was finite for all 1,000 frames. Timing was 62.0105 ms mean, 61.7898 ms median, 63.775 ms P99, and 70.5944 ms maximum. The complete loop including file output ran at 15.77 frames/s. Battery sensor temperature moved from 23°C to 24°C; the phone's GPU thermal sensor moved from 31.5°C to a maximum of 75.7°C. No cooling or clock changes were made.

The first and last 160-frame timing means were 61.7632 ms and 62.5360 ms respectively. This suggests a small warming drift in this run; it is not a controlled thermal comparison. The run remained about 15.9% faster in mean than the earlier 800-frame run, but that cross-session comparison had different duration and temperature conditions. Use the controlled A-B-A CPU-affinity measurements for the stronger speed comparison.

## Accuracy scope

All 1,000 outputs were checked for finiteness and have per-frame timing records. Frames 0–159 are byte-identical to the accepted 160-frame baseline and passed the upstream parity checker with original thresholds unchanged (exit code 0). This does not establish numerical parity for the remaining 840 recurrent frames. `prefix160-verification` contains the checker output, exact exit code, staged data, and per-frame hashes. The full 1,000-frame outputs remain on the phone, so the test did not transfer the roughly 74 MB of output tensors.

The result is a native prototype measurement. The installed Jetlink APK remains unchanged and does not expose this candidate as its production backend.
