# ORT node profile: accurate CPU+HTP candidate

## Result

The 16-frame run used the accepted shared-memory, `f8` affinity, original exact CPU GELU library, four ORT intra-op threads, and spinning disabled. The 16 output tensors were finite, 73,808 bytes each, and byte-identical to the accepted baseline. The original parity checker had already passed on the same outputs during previous runs; this short profile itself is additionally checked byte-for-byte.

Instrumentation increased observed mean execution time to 64.13 ms, so do not use this run for performance claims. The unprofiled 1,000-frame mean was 62.01 ms.

## Provider time in the ORT trace

| Provider | Node events / frame | Sum of node event durations / frame | Share of summed node event time |
|---|---:|---:|---:|
| CPUExecutionProvider | 63.0 | 6.000 ms | 9.5% |
| QNNExecutionProvider | 45.0 | 57.482 ms | 90.5% |

This is attribution from ORT event durations, not a hardware utilization measurement. The event sums closely track wall time but can include provider/runtime work differently; use the QNN optrace follow-up to split the HTP partition times internally.

## Largest provider events

See `ort-profile-summary.json` for the 20 largest unique provider op groups, their timings, shapes, and node indices. The full ORT JSON and captured outputs are in the ignored `outputs/jetlink-gelu-optimization-20261101/profile-20261102-a/` directory.

## Next step selected from this profile

The CPU GELU custom nodes account for about 6 ms/frame in the trace; the QNN provider partitions account for roughly 57 ms/frame. CPU table micro-optimization cannot recover the gap to 30 ms by itself. The next measurement should be QNN `optrace` for the largest HTP partitions, before making another graph change.
