# Pure NPU GELU lookup-table experiment

## Outcome

The standard-ONNX route that expands GELU into a half-domain index calculation followed by `Gather` was **not usable on this phone's QNN 2.42 HTP path**. The composed erf graph returned QNN error 1100 and then crashed. It did not produce a valid full model, and it was not installed in the app.

The component checks narrow the issue without identifying its exact cause:

| HTP diagnostic | Elements | Result |
|---|---:|---|
| FP16-to-INT32 Cast | 63,488 | Exact values, no mismatches |
| Gather with known-good host indices | 63,488 | Bit-identical, no mismatches |
| Composed index arithmetic | 63,488 | 60,317 mismatches; 56,689 indices out of bounds |

Therefore Cast and Gather individually work for these isolated cases, while the combined index graph is wrong. The evidence does not isolate which intermediate operator or numeric conversion causes the bad indices. See `cast-errors.json`, `gather-errors.json`, `index-errors.json`, and the raw native logs.

## CPU lookup proof is separate

The CPU exact lookup custom op is the accepted accuracy repair used by the best timing candidate. Its independent unrolled-kernel unit check covers 63,491 values per formula with no value or bit mismatches. A separate standard-op CPU test had signed-zero bit differences only; do not treat that result as bit-exact proof for the NPU path.

## Decision

Do not substitute the pure-NPU candidate or relax parity thresholds. Revisit only with operator-level profiling or a version-matched Qualcomm QNN 2.42 / HTP v73 development toolchain that can establish and validate a custom kernel.
