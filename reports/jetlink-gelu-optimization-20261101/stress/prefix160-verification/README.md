# 1000-frame stress test: prefix 0–159 verification

**Status: PASS for the 160-frame prefix only.** This does not validate frames 160–999.

- Found exactly 160 stress outputs (`out_0.bin` through `out_159.bin`). Each is 73,808 bytes (18,452 little-endian float32 values) and all values are finite.
- Every stress output matches the accepted `gelu-history-160-out/out_i.bin` baseline byte-for-byte.
- The upstream `verify_parity.py compare` command ran on the staged original `parity-160` `spec.json` and `out_ref_i.npy` files, with the original checker and thresholds unchanged. Exit code: 0; all slices and columns passed.
- No phone, model, thresholds, or source files were changed.

Artifacts: `verify_prefix160.py` reproduces the checks; `summary.json` records PASS and command status; `input-output-validation.json` records per-frame sizes, finite counts and SHA-256 hashes; `verify-parity.stdout.txt` and `verify-parity.exitcode.txt` preserve the original comparator run. The staged comparator dataset is under `compare/`.
