# Downstream last-query tail reduction

## Scope versus existing accepted prototype

The repository's `prepare_last_query.py` already prunes final attention queries, residuals, normalization and first MLP projection in partition 131_41. It restores six output boundaries to [1,288,C] with zero prefixes for compatibility with downstream cached partitions. Repeating that optimization is not new work.

This experiment moves last-token selection before the subsequent GELU and second projection. In three original branches, the sole use-def chain is GELU → second MatMul → bias Add → residual Add → Gather(axis=1,index=-1). All operations before Gather are token-local. Selecting the preactivation and residual first therefore preserves mathematical output semantics. Actual weights, GELU approximation and recurrent state semantics are unchanged. The isolated full-model rewrite intentionally does not modify attention or first projection, so its new contribution is separable from existing pruning.

Each second projection reduces from 288×1024×512 = 150,994,944 MACs to 524,288 MACs. GELU consumes 1,024 instead of 294,912 values. Across three original branches this removes 451,411,968 MACs per inference before any duplicate-branch elimination. These are graph-work counts, not measured speedups. The duplicate off-policy branches must not be counted twice after CSE.

## Verification

- Original teacher weights and index -1 verified. Sole consumer chains asserted. Original four output signatures retained. ONNX checker passed.
- Actual-weight tail CPU microbenchmark: full/compact random seed2861 outputs bit-identical. Median3.60ms versus0.156ms, 50 measurements after10 warmups, ORT CPU with2 threads and graph optimizations disabled. Another full-model CPU job was active, so these are indicative microbenchmarks, not controlled pipeline or target-device measurements.
- Full recurrent synthetic160 passes the original unchanged checker with independent candidate state. Evidence: synthetic160-parity.log.
- Night320 and day320 each PASS both all-frame and predeclared warmed-last160 checks, unchanged checker. See results.json and road score logs.

## Deployment implications

This cannot be spliced into old cached contexts without changing signatures. To realize all boundary benefits, retain compact [1,1,C] or [1,C] outputs from partition131_41 and rebuild downstream partitions/CPU bindings accordingly; do not reintroduce zero prefixes. The six original FP16 boundaries total2,654,208 bytes versus9,216 bytes compact, a structural reduction2,644,992 bytes before considering actual binding, aliasing or transfers. Compact GELU may remain CPU as an accuracy fallback; its small vector no longer requires processing the unused prefix. An all-HTP version must use the verified correct GELU implementation, not the known inaccurate native activation.

The standalone compact graph currently accepts original full-shaped inputs and gathers internally, deliberately isolating tail computation. It does not by itself demonstrate reduced partition-boundary transfer or full-model HTP execution. Native compact tail compilation, SoC43/v73 preparation, reload and host execution PASS. Rank2 GELU is reshaped to the verified rank4 package ABI. Gather-retained and compact-input variants are bit-identical over8 matched synthetic samples/4096 outputs; this is comparison between the two native variants, not a fullmodel target execution. Evidence: /workspace/shared/htp-tail-compile/result.json. No phone latency or 32ms claim.
