# Reconstructed compact partition 131_41

This deliverable exposes six compact FP16 boundaries without zero-prefix restoration. It preserves all 288 K/V tokens and original weights. It is ready for downstream integration and backend compilation, not accepted as a complete driving model.

## Source and provenance

The accepted QNN cached context and optimized partition files were unavailable. The baseline is reconstructed by the archived six-input/six-output cut from `/workspace/shared/jetlink-distillation/real-block/teacher-capture.onnx` and its original external weights. This source was produced from the public `light-speed.onnx`; the capture script lowered `org.tinygrad::Contiguous` to `Identity` and added two observation outputs. Neither custom nodes nor capture outputs lies inside this selected cut.

The reconstructed baseline has 87 nodes, not the archived accepted baseline's 111 nodes. It retains raw `MatMul` plus `Add` and native `LayerNormalization`; it is not byte-identical recovery of the accepted optimized/cached partition. No graph/model equality with that unavailable accepted artifact is claimed. Source-model, external-weight-file, boundary-manifest, per-initializer data-range and output artifact hashes are recorded in `artifacts/compact-partition-manifest.json`.

The earlier repository algorithm pruned Q/residual/first MLP projection and restored six `[1,288,C]` outputs. This implementation adapts the same pruning proof to the raw teacher by slicing before its token-local Q LayerNormalization, then **omits restoration**. It does not claim attention pruning itself as a new optimization.

## Artifacts and integration

- `artifacts/reconstructed-baseline-131-41.onnx`: 87 nodes, 13,059,888 bytes.
- `artifacts/compact-partition-131-41.onnx`: 90 nodes, 12,805,692 bytes, self-contained weights.
- `artifacts/compact-partition-manifest.json`: source provenance, names/order, shape and mask proofs.
- `artifacts/integration-cut-proof.json`: all 87 original node names to remove, exact external consumer edges.
- `artifacts/numeric-validation.json`: complete local numerical and CPU diagnostic evidence.
- `artifacts/final-audit.json`: final-script reproducibility and complete attention shape/lineage proof.

Input order and shape are unchanged:

1. `gelu_43`, `gelu_54`, `gelu_61`: FP16 `[1,288,1024]`.
2. `add_50`, `add_70`, `add_86`: FP16 `[1,288,512]`.

Compact output order and downstream name-only binding:

1. `add_52__last_query` → `add_52`: `[1,1,512]`.
2. `add_72__last_query` → `add_72`: `[1,1,512]`.
3. `add_88__last_query` → `add_88`: `[1,1,512]`.
4. `linear_110__last_query` → `linear_110`: `[1,1,1024]`.
5. `linear_151__last_query` → `linear_151`: `[1,1,1024]`.
6. `linear_180__last_query` → `linear_180`: `[1,1,1024]`.

Alias names when binding downstream inputs; do not pad or concatenate. The six FP16 output payloads shrink from 2,654,208 to 9,216 bytes (288×, 2,644,992 fewer bytes). These are logical tensor bytes, not measured DMA traffic. Old downstream cached contexts cannot consume the new shapes unchanged.

The cut's only external consumers are exactly the three final GELUs and three residual Adds, through these six declared boundaries. No selected intermediate has an undeclared external consumer or source graph observation output. Remove the 87 named source nodes, insert compact nodes and compact tails, bind the suffix names, prune unused weights, invalidate changed intermediate metadata, and topologically sort. The original source cut contains interleaved branch nodes: do not insert all new nodes at a single early point without sorting.

## Verification

- ONNX full checker and strict shape inference with type checks pass. Every node output has a concrete positive shape.
- All three last-query mask rows are verified fully unmasked before replacing `Where` with `Identity`.
- Three Q slices and three residual slices are `[287:288]`. All three compact Q/K/V MatMuls have verified shapes: Q `[1,8,1,64]`, Kᵀ `[1,8,64,288]`, scores `[1,8,1,288]`, V `[1,8,288,64]`, context `[1,8,1,64]`.
- Every ancestor of the final K/V inputs is byte-identical to the reconstructed baseline. All retained original initializers are byte-identical. No `Concat`, restoration node or zero-prefix tensor remains.
- Ten regression tests pass, including rejection of input reordering, boundary-shape drift, masked final queries, extra Q consumers and non-token-local Q normalization.
- Twelve matched synthetic cases compare 55,296 compact output values to baseline row 287. All outputs are finite and pass the predeclared local screen `atol=1e-3, rtol=1e-3`. **Not bit-exact:** 861 values differ; maximum absolute difference is 0.015625 on a last-token-only stress case. Ordinary unit-Gaussian cases have maximum absolute difference at most 0.001953125. This local threshold does not substitute for the unchanged upstream per-head acceptance checks.
- Changing only the input prefix while preserving every last row changes all six outputs, consistent with retaining history through K/V.
- ORT 1.23.2 CPU, optimizations disabled, two threads, 20 alternating paired measurements after three warmups: median baseline 50.09 ms, compact 25.08 ms. This is an isolated shared-host CPU diagnostic, not controlled full-pipeline or phone/HTP latency. CPU FP16 execution may include internal promotion.

Not run here: full recurrent model replay, road acceptance, QNN compilation or phone execution. These are integration-stage requirements. No device was accessed. Numeric differences require whole-model regression before accepting the composition.

## Reproduce

Use the existing dependencies, or an environment with ONNX 1.20.1, ONNX Runtime 1.23.2 and NumPy:

```sh
cd /workspace/shared/jetlink-compact-boundaries
export PYTHONPATH=/workspace/shared/jetlink-python
export OPENBLAS_NUM_THREADS=1
python prepare_compact_partition.py --output-dir fresh-artifacts
python validate_compact_partition.py \
  --baseline fresh-artifacts/reconstructed-baseline-131-41.onnx \
  --compact fresh-artifacts/compact-partition-131-41.onnx \
  --report fresh-artifacts/numeric-validation.json
python -m unittest -v test_compact_partition
```

Both CLIs refuse to overwrite existing evidence. `prepare_compact_partition.py --source ... --contract ...` allows explicit source/contract paths but rejects mismatched graph contracts. Extraction loads only the selected tensor ranges rather than all 790 MB of external weights. Source-weight file hashing reads it sequentially without materializing the whole model.
