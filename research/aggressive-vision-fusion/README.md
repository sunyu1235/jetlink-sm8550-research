# Aggressive vision fusion: experimental, not a phone speed result

## Why this route

The latest verified phone branch remains `codex/phone-compact-test-20261110`
at `4619484678610712fc132b4bf11848cb4a35e481`. Its compact candidate retains
63 exact CPU GELUs and 45 QNN EPContexts. Recomputed original CSV means,
excluding the first frame, are 53.200832 ms baseline and 51.998085 ms compact.
That is 1.202747 ms against this run's baseline, not against the historical
51.970485 ms best. The exact-CSE branch remains at `b72b1b3`; neither branch
contains a new CSE phone result. Both branch identities were checked again.

This experiment does not repeat CSE, final-query pruning, the slow six-op
activation projection, the failed full twelve-block SiLU student, or blanket
quantization. It joins the entire vision encoder, including its 36 correct
erf GELUs, into one native HTP context. The policy portion stays outside this
cut. The previous whole-context host crash is not retried unchanged.

## Exact cut and scope

- QNN converted region: 482 nodes, `cat_2` FP16 [1,24,128,256] to `view_39`
  FP16 [1,32,512]. All three downstream consumers remain connected.
- Corresponding ONNX cut: 421 nodes from `_to_copy_4` to `view_39`. The original
  uint8-to-FP16 cast remains unchanged. Integer images 0..255 are exactly
  representable in FP16.
- Only 1,572,864 input bytes and 32,768 output bytes cross this new region's
  public boundary. This is not a claim that those bytes are all copied or that
  all internal traffic disappears.
- GELU workload: 24,379,392 values per frame, 380,928 complete 64-value stripes.
  Separate logical FP16 reads/writes total 97,517,568 bytes (93 MiB), before
  conversions and other operations. These are work counts, not timings.
- `preserve_fixed.cpp/bin` retain the earlier arithmetic-preserving conversion
  and exact-value bias repair. The extractor does not alter selected node or
  constant bodies. Different backend fusion/layout decisions still require
  numerical validation.

The source is the reconstructed compact host model, SHA-256
`3c80a2e4ede9781a1e5aec41c503e12e2f2c739e987e168046de8ed87325800a`,
not the missing accepted Android cached model. Its required external weight
file SHA-256 is
`b356db4f9b2ed7cc563c2f4534715487ae4b82319e8225f657dcfa77d082bb75`.

## Completed gates

1. Twenty-six extractor tests pass. They cover closure, original graph outputs,
   live side outputs, malformed boundaries, and JSON/C++ descriptor agreement.
   Re-extraction after hardening reproduced the running C++ and weight archive
   byte for byte.
2. QNN 2.42 model library build, SoC43/v73 O3 context preparation, saved-context
   reload and actual host `graphExecute` pass. The one-frame host execution
   completed in 622 seconds. This is explicitly not a phone latency.
3. The source-only CPU cut control produces byte-identical heads and all three
   recurrent states on the first recorded-night frame.
4. Actual native vision produces 16,384 finite feature values. Against CPU:
   RMSE 0.0028205852, max absolute error 0.04296875, correlation 0.9999987997.
5. Feeding those native features into the unchanged CPU remainder passes the
   original full-head checker for that one frame. Head max absolute error is
   0.25. Image/desire states are byte-identical; feature-state max error is
   0.04296875. The candidate is not bit-identical to CPU.

The bridge and exact source-model replacement are implemented under `bridge/`
and `build_model.py`. Their structural/malformed-input checks pass. The separate
whole-model bridge invocation has now passed: all four outputs are byte-identical
to the passing split-native control. Evidence is in
`bridge/full-frame-check/result.json`. The final vector-package invocation also passed, with all four outputs
byte-identical to the same split-native scalar control, no nonfinite values, and
zero maximum error. See `bridge/vector-full-frame-check/result.json` and the
package/backend/system/weight provenance alongside it. Each package had exactly
one full-model invocation. The unchanged original checker also passed when
run directly on the saved actual vector output against the CPU baseline;
see `bridge/vector-full-frame-check/cpu-baseline-check/`. This remains only one
independent-zero-state frame. The approximately 600-second host times are only
functional checks, not performance evidence.

## Vector GELU development

The scalar exact lookup is only a correctness control. Moving a scalar lookup
over 24.38 million values onto DSP does not establish a useful speedup.

A companion true-HVX prototype is under
`../../experiments/gelu-polynomial` in this workspace. It approximates the normal
CDF in 32 intervals, multiplies by x, then applies exhaustively verified raw-half
corrections and special-value handling. Coefficient access uses register lookup,
not unsupported DDR gather or the slow 64-window VTCM gather experiment.
The corrected core passed 1,048,576 raw comparisons on each of native19 and
native21, and 524,288 on an actual compiled v73 ELF under QEMU. These include all
65,536 FP16 payloads, permuted mixed input, in-place operation and canaries; host
checks also exercise concurrent calls. A lane-ordering defect found by mixed
input was corrected before the final passes. Do not treat the old sorted-only
test as sufficient.

The four-layout QNN package passed a 24-graph baseline with 9,702,056 raw
comparisons and four forced-layout suites with 30 graphs and 9,734,492
comparisons each. All mismatches were zero. The forced Crouton-to-Plain suite
counts 73,878 real polynomial vectors, rather than accidentally testing only
scalar fallback. The v73 package builds with the expected DSP ELF target. QEMU
currently cannot execute custom QNN packages, so standalone HVX execution and
host QNN package execution are distinct evidence. Android loading and the full
recurrent acceptance remain unrun. The main integration result above is newer
than the companion package's earlier standalone-only report.

## Reproduction

Use the same original model, explicit source/weight/input hashes, a licensed QNN
2.42 installation, and the correct four-layout GELU package. SDK/compiler and
runtime files are not bundled. `run_native_region.sh` requires explicit source,
fresh output, input-list, tool-launcher and package paths. It never contacts a
phone, installs software or changes device policy.

`prepare_reference.py` extracts the original image preprocessing and CPU vision
reference. It records exact input and artifact hashes. `check_remainder.py`
checks only zero-state recorded-night frame 0 and saves original-checker inputs.
Set PYTHONPATH to include the ONNX/ORT environment and the checkout's
`vendor/jetlink`. Run the unchanged `scripts/verify_parity.py compare` on the
generated `native` comparison directory. One-frame acceptance is not recurrent
acceptance, even when a pooled one-frame check passes.

## Storage repack

The inference proof retained the complete source weight file while also keeping
vision weights inside the context. That temporary model data totaled
1,505,201,531 bytes. `repack_remainder.py` removes only unused root initializers
and streams live external tensor spans into a new file. It does not rewrite the
graph or change any live tensor value. Nineteen focused tests pass.

The measured repacked artifacts are 1,297,697 bytes ONNX, 90,789,536 bytes
remainder weights, and the unchanged 714,256,384-byte context: 806,343,617 bytes
total. This saves 698,857,914 bytes versus the duplicated proof and remains
1.77% larger than the original 792,305,602-byte model. Runtime libraries are
excluded. All 311 retained initializers have matching hashes; 347 unused ones
were removed. Every non-initializer model field is unchanged. The one unused,
pure Identity function is preserved byte-for-byte; arbitrary functions and
nested graph data remain rejected. See `repacked/evidence.json`.

The repack has static byte-equivalence and ONNX-checker evidence, not a separate
runtime invocation. Both full-model inference gates used the unrepacked proof
artifact. A phone integration may have a different remainder and final size.

```sh
python repack_remainder.py /absolute/path/bridge/model/vision-native.onnx \
  /absolute/new/repacked --context /absolute/path/vision/context/vision.bin \
  --allow-external-file /absolute/path/teacher-capture.weights
```

The last argument explicitly permits the frozen source external-weight symlink
used by this proof. It is unnecessary when weights are regular files beneath the
source model directory. Originals are never overwritten.

## Remaining gates and stop rules

- Compare the exhaustive raw-half map with an accepted Android-produced
  reference. Current full-domain checks use the preserved host reference.
- Reconfirm exact package results, all four layouts, tail behavior and saved
  context loading on the matching Android runtime. The completed host/QEMU
  evidence does not replace target execution.
- Run the complete independently recurrent synthetic160 gate and the existing
  held-out day/night320 gates with unchanged thresholds. The current frame-zero
  pass does not cover accumulated feature-state errors.
- Integrate the region with the accepted phone's accelerated remainder rather
  than deploying the host CPU remainder as though it were a complete fast model.
- Build/load the package on the matching phone toolchain. The current tools are
  alternate SDK6.4/Tools19 plus isolated native21 host support, not proof of the
  originally requested SDK5.5.5/Tools8.7.06 environment.
- Check placement and numerical behavior on the authorized device, then measure
  a controlled short A/B/A screen, excluding initialization and input I/O and
  recording all model/runtime hashes and thermal context. Longer acceptance is
  justified only if the timing is useful. Reject numerical failures rather than
  relaxing thresholds.

There is no new phone execution, sustained thermal result, accepted full model,
32 ms achievement, APK installation or vehicle deployment in this work.

## Final focused checks

The final Python pass ran 48 tests: 26 extractor, 3 exact-cut model, and 19
repack checks. Four ORT invalid-input tests and nine native descriptor cases
also pass, without initializing QNN. `final-validation.json` independently
rechecks all four saved vector output arrays against the split-native scalar
control. This count does not include the much larger GELU exhaustive suites.

## Source-only delivery boundary

Authored Python, shell, bridge C++ source, coefficient generators, tests and
JSON/text evidence may be packaged. Exclude generated QNN C++, model weights,
ONNX graphs, QNN context binaries, SDK headers/libraries, compiled shared objects,
raw road inputs and the SDK-derived interface implementation. Local generated
artifacts remain available only for reproducibility in this workspace.
