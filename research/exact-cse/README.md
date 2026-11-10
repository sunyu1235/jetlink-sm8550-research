# Exact CSE: source handoff and speed-first phone test

## Fetch this handoff

This is a source-only, additive follow-up to the accepted phone-test commit.
Fetch the branch without changing your existing remote configuration:

```sh
git fetch origin research/exact-cse-speed-screen-20261110
git switch --create exact-cse-phone-test --track origin/research/exact-cse-speed-screen-20261110
```

Keep any existing local changes on their current branch. This directory adds six
Python tools/tests and documentation only. It does not include model weights,
compiled caches, SDKs, executables, input captures or local evidence paths.
The accepted phone report and all other tracked files are unchanged.

## Scope and result

This work follows `codex/phone-compact-test-20261110` at
`4619484678610712fc132b4bf11848cb4a35e481`, specifically its
[Next step](https://github.com/sunyu1235/jetlink-sm8550-research/blob/4619484678610712fc132b4bf11848cb4a35e481/reports/jetlink-dot-compact-test-20261110/NEXT_STEP.md).
Only exact common-subexpression elimination is considered. No SiLU student,
quantization, operator approximation, extra last-query pruning, or training.
**Phone speed first; run the unchanged 160-frame numerical gate only if the
phone speed screen is promising. No new 160-frame run was performed.**

The new conservative tool reproduced **30 MatMul and 6 GELU eliminations** on
the already available, reconstructed compact whole-model source. Nodes change
from **1,007 to 850**. It preserves all four original output signatures, including
all three recurrent states. One zero-state CPU frame is bit-identical for every
head and every state value. This is only a build/smoke check, not recurrent
acceptance, a phone speed result, or proof of Android equivalence.

The tested source is `teacher-compact-boundaries.onnx`, SHA-256
`3c80a2e4ede9781a1e5aec41c503e12e2f2c739e987e168046de8ed87325800a`.
It is the prior source-level compact reconstruction, **not a recovered copy of
the accepted Android cached graph**. The candidate ONNX descriptor is
`4824a1926fb314bd12c8a53aa6312f60075e72716a7e6bab8a8d51292de10dff`.
External weight bytes are copied unchanged, not physically compacted.

## Has the phone compiler already removed the duplicates?

Evidence is partial, and the distinction matters:

- The accepted phone test report explicitly retains all 63 CPU exact GELUs.
  Its composition code replaces only two QNN EPContexts; it does not remove any
  CPU GELUs. The six duplicate pairs identified in source are `gelu_50→gelu_39`,
  `gelu_51→gelu_40`, `gelu_52→gelu_41`, `gelu_53→gelu_42`,
  `gelu_54→gelu_43`, and `gelu_55→gelu_44`.
- The older accepted ORT trace separately records 63 CPU kernel executions and
  45 QNN executions per frame. This is historical execution evidence, not a new
  profile of the latest compact candidate.
- Therefore the available record does **not** support saying all duplicates
  were already removed. The six GELUs remain explicit in the recorded pipeline.
- The 30 MatMuls are inside opaque QNN contexts. We cannot establish their
  separate runtime execution from source counts or boundary output names.
  A compiler dump or device operator trace remains necessary if that question
  affects whether to spend time rebuilding.

Sources:
[phone README](https://github.com/sunyu1235/jetlink-sm8550-research/blob/4619484678610712fc132b4bf11848cb4a35e481/reports/jetlink-dot-compact-test-20261110/README.md),
[phone composition](https://github.com/sunyu1235/jetlink-sm8550-research/blob/4619484678610712fc132b4bf11848cb4a35e481/reports/jetlink-dot-compact-test-20261110/prepare_cached_candidate.py),
[historical ORT profile](https://github.com/sunyu1235/jetlink-sm8550-research/blob/4619484678610712fc132b4bf11848cb4a35e481/reports/jetlink-gelu-optimization-20261101/profile-20261102/ort-profile-summary.json).

## Tools and safety checks

- `exact_cse.py`: byte-value initializer identity plus ordered input, domain,
  opset, overload, attribute and output-type signatures. Deterministic operations
  use an explicit allowlist. Random, training and unknown operations are barriers.
  Multi-output operators are removed atomically. Overridable initializers stay
  distinct. All graph inputs and outputs retain their protobuf signatures.
- Exact existing `jetlink.precision` opset-1 CPU GELUs can be opted in explicitly.
  The tool never changes their formula or installs another implementation.
- Cached `EPContext` graphs are rejected. Applying source CSE to an opaque
  context would not rewrite the compiled model and must not be treated as valid.
- Only root-graph external initializers are supported. External Constant/function
  tensors, nested graphs, training graphs and sparse initializers are rejected.
  Output/report/external-file collisions are rejected before writing.
- `audit_cached_graph.py`: read-only mapping of proven source aliases to actual
  cached graph producers. It explicitly leaves QNN internal execution unknown.
- `smoke_one_frame.py`: exactly one zero-state CPU frame. It fails if any output
  is nonfinite or differs bitwise. No timing is reported.
- `screen_speed.py`: validates existing short Android A/B/A CSVs. It checks frame
  completeness, positive finite timings, nonfinite outputs, supplied run identity
  and matched temperature context. It never launches a numerical replay.

### Source-only checks

From the repository root, use an environment containing NumPy, ONNX and
ONNX Runtime. The focused checks were rerun with Python 3.12.14, NumPy 2.5.4,
ONNX 1.20.1 and ONNX Runtime 1.23.2 (CPU). These versions describe the host
source checks, not a replacement for Android ORT 1.29 / QNN 2.42.

```sh
python -m unittest discover -s research/exact-cse -p 'test_*.py' -v
python -m compileall -q research/exact-cse
```

Seventeen focused tests pass, including finite-but-different smoke rejection,
initializer override, distinct recurrent inputs, no tolerance/commutation,
GELU-formula separation, random/custom barriers, multi-output ordering, external
weights and path collisions. Python compile checks pass. This is not a full
repository test-suite run.

## Run on the original phone workspace

The accepted files listed below were not included in the public repository and
are not present in this cloud checkout:

- `outputs/jetlink-dot-compact-test-20261110/compact-whole.onnx` and its two
  source partitions `upstream.onnx` / `downstream.onnx`.
- The original pre-partition exact-CPU-GELU source and its weights.
- The accepted QNN binary caches, Android runner and exact GELU library.

1. Read the actual accepted compact cached graph and resolve all external cache
   files. Confirm its 63 CPU GELUs, 45 EPContexts and four head/state interfaces.
   Run the read-only audit against the corresponding source:

   ```sh
   python research/exact-cse/audit_cached_graph.py \
     --source accepted-compact-source.onnx \
     --cached outputs/jetlink-dot-compact-test-20261110/compact-whole.onnx \
     --allow-jetlink-exact-gelu --report cache-audit.json
   ```

   `accepted-compact-source.onnx` is a required local preparation input, not a file
   produced by this patch. Reconstruct it from the accepted precision source and
   the actual compact `upstream.onnx` / `downstream.onnx` replacements. Verify
   cut inputs, outputs, weights and all unaffected operations. Do not substitute
   the host reconstruction and call it the accepted phone source.

2. Apply source CSE, keeping the current compact boundary shapes and exact CPU
   GELUs. Do not pass the cached wrapper to the rewrite command:

   ```sh
   python research/exact-cse/exact_cse.py \
     --source accepted-compact-source.onnx --output exact-cse/source.onnx \
     --allow-jetlink-exact-gelu --report exact-cse/source-proof.json
   ```

3. Rebuild affected Android policy partitions using the same ORT 1.29 / QNN 2.42
   configuration. Retain untouched contexts when their exact interfaces permit
   it. Verify that CPU placement is only the intended exact GELUs. Generate the
   shared-boundary wrapper **after** CSE, exposing only live canonical boundary
   tensors. Old extra observer outputs can otherwise force alias copies. Do not
   remove a head or recurrent output to get the benchmark to run.

4. Record hashes for models, context binaries, runner, exact GELU library and
   input files. Confirm no per-frame state files override autonomous recurrence.
   Use the accepted `partition_probe` invocation with shared buffers, `taskset f8`,
   burst mode, independent zero-state processes and identical varied inputs.
   Run **16-frame baseline / candidate / baseline** screens at matched starting
   temperatures. Use fresh output directories. Keep initialization, input I/O
   and Run timing separate, and do not alter APKs or system/thermal policy.

5. Analyze the three CSVs with `screen_speed.py`. Prepare a local JSON manifest
   with the required keys: `device_serial`, `device_model`, `input_sha256`,
   `runner_sha256`, `gelu_library_sha256`, `baseline_artifact_sha256`,
   `candidate_artifact_sha256`, `configuration`, and `placement_verified`.
   Use actual hashes; for a multi-file model, hash a deterministic artifact
   manifest containing every descriptor, weight and context hash. Set
   `placement_verified` to `true` only after checking the actual partition map.
   Record CPU affinity, burst/shared-buffer settings, runtime versions, input
   order and zero-state recurrence in `configuration`. Keep the device serial
   and generated run manifest local; redact private identifiers before any
   future publication. This tool checks the supplied manifest for completeness,
   not the truth of its claims or the artifacts' contents.

   ```sh
   python research/exact-cse/screen_speed.py \
     --baseline-a baseline-a.csv --candidate-b candidate-b.csv \
     --baseline-a2 baseline-a2.csv --frames 16 \
     --start-gpu-c ACTUAL_A_C ACTUAL_B_C ACTUAL_A2_C \
     --manifest screen-manifest.json --report speed-screen.json
   ```

   Replace the three temperature placeholders with actual pre-run readings.
   The optional screen heuristics
   require a gain of at least max(0.5 ms, 1%), faster than both baselines, greater
   than baseline drift, start GPU readings below 45 C and within 2 C. These are
   screening thresholds, not a guarantee of statistical significance or NPU
   thermal equivalence. If not promising, stop this candidate. If promising,
   verify the speed result, then run the unchanged 160-frame gate and compare
   both heads and recurrent state outputs. If speed does not improve, do not
   spend time on the 160-frame run for this candidate.

## Validation boundary and required local inputs

This publication contains source preparation and focused CPU tests only.
The exact accepted Android compact graph, its original source/weights, QNN
caches, Android runtime and phone executor are required in the original phone
workspace. They are not supplied by this branch. Actual accepted-cache
inspection, Android rebuild, placement verification and phone CSE speed testing
have not been performed for this handoff. No host timing substitutes for the
phone speed screen. No new 160-frame run, APK deployment or device setting
change was performed.

The source rewrite counts and single-frame bitwise result above summarize the
prior host preparation. Its raw local reports are not part of this source-only
publication. Regenerate proof reports against the actual accepted source in
your own workspace, without overwriting accepted artifacts. A host smoke pass
is not Android acceptance and does not justify skipping either the short phone
speed screen or, for a faster candidate, the unchanged 160-frame numerical gate.
