# Codex test plan: JetLink compact native research

## Goal and scope

Review this branch, reproduce the source-only checks, then assess whether a controlled SM8550 test can be prepared. Do not infer phone compatibility, latency, safety, or deployability from host simulation. Do not change the parity checker or refit against acceptance outputs. Preserve existing reports, frozen inputs, models, contexts, and library identities. Use new output directories and report missing prerequisites rather than inventing a successful test.

The native candidate invokes **two saved compact QNN contexts with the rest of the full model on CPU**. It is not an all-HTP model or a 10.8 MB full-model replacement. The complete current model data is about 802.686 MB before common runtime dependencies; see `model-size-audit-20261109.json`.

## 1. Checkout and read first

Start from the provided branch or commit URL, not an older checkout of main. Record `git rev-parse HEAD`, `git status --short`, and runtime versions before testing.

Read:

1. `research/README.md` for current accepted, rejected, and pending gates.
2. `research/compact-native-integration/README.md` and `bridge/README.md` for the exact host/native boundary.
3. `research/compact-integration/evidence/recovery-qualification.json`, `research/compact-native-integration/evidence/qualification.json`, and `research/compact-native-integration/evidence/recorded-roads/qualification-snapshot.json`.
4. `research/recorded-road-inputs/{day,night}/PREPROCESSING.md` before reconstructing camera inputs.

The completed native night320 and its predeclared final160 view pass. Day320 is still pending in this snapshot. Synthetic160 passes. Earlier prefix scores are diagnostics, not replacements for the predeclared full-road gate. A numerical agreement test against the teacher is not ground-truth driving accuracy.

## 2. Source-only checks that can run without a phone or model weights

Use a disposable Python environment on Linux. The recorded working versions are Python 3.12.14, NumPy 2.5.4, ONNX 1.20.1, and ONNX Runtime 1.23.2. The checks below need NumPy and ONNX; full replay also needs the runtime and separately obtained artifacts. Record the versions actually installed rather than claiming an exact recreation if they differ.

From the repository root, with that environment active:

```sh
python -m unittest discover -s research/requantization -p 'test_*.py' -v
python -m unittest discover -s research/compact-integration -p 'test_*.py' -v
python -m unittest discover -s research/distillation -p 'test_*.py' -v
python research/requantization/qnn-host-bridge/mixed-fullgraph/test_helpers.py
```

Expected: 7 boundary tests, 7 binding tests, 4 initializer tests, and six successful synthetic helper checks. The last file is a standalone script, not a unittest suite; `unittest discover` finding zero tests is not a pass for it.

The old accepted synthetic output arrays already present in the original repository can be rechecked independently:

```sh
PYTHONPATH=vendor/jetlink python vendor/jetlink/scripts/verify_parity.py compare --dir reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare
```

Expected: pass. This only compares saved outputs; it does not run the model or phone.

For the compact GELU host-reference prototype, leave the checked-in generated table unchanged and use a temporary copy:

```sh
tmp=$(mktemp -d)
cp experiments/compact-gelu/{reference.c,test.c,compact_gelu.h,gelu_tables.h,generate.py,storage-report.json} "$tmp/"
(
  cd "$tmp"
  gcc -std=gnu11 -O2 -Wall -Wextra -Werror reference.c -lm -o reference
  ./reference > reference.bin
  python generate.py reference.bin
  gcc -std=c11 -O3 -Wall -Wextra -Werror test.c -o test-release
  ./test-release reference.bin
  gcc -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined test.c -o test-sanitize
  ASAN_OPTIONS=detect_leaks=0 ./test-sanitize reference.bin
)
cmp "$tmp/gelu_tables.h" experiments/compact-gelu/gelu_tables.h
cmp "$tmp/storage-report.json" experiments/compact-gelu/storage-report.json
```

The recorded GCC 14.2 host reproduced the table/report exactly. Each executable passed 131,072 raw-bit comparisons, two full in-place cases, 520 length/canary cases, and invalid arguments. A different compiler/libm may change generated reference bits; report that rather than silently replacing the archived evidence. Do not use `-ffast-math`. LeakSanitizer is disabled above and is not a successful leak check. These tests do not establish Android GELU-table equivalence.

The ten `research/compact-integration/partition-source/test_compact_partition.py` tests additionally need the actual teacher capture model, its external weights, and the original partition contract. Their defaults contain old absolute paths. Review and adapt those defaults in a separate work branch before running; do not report them as covered by the source-only commands above.

## 3. Prerequisites for rebuilding native host evidence

The new source/evidence additions do not bundle:

- The 792,305,602-byte original `light-speed.onnx` (recorded SHA256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`), transformed ONNX models, or external weights.
- Proprietary QAIRT/QNN 2.42.0.251225 SDK headers/libraries/tools/licenses, matching Hexagon compiler/SDK, or Android runtime libraries.
- Compiled custom GELU OpPackages, compact bridge shared objects, or QNN context binaries.
- Raw camera videos/rlogs, reconstructed road inputs, new recurrent output arrays, or training arrays. The original repository's older synthetic comparison arrays are a separate pre-existing exception.
- A ready-to-run Android application, device transport, authorized connected phone, or vehicle deployment package.

Obtain missing dependencies through their legitimate sources and applicable terms. The source model's public URL and hashes are recorded in the original review request. Public road URLs, pinned source references, and preprocessing caveats are in `research/recorded-road-inputs/`. Hashes in evidence identify artifacts that are not included; they are not download links.

Some builders and launchers preserve experiment-specific absolute paths. Do not blindly run them. For example, `research/compact-native-integration/build_model.py` writes a wrapper beside the external teacher weights and refuses an existing output. Adapt paths deliberately in a separate branch and keep original artifacts untouched.

The bridge can be compiled with verified `QNN_SDK_ROOT` and `ORT_HEADERS` through `research/compact-native-integration/bridge/build.sh`. This produces a **Linux x86_64 host library**, not an Android-ready library. Its launcher also requires QNN host dependencies and explicit path adaptation. Runtime overrides include:

- `JETLINK_QNN_BACKEND`, `JETLINK_QNN_SYSTEM`
- `JETLINK_GELU_PACKAGE`
- `JETLINK_COMPACT_UPSTREAM`, `JETLINK_COMPACT_TAILS`

Rebuilding the contexts additionally needs the preserved-arithmetic converter launcher, QNN tool launcher, exact GELU XML/package, and correct SoC43/v73 configuration. `research/compact-integration/compile_contexts.sh` documents those inputs. Successful offline preparation is not successful execution. Compare official net-run and bridge outputs before starting recurrence, and verify all input/output tensor names, FP16 types, shapes, and the six 9,216-byte boundary buffers.

## 4. Controlled device test plan, not yet executed

Proceed only on an explicitly authorized SM8550 test device in a controlled bench setup, not while controlling a vehicle. Do not install this experimental model into a driving stack as part of validation.

1. Record device model/SoC, Android build, QNN/ORT versions, Hexagon architecture, compiler versions, and runtime/library/model/context hashes. Establish a separately measured accurate baseline on the same device. Historical 52.4086 ms is context, not a fresh measurement.
2. Port or integrate the native bridge into the Android harness with matching aarch64 host runtime and required v73 OpPackage support. Do not copy the x86_64 Linux `.so` and call that an Android test. Confirm ABI compatibility and backend placement with actual execution logs.
3. First verify formula-specific erf/tanh GELU behavior against the accepted Android implementation. The host reference table is not proof of device arithmetic. Test invalid input handling and recovery, then matched individual-context and joined-context samples. Check all six compact tensors without padding or undocumented conversions.
4. Run the full original synthetic160 acceptance sequence, with teacher and candidate propagating their own recurrent state. Run the unchanged upstream checker per frame and pooled, including quiet-column checks. Reject any nonfinite values or failing head/column. Do not tune thresholds to obtain a pass.
5. Run independent recorded day/night320 sequences with the documented resets and preprocessing. Check all320 and the **predeclared final160 after160 warmup**. The 160-prefix diagnostic with an 80-frame suffix is not that gate. Preserve logs, ordered-output hashes, all input/model/runtime hashes, fallback placement, and failures.
6. Only after device numerical gates pass, benchmark the same input, warmup, thermal state, power settings, and inference boundary in an A/B/A comparison. Report mean, p50/p99, sample count, CPU/HTP placement, transfer overhead, and complete single-frame/end-to-end latency separately. Host seconds/frame are not phone milliseconds; isolated context timings are not full-model latency.
7. Compare only like-for-like candidates. Do not merge quantization, distillation, and compact-boundary modifications solely because each passed a separate test. A combined model needs fresh complete numerical and performance gates. Stop rather than masking a failure or making a 32 ms claim without measurements.

## Requested Codex report

Return the exact tested commit, environment/dependency versions, commands, which checks passed/failed/were not run, the reason for each blocked stage, and paths to fresh logs and hashes. Separate source-only, saved-output comparison, host-native execution, and actual phone execution. Report device speed only when the device test actually ran.
