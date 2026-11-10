# JetLink research update, 2026-11-10

This is a source-and-evidence snapshot of three research tracks, built on main commit `da9d3af140e59ca1776eb1243752e85e931335a9`. It is not a deployable phone model. The existing reports and vendored upstream source are preserved unchanged.

For a fresh Codex review, start with [CODEX_TEST_PLAN.md](CODEX_TEST_PLAN.md).

## Current completed gates

- **Compact CPU integration:** all 160 synthetic frames and both 320-frame reconstructed day/night sequences pass the unchanged upstream checker, including each road's predeclared last 160 frames after warmup. See [compact integration](compact-integration/README.md) and `compact-integration/evidence/recovery-qualification.json`.
- **Native compact contexts with CPU remainder:** two saved QNN contexts execute inside independent whole-model recurrence. All 160 synthetic frames pass. The complete recorded night320 sequence now passes both all 320 frames and the predeclared final 160 frames after warmup. Day32 also passes, but the full native day320 qualification is still pending. Earlier night32 and night160 diagnostics are retained as history, not substituted for the now-completed full night gate. See `compact-native-integration/evidence/recorded-roads/` for the completed night evidence. See [native integration](compact-native-integration/README.md), `compact-native-integration/evidence/qualification.json`, and `compact-native-integration/evidence/recorded-road-prefixes/`.
- **Requantization:** selected A16/W8 first-projection native host execution with the remainder on CPU passed synthetic160 and one recorded night320 sequence. A complete mixed graph prepared for SoC43/v73, but whole-context execution and device latency have not passed. Rejected earlier candidates remain documented. See [requantization](requantization/README.md).
- **Distillation:** a scaled-SiLU replacement for one MLP passed synthetic160 and separate day/night320 comparisons. Expanding to twelve logical positions failed the night post-warmup check and one day-frame plan check, so the expanded candidate remains rejected. See [multi-block results](distillation/policy12/README.md), [single-block history](distillation/long-student-results.md), and their saved checker logs.
- **Compact GELU table representation:** exact host-reference bit tests pass and the two-map C-array payload is 93,184 bytes rather than 262,144 bytes. This is storage, not an HTP speed result. See [prototype](../experiments/compact-gelu/README.md).

No combination of independently passing tracks is automatically qualified. No new phone execution, 32 ms result, general driving accuracy, safety validation, merge, or deployment is claimed. Recorded inputs come from the public sources documented under [recorded-road-inputs](recorded-road-inputs/README.md); comparisons are against an independently recurrent teacher, not ground-truth driving labels.

## Model size: compare the complete package

The original monolithic ONNX is **792,305,602 bytes**. The compact CPU ONNX plus its required unchanged external weights is **792,133,560 bytes**, a saving of only **172,042 bytes (0.0217%)**. The two compact QNN contexts total **10,776,576 bytes**, but cover only the selected upstream partition and three tails. They are not a full-model replacement.

The current native wrapper, external weights, and two contexts total **802,686,144 bytes**, before common runtime dependencies. The 288-fold reduction from 2,654,208 to 9,216 bytes is boundary activation payload, **not model storage or measured latency**. Full accounting and caveats: [model-size-audit-20261109.json](model-size-audit-20261109.json).

## Reproduction and publication boundaries

- Published: original experiment scripts, host bridge source, generated mathematical reference tables, test code, text/JSON checker evidence, public-input provenance and hashes.
- Excluded: proprietary SDK source/headers/licenses, SDK/runtime libraries, compiled executables/shared objects, contexts, original or derived model weights, raw camera/log data, input/output arrays, caches, and credentials. Existing upstream source copyright/license notices are unchanged. No new license is assigned to third-party materials.
- Some scripts preserve absolute workspace paths, exact experimental defaults, and dependencies on unpublished artifacts. These are research recipes rather than a turnkey clone-and-run phone package. Adapt paths and obtain the matching dependencies/data lawfully; use recorded hashes to check identity. Hashes of excluded artifacts identify evidence, not bundled files.
- Historical README sections preserve chronological intermediate results and statements such as “not yet run” or “no push during recovery.” The current status above takes precedence for this publication snapshot; later evidence files do not rewrite the earlier manifests.
- Full-model runs were **not repeated for publication**. Their completed, immutable text evidence was copied; the live native road replay, checkpoint, source model and runtime artifacts were not modified.
- `publication-manifest.json` records byte lengths and SHA256 of the published additions plus the updated root README. The original root `snapshot-manifest.json` still describes the older imported snapshot.

## Checks rerun on the publication snapshot

See [publication-validation.json](publication-validation.json). Passed: 7 boundary-QDQ tests, 7 compact-binding tests, 10 compact-partition structural tests, 4 distillation-initializer tests, and 6 synthetic converter-helper checks. Compact GELU table regeneration reproduced the checked-in table/report byte-for-byte; release and address/undefined-sanitizer runs each passed 131,072 raw-bit comparisons, two full in-place cases, and 520 length/canary cases. LeakSanitizer was disabled in this traced environment and was not validated.

Python syntax and JSON parsing were checked for every new source/evidence file. A credential-pattern and binary-extension review found no matching secrets or binary artifacts among the additions. This is not a security certification. Native SDK-dependent runtime tests and full replays are represented by saved evidence rather than claimed as newly rerun.
