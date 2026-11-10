# Compact exact GELU kernel core (offline development, 2026-11-09)

This is a **host-tested scalar kernel core**, NOT a built QNN HTP OpPackage,
NOT an HVX-vectorized kernel, and NOT evidence of an accurate 32 ms model.
No phone, APK, device frequency, thermal policy, SDK, account, or remote branch
was changed. Source snapshot: `8f2bc5d31e06decaaa90190486e08c468e781c1f`.

## What was developed

The accepted CPU repair already uses full FP16 lookup tables. This prototype
keeps the two formulas and exact table semantics, but compresses both maps into
a shared bank of deduplicated 128-element pages. An input's upper 9 bits select
a page; its lower 7 bits select the entry. Pages that are entirely identity,
positive zero, or negative zero use tags instead of data. There is no runtime
transcendental computation, allocation, initialization, or calibration.

Unlike the rejected bounded-LUT ONNX graph, this is a single C kernel operation
intended for eventual SDK integration, not a graph expansion. Unlike a claim
that `vlut16` gathers a 128 KiB map in one instruction, it explicitly retains
a dependent indexed-memory lookup. Smaller footprint does NOT imply faster
HTP execution: the extra lookup and branching may make it slower. Even one
128-entry page exceeds a 128-byte HVX vector table, so no one-instruction
vectorization is implied.

## Verified results

- Generated the reference from the same double erf/tanh equations and final
  `_Float16` cast as the accepted CPU operator source in
  `reports/jetlink-gelu-repair-20261101/exact_gelu.c`.
- Both maps together: 262,144 bytes -> 93,184 bytes including page indices,
  a **64.453125% reduction**. This is C-array payload, excluding ELF alignment,
  metadata and code. 356 unique pages, 128 entries each.
- 131,072 exact raw-bit comparisons passed, including all 63,488 finite inputs
  per formula. Two full in-place tests, 520 length/canary tests (lengths 0..259
  per mode), invalid mode and null-pointer checks passed.
- GCC 14.2 release build and address/undefined sanitizers passed. LeakSanitizer
  itself cannot run in this traced environment; it was disabled for the ASan
  run, not reported as a successful leak check.
- erf and tanh tables differ on 3,695 finite input encodings. A single generic
  GELU mode is not equivalent to preserving the model's two formulas.

These are **host-reference equivalence** tests, not a new independent proof of
correctly rounded mathematical GELU. Existing reference arithmetic has negative
tail cancellation. Host libm/compiler rounding and NaN payloads are not assumed
to match Android's table generator. Before any deployment, dump the accepted
Android operator's actual tables, hash them, regenerate with that file, and
repeat these exact tests. The generator accepts two little-endian uint16 maps
ordered erf then tanh. Keep signed zero, infinities and NaN behavior explicit;
do not silently improve special-value behavior relative to the accepted model.

## Reproduce

```sh
cd experiments/compact-gelu
gcc -std=gnu11 -O2 -Wall -Wextra -Werror reference.c -lm -o reference
./reference > reference.bin
python generate.py reference.bin
gcc -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=address,undefined test.c -o test
ASAN_OPTIONS=detect_leaks=0 ./test reference.bin
gcc -std=c11 -O3 -Wall -Wextra -Werror test.c -o test-release
./test-release reference.bin
```

Do not use `-ffast-math` for reference generation. The checked-in generated
header and report are reproducible on the recorded host; generated reference
bytes SHA256 are in `storage-report.json`. Binaries are excluded from delivery.
`jl_gelu_run` validates mode and supports exact in-place operation; partial
input/output overlap is unsupported. `jl_gelu_half` is an internal helper
requiring validated mode. QNN tensor layout, padding, logical shape and strides
must be handled by the future adapter, not guessed from these raw buffers.

## Concrete next gates, not done

1. Obtain the version-matched QNN 2.42 legacy HTP OpPackage SDK, FP16 example,
   release-matched Hexagon v73 compiler/SDK and host package support. None are
   in this snapshot/environment. Current Qualcomm docs are not ABI evidence.
2. Build identity FP16 OpPackage on htp_v73 and htp_x86 using that release's
   generated wrapper. Preserve ordinary tensor layouts; verify padding/tails.
   Then adapt this scalar core as a correctness baseline. Also benchmark a
   full LUT baseline: compressed representation must win measured latency,
   not only size. HVX implementation is a separate measured optimization.
3. Ensure the real model builder lowers both formula-specific custom operators
   into the HTP package. A custom op merely registered on CPU is insufficient.
   Inspect provider placement and compiled graph to verify all 63 replacements
   remain on HTP without the 45 CPU/NPU boundaries. This probably requires
   QNN graph construction/custom builder work rather than stock ORT handling
   arbitrary custom-domain ONNX nodes. Exact API code is intentionally not
   fabricated in the absence of matching headers.
4. On an explicitly authorized test device, exhaustively compare all half
   encodings against the accepted tables. Then replay real activation captures,
   followed by full-model 160-frame recurrent validation with the unchanged
   upstream checker and independent candidate state. Longer varied held-out
   sequences remain necessary; finite outputs and single-op matches do not
   establish road accuracy or safety.
5. Measure same input, thermal state, warmup and inference boundary as the old
   full-HTP result. Existing means: wrong full HTP 32.475 ms, accepted repair
   52.4086 ms. Recovering the 19.9336 ms gap is a hypothesis, not a budget already
   recovered. A 35 ms intermediate gate allows at most 2.525 ms **net** increase
   versus the old graph, about 40 microseconds per replacement on average
   if spread across 63 nodes; this includes changes to fusion/scheduling and is
   not a per-op performance prediction. A strict 32 ms goal is already below
   the old 32.475 ms mean, so correct activations alone cannot guarantee it.

## Sources

- Repository README and review request:
  `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/dot-review-request.txt`
- Prior failures and accepted 160-frame result:
  `reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/README.md`
- Version/ABI/build constraints and Qualcomm primary links:
  `reports/jetlink-gelu-optimization-20261101/sdk/README.md`
- Numerical reference: `reports/jetlink-gelu-repair-20261101/exact_gelu.c`

No new device measurements or whole-model outputs were produced. The missing
SDK and actual activation/model inputs block those stages, not the completed
offline kernel tests. There was no denied operation or uncertain device action.

## Follow-up SDK acquisition check

The concrete official QAIRT 2.42 package URL was found and checked:
https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip

Google's LiteRT QAIRT SDK documentation names this exact version/URL:
https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/qualcomm/doc/QAIRT_SDK.md
Qualcomm's own sample-app setup documents the same official URL scheme:
https://github.com/qualcomm/sample-apps-for-qualcomm-linux/blob/main/GenAI-Solutions/GenAI-Studio/docs/setup/DEVICE_SETUP.md

A HEAD request followed the official redirect to
`apigwx-aws.qualcomm.com/qsc/public/v1/api/download/...` and returned HTTP 403.
No package bytes were obtained; no alternate download intended to circumvent
that denial was attempted. The public Software Center catalog is JS-rendered
and yielded no readable version/download controls in the web reader.

The official Hexagon SDK product page states that download requires login:
https://www.qualcomm.com/developer/software/hexagon-npu-sdk?redirect=qdn
No account login, credential creation, license acceptance or software install
was performed. A bounded filesystem search found no QNN/Hexagon SDK under
`/opt`, `/workspace`, or `/usr/local`. Actual HTP compilation therefore remains
blocked pending accessible official matching packages and compiler. A model
alone cannot replace the compiler or supply device measurements.

## Corrected acquisition status after explicitly requested retry

The earlier anonymous HEAD 403 did not establish that anonymous GET was blocked.
After the user reported successful anonymous access and requested retry, a normal
GET with a byte range on the same official URL returned HTTP 206. Total ZIP size:
1,543,955,191 bytes. Selective ordinary Range reads obtained the ZIP index and
version-specific docs/source/license with 2,678,766 bytes transferred; extracted
entries passed ZIP CRC verification. Full archive SHA256 is not claimed.

Actual build ID: `v2.42.0.251225135753_193295`.
Actual archive prefixes use `examples/QNN/`, rather than some older docs' paths.
Both HTP example and generator Makefiles specify:
- v73: `hexagon-sdk-5.5.5`, Hexagon Tools `8.7.06`.
- host x86: `hexagon-sdk-6.4.0` and Tools `21.0.0`, including `libnative`.
These are now verified version-matched package settings, not assumptions from
current online docs. Some SDK README paths are stale; actual Makefiles govern.

The AI Stack License included in this archive states downloading/using accepts
its terms. Section 1 allows development, source modification and benchmarking;
redistribution is limited to object code incorporated in one's application, not
standalone SDK redistribution. No proprietary SDK source/binary is included in
this prototype archive or published to a repository. Access to SDK material is
now available; matching external Hexagon SDK/compiler acquisition, actual
OpPackage builds and device validation remain unresolved.
