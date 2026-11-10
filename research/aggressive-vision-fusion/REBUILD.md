# Rebuild the aggressive vision artifacts locally

This route uses the published authored source, the original model already held by the user, and separately licensed local Qualcomm dependencies. The script writes local outputs only. It does not transfer SDK/runtime bytes, contact a phone, or upload files. No context binary, shared library, SDK or model weights are distributed by this handoff.

## Source revision

Repository: `sunyu1235/jetlink-sm8550-research`

Pinned source revision: `d1cd54c6be25f594091358725a95dd5db69fc7ca`, branch `research/aggressive-vision-fusion-20261110`.

Required paths:

- `research/aggressive-vision-fusion/preparation/`
- `research/aggressive-vision-fusion/extract_region.py`
- `experiments/gelu-polynomial/`

The remote branch and all 98 original source/evidence payload blobs were verified against that revision before adding this guide. The rebuild handoff is an additive commit on the same branch and does not change those inputs. Use a clean checkout of this branch. Local source changes create a different candidate and require new validation.

## Required local environment

- Linux x86_64 host. This is not a Windows/macOS-native build recipe.
- Original embedded-weight Light Speed ONNX, 792,305,602 bytes, SHA256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`. A phone EPContext/cache model is not a substitute.
- QAIRT/QNN `2.42.0.251225`, build `v2.42.0.251225135753_193295`, including converter Python files, x86_64 tools/backend, HTP headers and ExampleOpPackageInterface.cpp.
- For the demonstrated target build: Hexagon SDK `6.4.0.2`, Tools `19.0.04`, including `bin/clang-19`, `bin/hexagon-clang++`, QuRT computev73 headers and target libraries.
- For demonstrated host package preparation: native `21.0.04` headers and `libnative.a`; host Clang/LLVM19, libc++19/libc++abi/libunwind; GCC with `_Float16` support.
- Activate the SDK-compatible Python environment before invoking the script. The original converter used Python 3.10.22, NumPy 1.26.4 and ONNX 1.17.0. The script expects its interpreter to be available as `python` and does not install dependencies.
- Allow roughly 10 GB of free scratch space for copied models, converter data, intermediate libraries and the 714 MB context. This is planning headroom, not a measured minimum.

## Fetch without disturbing the accepted phone workspace

```sh
git fetch origin research/aggressive-vision-fusion-20261110
git worktree add ../jetlink-aggressive-rebuild -b review/aggressive-rebuild-20261110 origin/research/aggressive-vision-fusion-20261110
cd ../jetlink-aggressive-rebuild
```

Use fresh local names if either already exists. Do not reset or clean the existing phone workspace.

## Run

Review `rebuild-aggressive-vision-artifacts.sh`, then set these variables to actual local paths:

```sh
export REPO_ROOT="$PWD" # The fresh worktree above, or another clean checkout of this branch.
export ORIGINAL_MODEL=/absolute/path/to/light-speed.onnx
export WORK_DIR=/absolute/path/to/new-aggressive-build
export QNN_SDK_ROOT=/absolute/path/to/qairt/2.42.0.251225
export HEXAGON_SDK_ROOT=/absolute/path/to/Hexagon_SDK/6.4.0.2
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools"
export NATIVE_TOOLS=/absolute/path/to/native21/Tools
export HOST_TOOLS_ROOT=/absolute/path/to/llvm19-root
# If LLVM19 and libc++19 use Debian's system /usr layout, HOST_TOOLS_ROOT=/ is suitable.
bash "$REPO_ROOT/research/aggressive-vision-fusion/rebuild-aggressive-vision-artifacts.sh"
```

WORK_DIR must be a new absolute directory whose parent already exists. The script refuses an existing path (including a dangling symlink), the wrong original-model hash, missing required tools, or an invalid RUN_PACKAGE_TESTS value. It copies an explicit source-file list into the new output directory and regenerates the licensed interface locally, so stale build products are not reused. A failed build leaves its logs and partial outputs for inspection; use a different new WORK_DIR for a retry. It prints the resulting paths, sizes and SHA256 values:

- `$WORK_DIR/vision/context/vision.bin`
- `$WORK_DIR/gelu-polynomial/qnn-package/build/libJetlinkGeluV73.so`

The intermediate x86_64 host package remains at `$WORK_DIR/gelu-polynomial/qnn-package/build/libJetlinkGeluHost.so`. It is required for offline context generation and must not be substituted for the DSP target library on Android.

## What the pipeline actually does

1. Generates the licensed interface locally from that installation's QNN example, then builds the v73 and host GELU packages. The script does not execute the DSP library.
2. By default, regenerates and hash-checks the host reference and runs the standalone core, 24-graph baseline and four forced-layout package tests. `RUN_PACKAGE_TESTS=0` skips these, but that removes an important local correctness gate.
3. Maps the original 36 erf and 27 tanh GELUs, promotes only the tested image input boundary, and converts with the QNN2.42 arithmetic-preservation wrapper.
4. Repairs exactly 19 mixed-type FC biases only if FP32→FP16→FP32 preserves their represented values. It will not weaken that assertion for another model.
5. Extracts `cat_2` → `view_39` as `vision_merged`, builds its x86_64 model library, then invokes qnn-context-binary-generator for SoC43/SM8550, v73 and O3. It uses the host package's `exampleInterfaceProvider:CPU` registration during this host-side preparation.
6. Stops after writing the context. No road input tensors are required for this generation step and no full-model inference is automatically run.

`run_native_region.sh` is not the shortest answer to this request: it assumes `preserve_net.json` and repaired converter files already exist, takes a separate tool-launcher path, and immediately executes a long host inference. The supplied wrapper instead runs only its generation stages directly with the configured SDK.

## SDK/version risk

The phone handoff originally requested SDK5.5.5/Tools8.7.06. The published build script hardcodes `bin/clang-19`; pointing HEXAGON_TOOLS_ROOT at Tools8.7.06 does not make it portable. If that is the only installed compiler, this recipe deliberately stops. Use a separately licensed demonstrated stack, or adapt and validate the source for the phone-prescribed toolchain as a new build task. Merely renaming or symlinking an older compiler to clang-19 does not establish compatibility.

The compiled library is ELF32, little-endian Qualcomm DSP6, v73, with QuRT/HVX128 build settings, not an Android ARM64 JNI library. Its QNN package interface expects core API 2.32.0 and HTP API 5.41.0 from this SDK. Matching QNN runtime/skel and package loading on the physical phone still need verification. Compilation and standalone QEMU results do not establish Android loading, full recurrent correctness or speed.

## Verification performed for this handoff

- Matched the 98 original payload files to their remote Git blob hashes at the pinned source revision. Parsed 18 source Python files and checked all 10 original shell scripts plus the new wrapper with `bash -n`.
- Reran the 48 extractor/model/repack unit tests: all passed (26 extractor, 3 model-cut, 19 repack). These are focused tests, not the full repository suite.
- Ran 10 isolated wrapper tests with mocked tools: the default and skipped package-test paths, paths containing spaces, explicit source-only copying, refusal of existing/dangling/relative output paths, invalid test flags, wrong model hash, missing compiler/SDK, and stopping after a failed build stage. These test orchestration only, not QNN numerics or generated artifact validity.
- Copied the source-only qnn-package to a fresh temporary folder and ran its unmodified `build_v73.sh` against the demonstrated SDK/Tools stack. It compiled a 184,664-byte ELF32 DSP6/v73 library. The output was not executed or uploaded.
- That fresh target build had SHA256 `b8798e4d2f39ab06d3c9981f7c5c827eb88fd1df239ee02142b50d0fdf82b295`, matching the historical target artifact. An earlier fresh build had hash `cadd6eecacc8e9a53cda9f3a103617d510e8641ed6368cc2e2a914eb86f44de6`; a byte-identical rebuild on another installation is not promised. A matching file hash does not establish Android loading or performance.
- The complete mapping/conversion/context pipeline was inspected against the existing successful scripts and logs, but was not rerun end-to-end during this handoff. The host exhaustive/package suites were not rerun for publication. The wrapper runs them by default when invoked locally. New output hashes and numerical results need their own verification.

## Licensing and publication boundary

Only authored wrapper source and documentation are added. Existing source notices remain unchanged; this guide does not extend any vendor or model license. Obtain and use the model, SDK, compiler, headers and runtime under their applicable licenses. The locally generated SDK-derived interface, model data, context and linked shared libraries are not included in this public handoff. Do not commit the output directory or assume that a successful local build authorizes redistribution of its outputs.

The original SOURCE_MANIFEST and PUBLICATION_MANIFEST describe the earlier source publication. They are not expanded to claim coverage of this later rebuild guide and wrapper.

## Historical artifacts for comparison

The existing context is 714,256,384 bytes, SHA256 `4427131892da05dcf5aa351fe675e07c9949e9c7b54072d4829b9cd23e2bfd8f`.

Historical extractor inputs:

- preserve_net.json: `638536c27d0ce270608ebe32f3d229ad3ab03d84dc2f7187e34b7f955efcd0ac`
- preserve_fixed.cpp: `b8d60b95d25e32cc1b703f44d4c8906e33db2e18c3a1f411fff4120c86a07838`
- preserve_fixed.bin: `ccc603b554aff8caaacfe7d1a13a5da0c972eb755ce54da6ea74669172990040`

Historical extracted region:

- region.cpp: `40ae47d23717a0d4bd927ea557c885e7964703e2d17a986fdf45a095dc7fa780`
- region.bin: `789d0299b79c1febcec9fc7ac1f7f5eef04cc2571e3fa7da5a03988f57ccf000`

Investigate mismatches rather than treating a newly generated artifact as the already-tested candidate. Context or ELF hashes can differ across build provenance; a hash match is useful provenance, not a substitute for target execution.
