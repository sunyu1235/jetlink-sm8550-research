# Corrected HVX polynomial erf GELU candidate

Standalone authored core and separate experimental QNN package. No device latency claim.

## Core

`core.cpp` uses 32 intervals of width 0.5 on [-8,8]. Degree-six polynomials approximate Phi(x), followed by multiplication by x. `generate.py` fits 64 Chebyshev samples per interval using NumPy 1.26.4, rounds coefficients to FP32, and writes `coefficients.h`. Polynomial coefficients occupy 896 bytes; interval midpoints add 128 bytes. Register-based vlut16 selects the coefficient words. There is no memory gather, VTCM, allocation, or mutable state in the core.

FP32/qf32 operations and explicit conversions are deliberately retained. Do not enable fast-math, contract operations, or change conversion order without exhaustive revalidation. Numeric coefficients and correction values are not universal across arithmetic implementations.

The public `vector_core.h` helper accepts/returns one HVX vector of 64 raw half payloads. `core.h` exposes an aligned pointer wrapper and mechanics probes for standalone tests. Exact in-place execution is supported; partial overlap is not.

The first sorted-input test concealed an ordering error: coefficient words initially used contiguous lane groups, whereas half-to-qf32 widening uses even/odd half lanes. Explicit word deinterleave now aligns coefficients with widening. Independent lookup tests and widely mixed input permutations guard this regression.

Native21 exhaustive characterization required three isolated replacements:
- Input 0x0e55 -> output 0x0a55.
- Input 0x157c -> output 0x117d.
- Input 0x2e9e -> output 0x2b29.

Tiny negative inputs with magnitude below 0x0032 use the exact sign-bit plus magnitude-right-shift result. This covers a contiguous cluster of 25 otherwise mismatching odd subnormal inputs, and preserves negative zero. Tail outputs and specials are handled with integer masks: large positive finite inputs are unchanged, large negative finite inputs become negative zero, NaNs retain sign/payload and are quieted, positive infinity is retained, negative infinity becomes 0xfe00. Exhaustive tests verify these masks against the existing reference map rather than assuming mathematical behavior.

## Verified tests

- Native21: 1,048,576 raw comparisons, zero mismatches.
- Native19: 1,048,576 raw comparisons, zero mismatches.
- Compiled v73 target ELF in the locally patched QEMU: 524,288 raw comparisons, zero mismatches.
- Tests cover every raw half encoding, large-odd-multiplier permutations mixing signs/exponents/specials in individual vectors, exact in-place operation, input preservation and output canaries. Host tests include four simultaneous standalone threads. Lookup and interval-index tests are independent of end-to-end GELU checks.
- v73 assembly contains vlut16, qf32 arithmetic, shuffles/deinterleaves and vector comparison masks, with no vgather. The vector helper is a leaf with no vector spills in the current compilation. This is assembly evidence, not a throughput estimate.

`build_test.sh` defaults to native21. Set NATIVE_TOOLS to the existing Tools19 directory for that variant. `build_v73.sh` generates object/assembly. `run_qemu.sh` compiles and runs the target ELF with a 60-second bound using the existing patched simulator. Compiler/native19 is the alternate Tools19.0.04 stack; native21 is21.0.04, not the exact prescribed toolchain. Simulator results are not physical DSP execution.

## QNN package

`qnn-package` is separate from every prior package/artifact. It retains the existing package/operator names for controlled replacement testing. It must not be loaded alongside another registration with the same names.

- Erf uses the corrected vector core; tanh retains the exact compact lookup implementation.
- All four Plain/Crouton registrations remain available, with MainMemory properties and unchanged SNAIL cost classification.
- Plain-to-Plain processes full 64-half depth runs.
- Any Crouton path uses complete adjacent-width-pair/depth32 stripes only. Crouton-to-Plain deinterleaves the result and writes exactly two independently addressed 64-byte depth runs. Plain-to-Crouton performs the reverse. No cross-row or cross-block adjacency is assumed for Plain tensors.
- Partial widths/depths and incompatible padded origins fall back to exact scalar lookup. Tails never read or write padding.
- Host builds hide native archive symbols and enable a diagnostic vector counter; target builds omit that counter.
- Baseline24-graph suite passes. Four forced-layout variants each pass30 graphs including odd widths/depths, both formulas, changed inputs, invalid configurations, canaries and serialization/reload. Counters prove vector erf execution, including the actual Crouton-input/Plain-output path.
- `qnn-package/build/libJetlinkGeluHost.so` and `libJetlinkGeluV73.so` are local test outputs only. No full-vision graph was rerun with this package.

Run `qnn-package/build_host.sh`, `qnn-package/run_tests.sh`, `python qnn-package/test_layout_matrix.py`, and `qnn-package/build_v73.sh`. Existing scripts reference local licensed SDK/tool paths.

## Remaining gates

Accepted Android-produced full-domain reference comparison; package loading and correctness on physical v73; full vision boundary and recurrent replay with this package; physical layout/dispatch performance and device latency. Host/simulator timings must not be reported as NPU speedup or32ms attainment.

## Redistribution boundary

Authored source, numeric coefficients, tests and logs may be separated from local build inputs. Do not redistribute LocalLicensedInterface.cpp, SDK headers/libraries, libnative, compiler/runtime bytes, QEMU coprocessor plugins, or linked artifacts containing those bytes. `qnn-package` uses local symlinks to licensed interface/reference inputs; do not dereference those indiscriminately when packaging. The prior authored compact GELU table/source is also linked locally. Original full-model/context/package files were not modified.

## Clean source-only build

The archive contains regular authored source files, numeric lookup/coefficient tables, and test evidence only. No pre-generated licensed interface, SDK/toolchain/runtime, linked object, learned model weight, ONNX, or raw reference map is included. Install/license those external dependencies independently. The checked versions are QNN2.42.0.251225, alternate Hexagon Tools19.0.04/SDK6.4.0.2, native21.0.04, host Clang19/libc++, GCC with _Float16 support, Python3 and NumPy1.26.4 for coefficient regeneration.

From the extracted archive root, set paths for your installation (the existing workspace defaults are conveniences, not bundled dependencies):

    export QNN_SDK_ROOT=/path/to/qairt/2.42.0.251225
    export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
    export HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.04/Tools
    export NATIVE_TOOLS=/path/to/native21/Tools
    export HOST_TOOLS_ROOT=/path/to/host-tools
    export HOST_CXX="$HOST_TOOLS_ROOT/usr/lib/llvm-19/bin/clang++"

HOST_TOOLS_ROOT uses the demonstrated extracted LLVM19 layout, including usr/lib/llvm-19/include/c++/v1, its lib directory, and usr/lib/x86_64-linux-gnu. HOST_CXX may override the compiler executable without changing that libc++ layout. CC overrides the reference generator compiler.

    sh build_reference.sh
    export GELU_REFERENCE="$PWD/build/reference.bin"
    sh build_test.sh
    sh build_v73.sh
    sh qnn-package/build_host.sh
    sh qnn-package/run_tests.sh
    python qnn-package/test_layout_matrix.py
    sh qnn-package/build_v73.sh

build_reference.sh regenerates both raw maps from the authored reference equations and checks the combined SHA256 against manifest.json. A different libm/compiler/special-value convention must be investigated rather than silently accepted. This host reference still does not replace an Android-device-produced reference gate.

For a second standalone host run with Tools19:

    NATIVE_TOOLS="$HEXAGON_TOOLS_ROOT" sh build_test.sh

Package build scripts automatically regenerate the excluded LocalLicensedInterface.cpp if missing, using the included authored prepare_interface.py and this licensed source:

    python qnn-package/prepare_interface.py \
      "$QNN_SDK_ROOT/examples/QNN/OpPackage/HTP/ExampleOpPackageInterface.cpp" \
      qnn-package/LocalLicensedInterface.cpp

That generated output remains local and must not be redistributed. The initial source archive contains no dangling symlinks or assumptions that this excluded file already exists.

Coefficient regeneration, only with matching NumPy and validation afterward:

    python generate.py
    sh build_test.sh

Optional target simulator test requires the separately available corrected simulator and licensed coprocessor plugin:

    export QEMU_HEXAGON=/path/to/patched/qemu-system-hexagon
    export QEMU_COPROC_ROOT=/path/to/Tools/QEMUCoprocPlugin
    sh run_qemu.sh

The simulator patch is not bundled here; unpatched simulator compatibility is not claimed. Logs under evidence/ preserve the verified original runs. SOURCE_MANIFEST.json lists every other archived file with size and SHA256; the external archive manifest also records the tar.gz hash.
