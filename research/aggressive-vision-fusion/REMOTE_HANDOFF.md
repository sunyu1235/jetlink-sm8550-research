# Aggressive vision fusion: phone-workspace handoff

This isolated source/evidence branch starts at the accepted phone commit `4619484678610712fc132b4bf11848cb4a35e481`. Existing files and branches are unchanged. No phone run, merge, deployment or 32 ms result is claimed.

## Fetch into a separate worktree

```sh
git fetch origin research/aggressive-vision-fusion-20261110
git worktree add ../jetlink-aggressive-review -b review/aggressive-vision-fusion-20261110 origin/research/aggressive-vision-fusion-20261110
cd ../jetlink-aggressive-review
```

Choose fresh local names if those already exist. Keep the accepted phone artifacts and any uncommitted changes in their existing workspace.

## Start here

1. Read [the experiment](README.md), [bridge contract and build instructions](bridge/README.md), and [corrected HVX GELU package](../../experiments/gelu-polynomial/README.md).
2. Verify the accepted Android full-domain activation reference and all package layouts/tails on the matching toolchain. The cloud uses alternate SDK6.4/Tools19 plus isolated native21, not proof of equivalence to SDK5.5.5/Tools8.7.06.
3. Integrate the single vision context with the accepted phone's accelerated remainder. The host proof uses a CPU remainder and is not a complete fast phone model. Preserve all heads, recurrent states and thresholds.
4. After target placement and short correctness checks, run a matched-temperature baseline/candidate/baseline speed screen. Record model/runtime hashes and separate initialization from Run latency. Host or QEMU times cannot estimate phone speed.
5. Only if the screen is promising, run unchanged independently recurrent synthetic160 and held-out day/night320 gates. Reject numerical failures rather than relaxing thresholds. The repack has static byte-equivalence evidence only and requires runtime validation.

## Evidence and limits

The original 99-file aggressive source/evidence bundle is preserved byte-for-byte. Its `SOURCE_MANIFEST.json` is relative to the repository root and covers the original 98 payload files. `PUBLICATION_MANIFEST.json` beside this handoff covers all added files except itself.

Scalar and vector GELU full-model bridges each passed exactly one independent-zero-state recorded-night frame. All four vector outputs are byte-identical to the split-native scalar control. The saved vector head passes the unchanged full-head checker against CPU; the native candidate is not bitwise identical to CPU. The companion polynomial README's earlier standalone-only status predates this integrated evidence; use `README.md` and `bridge/vector-full-frame-check/result.json` for the later result.

For publication, 48 Python tests were rerun successfully (26 extractor, 3 model-cut, 19 repack), the bridge built, four malformed ORT inputs were rejected before QNN initialization, and nine native tensor descriptor cases passed without loading QNN. Source syntax checks also passed. See `publication-validation.json` and `publication-focused-tests.log`. Stored exhaustive GELU, QEMU and one-frame inference evidence was not rerun during publication. No full repository suite, fresh QNN inference, Android build, phone timing, recurrent acceptance or repacked-model inference is claimed.

## Focused source checks

With Python, NumPy and ONNX, the 45 fixture tests require no learned model or SDK:

```sh
python -m unittest discover -s research/aggressive-vision-fusion -p test_extract_region.py -v
python -m unittest discover -s research/aggressive-vision-fusion -p test_repack_remainder.py -v
```

The three model-cut tests require the local reconstructed source with SHA-256 `3c80a2e4ede9781a1e5aec41c503e12e2f2c739e987e168046de8ed87325800a`:

```sh
export JETLINK_SOURCE_MODEL=/absolute/path/to/teacher-compact-boundaries.onnx
python -m unittest discover -s research/aggressive-vision-fusion -p test_build_model.py -v
```

That source is not the accepted Android cached graph. Full builds/inference require the original local model, weights, inputs, licensed SDK/runtime and newly built context. Follow the bridge and GELU READMEs for explicit local paths. They are not provided by cloning this branch. Do not overwrite accepted artifacts or commit generated outputs.

## Publication boundary

No model weights, ONNX, generated QNN C++, raw road data, tensor dumps, context binaries, SDK headers/libraries, linked binaries, compiler/runtime files or secrets are added. Numeric activation tables and coefficients are authored arithmetic data, not learned weights. `prepare_interface.py` locally adapts a licensed SDK example; its output is excluded and must not be redistributed. Existing files retain their licenses. Simulator source/build material is a separate supplement and is not included in this initial phone-workspace handoff.
