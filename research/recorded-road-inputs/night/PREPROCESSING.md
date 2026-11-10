# Public dual-camera offline input sample

## What is ready

`inputs320/` contains 320 consecutive dual-camera inputs for the exact original Light Speed ONNX teacher identified by SHA-256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21` and checkpoint `f22e06e5-3b3f-48d2-a6b9-2f32dc20e707/56320/f22b8b34-0983-4185-a453-a87b73798d61/16384`.

- `in_warped_0.npy` through `in_warped_319.npy`: uint8 `[2,6,128,256]`, narrow then wide camera.
- `in_packed_0.npy` through `in_packed_319.npy`: float32 `[12]`, with desire in `[0:8]`, traffic convention in `[8:10]`, action time in `[10:12]`.
- `spec.json`: byte-identical copy of the established teacher specification. Model identity was not changed.
- `new_img.npy`: the same images as a memory-mappable uint8 `[320,2,6,128,256]` array.
- `frame_index.json`, `transforms.npy`, and the three scalar arrays preserve alignment, calibration, and contextual evidence.

The source is real road imagery. Context is reconstructed from logs using the original Light Speed-era modeld formulas. This is an offline, controlled teacher/student comparison, **not** a reconstruction of the original model's exact live state and not evidence of driving safety. The recording used openpilot 0.10.2 on `mici`, not the later Light Speed teacher. Its existing `modelV2` predictions are not Light Speed ground truth.

## Source, size, and access

Official openpilot model replay selects route `98395b7c5b27882e|0000002b--2686b5a2d0`, segment 1. The files were downloaded through unauthenticated HTTPS from comma's public CI artifact container. HEAD returned 200 and ranged video GET returned 206.

| File | Bytes | SHA-256 |
|---|---:|---|
| fcamera.hevc | 37,668,943 | 55bbabc1f481fb86170c4203d629264dca44b0c09cdbe93a577cb5446e00261e |
| ecamera.hevc | 37,660,514 | d42b9ccc13b35ccbcb77787bf311a7a8ee1e60255412df4e6c8dacde65e04e8a |
| rlog.zst | 11,220,963 | bb74414f3ef0d9c1f92713c52baa775ca96602312ff8ff8413491ccb5f238a14 |

Total: 86,550,420 bytes, approximately 82.54 MiB. `source_manifest.json` contains exact download URLs. No driver-camera video was downloaded. Raw recordings remain local.

Access and provenance are verified. A separate dataset license explicitly granting model-training or redistribution rights for these blob files was not located. The repository's MIT license explicitly concerns software and associated documentation; it should not be assumed to license all separately hosted road recordings. Official publication supports their intended use as replay fixtures, but this work makes no broad data-reuse, commercial-training, or redistribution-clearance claim.

## Alignment and calibration

Both source videos are HEVC, 1344×760, decoded directly to planar YUV420 at approximately 20 Hz. The log contains 1,200 valid camera states and 1,200 valid encode indexes for each road camera. All 240 calibration messages are valid and calibrated. The recorded hardware is `mici`, sensor `os04c10`.

**Do not zip camera-state records directly with decoded frames.** The video encode indexes map segmentId 0..1199 to frameId 1202..2401. The camera states in this segment instead cover frameId 1203..2402. Decode position is joined through each stream's `EncodeIdx.segmentId` and `frameId`, then camera states are joined by frameId.

The prepared pack uses segmentId 20..339, frameId 1222..1541. All IDs are consecutive. The maximum narrow/wide start-of-frame difference is 18,124 ns. The entire segment's paired camera states differ by less than 20 μs. This is measured from the log, not assumed from filenames.

For each prepared pair, the metadata cutoff is the later of the two camera-state log timestamps. The latest valid calibration/device/context event at or before that cutoff is selected. There are three calibration values in this 320-frame pack. This selection is causal at the chosen logged-camera publication time, but the log does not preserve the exact SubMaster polling instant of every original inference.

## Exact source geometry, not just matching tensor dimensions

The target geometry was checked against original Light Speed PR #39173 at immutable openpilot commit `c1d94adad4facfce493db6b58e0e3d5e4e09ae4c`, and its tinygrad submodule `d39227beebd0c3a340dc42b2cdc1e3db1c36aed9`. Geometry and warp source blobs match the newer files originally inspected. Jetlink's pinned `a863c3801da6fd925d14eb1f5b1e250f8efcc649` adapter consumes the upstream driving warp.

- Source narrow camera: focal length 1141.5, principal point `(672,380)`.
- Source wide camera: focal length 425.25, principal point `(672,380)`.
- Target narrow/MED: 512×256, focal length 910, principal point `(256,47.6)`.
- Target wide/SBIG: 512×256, focal length 455, principal point `(256,151.8)`.
- The inverse sampling matrix is `K_source @ view_from_device @ R(rpyCalib) @ inverse(K_target @ view_from_device)`.
- `rpyCalib` is first converted to float32, matching modeld. Rotation uses original Z-Y-X matrix multiplication. Final matrices are float32.
- The log also contains `wideFromDeviceEuler`; the original target modeld does not apply this field in these warp matrices, so this preprocessing does not add it independently.

## Pixel preparation and history

`preprocess_road.py` is a NumPy port of the pinned upstream `compile_warp.py`:

1. Decode source HEVC to original-resolution Y, U, V planes, avoiding an RGB round trip.
2. Warp luma with the inverse homography and chroma with the corresponding half-resolution transform `S @ M @ inverse(S)`.
3. Use nearest-neighbor sampling with half-to-even rounding and coordinate clamping. Do not substitute bilinear resize or zero borders.
4. Pack channels in this exact order: Y[even row, even column], Y[odd row, even column], Y[even row, odd column], Y[odd row, odd column], U, V.
5. Stack narrow then wide. Leave values uint8; do not normalize to `[0,1]` or subtract a mean.

Each six-channel tensor is **one camera frame**, not two temporal frames. Temporal image history belongs in the model's `state_img_q [2,5,6,128,256]`. Do not stack previous video frames into the channel dimension and do not skip three of every four video frames because the spec contains `frame_skip=4`.

Reset all model state queues to zero once before the first prepared input. Carry `next_state_*` outputs forward into each model's own recurrent state. Teacher and student must each use their own state, not share teacher state. A useful bounded experiment warms up on the first 160 frames and scores the last 160. This deliberately differs from the unknowable hidden state at this point in the original drive.

## Scalar context fidelity

The full segment has zero `carControl.latActive` events and all recorded lane-change states are off. Thus no lane-change desire is activated; the input pulse vector is eight zeros. This conclusion uses actual recorded state and is not derived from predicted desire probabilities.

Every recorded driver-monitoring state has `isRHD=false`, yielding traffic convention `[1,0]`. The current parsing schema calls that older stream `driverMonitoringStateDEPRECATED`; the `isRHD` field is retained.

The log's lateral delay is constant at `0.28665614128112793` s. The segment's static `carParams.longitudinalActuatorDelay` is 0.5 s. Original Light Speed-era modeld uses lateral smoothing 0, longitudinal smoothing 0.3 s, frame delay 0.05 s, and action delay 0.025 s. The resulting float32 action time is `[0.361656129360199,0.875]`.

`carParams` is emitted later in this segment and is treated as static vehicle metadata, not a causal per-frame message. The values reconstruct the target teacher's context formula; exact original invocation tensors were not logged. The original source version, message times, and raw rlog are retained for correction. This is not a claim of bit-exact original driving replay.

## Validation

`validate_warp.py` runs the actual pinned official tinygrad `make_frame_prepare` source on a CPU with LLVM, using padded NV12 frames matching the original source layout. It compares against the NumPy implementation on both cameras for all three distinct calibration values.

- Five of six camera samples were byte-exact across all 196,608 pixels.
- One sample differed at 2 of 196,608 pixels, each by one uint8 level; mean absolute difference was approximately 0.00001017.
- A first attempt using CPU:CLANG failed before warp execution because the clang executable was missing. The completed run used **CPU:LLVM**.
- A coordinate arithmetic difference near a nearest-neighbor rounding boundary is plausible but was not proven as the cause. No QCOM GPU/ISP bit-parity claim is made.

`warp_validation.json` records the measured details. The input bytes were frozen before teacher/student replay; subsequent documentation edits do not change them. Both models receive the same frozen inputs, which is appropriate for the controlled comparison with the stated preprocessing limit.

## Reproduce locally

Dependencies: Python, NumPy, pycapnp, zstandard, and ffmpeg. For source-function comparison, also use the pinned tinygrad source and a supported CPU backend. No pickle execution is required.

1. Obtain the three public files from `source_manifest.json` and check byte counts and hashes.
2. Run `python extract_metadata.py` with the included Cap'n Proto schemas. Schema hashes and provenance are recorded separately.
3. Run `python test_preprocessing.py`.
4. Run `python preprocess_road.py --start 20 --frames 320`.
5. Run `python export_flat_inputs.py --inputs inputs320 --spec PATH_TO_ORIGINAL_SPEC_JSON`.
6. Optionally run `DEV=CPU:LLVM python validate_warp.py --tinygrad PATH_TO_PINNED_TINYGRAD_SOURCE`.

For the already frozen pack, do not rerun these writers during an evaluation. `frozen_manifest.json` records the finalized input file hashes. This artifact does not connect to a car, run controls, or send any vehicle commands.

## Primary sources

- Official replay fixture: https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/selfdrive/test/process_replay/model_replay.py
- Public blob URL construction: https://github.com/commaai/openpilot/blob/f5dfddb72afdc5182cb28b63b3c692a9f00535fe/openpilot/tools/lib/openpilotci.py
- Logging format: https://docs.comma.ai/concepts/logs/
- Light Speed PR: https://github.com/commaai/openpilot/pull/39173
- Original target geometry: https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/common/transformations/model.py
- Source camera geometry: https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/common/transformations/camera.py
- Original modeld transforms/context: https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/selfdrive/modeld/modeld.py
- Warp build arguments: https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/selfdrive/modeld/SConscript
- Official warp implementation: https://github.com/tinygrad/tinygrad/blob/d39227beebd0c3a340dc42b2cdc1e3db1c36aed9/examples/openpilot/compile_warp.py
- Jetlink warp contract: https://github.com/zoompilot/jetlink/blob/a863c3801da6fd925d14eb1f5b1e250f8efcc649/jetlink/openpilot/warp.py
- Jetlink packed scalar order: https://github.com/zoompilot/jetlink/blob/a863c3801da6fd925d14eb1f5b1e250f8efcc649/jetlink/spec.py
