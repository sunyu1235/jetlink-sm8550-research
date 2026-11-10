# Daytime public dual-camera held-out replay inputs

## Ready artifact

`inputs320/` contains 320 consecutive aligned daytime camera pairs for the original Light Speed ONNX teacher, SHA-256 `6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`.

- `in_warped_0.npy` … `in_warped_319.npy`: uint8 `[2,6,128,256]`, narrow camera then wide camera.
- `in_packed_0.npy` … `in_packed_319.npy`: float32 `[12]`, desire `[0:8]`, traffic convention `[8:10]`, action time `[10:12]`.
- `spec.json`: byte-identical to `/workspace/shared/jetlink-history-regenerated/spec.json`; file SHA-256 `721b92094842200653744c988ae3b238dc2fb8947931316ec9ea0c19d479b623`.
- `new_img.npy`, scalar arrays, `transforms.npy`, and `frame_index.json` retain the batched inputs and per-frame source/context evidence.
- `frozen_manifest.json` in the parent directory freezes every input-pack file. Do not modify these inputs during evaluation.

This route is distinct from the earlier night route and held out from the local student-fitting experiment. This does **not** establish that the original teacher was never trained on this route. These are real public images with reconstructed target context, not a historical capture of Light Speed invocation tensors or live recurrent state.

No model inference, model training, vehicle connection, control commands, publishing, uploading, or redistribution occurred while preparing this pack. The original night pack was not changed: all 641 files in its existing frozen contract manifest were checked against their recorded hashes.

## Source and rights limits

Official `DEMO_ROUTE` is `5beb9b58bd12b691/0000010a--a51155e496/0`, defined in [the pinned official replay header](https://github.com/commaai/openpilot/blob/f5dfddb72afdc5182cb28b63b3c692a9f00535fe/openpilot/tools/replay/replay.h). The copied header was fetched and the exact route declaration checked. The selected start-frame image was inspected: it is daylight, approaching a stop-marked intersection. The route's initial residential preview does not mean every selected frame has residential surroundings.

All three full files returned unauthenticated HTTPS 200. Their sizes match the earlier official source listing. Full-file SHA-256s are local checksums for reproducibility, not publisher-signed checksums.

| File | Bytes | SHA-256 |
|---|---:|---|
| `fcamera.hevc` | 37,463,690 | `03c86cd00aba7fe627b224f463cfac4836d3ab385c628224f137d916c597e338` |
| `ecamera.hevc` | 37,558,900 | `9b377fb1d9fa2e4c33f95af7c8b829fe0c9cbaae0176cc2ef1c93cc8980dd012` |
| `rlog.zst` | 10,742,090 | `0d4eb61a36e83cf0b626060169cfe77e78f65c8a25daf19bfdaaa4887ea45c82` |

Total: 85,764,680 bytes. Exact public download URLs, HTTP status, byte counts, ETags, and modification times are in `source_manifest.json`. No cabin/driver-camera video was downloaded. Raw files and derived images remain local.

No separate explicit license granting model-training or redistribution rights for these public blob recordings was found in the inherited provenance research. The software repository's MIT license must not be assumed to cover separately hosted recordings. This work claims verified official replay-fixture provenance and availability, not broad commercial, training, privacy, or redistribution clearance.

The raw log identifies openpilot `0.10.4`, commit `1e627a73985e531ff146497262720f06a6fc3e2b`, nightly branch, `mici`, and recording initialization 2026-02-25 17:30:29.609080 UTC. Blob modification time is 2026-09-03 00:44:53 UTC. These dates have different meanings. Recorded `modelV2` outputs are not Light Speed ground truth.

## Full-source and selected-frame alignment

Each video fully decodes to 1,200 HEVC/YUV420p frames at 1344×760 and approximately 20 Hz. The full log has 1,200 valid encode indexes and 1,203 valid camera-state events for each road camera. All keys are unique.

- Video decode position/`EncodeIdx.segmentId` 0…1199 maps to road `frameId` 2…1201.
- Camera-state events cover `frameId` 0…1202. Never zip their list positions with decoded frames.
- Every encode timestamp was checked against the matching camera-state `frameId` for both SOF and EOF, across all 1,200 pairs.
- Full-segment maximum narrow/wide SOF difference: 45,363 ns.
- Selected sequence: `segmentId` 200…519, `frameId` 202…521, no frame gaps or subsampling.
- Selected maximum narrow/wide SOF difference: 15,468 ns.
- Selected first/last narrow SOF: 429545323336417 / 429561273300338 ns, a 15.949964-second first-to-last interval at 20 Hz.

The full log has 233 calibration messages: first 9 invalid, then 224 valid. The first valid calibration is at 429539538201998 ns. All required dynamic context is usable from around segmentId 110; the chosen slice begins at 200, beyond startup. Every chosen frame has valid, calibrated latest metadata, and the slice uses one calibration vector:

`[-0.00040542264468967915, 0.164722740650177, 0.004513291642069817]`

The metadata cutoff for each pair is the later of its two camera-state publication timestamps. Latest events at or before that cutoff are selected and their validity asserted. `frame_index.json` records selected source timestamps for calibration, device, lateral delay, driver monitoring, control, car state, and car parameters. This is causal at that chosen publication time; the exact historical modeld/SubMaster polling instant is not in the log. Encode-index publication may occur later and is used for offline identity joins, not to backdate context.

## Geometry and pixels

The implementation is copied from the verified night preprocessing and adapted only for this route's alignment/context selection. Geometry is pinned to original Light Speed PR #39173 at openpilot commit `c1d94adad4facfce493db6b58e0e3d5e4e09ae4c`, with tinygrad commit `d39227beebd0c3a340dc42b2cdc1e3db1c36aed9`.

- Source narrow: focal 1141.5, center `(672,380)`; source wide: focal 425.25, same center.
- Target narrow MED: 512×256, focal 910, center `(256,47.6)`.
- Target wide SBIG: 512×256, focal 455, center `(256,151.8)`.
- Inverse sampling matrix: `K_source @ view_from_device @ R(rpyCalib) @ inverse(K_target @ view_from_device)`.
- Calibration first converts to float32; rotation uses upstream Z-Y-X multiplication; final matrix is float32.
- The logged `wideFromDeviceEuler` is preserved but not added to this transform, matching the original target modeld.
- Decode directly to planar YUV420 without RGB conversion. Apply nearest-neighbor homography, ties-to-even rounding, clamped borders, and corresponding half-resolution chroma transform.
- Channels are `Y[even,even]`, `Y[odd,even]`, `Y[even,odd]`, `Y[odd,odd]`, `U`, `V`. Values remain uint8, not normalized.

Each six-channel image is one camera frame, not two temporal frames. Temporal image history is the separate model state queue. The `frame_skip=4` spec field does not authorize dropping three of every four replay inputs.

## Scalar reconstruction and exact limits

The chosen slice has 198 frames with lateral control inactive followed by 122 active frames. All 320 recorded lane-change states are `off`. All 1,594 high-rate car-state messages within the selected interval are valid and have both blinkers false. The first selected lateral control is inactive, forcing the target DesireHelper into `off`; without blinkers it cannot trigger a lane-change desire. The model's rising-edge desire pulse is therefore eight zeros. This conclusion is from this slice's raw context and target logic, not inherited from the night route or generated desire probabilities.

Later in the full daytime route there are 5 pre-lane-change, 101 lane-change-starting and 20 lane-change-finishing model states, so a whole-route zero-desire claim would be false. This pack deliberately does not include those events.

Every selected driver-monitoring state has `isRHD=false`, yielding `[1,0]`. The parsing schema exposes the older stream as `driverMonitoringStateDEPRECATED`; the `isRHD` field is present.

The route's lateral delay is 0.4275784194469452 s. Both valid `carParams` records agree on longitudinal actuator delay 0.15000000596046448 s. For this slice the first parameter record was already available, so no future static record is needed. The original target modeld adds lateral smoothing 0, longitudinal smoothing 0.3 s, frame delay 0.05 s, and action delay 0.025 s. Final float32 action time is:

`[0.5025784373283386, 0.5249999761581421]`

Selected logged speed ranges from 1.1034455299377441 to 19.251502990722656 m/s. Source raw timestamps and reconstruction formulas are retained. These scalar tensors are suitable for a controlled teacher/student comparison, but are not claimed to be bit-exact tensors used by the older on-vehicle model.

## Required recurrence discipline

Zero all `state_*` queues exactly once before input 0. On each consecutive frame, carry that model's own `next_state_*` values forward. Teacher and student each own their independent recurrent state; never feed the teacher's recurrent outputs into the student. Suggested bounded evaluation: warm up on frames 0…159, score frames 160…319. Report that reset/warmup protocol. It does not reconstruct the original car's hidden state or establish driving safety.

## Validation

`test_preprocessing.py` passes channel-packing/identity, rounding/clamping, and exact source-derived geometry checks. `audit_inputs.py` passes full-source joins, counts, hashes, per-frame causal/valid context, repeated vehicle metadata, all 320 flat/batched input equivalences, exact spec identity, and unchanged night-pack hashes. See `audit.json` for checked values and maximum metadata ages.

The complete official tinygrad source archive for the exact pinned commit was downloaded locally. Its `compile_warp.py` bytes match the inherited pinned source. `validate_warp.py` imports the official source function, prepares padded NV12 input, and runs CPU:LLVM, not the ONNX model, at sequence indexes 0, 159, and 319 for both cameras. `CACHELEVEL=0` disables serialized disk-cache reads; the script does not call the upstream pickle-producing CLI or load a model/checkpoint pickle.

- All six recomputed NumPy arrays match the saved pack exactly.
- All three wide-camera samples and narrow index 159 are byte-exact against tinygrad.
- Narrow index 0 differs at 1 of 196,608 uint8 elements, by 1 level.
- Narrow index 319 differs at 2 of 196,608 elements, each by 1 level.
- Maximum per-sample mean absolute difference: 0.000010172526041666666.

`rounding_diagnostic.json` tests a source-coordinate ramp at the same pinned warp. It identifies one narrow-chroma coordinate difference at destination `(68,30)`: NumPy's float32 x coordinate is exactly 262.5 and rounds to source `(262,103)`, while the pinned CPU warp selects `(263,103)`. This localizes the discrepancy to a sampling/rounding boundary; it does not prove the compiler-level arithmetic cause. It does not change the frozen image bytes. CPU sample comparison is not a claim of QCOM GPU/ISP parity. An initial standalone CPU backend probe failed before warp execution because its default cache path was read-only; the successful runs used `CACHELEVEL=0` and a local cache path.

## Reproduce in a new output directory

Dependencies: Python, NumPy, pycapnp, zstandard, ffmpeg/ffprobe; exact pinned tinygrad and a working LLVM CPU backend for the optional source-function comparison. Required source files and hashes are preserved locally. Do not rerun writers over the frozen pack.

1. Download the three `source_manifest.json` URLs and verify byte counts and hashes.
2. Run `extract_metadata.py` using the included schemas.
3. Run `test_preprocessing.py`.
4. Run `preprocess_road.py --start 200 --frames 320` in the fresh output directory.
5. Run `export_flat_inputs.py --inputs inputs320 --spec /workspace/shared/jetlink-history-regenerated/spec.json`.
6. Run `DEV=CPU:LLVM CACHELEVEL=0 XDG_CACHE_HOME=./cache python validate_warp.py --tinygrad source/tinygrad-d39227beebd0c3a340dc42b2cdc1e3db1c36aed9`.
7. Run `audit_inputs.py` after recording full `ffprobe -count_frames` metadata, then freeze checksums before any model evaluation.

## Primary implementation sources

- [Original Light Speed PR](https://github.com/commaai/openpilot/pull/39173)
- [Original target geometry](https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/common/transformations/model.py)
- [Original source-camera geometry](https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/common/transformations/camera.py)
- [Original modeld context/transforms](https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/selfdrive/modeld/modeld.py)
- [Original desire helper](https://github.com/commaai/openpilot/blob/c1d94adad4facfce493db6b58e0e3d5e4e09ae4c/openpilot/selfdrive/controls/lib/desire_helper.py)
- [Pinned official warp](https://github.com/tinygrad/tinygrad/blob/d39227beebd0c3a340dc42b2cdc1e3db1c36aed9/examples/openpilot/compile_warp.py)
- [Jetlink packed scalar contract](https://github.com/zoompilot/jetlink/blob/a863c3801da6fd925d14eb1f5b1e250f8efcc649/jetlink/spec.py)
- [Official logging documentation](https://docs.comma.ai/concepts/logs/)
