## Completed recorded-night gate, 2026-11-10

The native compact candidate now passes the complete 320-frame recorded-night sequence and the predeclared final 160 frames after warmup with the unchanged checker. Both saved QNN contexts execute in independent full-model recurrence; unrelated nodes remain on CPU. Full native day320 qualification is still pending. The completed day32 prefix passes. See `evidence/recorded-roads/qualification-snapshot.json` and `evidence/recorded-roads/night320/`.

These are host-functional numerical comparisons, not phone latency or driving-safety validation. This new gate supersedes the earlier road-pending status below. Frozen replay/model/runtime artifacts were not modified for publication.

# Native compact-context recurrent integration

This extends the source-level compact CPU integration with actual QNN host execution of two saved contexts inside the whole-model recurrence. It is not a device test or complete accelerated model: unrelated computation still runs on CPU.

The bridge loads existing saved compact-upstream and three-tail contexts once, checks their tensor ABI by name/type/shape, and shares six FP16 boundary buffers totaling 9,216 bytes. There is no padding. Eight matched inputs (12,288 output values) are bit-identical to official QNN net-run outputs. Invalid shape, dtype and nonfinite inputs are rejected; subsequent valid execution still succeeds.

The model builder proves the cut's external inputs/outputs, removes exactly 102 source nodes and inserts one custom node. All unaffected nodes remain byte-identical, and graph I/O signatures match the already-passing compact CPU model. Saved context arithmetic is not assumed identical to CPU arithmetic.

Completed full-model recurrent gates: all 160 synthetic acceptance frames and last 80; prefixes 4/32/64/128 and their half-length suffixes also pass. The original checker passes every view without threshold changes. Matching CPU control views also pass. The native candidate has its own propagated state. The full 160-frame result is a host-functional numerical acceptance, not phone or road qualification.

The 160-frame run completed successfully. All 2,952,320 emitted head values are finite; original bridge, package and context hashes remain unchanged. Both saved contexts execute for each frame, with independent recurrent state across checkpointed stages. No weights, quantizer settings or thresholds were refitted. Host simulation costs about 20–24 seconds/frame and is not phone timing. Native day/night replay and actual device qualification remain pending.

Completed local run: /workspace/shared/jetlink-compact-native-replay. Inspect logs and checkpoint before any restart. The run_staged.sh script uses flock and refuses existing output directories; intentional recovery should use replay_checkpoint.py --resume with unchanged model/library/context identities, then score the completed set. Original bridge/context hashes are retained in bridge/artifact-hashes.json.

No phone execution, target performance, road safety, repo publication or deployment is claimed. Source archive excludes model weights, SDK binaries, compiled shared objects, contexts and raw replay data.
