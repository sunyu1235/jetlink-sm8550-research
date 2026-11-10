# Compact partition and three-tail integration

## Recovery result, 2026-11-09

The reconstructed last-query partition and all three compact tails now pass the unchanged upstream numerical checker in the source-level CPU whole model:

| Sequence | All frames | Predeclared last 160 after warmup |
| --- | --- | --- |
| Original synthetic acceptance | 160 PASS | Not a separate test |
| Recorded night inputs | 320 PASS | PASS |
| Recorded day inputs | 320 PASS | PASS |

Teacher and candidate propagate their own recurrent states. No weights, thresholds or quantizer settings were refitted during this recovery. These are numerical agreement tests on reconstructed recorded-camera inputs, not ground-truth driving accuracy or safety validation.

The interrupted night run had written through frame 306. Its saved recurrent checkpoint was at frame 287; the recovery resumed at 288 and recomputed the remaining suffix through 319. It did not pretend that the checkpoint contained state through 306. Day ran fresh. The qualification script verified 320 finite 18,452-value outputs for each road and recorded ordered-output, model and checker hashes. All four road checker invocations exited zero.

## What the optimization changes

All 288 history K/V tokens remain. Only the final query/residual row and final token-local tails are computed at the compact boundary. Six FP16 boundary payloads decrease from 2,654,208 bytes to 9,216 bytes, a factor of 288. This is logical tensor payload, not measured DMA traffic or latency. There is no zero-prefix restoration. The final source model removes 102 source nodes and inserts 90 compact partition nodes plus 15 tail nodes.

The source partition is reconstructed from the original teacher, not recovered byte-identical accepted phone cached contexts. Old downstream cached contexts cannot consume new shapes unchanged. Existing source/rebuild integration manifests retain their historical `numerical_validation: pending` field; the later `evidence/recovery-qualification.json` records the completed CPU numerical results without rewriting those original manifests.

## Separate native HTP-host evidence

Before recovery, two compatible serialized contexts (compact upstream and three tails) passed offline SoC43/v73 preparation and host reload/execution. Eight deterministic samples joined six native FP16 buffers without padding, producing 12,288 finite values. Against the compact CPU pipeline: RMSE 0.0027571, maximum absolute error 0.03125, minimum per-output correlation 0.9999821. This is not bit-exact agreement and is not the full-model road test above.

Contexts are 8,617,984 and 2,158,592 bytes. Native file handoff does not prove Android shared-memory allocation or zero-copy DMA. No Android or phone execution occurred. Shared-host isolated CPU timing from the upstream partition report is diagnostic only and cannot establish SM8550 speed.

## Checks rerun after recovery

- Seven compact binding tests pass, including shape, dtype, finite-value and contiguity rejection.
- Ten compact partition tests pass, including mask, Q-consumer and normalization guards.
- Both road replays and all four unchanged-checker views pass.
- Qualification script compiles and validates the completed output sets.

Evidence is in `evidence/`. The source archive also includes the partition preparation and validation scripts in a sibling folder. Scripts retain explicit workspace paths from the experiment; adapt them to matching source models, dependencies and artifacts. They are not a turnkey phone deployment package.

## Three-track status and next gate

1. **Operators/partitioning:** compact integration has the CPU acceptance above and a separate small native-host pipeline pass. Full native deployed-model execution and Android arithmetic equivalence remain unverified. Earlier full host-context execution crashed; do not call an offline preparation pass a full runtime pass.
2. **Requantization:** earlier selected A16/W8 first-projection native QNN execution with CPU remainder passed synthetic160 and night320, including post-warmup. The full mixed context prepared, but full-context accuracy and phone speed remain unverified. This recovery did not combine it with compact boundaries or retest that combination.
3. **Distillation:** earlier single-block SiLU replacement passed synthetic160 and day/night320 views. Expansion to twelve logical positions failed night post-warmup and one day-frame plan check, so that expanded candidate remains rejected. This recovery did not retune or combine it with compact boundaries.

The next meaningful qualification is integrating actual compact native context execution into the recurrent whole-model replay, followed by matching-device checks and controlled full-pipeline timing when hardware is available. Offline source-level agreement does not certify the custom HTP arithmetic or a 32 ms target. Combining individually passing tracks requires a new full-model check.

No weights, raw inputs/outputs, licensed SDK files, compiled binaries or contexts are included in the source/evidence archive. No commit, push or deployment was performed during recovery.
