# Unrolled GELU library A-B-A at f8/shared=1

## Result

The same 160-frame input, context, runner, `f8` CPU affinity and `shared=1` option were used for all rounds. The only change was `JL_CUSTOM_OP_LIBRARY`: original → unrolled → original. Each round completed with runner exit 0, produced 160 finite float32 outputs of 73,808 bytes, passed the unchanged upstream `verify_parity.py compare`, and matched the accepted original shared=1 baseline byte-for-byte for every frame. The three rounds were also byte-for-byte identical to one another.

| Round | Library | Mean ms | P50 ms | P99 ms | First CSV battery temp |
|---|---|---:|---:|---:|---:|
| A0 | original | 61.717 | 61.511 | 64.262 | 23.0 °C |
| B1 | unrolled | 61.392 | 61.202 | 63.883 | 23.0 °C |
| A2 | original | 61.795 | 61.617 | 64.537 | 23.0 °C |
| A average | original | 61.756 | 61.564 | 64.399 | — |
| B − A average | unrolled | **−0.364 (−0.59%)** | **−0.362 (−0.59%)** | **−0.516 (−0.80%)** | — |

The original library reproduced its prior `f8/shared=1` mean of about 61.8 ms. The unrolled library was slightly faster in this single B run, but the change is under 1% and does not establish a repeatable optimization. Treat the performance result as **inconclusive / not accepted for promotion**; retain the original library as the operational baseline unless a later repeated test shows a clearer gain. No app or model was changed.

The source unit proof at [unit-proof.json](../unit-proof.json) records 63,491 bit-exact cases for each formula. This A-B-A model-level comparison also confirms that the unrolled library produces exactly the same outputs as the original on this 160-frame sequence.

## Temperature and process checks

All three runs started at the same first-row CSV battery sample (230 deci-°C = 23.0 °C). The runner cannot read the GPU sensor (`gpu_milli_c=-1`), so root-only KGSL temperature snapshots are kept separately: starts A0/B1/A2 26.8/29.2/29.9 °C; ends 51.3/52.8/55.2 °C. GPU frequency was not collected. These snapshots are ancillary and are not mixed with CSV temperatures.

The wrapper confirmed `Cpus_allowed=f8` and `Cpus_allowed_list=3-7` before each runner `exec`. Both libraries, runner, context model, and the 160-frame input manifest are SHA-256 recorded in `hash-manifest.json`. The local unrolled library hash matches the deployed phone library hash. Each run used the exact same runner invocation and options; only the custom-op library path changed.

## Reproduction artifacts

- `run-unrolled-aba.ps1` runs the three phone rounds and saves logs, CSV, exit status, battery/GPU temperature snapshots, and pulled raw frame outputs.
- `analyze_unrolled.py` validates outputs, checks byte identity against the accepted original shared=1 baseline, stages arrays with references, and runs the original parity comparator with its original thresholds.
- `summary.json` contains full per-round metrics, parity status, byte-equality results, and constraints.
- `hash-manifest.json` records hashes for the model, runner, both libraries, input manifest, outputs, CSV, commands and logs.
- The comparison stdout and exit status for each round are in `compare/<round>/`.
- Raw outputs are under `outputs/jetlink-gelu-optimization-20261101/cpu/ab/`.
