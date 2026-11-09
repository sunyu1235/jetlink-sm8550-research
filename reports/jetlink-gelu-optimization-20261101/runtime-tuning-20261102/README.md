# Snapdragon 8 Gen 2 QNN tuning: evidence and next controlled check

**2026-11-09 correction:** `soc_model` / `vtcm_mb` must be tested with newly compiled graphs. Applying them to the existing EPContext does not recompile its schedule. The next prepared comparison is therefore an isolated 13 MB policy partition, default / SoC43 / default; VTCM is deferred pending device-capacity confirmation. See [the revised partition experiment](../partition-tuning-20261109/README.md). No phone run for either knob has completed yet.

**State as of 2026-11-02.** The phone is not currently reachable over wireless ADB (`adb devices -l` returns no devices), so no new phone run or app change was made in this step. A small Android probe was prepared and cross-compiled locally to isolate two QNN graph configuration options.

## What the measurements say

- The accepted, precision-preserving CPU GELU + HTP configuration remains about **62 ms/frame** in the unprofiled sustained replay. In its 16-frame ORT trace, CPU provider nodes account for **6.000 ms/frame** and QNN provider nodes for **57.482 ms/frame**. The trace is instrumented, so these are attribution numbers rather than production latency. See [the ORT profile report](../profile-20261102/README.md) and [`ort-profile-summary.json`](../profile-20261102/ort-profile-summary.json).
- Four QNN partitions in that accurate trace consume about **18.81 ms/frame** combined (5.175, 4.583, 4.529, and 4.521 ms). They produce groups of policy heads. That makes CPU GELU loop tuning an insufficient route to 30 ms.
- The separate QNN optrace that did complete ran the **all-HTP graph with native GELU**, which is already known to fail model parity. It is useful only as a rough workload map: HTP spent the most cycles in multiply, layer normalization, GELU, reshape, matrix multiply, slice, and softmax work. Do not use its op ranking as a measured breakdown of the accurate hybrid graph.
- That optrace run took 149.405 ms/frame while instrumentation was enabled. This is profiling overhead, not an unprofiled performance result. Its eight accelerator-cycle samples average 97.18 million cycles; those samples matched the op-cycle sum and root accelerator counter.
- A direct optrace of the accurate graph stopped during QNN context creation when wireless ADB went offline. It produced no usable accurate-model QNN optrace.

## Why the next variables are specific

The accepted runner already requests `htp_performance_mode=burst`, `qnn_context_priority=high`, `enable_htp_shared_memory_allocator=1`, and `htp_graph_finalization_optimization_mode=3`. Raising those again is not a new experiment.

The current ONNX Runtime QNN provider reference documents two relevant options absent from the accepted runner: `soc_model` defaults to unknown (`0`), and `vtcm_mb` is unset (`0`). Qualcomm identifies SM8550 / Snapdragon 8 Gen 2 as SoC model **43** with HTP **V73**. These provide two concrete, separately testable questions: whether explicitly targeting the phone's SoC changes graph scheduling, and whether reserving 8 MiB of VTCM reduces the cost of the large HTP workloads. Neither option is assumed to help. The VTCM value must be accepted by the installed runtime; a rejected or slower context is a failed experiment, not a reason to alter the installed app.

Sources: [ONNX Runtime QNN provider options](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options), [Qualcomm's SM8550 target table](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372), and [Qualcomm HTP documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html).

## Prepared probe and decision rule

[`history_probe_runtime_tuning.c`](history_probe_runtime_tuning.c) is the existing diagnostic runner with only optional `soc_model` and `vtcm_mb` provider settings added. The compiled Android binary and hash are in the build manifest; [`build_probe.ps1`](build_probe.ps1) reproduces the build. With both variables unset, it retains the accepted runner's HTP settings. Set only one variable per run:

- baseline: leave `JL_SOC_MODEL` and `JL_VTCM_MB` unset;
- SoC targeting: `JL_SOC_MODEL=43`;
- VTCM: `JL_VTCM_MB=8`.

For each candidate, use the same accurate hybrid model, the same 160-frame input sequence, `shared=1`, four CPU threads, and the original exact GELU library. Compare baseline/candidate/baseline on the same device session conditions; capture the output tensors and run the unchanged parity checker. Promote a candidate only if parity passes and the A-B-A mean improves by at least 3% without a worse P99. First run a short smoke and monitor temperature. Do not use the all-HTP native-GELU graph for this comparison.

If those documented runtime options do not produce a repeatable gain, the remaining plausible path toward 30 ms is a QNN HTP custom fused kernel for the large attention or policy-head subgraphs, with exact GELU semantics retained. That requires a matching QNN 2.42 development toolchain and per-head parity work; it is a separate compiler/kernel project, not a safe one-line model rewrite. The previous unvalidated all-HTP route is not a valid substitute.

## Connection and scope

The last wireless ADB check found no attached device, and the accurate optrace log ends after ORT initialization. The runner was only built on the host. No APK, QNN/system library, root setting, clock, or thermal policy was changed.
