# Phone CPU32 capture log

Device: `192.168.1.49:5555` (`XT2301`, `rtwo`). ORT reported `1.29.0`. The two runs used the CPU EP only, 4 intra-op threads from the probe's CPU branch, and no HTP/GPU EP. No app, root state, or thermal settings were changed.

All frames' input files were checked on the phone before inference. The vision test had 16/16 `new_img` inputs, each 1,572,864 bytes. `state_img_q` started zero and was fed back through the probe's alternating state buffers. Vision outputs were checked as 16/16 files at exactly 65,536 bytes. The policy test had 64/64 supplied tensors at exact expected sizes: `view_39` 65,536, `desire` 32, `traffic_convention` 8, `action_t` 8 bytes. Its two state tensors started zero and were fed back.

## Vision CPU32 on phone

Started on phone: 2026-11-01 13:34:43 +08:00; output frame 15 written at 13:34:52.477 +08:00.

```text
C:\adb\adb.exe shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64:/vendor/lib64; export ADSP_LIBRARY_PATH=/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64; /data/local/tmp/qnn_context_probe /data/local/tmp/vision-fp32.onnx 0 0 none 16 /data/local/tmp/vision-fp32-inputs /data/local/tmp/vision-phone-cpu32-audit-20261101 varied cpu'
```

Process exit code: `0`.

```text
ORT 1.29.0
in new_img 1572864 bytes
in state_img_q 7864320 bytes
out view_39 65536 bytes
out next_state_img_q 7864320 bytes
shared=0 rpc_us=0 n=16 mean_ms=608.229 p50_ms=613.225 p99_ms=633.318 min_ms=573.280 max_ms=638.649 over50=16
```

## Policy CPU32 on phone

Started on phone: 2026-11-01 13:35:51.121 +08:00; output frame 15 written at 13:35:53.321 +08:00.

```text
C:\adb\adb.exe shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64:/vendor/lib64; export ADSP_LIBRARY_PATH=/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64; /data/local/tmp/qnn_context_probe /data/local/tmp/policy-fp32.onnx 0 0 none 16 /data/local/tmp/phone-policy-cpu32-inputs /data/local/tmp/policy-phone-cpu32-audit-20261101-output varied cpu'
```

Process exit code: `0`.

```text
ORT 1.29.0
in view_39 65536 bytes
in desire 32 bytes
in traffic_convention 8 bytes
in action_t 8 bytes
in state_desire_q 4224 bytes
in state_feat_q 8388608 bytes
out outputs 73808 bytes
out next_state_desire_q 4224 bytes
out next_state_feat_q 8388608 bytes
output_type=1 nonfinite=0/18452
shared=0 rpc_us=0 n=16 mean_ms=150.245 p50_ms=144.962 p99_ms=166.653 min_ms=143.475 max_ms=206.076 over50=16
```

The phone policy outputs were compared against the matching `parity-varied` PC CPU reference frames with the unmodified upstream compare command. Exit code was 0; maximum absolute error across outputs was 0.2101. See `phone_cpu32_full.compare.log`.
