#!/system/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64:/vendor/lib64
export ADSP_LIBRARY_PATH=/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64
export JL_CUSTOM_OP_LIBRARY=/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so
export JL_FRAME_CSV=/data/local/tmp/gelu-best-stress-1000.csv
export JL_INPUT_MODULO=160
mkdir -p /data/local/tmp/gelu-best-stress-1000-out
exec taskset f8 /system/bin/sh -c 'grep Cpus_allowed /proc/self/status; exec /data/local/tmp/jetlink-gelu-repair-20261101/history_probe /data/local/tmp/light-speed-gelu-both-ctx.onnx 1 0 none 1000 /data/local/tmp/jetlink-history-160 /data/local/tmp/gelu-best-stress-1000-out varied'
