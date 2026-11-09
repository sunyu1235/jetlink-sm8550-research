#!/system/bin/sh
set -e
mkdir -p /data/local/tmp/gelu-best-stress-prefix160
i=0
while [ "$i" -lt 160 ]; do
  cp "/data/local/tmp/gelu-best-stress-1000-out/out_${i}.bin" "/data/local/tmp/gelu-best-stress-prefix160/out_${i}.bin"
  i=$((i + 1))
done
