#!/bin/bash
set -euo pipefail
Q=/workspace/shared/htp-dependencies/build-subset/qairt/2.42.0.251225
H=/workspace/shared/htp-dependencies/build-subset/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/lib
HOST=/workspace/shared/jetlink-exact-integration/host-deps/unpacked
export LD_LIBRARY_PATH="$Q/lib/x86_64-linux-clang:$HOST/usr/lib/x86_64-linux-gnu:$H:$HOST/usr/lib/llvm-19/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/workspace/shared/jetlink-python:${PYTHONPATH:-}"
exec python "$@"
