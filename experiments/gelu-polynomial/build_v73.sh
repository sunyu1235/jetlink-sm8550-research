#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
T=${HEXAGON_TOOLS_ROOT:-/workspace/shared/htp-dependencies/build-subset/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools}
H=${HOST_TOOLS_ROOT:-/workspace/shared/htp-dependencies/host-tools}
export LD_LIBRARY_PATH="$H/usr/lib/x86_64-linux-gnu:$H/usr/lib/llvm-19/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$T/bin/clang-19" -mv73 -mhvx -mhvx-length=128B -std=c++17 -O2 -Wall -Werror -c core.cpp -o build/core-v73.o
"$T/bin/clang-19" -mv73 -mhvx -mhvx-length=128B -std=c++17 -O2 -S core.cpp -o build/core-v73.s
if grep -q vgather build/core-v73.s; then echo 'Unexpected memory gather'; exit 1; fi
printf 'Target assembly instruction evidence:\n'
grep -E 'vlut16|qf32|vshuff|vcmp.eq' build/core-v73.s | head -20
