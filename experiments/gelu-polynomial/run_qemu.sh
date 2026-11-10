#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
D=$(pwd)
T=${HEXAGON_TOOLS_ROOT:-/workspace/shared/htp-dependencies/build-subset/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools}
H=${HOST_TOOLS_ROOT:-/workspace/shared/htp-dependencies/host-tools}
export LD_LIBRARY_PATH="$H/usr/lib/x86_64-linux-gnu:$H/usr/lib/llvm-19/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$T/bin/hexagon-clang++" -mv73 -mhvx -mhvx-length=128B -std=c++17 -O2 -DJL_NO_THREADS core.cpp test_core.cpp -o build/test_core-v73.elf
Q=${QEMU_HEXAGON:-/workspace/shared/htp-dependencies/simulator-subset/qemu-build/qemu-system-hexagon}
P=${QEMU_COPROC_ROOT:-/workspace/shared/htp-dependencies/simulator-subset/Hexagon_SDK/6.4.0.2/tools/Tools/QEMUCoprocPlugin}
timeout -k 5 60 "$Q" -M V73NA_1024 -m 6G -display none -monitor none -cpu any,coproc="$P" -kernel "$D/build/test_core-v73.elf" -append "${GELU_REFERENCE:?Set GELU_REFERENCE to an absolute reference.bin path} $D/build/qemu-actual.bin"
