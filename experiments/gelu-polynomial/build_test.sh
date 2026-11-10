#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
H=${HOST_TOOLS_ROOT:-/workspace/shared/htp-dependencies/host-tools}
T=${NATIVE_TOOLS:-/workspace/shared/htp-dependencies/libnative21/Tools}
export LD_LIBRARY_PATH="$H/usr/lib/x86_64-linux-gnu:$H/usr/lib/llvm-19/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"${HOST_CXX:-$H/usr/lib/llvm-19/bin/clang++}" -std=c++17 -nostdinc++ -isystem "$H/usr/lib/llvm-19/include/c++/v1" -I"$T/libnative/include" -D__HVXDBL__ -O2 -Wall -Werror core.cpp test_core.cpp "$T/libnative/lib/libnative.a" -L"$H/usr/lib/llvm-19/lib" -stdlib=libc++ -lpthread -o build/test_core
build/test_core "${GELU_REFERENCE:?Set GELU_REFERENCE to the generated/verified reference.bin}" build/actual.bin
