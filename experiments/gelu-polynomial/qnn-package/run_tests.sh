#!/bin/sh
set -eu
Q=${QNN_SDK_ROOT:-/workspace/shared/htp-dependencies/build-subset/qairt/2.42.0.251225}
H=${HOST_TOOLS_ROOT:-/workspace/shared/htp-dependencies/host-tools}
export LD_LIBRARY_PATH="$H/usr/lib/x86_64-linux-gnu:$H/usr/lib/llvm-19/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$(dirname "$0")"
mkdir -p build
gcc -std=c11 -Wall -Wextra -Werror -I"$Q/include/QNN" test_qnn_gelu.c -ldl -o build/test_qnn_gelu
build/test_qnn_gelu "$Q/lib/x86_64-linux-clang/libQnnHtp.so" "$PWD/build/libJetlinkGeluHost.so" "${GELU_REFERENCE:?Set GELU_REFERENCE to the generated/verified reference.bin}" > roundtrip-test.log 2>&1
grep 'RESULT passed_graphs=24 expected_graphs=24 status=PASS' roundtrip-test.log
