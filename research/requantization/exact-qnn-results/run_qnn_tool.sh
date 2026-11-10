#!/bin/bash
set -euo pipefail
Q=/workspace/shared/htp-dependencies/build-subset/qairt/2.42.0.251225
P=/workspace/shared/jetlink-python310/cpython-3.10.22-linux-x86_64-gnu
H=/workspace/shared/htp-dependencies/build-subset/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/lib
HOST=/workspace/shared/jetlink-exact-integration/host-deps/unpacked
export PATH="$HOST/usr/lib/llvm-19/bin:$PATH"
export QNN_SDK_ROOT="$Q"
export PYTHONPATH="$Q/lib/python:/workspace/shared/jetlink-converter-python"
export LD_LIBRARY_PATH="$Q/lib/x86_64-linux-clang:$HOST/usr/lib/x86_64-linux-gnu:$H:$P/lib:/workspace/shared/jetlink-exact-integration/host-deps/unpacked/usr/lib/llvm-19/lib"
case "$1" in
 qnn-context-binary-generator|qnn-net-run) exec "$Q/bin/x86_64-linux-clang/$1" "${@:2}";;
 *) exec "$P/bin/python3.10" "$Q/bin/x86_64-linux-clang/$1" "${@:2}";;
esac
