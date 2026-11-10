#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
Q=${QNN_SDK_ROOT:-/workspace/shared/htp-dependencies/build-subset/qairt/2.42.0.251225}
ORT_HEADERS=${ORT_HEADERS:-/workspace/shared/jetlink-exact-integration/include}
${CXX:-g++} -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -Wall -Wextra -ffp-contract=off -frounding-math bridge.cpp -I"$ORT_HEADERS" -I"$Q/include/QNN" -I"$Q/share/QNN/converter/jni" -ldl -pthread -o libjetlink_compact_bridge.so
