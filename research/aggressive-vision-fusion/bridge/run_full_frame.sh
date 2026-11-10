#!/bin/bash
# Explicit licensed dependency paths; no installation and no implicit inference retry.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec "${PYTHON:-python}" "$HERE/test_full_frame.py" "$@"
fi
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the licensed QNN SDK}"
QNN_HOST_LIB_DIR=${QNN_HOST_LIB_DIR:-"$QNN_SDK_ROOT/lib/x86_64-linux-clang"}
export JETLINK_QNN_BACKEND=${JETLINK_QNN_BACKEND:-"$QNN_HOST_LIB_DIR/libQnnHtp.so"}
export JETLINK_QNN_SYSTEM=${JETLINK_QNN_SYSTEM:-"$QNN_HOST_LIB_DIR/libQnnSystem.so"}
: "${JETLINK_GELU_PACKAGE:?Set the scalar or vector GELU package path}"
# Context, road, vendor and split-control can be passed as Python CLI arguments.
export LD_LIBRARY_PATH="$QNN_HOST_LIB_DIR${QNN_EXTRA_LIBRARY_PATH:+:$QNN_EXTRA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${PYTHON:-python}" "$HERE/test_full_frame.py" "$@"
