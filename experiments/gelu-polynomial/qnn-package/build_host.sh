#!/bin/sh
set -eu
H=${HOST_TOOLS_ROOT:-/workspace/shared/htp-dependencies/host-tools}
export LD_LIBRARY_PATH="$H/usr/lib/x86_64-linux-gnu:$H/usr/lib/llvm-19/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
B=/workspace/shared/htp-dependencies/build-subset
Q=${QNN_SDK_ROOT:-$B/qairt/2.42.0.251225}
T=${NATIVE_TOOLS:-/workspace/shared/htp-dependencies/libnative21/Tools}
cd "$(dirname "$0")"
mkdir -p build
export QNN_SDK_ROOT="$Q"
if [ ! -f LocalLicensedInterface.cpp ]; then
 python prepare_interface.py "$Q/examples/QNN/OpPackage/HTP/ExampleOpPackageInterface.cpp" LocalLicensedInterface.cpp
fi
for src in GeluFp16 LocalLicensedInterface core; do
 "${HOST_CXX:-$H/usr/lib/llvm-19/bin/clang++}" -std=c++17 -nostdinc++ -isystem "$H/usr/lib/llvm-19/include/c++/v1" -isystem "$H/usr/lib/llvm-19/lib/clang/19/include" -I"$Q/include/QNN" -I"$T/libnative/include" -DJL_POLY_COUNT -DTHIS_PKG_NAME=jetlink_gelu -D__HVXDBL__ -DUSE_OS_LINUX -fPIC -O2 -Wall -Werror -Wno-missing-braces -Wno-unused-function -Wno-format -c "$src.cpp" -o "build/clang-host-$src.o"
done
"${HOST_CXX:-$H/usr/lib/llvm-19/bin/clang++}" -shared -stdlib=libc++ -L"$H/usr/lib/llvm-19/lib" build/clang-host-GeluFp16.o build/clang-host-LocalLicensedInterface.o build/clang-host-core.o -Wl,--exclude-libs,ALL -Wl,--whole-archive "$T/libnative/lib/libnative.a" -Wl,--no-whole-archive -lpthread -o build/libJetlinkGeluHost.so
python - <<'PY'
import ctypes,os
q=ctypes.CDLL(os.environ['QNN_SDK_ROOT']+'/lib/x86_64-linux-clang/libQnnHtp.so',mode=os.RTLD_GLOBAL|os.RTLD_NOW);print('QNN host dlopen OK')
s=ctypes.CDLL('./build/libJetlinkGeluHost.so',mode=os.RTLD_NOW);print('package dlopen OK')
PY
