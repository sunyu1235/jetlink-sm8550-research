#!/bin/sh
set -eu
B=/workspace/shared/htp-dependencies/build-subset
Q=${QNN_SDK_ROOT:-$B/qairt/2.42.0.251225}
S=${HEXAGON_SDK_ROOT:-$B/Hexagon_SDK/6.4.0.2}
T=${HEXAGON_TOOLS_ROOT:-$S/tools/HEXAGON_Tools/19.0.04/Tools}
cd "$(dirname "$0")"
mkdir -p build
mkdir -p build
if [ ! -f LocalLicensedInterface.cpp ]; then
 python prepare_interface.py "$Q/examples/QNN/OpPackage/HTP/ExampleOpPackageInterface.cpp" LocalLicensedInterface.cpp
fi
for src in GeluFp16 LocalLicensedInterface core; do
 "$T/bin/clang-19" -std=c++17 -I"$Q/include/QNN" -fPIC -Wall -Werror -Wno-format -Wno-unused-function -Wno-unused-command-line-argument -stdlib=libc++ -DTHIS_PKG_NAME=jetlink_gelu -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -O2 -DPREPARE_DISABLED -mv73 -I"$S/rtos/qurt/computev73/include/qurt" -I"$S/rtos/qurt/computev73/include/posix" -I"$S/incs" -I"$S/incs/stddef" -c "$src.cpp" -o "build/$src.o"
done
"$T/bin/hexagon-clang++" -mv73 -fPIC -std=c++17 -shared build/GeluFp16.o build/LocalLicensedInterface.o build/core.o -o build/libJetlinkGeluV73.so
