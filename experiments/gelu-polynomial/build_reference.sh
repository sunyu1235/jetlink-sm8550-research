#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
"${CC:-gcc}" -std=gnu11 -O2 -fno-fast-math reference.c -lm -o build/make_reference
build/make_reference > build/reference.bin
python - <<'PY'
from pathlib import Path
import hashlib,json
actual=hashlib.sha256(Path('build/reference.bin').read_bytes()).hexdigest()
expected=json.loads(Path('manifest.json').read_text())['reference_file_sha256']
print('Reference SHA256:',actual)
if actual != expected:
 raise SystemExit('Reference differs from tested host map. Investigate toolchain/libm/rounding before treating tests as equivalent.')
PY
