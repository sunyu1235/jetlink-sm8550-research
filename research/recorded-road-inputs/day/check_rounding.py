#!/usr/bin/env python3
"""Diagnose sampling coordinates at the observed narrow-chroma rounding boundary."""
import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).parent
sys.path.insert(0, str(root/'source/tinygrad-d39227beebd0c3a340dc42b2cdc1e3db1c36aed9'))
from tinygrad import Tensor
from examples.openpilot.compile_warp import warp_perspective_tinygrad
from preprocess_road import sampling_indices, UV_SCALE

matrix = np.load(root/'inputs320/transforms.npy',allow_pickle=False)[0,0]*UV_SCALE
sw, sh = 672,380
ramp = np.arange(sw*sh,dtype=np.int32)
actual = warp_perspective_tinygrad(Tensor(ramp),Tensor(matrix),(256,128),(sh,sw),0).numpy().reshape(128,256)
yy,xx = sampling_indices(matrix,(256,128),(sh,sw))
expected = yy*sw+xx
differences = []
for y,x in np.argwhere(actual != expected):
    fx,fy = np.float32(x),np.float32(y)
    numerator = matrix[0,0]*fx + matrix[0,1]*fy + matrix[0,2]
    denominator = matrix[2,0]*fx + matrix[2,1]*fy + matrix[2,2]
    differences.append({'dst_uv_x_y': [int(x),int(y)],
                        'numpy_src_x_y': [int(xx[y,x]),int(yy[y,x])],
                        'tinygrad_src_x_y': [int(actual[y,x]%sw),int(actual[y,x]//sw)],
                        'numpy_float32_src_x_before_round': float(numerator/denominator)})
report = {'runtime':'CPU:LLVM','method':'Same pinned warp, source int32 linear-index ramp; independent coordinate diagnostic, inputs unchanged.',
          'different_sample_coordinates':len(differences),'differences':differences,
          'limit':'The diagnostic identifies a CPU coordinate-rounding boundary difference, not QCOM parity or the compiler-level arithmetic cause.'}
(root/'rounding_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
