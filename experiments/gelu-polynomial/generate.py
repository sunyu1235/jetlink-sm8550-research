"""Authored coefficient generator. NumPy 1.26.4; no SDK dependencies."""
import math, struct
from pathlib import Path
import numpy as np
out=Path(__file__).parent
rows=[]
for j in range(32):
 mid=-8+(j+.5)*.5
 t=np.cos(np.pi*(np.arange(64)+.5)/64)*.25
 vals=np.array([.5*math.erfc(-(mid+z)/math.sqrt(2)) for z in t])
 c=np.polynomial.polynomial.polyfit(t,vals,6).astype(np.float32)
 rows.append([mid]+list(c))
a=np.array(rows,dtype='<f4').T.view('<u4')
text='#pragma once\n#include <stdint.h>\n// Rows: midpoint then polynomial coefficients c0..c6.\n'
text+='alignas(128) static const uint32_t coefficients[8][32] = {\n'
text+='\n'.join('{'+','.join('0x%08xu'%int(x) for x in row)+'},' for row in a)+'\n};\n'
(out/'coefficients.h').write_text(text)
print('Generated',a.nbytes,'bytes of coefficients.')
