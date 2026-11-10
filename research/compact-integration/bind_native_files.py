"""Validate native context ABI and reuse upstream files without zero-padding/copy."""
from pathlib import Path
import json,numpy as np
from compact_bindings import ORDER,SHAPES,ALIASES
p=Path(__file__).parent;up=p/'upstream-host-output';target=json.load(open(p/'qnn/tails_net.json'))['graph']['tensors'];lines=[];records=[]
for n in ORDER:
 meta=json.load(open(up/(ALIASES[n]+'_native.raw.json')));assert meta['Datatype']=='QNN_DATATYPE_FLOAT_16' and tuple(meta['Dimensions'])==SHAPES[n]
 assert target[n]['data_type']==534 and tuple(target[n]['dims'])==SHAPES[n]
for i in range(8):
 parts=[];total=0
 for n in ORDER:
  f=up/f'Result_{i}'/(ALIASES[n]+'_native.raw');expected=2*np.prod(SHAPES[n]).item();assert f.stat().st_size==expected;assert np.isfinite(np.fromfile(f,np.float16)).all();total+=expected;parts.append(n+':='+str(f))
 assert total==9216;lines.append(' '.join(parts));records.append(dict(sample=i,boundary_bytes=total,raw_files_borrowed_no_copy=True))
(p/'tails-native-inputs.txt').write_text('\n'.join(lines)+'\n');(p/'native-binding-proof.json').write_text(json.dumps(dict(scope='serialized QNN host file handoff; not Android shared-allocation proof',samples=records,padding_bytes=0,dtype_casts=0),indent=2))
