"""Repair QNN mixed FP16 FC bias metadata only after exact value preservation proof."""
from pathlib import Path
import argparse,json,re,tarfile,io,copy,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();p=a.directory
for name in ['preserve_fixed.cpp','preserve_fixed.bin','preserve-bias-repair.json']:
 if (p/name).exists():raise FileExistsError(p/name)
m=json.load(open(p/'preserve_net.json'))['graph'];names=set()
for n in m['nodes'].values():
 if n['type']=='FullyConnected' and len(n['input_names'])==3:
  a,w,b=n['input_names'];o=n['output_names'][0]
  if [m['tensors'][x]['data_type'] for x in [a,w,b,o]]==[534,534,562,534]:names.add(b)
assert len(names)==19
# No other operation consumes these changed constants.
for n in m['nodes'].values():
 if names.intersection(n['input_names']):assert n['type']=='FullyConnected' and n['input_names'][2] in names
cpp=(p/'preserve.cpp').read_text();report=[]
with tarfile.open(p/'preserve.bin','r') as src,tarfile.open(p/'preserve_fixed.bin','w') as dst:
 for member in src:
  f=src.extractfile(member) if member.isfile() else None
  name=member.name.removesuffix('.raw')
  if name in names:
   original=f.read();x=np.frombuffer(original,dtype='<f4');h=x.astype('<f2');back=h.astype('<f4');assert np.array_equal(x.view('<u4'),back.view('<u4')),(name,'not exactly representable as half')
   data=h.tobytes();mi=copy.copy(member);mi.size=len(data);dst.addfile(mi,io.BytesIO(data));report.append(dict(tensor=name,elements=x.size,exact_float32_roundtrip=True))
   pat=r'(static ModelError_t addTensor_'+re.escape(name)+r'\(QnnModel& model\)\{.*?)(?=\nstatic |\Z)';match=re.search(pat,cpp,re.S);assert match
   chunk=match.group();assert chunk.count('QNN_DATATYPE_FLOAT_32')==1;new=chunk.replace('QNN_DATATYPE_FLOAT_32','QNN_DATATYPE_FLOAT_16');cpp=cpp[:match.start()]+new+cpp[match.end():]
  else:dst.addfile(member,f)
assert len(report)==19
(p/'preserve_fixed.cpp').write_text(cpp);(p/'preserve-bias-repair.json').write_text(json.dumps(dict(changes=report,changed_numeric_values=0,warning='post-converter format repair; full-model parity still required'),indent=2));print('19 biases restored FP16; all values exact FP32 roundtrip')
