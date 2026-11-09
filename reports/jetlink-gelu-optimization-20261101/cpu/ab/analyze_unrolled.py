from __future__ import annotations
import csv, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\sunbo\Documents\openpilot')
REPORT=ROOT/'reports/jetlink-gelu-optimization-20261101/cpu/ab'
RAW=ROOT/'outputs/jetlink-gelu-optimization-20261101/cpu/ab'
REF=ROOT/'outputs/jetlink-gelu-history-20261101/parity-160'
BASE=ROOT/'outputs/jetlink-gelu-optimization-20261101/ab/b1-shared1/b1-shared1-out'
UPSTREAM=Path(r'C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research\scripts\verify_parity.py')
RUNS=['a0-original-f8','b1-unrolled-f8','a2-original-f8']
N=160; BYTES=73808

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ints(p):
 try:return [int(x.strip()) for x in Path(p).read_text().splitlines() if x.strip()]
 except:return []
def analyze(name):
 folder=RAW/name/(name+'-out')
 files=sorted(folder.glob('out_*.bin'),key=lambda p:int(re.search(r'out_(\d+)',p.name).group(1)))
 compare=REPORT/'compare'/name; compare.mkdir(parents=True,exist_ok=True)
 shutil.copy2(REF/'spec.json',compare/'spec.json')
 issues=[]
 if len(files)!=N:issues.append(f'output_count_{len(files)}')
 baseline_match=0; hashes=[]; finite=True; size_ok=True
 for i in range(N):
  src=folder/f'out_{i}.bin'; refp=REF/f'out_ref_{i}.npy'; base=BASE/f'out_{i}.bin'
  if not src.exists():issues.append(f'missing_{i}'); continue
  size_ok &= src.stat().st_size==BYTES
  if src.stat().st_size!=BYTES:issues.append(f'bad_size_{i}_{src.stat().st_size}')
  arr=np.fromfile(src,dtype='<f4')
  if arr.size!=18452:issues.append(f'bad_elements_{i}_{arr.size}')
  if not np.isfinite(arr).all():finite=False;issues.append(f'nonfinite_{i}')
  if base.exists() and src.read_bytes()==base.read_bytes():baseline_match+=1
  shutil.copy2(refp,compare/refp.name)
  ref=np.load(refp,allow_pickle=False)
  np.save(compare/f'out_link_{i}.npy',arr.reshape(ref.shape))
  hashes.append(sha(src))
 rows=[]
 with (REPORT/f'{name}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=N:issues.append(f'csv_rows_{len(rows)}')
 latency=np.array([float(x['execute_ms']) for x in rows])
 nf=sum(int(x['nonfinite']) for x in rows)
 if nf:issues.append(f'csv_nonfinite_{nf}')
 stdout=(REPORT/f'{name}.stdout.txt').read_text(encoding='utf-8',errors='replace')
 mask=re.search(r'Cpus_allowed:\s*([0-9a-f]+)',stdout)
 runner_exit=re.findall(r'RUNNER_EXIT=(\d+)',stdout)
 adb_exit=int((REPORT/f'{name}.adb-exitcode.txt').read_text().strip())
 cmd=[sys.executable,str(UPSTREAM),'compare','--dir',str(compare)]
 env=os.environ.copy();env['PYTHONPATH']=str(UPSTREAM.parent.parent)+os.pathsep+env.get('PYTHONPATH','')
 p=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (compare/'compare.stdout.txt').write_text('$ '+' '.join(cmd)+'\n\n'+p.stdout,encoding='utf-8')
 (compare/'compare.exitcode.txt').write_text(str(p.returncode)+'\n',encoding='ascii')
 parity=p.returncode==0 and '\nOK: every slice' in p.stdout
 failheads=[line.strip() for line in p.stdout.splitlines() if '[FAIL]' in line]
 return {'name':name,'custom_op':'unrolled' if 'unrolled' in name else 'original','effective_affinity_mask':mask.group(1) if mask else None,
  'frame_count':len(files),'all_outputs_73808_bytes':bool(size_ok and len(files)==N),'all_finite':finite,'csv_frames':len(rows),
  'first_frame_csv_temp':{'battery_deci_c':int(rows[0]['battery_deci_c']),'gpu_milli_c':int(rows[0]['gpu_milli_c'])} if rows else None,
  'timing_ms':{'mean':float(latency.mean()),'p50':float(np.percentile(latency,50)),'p99':float(np.percentile(latency,99)),'min':float(latency.min()),'max':float(latency.max())},
  'gpu_temp_root_snapshot_milli_c':[ints(REPORT/f'{name}.gpu-before.txt'),ints(REPORT/f'{name}.gpu-after.txt')],
  'process_exit':{'adb':adb_exit,'runner':int(runner_exit[-1]) if runner_exit else None},
  'byte_equal_to_accepted_shared1_baseline_frames':baseline_match,'frame_sha256':hashes,
  'parity':{'pass':parity,'exit_code':p.returncode,'failed_lines':failheads,'log':str(compare/'compare.stdout.txt')},'issues':issues}

results=[analyze(x) for x in RUNS]
# cross-run equality
perframe=[]
for i in range(N):
 vals=[(RAW/n/(n+'-out')/f'out_{i}.bin').read_bytes() for n in RUNS]
 perframe.append(vals[0]==vals[1]==vals[2])
a0=results[0]['timing_ms'];b=results[1]['timing_ms'];a2=results[2]['timing_ms']
a={k:(a0[k]+a2[k])/2 for k in ['mean','p50','p99','min','max']}
delta={k:b[k]-a[k] for k in a}
summary={'test':'Unrolled GELU A-B-A, f8 affinity, shared=1, 160 same frames','only_changed_variable':'JL_CUSTOM_OP_LIBRARY: original / unrolled / original','model':'light-speed-gelu-both-ctx.onnx; unchanged','inputs':'jetlink-history-160; same 160-frame sequence and state initialization','runs':results,
 'outputs_byte_identical_across_all_runs':all(perframe),'identical_frame_count':sum(perframe),'outputs_equal_accepted_original_shared1_baseline':all(r['byte_equal_to_accepted_shared1_baseline_frames']==N for r in results),
 'all_parity_pass':all(r['parity']['pass'] for r in results),'all_processes_successful':all(r['process_exit']=={'adb':0,'runner':0} for r in results),'all_outputs_finite_and_correct_size':all(r['all_finite'] and r['all_outputs_73808_bytes'] for r in results),
 'latency_comparison':{'original_A_average_ms':a,'unrolled_B_ms':b,'B_minus_A_ms':delta,'B_minus_A_percent':{k:100*delta[k]/a[k] for k in a}},
 'temperature_note':'First CSV battery_deci_c is the common in-run sample; all three starts are 230 (23.0 C). Runner GPU temperature is -1. Separate root-only KGSL temp snapshots are retained, not mixed with CSV samples.',
 'thermal_root_snapshot_note':'A0/B1/A2 starts are '+ '/'.join(str(r['gpu_temp_root_snapshot_milli_c'][0][0]) for r in results)+' milli-C; ends are '+ '/'.join(str(r['gpu_temp_root_snapshot_milli_c'][1][0]) for r in results)+' milli-C.',
 'limits':['Single B run; performance conclusion is only this sequence and run order.','The 4-thread setting is unchanged because the exact same history_probe invocation and runner were used; custom-op library was the sole changed environment variable.']}
(REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
for r in results:print(r['name'],r['timing_ms'],r['byte_equal_to_accepted_shared1_baseline_frames'],r['parity'],r['issues'])
