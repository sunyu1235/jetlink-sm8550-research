from __future__ import annotations
import csv, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(r'C:\Users\sunbo\Documents\openpilot')
REPORT=ROOT/'reports/jetlink-gelu-optimization-20261101/threads'
RAW=ROOT/'outputs/jetlink-gelu-optimization-20261101/threads'
BASE=ROOT/'outputs/jetlink-gelu-history-20261101/gelu-history-160-out'
PARITY=ROOT/'outputs/jetlink-gelu-history-20261101/parity-160'
UPSTREAM=Path(r'C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research\scripts\verify_parity.py')
RUNS=['a0-threads4','b1-threads1','a2-threads4'];N=160;BYTES=73808;VALUES=18452

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def check_run(name):
 rawdir=RAW/name/(name+'-out')
 compare=REPORT/'compare'/name;compare.mkdir(parents=True,exist_ok=True)
 shutil.copy2(PARITY/'spec.json',compare/'spec.json')
 errors=[];frame_hashes=[];finite=True;sizeok=True;base_match=0
 for i in range(N):
  stress=rawdir/f'out_{i}.bin';baseline=BASE/f'out_{i}.bin';refp=PARITY/f'out_ref_{i}.npy'
  if not stress.is_file():errors.append(f'missing output {i}');sizeok=False;continue
  if stress.stat().st_size!=BYTES:errors.append(f'frame {i}: got {stress.stat().st_size} bytes');sizeok=False
  arr=np.fromfile(stress,dtype='<f4')
  if arr.size!=VALUES:errors.append(f'frame {i}: got {arr.size} float32');sizeok=False
  nf=int((~np.isfinite(arr)).sum());
  if nf:errors.append(f'frame {i}: {nf} nonfinite');finite=False
  frame_hashes.append(sha(stress))
  if baseline.is_file() and stress.read_bytes()==baseline.read_bytes():base_match+=1
  elif not baseline.is_file():errors.append(f'missing accepted baseline {i}')
  else:errors.append(f'frame {i}: raw differs from accepted baseline')
  ref=np.load(refp,allow_pickle=False)
  if ref.size!=VALUES:errors.append(f'frame {i}: reference size {ref.size}')
  elif arr.size==VALUES:
   np.save(compare/f'out_link_{i}.npy',arr.reshape(ref.shape))
   shutil.copy2(refp,compare/refp.name)
 rows=[]
 with (REPORT/f'{name}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=N:errors.append(f'csv has {len(rows)} rows')
 ms=np.array([float(x['execute_ms']) for x in rows],dtype=np.float64)
 if sum(int(x['nonfinite']) for x in rows):errors.append('CSV reports nonfinite values')
 stdout=(REPORT/f'{name}.stdout.txt').read_text(encoding='utf-8',errors='replace')
 m=re.search(r'Cpus_allowed:\s*([0-9a-f]+)',stdout)
 t=re.search(r'intra_op\.num_threads=(\d+)',stdout)
 runner=re.findall(r'RUNNER_EXIT=(\d+)',stdout)
 adb=int((REPORT/f'{name}.adb-exitcode.txt').read_text().strip())
 cmd=[sys.executable,str(UPSTREAM),'compare','--dir',str(compare)]
 env=os.environ.copy();env['PYTHONPATH']=str(UPSTREAM.parent.parent)+os.pathsep+env.get('PYTHONPATH','')
 (compare/'command.txt').write_text(' '.join(cmd)+'\nPYTHONPATH='+env['PYTHONPATH']+'\n',encoding='utf-8')
 proc=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (compare/'compare.stdout.txt').write_text(proc.stdout,encoding='utf-8')
 (compare/'compare.exitcode.txt').write_text(str(proc.returncode)+'\n',encoding='ascii')
 parity=proc.returncode==0 and '\nOK: every slice' in proc.stdout
 fail=[line.strip() for line in proc.stdout.splitlines() if '[FAIL]' in line]
 return {'name':name,'expected_intra_op_threads':4 if name!='b1-threads1' else 1,
  'logged_intra_op_threads':int(t.group(1)) if t else None,'affinity_mask':m.group(1) if m else None,
  'outputs':{'count':len(list(rawdir.glob('out_*.bin'))),'all_73808_bytes':sizeok and len(list(rawdir.glob('out_*.bin')))==N,
             'all_finite':finite,'byte_identical_accepted_baseline_frames':base_match,'sha256_per_frame':frame_hashes},
  'csv':{'rows':len(rows),'first_frame_battery_deci_c':int(rows[0]['battery_deci_c']) if rows else None,
         'first_frame_gpu_milli_c':int(rows[0]['gpu_milli_c']) if rows else None,
         'timing_ms':{'mean':float(ms.mean()),'p50':float(np.percentile(ms,50)),'p99':float(np.percentile(ms,99)),'min':float(ms.min()),'max':float(ms.max())}},
  'gpu_root_temp_milli_c':{'before':int((REPORT/f'{name}.gpu-before.txt').read_text().strip()),'after':int((REPORT/f'{name}.gpu-after.txt').read_text().strip())},
  'process_exit':{'adb':adb,'runner':int(runner[-1]) if runner else None},
  'parity':{'pass':parity,'exit_code':proc.returncode,'failed_lines':fail,'log':str(compare/'compare.stdout.txt')},
  'errors':errors}

runs=[check_run(n) for n in RUNS]
identical=all((RAW/RUNS[0]/(RUNS[0]+'-out')/f'out_{i}.bin').read_bytes()==(RAW/n/(n+'-out')/f'out_{i}.bin').read_bytes() for n in RUNS[1:] for i in range(N))
a0=runs[0]['csv']['timing_ms'];b=runs[1]['csv']['timing_ms'];a2=runs[2]['csv']['timing_ms']
keys=['mean','p50','p99','min','max'];a={k:(a0[k]+a2[k])/2 for k in keys};delta={k:b[k]-a[k] for k in keys}
checks={'all_160_frames_present_and_size_correct':all(r['outputs']['count']==N and r['outputs']['all_73808_bytes'] for r in runs),
 'all_outputs_finite':all(r['outputs']['all_finite'] for r in runs),
 'all_outputs_identical_to_accepted_baseline':all(r['outputs']['byte_identical_accepted_baseline_frames']==N for r in runs),
 'all_outputs_identical_across_aba':identical,
 'thread_counts_logged_4_1_4':[r['logged_intra_op_threads'] for r in runs]==[4,1,4],
 'affinity_f8_all_runs':[r['affinity_mask'] for r in runs]==['f8']*3,
 'all_original_parity_checks_pass':all(r['parity']['pass'] for r in runs),
 'all_processes_exit_zero':all(r['process_exit']=={'adb':0,'runner':0} for r in runs)}
summary={'status':'PASS' if all(checks.values()) else 'FAIL','experiment':'160-frame shared=1, f8 affinity, original CPU GELU library, spinning=0; threads 4/1/4','only_variable':'JL_CPU_THREADS=4,1,4; same runner/model/input/other options','checks':checks,
 'latency_comparison_ms':{'four_thread_A_average':a,'one_thread_B':b,'B_minus_A':delta,'B_minus_A_percent':{k:100*delta[k]/a[k] for k in keys}},
 'temperature_sampling':'CSV first-row battery_deci_c is the in-run sample; runner gpu_milli_c is -1. Root-only KGSL temp was read immediately before and after each run, with a conservative stop at >=60000 milli-C.',
 'thermal_stop_threshold_milli_c':60000,'runs':runs,
 'limits':['A-B-A contains one B1 only; timing difference does not establish a universal optimum.','No frequency, thermal policy, app, model, system library, or original runner was changed.'],
 'overall_pass_requires_all_checks':True}
(REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(REPORT/'checks.json').write_text(json.dumps({'checks':checks,'errors':{r['name']:r['errors'] for r in runs}},indent=2),encoding='utf-8')
print(json.dumps({'status':summary['status'],'checks':checks,'latency':summary['latency_comparison_ms']},indent=2))
for r in runs:print(r['name'],r['outputs']['byte_identical_accepted_baseline_frames'],r['parity']['exit_code'],r['csv']['timing_ms'])
raise SystemExit(0 if summary['status']=='PASS' else 1)

