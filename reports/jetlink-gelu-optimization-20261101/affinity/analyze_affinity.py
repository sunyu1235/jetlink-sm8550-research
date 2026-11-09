from __future__ import annotations
import csv, hashlib, json, math, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'C:\Users\sunbo\Documents\openpilot')
BASE = ROOT / 'outputs/jetlink-gelu-optimization-20261101/affinity'
REPORT = ROOT / 'reports/jetlink-gelu-optimization-20261101/affinity'
REF = ROOT / 'outputs/jetlink-gelu-history-20261101/parity-160'
UPSTREAM = Path(r'C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research\scripts\verify_parity.py')
PYTHON = Path(sys.executable)
RUNS = ['a0-all-ff', 'b1-nonlittle-f8', 'a2-all-ff']
N = 160
NBYTES = 18452 * 4

def sha(path: Path) -> str:
  h=hashlib.sha256()
  with path.open('rb') as f:
    for b in iter(lambda:f.read(1<<20), b''): h.update(b)
  return h.hexdigest()

def read_root_temp(path: Path):
  try: return [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]
  except Exception: return []

def check_run(name):
  rawdir=BASE/name/(name+'-out')
  files=sorted(rawdir.glob('out_*.bin'),key=lambda p:int(re.search(r'out_(\d+)',p.name).group(1)))
  issues=[]
  if len(files)!=N: issues.append(f'file_count={len(files)} expected={N}')
  compare_dir=REPORT/'compare'/name
  compare_dir.mkdir(parents=True,exist_ok=True)
  shutil.copy2(REF/'spec.json',compare_dir/'spec.json')
  for i in range(N):
    ref_path=REF/f'out_ref_{i}.npy'
    if not ref_path.is_file(): issues.append(f'missing_reference={i}'); continue
    shutil.copy2(ref_path,compare_dir/ref_path.name)
    raw=rawdir/f'out_{i}.bin'
    if not raw.is_file(): issues.append(f'missing_output={i}'); continue
    if raw.stat().st_size!=NBYTES: issues.append(f'bad_bytes_{i}={raw.stat().st_size}')
    arr=np.fromfile(raw,dtype='<f4')
    if arr.size!=18452: issues.append(f'bad_elements_{i}={arr.size}')
    nonfinite=int((~np.isfinite(arr)).sum())
    if nonfinite: issues.append(f'nonfinite_{i}={nonfinite}')
    ref=np.load(ref_path,allow_pickle=False)
    if arr.size==ref.size:
      np.save(compare_dir/f'out_link_{i}.npy',arr.reshape(ref.shape))
    else: issues.append(f'ref_size_mismatch_{i}={ref.shape}')
  rows=[]
  with (REPORT/f'{name}.csv').open(newline='') as f: rows=list(csv.DictReader(f))
  if len(rows)!=N: issues.append(f'csv_rows={len(rows)} expected={N}')
  execute=np.array([float(r['execute_ms']) for r in rows],dtype=np.float64)
  nf=sum(int(r['nonfinite']) for r in rows)
  if nf: issues.append(f'csv_nonfinite={nf}')
  # parity compare, with unchanged upstream command and environment
  cmd=[str(PYTHON),str(UPSTREAM),'compare','--dir',str(compare_dir)]
  env=__import__('os').environ.copy()
  env['PYTHONPATH']=str(UPSTREAM.parent.parent)+__import__('os').pathsep+env.get('PYTHONPATH','')
  oldlog=compare_dir/'compare.stdout.txt'
  if oldlog.exists() and 'ModuleNotFoundError: No module named \"jetlink\"' in oldlog.read_text(encoding='utf-8',errors='replace'):
    oldlog.replace(compare_dir/'compare.stdout.txt.import-failure')
    (compare_dir/'compare.exitcode.txt').replace(compare_dir/'compare.exitcode.txt.import-failure')
  proc=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (compare_dir/'compare.stdout.txt').write_text('$ '+' '.join(cmd)+'\n\n'+proc.stdout,encoding='utf-8')
  (compare_dir/'compare.exitcode.txt').write_text(str(proc.returncode)+'\n',encoding='ascii')
  stdout=proc.stdout
  failheads=[]
  for line in stdout.splitlines():
    if '[FAIL]' in line or ' FAIL' in line: failheads.append(line.strip())
  ok='\nOK: every slice' in stdout and proc.returncode==0
  return {
    'name':name,'frames':len(files),'all_file_bytes_valid':all((rawdir/f'out_{i}.bin').is_file() and (rawdir/f'out_{i}.bin').stat().st_size==NBYTES for i in range(N)),
    'all_finite':nf==0 and not any(x.startswith('nonfinite_') for x in issues),
    'raw_sha256_by_frame':[sha(rawdir/f'out_{i}.bin') for i in range(N)],
    'csv_rows':len(rows),'first_csv_battery_deci_c':int(rows[0]['battery_deci_c']) if rows else None,
    'first_csv_gpu_milli_c':int(rows[0]['gpu_milli_c']) if rows else None,
    'latency_ms':{'mean':float(execute.mean()),'p50':float(np.percentile(execute,50)),'p99':float(np.percentile(execute,99)),'min':float(execute.min()),'max':float(execute.max())},
    'gpu_temp_raw_milli_c_before_after':read_root_temp(REPORT/f'{name}.gpu-before.txt')+read_root_temp(REPORT/f'{name}.gpu-after.txt'),
    'effective_mask':re.search(r'Cpus_allowed:\s*([0-9a-f]+)',(REPORT/f'{name}.stdout.txt').read_text()).group(1),
    'process_exit':{'adb':int((REPORT/f'{name}.adb-exitcode.txt').read_text()),'runner':int(re.findall(r'RUNNER_EXIT=(\d+)',(REPORT/f'{name}.stdout.txt').read_text())[-1])},
    'parity':{'pass':ok,'exit_code':proc.returncode,'failed_heads':failheads,'log':str(compare_dir/'compare.stdout.txt')},
    'issues':issues,
  }

results=[check_run(n) for n in RUNS]
# Verify exact output reproducibility across the three affinity runs.
identical=[]
for i in range(N):
  blobs=[(BASE/n/(n+'-out')/f'out_{i}.bin').read_bytes() for n in RUNS]
  identical.append(blobs[0]==blobs[1]==blobs[2])
means={r['name']:r['latency_ms']['mean'] for r in results}
med_a=(means['a0-all-ff']+means['a2-all-ff'])/2
summary={'test':'A-B-A CPU affinity only; shared=1; same model/custom-op/runner/inputs/options','mask_map':{'ff':'all CPUs 0-7','f8':'non-little CPUs 3-7'},'input':'160-frame synthetic roadlike history sequence, independent zero-initialized state each run','result_runs':results,'outputs_byte_identical_all_frames':all(identical),'byte_identical_frame_count':sum(identical),'all_runs_parity_pass':all(r['parity']['pass'] for r in results),'all_runs_finite_and_correct_size':all(r['all_finite'] and r['all_file_bytes_valid'] for r in results),'mean_latency_delta_nonlittle_vs_A_average_ms':means['b1-nonlittle-f8']-med_a,'mean_latency_delta_percent_vs_A_average':100*(means['b1-nonlittle-f8']/med_a-1),'warnings':['Root-only KGSL temperature values are ancillary snapshots; runner CSV GPU temperature is unavailable (-1). Compare first-row CSV battery_deci_c as requested: A0=220 (22.0 C), B1=220 (22.0 C), A2=230 (23.0 C). GPU sensor start snapshots differ by more than 2 C between A0 and B1; preserve this as a thermal caveat.','Toybox taskset -p display is misleading on this image; wrapper recorded /proc/self/status Cpus_allowed and Cpus_allowed_list instead.']}
(REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('result_runs',)},indent=2))
for r in results: print(r['name'],r['latency_ms'],r['effective_mask'],r['parity']['pass'],r['parity']['failed_heads'],r['issues'])
