from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'C:\Users\sunbo\Documents\openpilot')
STRESS = ROOT / 'outputs/jetlink-gelu-optimization-20261101/stress/gelu-best-stress-prefix160'
BASELINE = ROOT / 'outputs/jetlink-gelu-history-20261101/gelu-history-160-out'
PARITY = ROOT / 'outputs/jetlink-gelu-history-20261101/parity-160'
UPSTREAM = Path(r'C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research\scripts\verify_parity.py')
REPORT = ROOT / 'reports/jetlink-gelu-optimization-20261101/stress/prefix160-verification'
EXPECTED_N = 160
EXPECTED_BYTES = 73808
EXPECTED_FLOATS = 18452


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
      h.update(chunk)
  return h.hexdigest()


def numbered_outputs(directory: Path, pattern: str):
  found = {}
  for path in directory.glob(pattern):
    m = re.fullmatch(r'out_(\d+)\.(?:bin|npy)', path.name)
    if m:
      found[int(m.group(1))] = path
  return found


def main() -> int:
  REPORT.mkdir(parents=True, exist_ok=True)
  compare_dir = REPORT / 'compare'
  compare_dir.mkdir(parents=True, exist_ok=True)
  errors = []
  stress_files = numbered_outputs(STRESS, 'out_*.bin') if STRESS.is_dir() else {}
  baseline_files = numbered_outputs(BASELINE, 'out_*.bin') if BASELINE.is_dir() else {}
  refs = numbered_outputs(PARITY, 'out_ref_*.npy') if PARITY.is_dir() else {}
  # The helper above recognizes out_N.npy only; construct the reference map explicitly.
  refs = {int(m.group(1)): p for p in PARITY.glob('out_ref_*.npy')
          if (m := re.fullmatch(r'out_ref_(\d+)\.npy', p.name))}
  if len(stress_files) != EXPECTED_N:
    errors.append(f'stress_output_count={len(stress_files)} expected={EXPECTED_N}')
  if len(baseline_files) < EXPECTED_N:
    errors.append(f'baseline_output_count={len(baseline_files)} expected_at_least={EXPECTED_N}')
  if len(refs) < EXPECTED_N:
    errors.append(f'reference_count={len(refs)} expected_at_least={EXPECTED_N}')
  if not (PARITY / 'spec.json').is_file():
    errors.append('missing spec.json in original parity-160 configuration')
  if not UPSTREAM.is_file():
    errors.append(f'missing upstream verifier: {UPSTREAM}')

  rows=[]
  size_ok=True; finite_ok=True; same_as_baseline=True; reference_inputs_ok=True
  shutil.copy2(PARITY / 'spec.json', compare_dir / 'spec.json') if (PARITY / 'spec.json').is_file() else None
  for i in range(EXPECTED_N):
    stress = stress_files.get(i); base = baseline_files.get(i); ref_path = refs.get(i)
    row={'frame':i,'stress_bytes':None,'stress_float_count':None,'nonfinite':None,
         'baseline_bytes':None,'byte_identical_to_accepted_baseline':False,
         'reference_present':bool(ref_path),'stress_sha256':None,'baseline_sha256':None}
    if stress is None:
      errors.append(f'missing stress frame {i}')
    else:
      row['stress_bytes']=stress.stat().st_size
      row['stress_sha256']=sha256(stress)
      if stress.stat().st_size != EXPECTED_BYTES:
        size_ok=False; errors.append(f'frame {i}: stress size {stress.stat().st_size}, expected {EXPECTED_BYTES}')
      arr=np.fromfile(stress,dtype='<f4')
      row['stress_float_count']=int(arr.size)
      row['nonfinite']=int((~np.isfinite(arr)).sum()) if arr.size else 0
      if arr.size != EXPECTED_FLOATS:
        size_ok=False; errors.append(f'frame {i}: decoded float count {arr.size}, expected {EXPECTED_FLOATS}')
      if row['nonfinite']:
        finite_ok=False; errors.append(f'frame {i}: {row["nonfinite"]} nonfinite values')
    if base is None:
      same_as_baseline=False; errors.append(f'missing accepted baseline frame {i}')
    else:
      row['baseline_bytes']=base.stat().st_size
      row['baseline_sha256']=sha256(base)
      if stress is not None:
        row['byte_identical_to_accepted_baseline'] = stress.read_bytes() == base.read_bytes()
        if not row['byte_identical_to_accepted_baseline']:
          same_as_baseline=False; errors.append(f'frame {i}: stress raw differs from accepted baseline')
    if ref_path is None:
      reference_inputs_ok=False; errors.append(f'missing original reference frame {i}')
    elif stress is not None:
      ref=np.load(ref_path,allow_pickle=False)
      if ref.size != EXPECTED_FLOATS:
        reference_inputs_ok=False; errors.append(f'frame {i}: reference element count {ref.size}')
      else:
        np.save(compare_dir/f'out_link_{i}.npy',np.fromfile(stress,dtype='<f4').reshape(ref.shape))
        shutil.copy2(ref_path,compare_dir/ref_path.name)
    rows.append(row)

  check={'scope':{'frames':'stress prefix frames 0..159 only','stress_directory':str(STRESS),
                  'accepted_raw_baseline':str(BASELINE),'original_reference_and_spec':str(PARITY),
                  'dtype':'little-endian float32','expected_bytes_per_frame':EXPECTED_BYTES,
                  'expected_values_per_frame':EXPECTED_FLOATS},
         'counts':{'stress_files':len(stress_files),'baseline_files':len(baseline_files),'reference_files':len(refs)},
         'checks':{'all_stress_files_expected_size':size_ok and len(stress_files)==EXPECTED_N,
                   'all_stress_frames_finite':finite_ok and len(rows)==EXPECTED_N,
                   'all_stress_raw_equal_accepted_baseline':same_as_baseline and all(x['byte_identical_to_accepted_baseline'] for x in rows),
                   'reference_inputs_complete':reference_inputs_ok and (compare_dir/'spec.json').is_file()},
         'frames':rows,'pre_compare_errors':errors}
  (REPORT/'input-output-validation.json').write_text(json.dumps(check,indent=2),encoding='utf-8')

  parity={'command':None,'exit_code':None,'pass':False,'stdout_log':str(REPORT/'verify-parity.stdout.txt')}
  if UPSTREAM.is_file() and (compare_dir/'spec.json').is_file() and len(list(compare_dir.glob('out_ref_*.npy')))==EXPECTED_N and len(list(compare_dir.glob('out_link_*.npy')))==EXPECTED_N:
    cmd=[sys.executable,str(UPSTREAM),'compare','--dir',str(compare_dir)]
    env=os.environ.copy()
    env['PYTHONPATH']=str(UPSTREAM.parent.parent)+os.pathsep+env.get('PYTHONPATH','')
    parity['command']=' '.join(cmd)
    proc=subprocess.run(cmd,cwd=str(ROOT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (REPORT/'verify-parity.stdout.txt').write_text('$ '+' '.join(cmd)+'\nPYTHONPATH='+env['PYTHONPATH']+'\n\n'+proc.stdout,encoding='utf-8')
    (REPORT/'verify-parity.exitcode.txt').write_text(str(proc.returncode)+'\n',encoding='ascii')
    parity.update({'exit_code':proc.returncode,'pass':proc.returncode==0 and '\nOK: every slice' in proc.stdout})
  else:
    reason='not run: upstream verifier, spec, or complete staged arrays unavailable'
    (REPORT/'verify-parity.stdout.txt').write_text(reason+'\n',encoding='utf-8')
    (REPORT/'verify-parity.exitcode.txt').write_text('NOT_RUN\n',encoding='ascii')

  all_checks=all(check['checks'].values()) and parity['pass']
  summary={'status':'PASS' if all_checks else 'FAIL','pass_requires_all_file_checks_baseline_identity_and_original_parity':True,
           'scope':'Only stress test frames 0-159, compared to the accepted 160-frame raw baseline and original parity-160 references/spec. No model, phone, threshold, or source changed.',
           'checks':check['checks'],'parity':parity,'pre_compare_errors':errors,
           'frame_count':len(rows),'stress_output_hashes':[x['stress_sha256'] for x in rows],
           'baseline_hashes':[x['baseline_sha256'] for x in rows]}
  (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
  print(json.dumps({k:v for k,v in summary.items() if k not in ('stress_output_hashes','baseline_hashes')},indent=2))
  return 0 if all_checks else 1

if __name__=='__main__':
  raise SystemExit(main())

