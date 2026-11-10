"""Run the unchanged upstream checker on full and predeclared warmup-excluded views."""
import argparse,os,sys,shutil,subprocess,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--teacher',type=Path,required=True);ap.add_argument('--student',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--jetlink',type=Path,required=True);ap.add_argument('--frames',type=int,default=320);ap.add_argument('--warmup',type=int,default=160);a=ap.parse_args()
if not 0<=a.warmup<a.frames:raise ValueError('invalid warmup')
if a.out.exists():raise FileExistsError('refuse to overwrite score evidence')
a.out.mkdir(parents=True);summary={}
for label,start in [('all_frames',0),('after_warmup',a.warmup)]:
 d=a.out/label;d.mkdir();shutil.copyfile(a.teacher/'spec.json',d/'spec.json')
 for j,i in enumerate(range(start,a.frames)):
  for source,prefix in [(a.teacher,'out_ref'),(a.student,'out_link')]:
   f=source/f'{prefix}_{i}.npy';v=np.load(f,allow_pickle=False)
   if v.shape!=(18452,) or not np.isfinite(v).all():raise ValueError(f'invalid complete output: {f}')
   shutil.copyfile(f,d/f'{prefix}_{j}.npy')
 env=dict(os.environ,PYTHONPATH=str(a.jetlink.resolve()))
 p=subprocess.run([sys.executable,str(a.jetlink/'scripts/verify_parity.py'),'compare','--dir',str(d)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (a.out/(label+'.log')).write_text(p.stdout);summary[label]={'first_original_frame':start,'frames':a.frames-start,'checker_exit':p.returncode,'last_line':p.stdout.strip().splitlines()[-1]}
 print(label,p.returncode,summary[label]['last_line'],flush=True)
(a.out/'summary.json').write_text(json.dumps({'status':'RECORDED_IMAGE_NUMERICAL_CHECK_NOT_DRIVING_SAFETY_OR_TARGET_SPEED','checker_modified':False,'views':summary},indent=2))
