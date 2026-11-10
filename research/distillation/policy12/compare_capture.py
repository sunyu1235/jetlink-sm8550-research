import argparse,json,shutil,os,subprocess,sys
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('split',choices=['train','validation']);a=ap.parse_args();root=Path(__file__).resolve().parent;d=root/('cse-'+a.split+'160');d.mkdir(exist_ok=False)
shutil.copyfile('/workspace/shared/jetlink-history-regenerated/spec.json',d/'spec.json');ref=np.load('/workspace/shared/jetlink-long-calibration/'+a.split+'-teacher-outputs.npy');same=0;mx=0
for i in range(160):
 r=ref[i].reshape(-1);v=np.load(root/f'{a.split}_out_{i:03d}.npy');assert v.shape==r.shape==(18452,);assert np.isfinite(v).all();same+=int(np.array_equal(v,r));mx=max(mx,float(abs(v-r).max()));np.save(d/f'out_ref_{i}.npy',r);np.save(d/f'out_link_{i}.npy',v)
jetlink=Path('/workspace/scratch/a607d96f1cc1/htp-work/vendor/jetlink');env=dict(os.environ,PYTHONPATH=str(jetlink));r=subprocess.run([sys.executable,str(jetlink/'scripts/verify_parity.py'),'compare','--dir',str(d)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(root/('cse-'+a.split+'160.log')).write_text(r.stdout);summary=dict(split=a.split,frames=160,checker_exit=r.returncode,identical_frames=same,max_abs=mx,last_line=r.stdout.strip().splitlines()[-1]);(d/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
