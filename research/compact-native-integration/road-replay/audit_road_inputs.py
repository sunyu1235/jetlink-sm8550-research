"""Read-only provenance audit for the resumed native road replay."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).parent
B=Path('/workspace/shared')
R=Path('/workspace/scratch/a607d96f1cc1/htp-work')
report={'scope':'Offline host functional replay provenance, not phone runtime or driving safety','roads':{},'checker':{}}
checker=R/'vendor/jetlink/scripts/verify_parity.py'
report['checker']={'path':str(checker),'sha256':hashlib.sha256(checker.read_bytes()).hexdigest()}
for road,base,teacher in [('night',B/'jetlink-public-road',B/'jetlink-distillation/real-block/road320-teacher-capture'),('day',B/'jetlink-day-road',B/'jetlink-distillation/policy12/day320-teacher')]:
 manifest=json.loads((base/'frozen_manifest.json').read_text());inputs=base/'inputs320'
 for name,v in manifest['files'].items():
  p=inputs/name
  assert p.stat().st_size==v['bytes'],p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],p
 assert (teacher/'spec.json').read_bytes()==(inputs/'spec.json').read_bytes()
 run=json.loads((teacher/'out_ref-run.json').read_text())
 assert run['frames']==320 and run['reset_at_first_frame'] and run['independent_recurrence']
 assert run['inputs']==str(inputs)
 h=hashlib.sha256()
 for i in range(320):
  p=teacher/f'out_ref_{i}.npy';v=np.load(p,allow_pickle=False)
  assert v.shape==(18452,) and np.isfinite(v).all(),p
  h.update(p.read_bytes())
 report['roads'][road]={'frozen_input_files_verified':len(manifest['files']),'input_manifest_sha256':hashlib.sha256((base/'frozen_manifest.json').read_bytes()).hexdigest(),'teacher_frames_validated':320,'teacher_outputs_ordered_sha256':h.hexdigest(),'input_teacher_specs_byte_identical':True,'reference_model':run['model'],'independent_teacher_recurrence':True}
(P/'road-recovery-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
