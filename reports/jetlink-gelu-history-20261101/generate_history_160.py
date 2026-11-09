#!/usr/bin/env python3
"""Generate the upstream deterministic 160-frame history and phone raw inputs."""
import hashlib, json, os, runpy, sys
from pathlib import Path
import numpy as np

ROOT=Path(r"C:\Users\sunbo\Documents\openpilot")
UPSTREAM=Path(r"C:\Users\sunbo\AppData\Local\Temp\codex-jetlink-research")
MODEL=Path(r"C:\Users\sunbo\Documents\moto-x40-root\light-speed.onnx")
BASE=ROOT/"outputs/jetlink-gelu-history-20261101/parity-160"
SPEC_SOURCE=Path(r"C:\Users\sunbo\Documents\moto-x40-root\parity-varied\spec.json")
N=160; SEED=1
os.environ["PYTHONPATH"]=str(UPSTREAM)+os.pathsep+os.environ.get("PYTHONPATH","")
sys.path.insert(0,str(UPSTREAM))
from jetlink.spec import ModelSpec
verify=runpy.run_path(str(UPSTREAM/"scripts/verify_parity.py"))

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(4<<20),b""): h.update(b)
 return h.hexdigest()

def main():
 if BASE.exists() and any(BASE.iterdir()): raise SystemExit(f"refusing nonempty output {BASE}")
 BASE.mkdir(parents=True,exist_ok=True)
 spec=ModelSpec.load(SPEC_SOURCE)
 if MODEL.stat().st_size!=spec.nbytes or sha(MODEL)!=spec.sha256: raise SystemExit("source spec does not match original light-speed.onnx")
 (BASE/"spec.json").write_text(json.dumps(spec.to_dict(),indent=2)+"\n",encoding="utf-8")
 phone=BASE/"jetlink-history-160"; phone.mkdir()
 entries=[]
 for i,(warped,packed) in enumerate(verify["make_inputs"](spec,N,seed=SEED)):
  wp=BASE/f"in_warped_{i}.npy"; pp=BASE/f"in_packed_{i}.npy"
  np.save(wp,warped); np.save(pp,packed)
  by={name:(packed[at].reshape(shape)) for name,(at,shape) in spec.packed_layout.items()}
  image_path=phone/f"{i}_new_img.bin"
  np.asarray(warped,dtype="<f2").tofile(image_path)
  scalars={}
  for name in ("desire","traffic_convention","action_t"):
   path=phone/f"{i}_{name}.bin"; np.asarray(by[name],dtype="<f4").tofile(path)
   scalars[name]={"path":str(path.relative_to(BASE)),"bytes":path.stat().st_size,"sha256":sha(path)}
  entries.append({"frame":i,"in_warped_npy":{"path":wp.name,"bytes":wp.stat().st_size,"sha256":sha(wp)},"in_packed_npy":{"path":pp.name,"bytes":pp.stat().st_size,"sha256":sha(pp)},"phone_inputs":{"new_img":{"path":str(image_path.relative_to(BASE)),"dtype":"little-endian float16","shape":list(warped.shape),"bytes":image_path.stat().st_size,"sha256":sha(image_path)},**scalars}})
 manifest={"source_model":str(MODEL),"source_model_sha256":spec.sha256,"source_model_bytes":spec.nbytes,"spec_source":str(SPEC_SOURCE),"n":N,"seed":SEED,"generator":"upstream scripts/verify_parity.py make_inputs(spec,160,seed=1)","frame_skip":spec.frame_skip,"phone_input_note":"Per-frame new_img exported as little-endian FP16 as requested; non-state desire, traffic_convention and action_t exported as little-endian FP32. No state inputs are exported.","hashes":entries}
 (ROOT/"reports/jetlink-gelu-history-20261101/input-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"frames":N,"seed":SEED,"output":str(BASE),"phone_input_dir":str(phone),"source_model_sha256":spec.sha256,"phone_raw_file_count":len(entries)*4,"array_and_raw_file_count":len(entries)*6,"hash_manifest":str(ROOT/"reports/jetlink-gelu-history-20261101/input-manifest.json")}))
if __name__=="__main__": main()
