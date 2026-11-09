#!/usr/bin/env python3
"""Validate pulled shared allocator A/B runs and prepare unchanged parity inputs."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np

ROOT=Path(r"C:\Users\sunbo\Documents\openpilot")
OUT=ROOT/"outputs/jetlink-gelu-optimization-20261101/ab"
REP=ROOT/"reports/jetlink-gelu-optimization-20261101/ab"
HISTORY=ROOT/"outputs/jetlink-gelu-history-20261101/parity-160"
PAR=REP/"verify-parity"
RUNS={"smoke16-shared1":16,"a0-shared0":160,"b1-shared1":160,"a2-shared0":160,"a2r-shared0":160}

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def read_temp(p):
 try:return [int(x) for x in p.read_text().split()]
 except:return None

def main():
 report={"runs":{},"raw_pairwise":{},"parity_dirs":{}}
 for run,n in RUNS.items():
  d=OUT/run/f"{run}-out"; rows=[]; raw=[]; hashes=[]
  files=sorted(d.glob("out_*.bin"),key=lambda p:int(p.stem.split("_")[1]))
  if len(files)!=n:raise SystemExit(f"{run}: expected {n} output files, got {len(files)}")
  for i,p in enumerate(files):
   if p.name!=f"out_{i}.bin" or p.stat().st_size!=73808:raise SystemExit(f"{run}: bad/missing frame {i}: {p}")
   a=np.fromfile(p,dtype=np.float32)
   if a.size!=18452 or not np.isfinite(a).all():raise SystemExit(f"{run}: invalid float32 frame {i}")
   raw.append(a); hashes.append({"frame":i,"sha256":sha(p)})
  csvpath=REP/f"{run}.csv"
  with csvpath.open(newline="") as f: rows=list(csv.DictReader(f))
  if len(rows)!=n or any(int(r["frame"])!=i for i,r in enumerate(rows)):raise SystemExit(f"{run}: malformed CSV rows")
  ms=[float(r["execute_ms"]) for r in rows]; sm=sorted(ms); p50=sm[(n-1)//2]; p99=sm[(n-1)*99//100]
  if any(int(r["nonfinite"])!=0 for r in rows):raise SystemExit(f"{run}: nonfinite CSV count")
  temps={k:read_temp(REP/f"{run}.{k}.txt") for k in ("temp-before","temp-after") if (REP/f"{run}.{k}.txt").exists()}
  # The A2 retry's exact adjacent root read is pulled from the device, all other runs used local saved reads.
  if run=="a2r-shared0":
   temps={"temp-before":read_temp(REP/"a2r-shared0.temp-before.txt"),"temp-after":read_temp(REP/"a2r-shared0.temp-after.txt")}
  report["runs"][run]={"n":n,"shared":1 if "shared1" in run else 0,"output_bytes_per_frame":73808,"output_type":1,"all_finite":True,"frame_output_hashes":hashes,"csv_sha256":sha(csvpath),"csv_rows":n,"first_frame_csv_temperature":{"battery_deci_c":int(rows[0]["battery_deci_c"]),"gpu_milli_c":int(rows[0]["gpu_milli_c"]),"gpu_clock_hz":int(rows[0]["gpu_clock_hz"])},"latency_ms":{"mean":sum(ms)/n,"p50":p50,"p99":p99,"min":min(ms),"max":max(ms)},"nonfinite_total":sum(int(r["nonfinite"]) for r in rows),"external_gpu_temperature_read_milli_c":temps,"stdout_sha256":sha(REP/f"{run}.stdout.txt"),"exit_code":int((REP/f"{run}.exitcode.txt").read_text())}
  # Create parity input directories for each full run; frame arrays are exactly the downloaded raw bytes interpreted as float32.
  if n==160:
   pdir=PAR/run
   if pdir.exists():raise SystemExit(f"refusing overwrite {pdir}")
   pdir.mkdir(parents=True)
   shutil.copy2(HISTORY/"spec.json",pdir/"spec.json")
   for i,a in enumerate(raw):
    shutil.copy2(HISTORY/f"out_ref_{i}.npy",pdir/f"out_ref_{i}.npy")
    np.save(pdir/f"out_link_{i}.npy",a)
   report["parity_dirs"][run]=str(pdir)
 # Byte comparisons among every complete run, including the temperature-mismatched diagnostic A2.
 full=[x for x,n in RUNS.items() if n==160]
 for i,a in enumerate(full):
  for b in full[i+1:]:
   diffs=[]
   for frame in range(160):
    pa=OUT/a/f"{a}-out"/f"out_{frame}.bin"; pb=OUT/b/f"{b}-out"/f"out_{frame}.bin"
    if pa.read_bytes()!=pb.read_bytes():diffs.append(frame)
   report["raw_pairwise"][f"{a}__vs__{b}"]={"identical_frames":160-len(diffs),"different_frames":diffs,"byte_identical_all_frames":not diffs}
 manifest={"runner_path":"/data/local/tmp/jetlink-gelu-repair-20261101/history_probe","runner_sha256":"377147782e0c0c7c923f18dee72d337f6d8a02af402d2f4a1e363d7917b6fbb3","model_path":"/data/local/tmp/light-speed-gelu-both-ctx.onnx","model_sha256":"2c8b28e972af55ca3592c386b049674f9da946fce2fd6a82b0e5877c98d6d68f","custom_op_library":"/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so","custom_op_library_sha256":"3f682906d26ff829cdf8ecda2f32d9bcb05f4450f1c9afc211fcb0991082f700","input_directory":"/data/local/tmp/jetlink-history-160","input_manifest_path":str(ROOT/"reports/jetlink-gelu-history-20261101/input-manifest.json"),"input_manifest_sha256":sha(ROOT/"reports/jetlink-gelu-history-20261101/input-manifest.json"),"inference_parameters_except_shared":"rpc_us=0; profiling=none; runs=160; same context, inputs, output dtype, custom library, environment and frequency/thermal settings","runs":report["runs"],"raw_pairwise":report["raw_pairwise"],"parity_dirs":report["parity_dirs"],"csv_gpu_temp_limitation":"history_probe runs as shell; KGSL temp/clock are unreadable to shell, so CSV gpu_milli_c/gpu_clock_hz are -1. Battery temp in first CSV row is readable and consistent. Separate root-only pre/post reads are recorded where available."}
 (REP/"hash-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"runs":{k:{"latency_ms":v["latency_ms"],"gpu_csv_first":v["first_frame_csv_temperature"],"root_gpu_temp":v["external_gpu_temperature_read_milli_c"],"exit":v["exit_code"]} for k,v in report["runs"].items()},"raw_pairwise":report["raw_pairwise"]},indent=2))
if __name__=="__main__":main()
