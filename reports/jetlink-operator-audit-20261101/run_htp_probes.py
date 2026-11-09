#!/usr/bin/env python3
"""Run strict one-node HTP FP16 probes on the already connected phone."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ARTIFACTS = WORKSPACE / "outputs" / "jetlink-operator-audit-20261101"
REPORTS = Path(__file__).resolve().parent
ADB = Path(r"C:\adb\adb.exe")
DEVICE = "192.168.1.49:5555"
REMOTE_ROOT = "/data/local/tmp/jetlink-operator-audit-20261101"
STRICT_PROBE = "/data/local/tmp/strict_probe"
APP_LIB = "/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64"
OPS = ["depthwise_conv", "layer_normalization", "matmul_1", "matmul_2", "gelu"]


def run(args, check=False):
    p = subprocess.run([str(ADB), *args], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if check and p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {args}\n{p.stdout}")
    return p.returncode, p.stdout


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not ADB.is_file():
        raise SystemExit(f"adb not found: {ADB}")
    if any((ARTIFACTS / name / "htp-output.bin").exists() for name in OPS):
        raise SystemExit("HTP output exists; refusing overwrite")

    code, devices = run(["devices", "-l"], check=True)
    if DEVICE not in devices or f"{DEVICE}      device " not in devices:
        raise SystemExit(f"expected existing ADB device {DEVICE}; got:\n{devices}")
    code, strict_hash = run(["shell", f"sha256sum {STRICT_PROBE}"], check=True)
    expected_probe_hash = sha(WORKSPACE / "outputs" / "jetlink-block-case-20261101" / "strict_probe")
    if expected_probe_hash not in strict_hash.lower():
        raise SystemExit(f"device strict_probe hash mismatch; expected {expected_probe_hash}, got {strict_hash}")
    code, free = run(["shell", f"test ! -e {REMOTE_ROOT} && echo free"], check=True)
    if "free" not in free:
        raise SystemExit(f"remote diagnostic directory already exists: {REMOTE_ROOT}")

    run(["push", str(ARTIFACTS), "/data/local/tmp/"], check=True)
    checks = []
    for op in OPS:
        oprec = manifest["operators"][op]
        for inp in oprec["dynamic_inputs"]:
            remote_input = f"{REMOTE_ROOT}/{op}/inputs/0_{inp['name']}.bin"
            command = f"test -f {remote_input} && test $(stat -c %s {remote_input}) = {inp['bytes']} && echo ok"
            check_code, check_out = run(["shell", command], check=True)
            if "ok" not in check_out:
                raise SystemExit(f"remote input preflight failed for {op}/{inp['name']}: {check_out}")
            checks.append({"operator": op, "input": inp["name"], "expected_bytes": inp["bytes"],
                           "device_check": "exists and exact size"})
    run(["shell", f"mkdir -p " + " ".join(f"{REMOTE_ROOT}/{op}/htp-out" for op in OPS)], check=True)

    results = []
    for op in OPS:
        model = f"{REMOTE_ROOT}/{op}/operator.onnx"
        inputs = f"{REMOTE_ROOT}/{op}/inputs"
        outdir = f"{REMOTE_ROOT}/{op}/htp-out"
        shell = (f"export LD_LIBRARY_PATH=/data/local/tmp/qnn-probe:{APP_LIB}:/vendor/lib64; "
                 f"export ADSP_LIBRARY_PATH={APP_LIB}; "
                 f"{STRICT_PROBE} {model} 0 0 none 1 {inputs} {outdir} oneop htp")
        cmd = ["shell", shell]
        rc, output = run(cmd)
        log = REPORTS / f"{op}.htp16.log"
        log.write_text("$ adb " + " ".join(cmd) + f"\nexit_code={rc}\n\n" + output, encoding="utf-8")
        rec = {"operator": op, "command": shell, "exit_code": rc,
               "log": str(log), "device_output": output}
        if rc == 0:
            out_remote = f"{outdir}/out_0.bin"
            expected = (ARTIFACTS / op / "ref.bin").stat().st_size
            check_command = f"test -f {out_remote} && test $(stat -c %s {out_remote}) = {expected} && echo ok"
            check_rc, check_out = run(["shell", check_command], check=True)
            if "ok" not in check_out:
                raise RuntimeError(f"wrong HTP output size for {op}: {check_out}")
            local_out = ARTIFACTS / op / "htp-output.bin"
            pull_rc, pull_out = run(["pull", out_remote, str(local_out)], check=True)
            if local_out.stat().st_size != expected:
                raise RuntimeError(f"pulled output size mismatch for {op}: {local_out.stat().st_size} != {expected}")
            rec.update({"output_bytes": local_out.stat().st_size, "output_sha256": sha(local_out),
                        "output_local": str(local_out), "pull_output": pull_out})
        results.append(rec)
        print(f"{op}: exit={rc}; " + ("output pulled" if rc == 0 else output.strip().splitlines()[-1]))

    result = {"device": DEVICE, "device_model": "XT2301_5 / rtwo", "device_clock": run(["shell", "date -Is"])[1].strip(),
              "strict_probe_sha256": expected_probe_hash,
              "strict_cpu_fallback_disabled": True, "backend": "QNN HTP FP16 (enable_htp_fp16_precision=1)",
              "input_preflight": checks, "runs": results}
    (REPORTS / "htp16-runs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if any(r["exit_code"] != 0 for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
