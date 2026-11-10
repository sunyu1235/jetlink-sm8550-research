"""Small, reversible two-partition compile and phone speed screen."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/jetlink-dot-compact-test-20261110"
PHONE = "/data/local/tmp/jetlink-dot-compact-20261110"
ADB = ["C:/mp-sdk/platform-tools/adb.exe", "-s", "ZY22GVVTVZ"]
RUNNER = "/data/local/tmp/jetlink-shared-boundaries-20261109/last-query-full160/partition_probe"
LIB = "/data/app/~~VpmS9jlwALEhgoyt9-46dg==/io.zoompilot.jetlink.android-EWl-c1N0tgoTnf5ottkdxQ==/lib/arm64"
ENV = ("export LD_LIBRARY_PATH=" + shlex.quote("/data/local/tmp/qnn-probe:" + LIB + ":/vendor/lib64") + "; "
       "export ADSP_LIBRARY_PATH=" + shlex.quote(LIB) + "; "
       "unset JL_CONTEXT_OUTPUT JL_SOC_MODEL JL_VTCM_MB JL_ORT_PROFILE JL_DUMP_TENSORS "
       "JL_DUMP_FRAME_LIMIT JL_INPUT_MODULO JL_HTP_OPT_MODE JL_CUSTOM_OP_LIBRARY;")


def adb(*args, timeout=60):
    r = subprocess.run(ADB + list(args), capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stdout.decode("utf-8", "replace")[-3000:] + r.stderr.decode("utf-8", "replace")[-3000:])
    return r.stdout


def native(command, label):
    script = ENV + command
    cmd = "taskset f8 /system/bin/sh -c " + shlex.quote(script)
    r = subprocess.run(ADB + ["shell", cmd], capture_output=True, timeout=160)
    log = r.stdout + r.stderr
    (REPORT / (label + ".log")).write_bytes(log)
    print(json.dumps({"stage": label, "exit": r.returncode,
                      "log": str(REPORT / (label + ".log"))}), flush=True)
    print(log.decode("utf-8", "replace")[-2000:], flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)


def compile_parts():
    adb("shell", "mkdir -p " + PHONE)
    for label in ("upstream", "downstream"):
        adb("push", str(OUT / (label + ".onnx")), PHONE + "/" + label + ".onnx")
        native("export JL_SOC_MODEL=43; export JL_CONTEXT_OUTPUT=" + PHONE + "/" + label + "-ctx0.onnx; "
               + RUNNER + " " + PHONE + "/" + label + ".onnx 0 0 none 1", label + "-compile0")
        adb("pull", PHONE + "/" + label + "-ctx0.onnx", str(OUT / (label + "-ctx.onnx")))


def run(frames, label):
    assert frames in (16, 160)
    directory = PHONE + "/" + label
    adb("shell", "test ! -e " + directory + " && mkdir " + directory)
    if label.startswith("baseline"):
        model = "/data/local/tmp/last-query-full160.onnx"
    else:
        model = "/data/local/tmp/dot-compact-full-20261110.onnx"
        adb("push", str(OUT / "compact-whole.onnx"), model)
    before = adb("shell", "su -c 'cat /sys/class/kgsl/kgsl-3d0/temp'")
    (REPORT / (label + "-gpu-before.txt")).write_bytes(before)
    native("export JL_CUSTOM_OP_LIBRARY=/data/local/tmp/jetlink-gelu-repair-20261101/libspecifiedgelu.so; "
           "export JL_FRAME_CSV=" + directory + "/timing.csv; " + RUNNER + " " + model + " 1 0 none "
           + str(frames) + " /data/local/tmp/jetlink-history-160 " + directory + " varied", label)
    after = adb("shell", "su -c 'cat /sys/class/kgsl/kgsl-3d0/temp'")
    (REPORT / (label + "-gpu-after.txt")).write_bytes(after)
    adb("pull", directory, str(OUT / label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compile", "baseline16", "candidate16", "baseline160", "candidate160"))
    args = parser.parse_args()
    compile_parts() if args.mode == "compile" else run(160 if args.mode.endswith("160") else 16, args.mode)
