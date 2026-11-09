#!/usr/bin/env python3
"""Run CPU FP32 policy on replayed whole-model hidden_state slices."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(r"C:\Users\sunbo\Documents\openpilot")
PHONE_RAW = ROOT / "outputs/jetlink-gelu-repair-20261101/full-replay-raw/light-speed-exact-gelu-replay"
INPUTS = Path(r"C:\Users\sunbo\Documents\moto-x40-root\parity-varied")
POLICY = Path(r"C:\Users\sunbo\Documents\moto-x40-root\policy-fp32.onnx")
OUTDIR = ROOT / "reports/jetlink-gelu-repair-20261101/full-replay-policy-fp32"
NFRAMES = 16
VIEWSLICE = slice(2066, 18450)
VIEW_SHAPE = (1, 32, 512)

def main() -> None:
    if OUTDIR.exists():
        raise SystemExit(f"refusing to overwrite {OUTDIR}")
    OUTDIR.mkdir(parents=True)
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    names = {x.name for x in session.get_inputs()}
    required = {"view_39", "desire", "traffic_convention", "action_t", "state_desire_q", "state_feat_q"}
    if names != required:
        raise SystemExit(f"unexpected policy inputs: {sorted(names)}")
    state = {
        "state_desire_q": np.zeros((132, 1, 8), np.float32),
        "state_feat_q": np.zeros((128, 1, 16384), np.float32),
    }
    outputs = []
    checks = []
    for i in range(NFRAMES):
        raw = PHONE_RAW / f"out_{i}.bin"
        if not raw.is_file() or raw.stat().st_size != 18452 * 4:
            raise SystemExit(f"missing/incorrect phone frame {i}: {raw}")
        flat = np.fromfile(raw, dtype=np.float32)
        if flat.size != 18452 or not np.isfinite(flat).all():
            raise SystemExit(f"invalid phone float32 output {i}")
        view = np.ascontiguousarray(flat[VIEWSLICE].reshape(VIEW_SHAPE), dtype=np.float32)
        packed = np.load(INPUTS / f"in_packed_{i}.npy").reshape(-1).astype(np.float32)
        if packed.size != 12:
            raise SystemExit(f"unexpected packed input length frame {i}: {packed.size}")
        feed = {
            "view_39": view,
            "desire": packed[0:8].copy(),
            "traffic_convention": packed[8:10].reshape(1, 2).copy(),
            "action_t": packed[10:12].reshape(1, 2).copy(),
            **state,
        }
        result = session.run(None, feed)
        result_names = [o.name for o in session.get_outputs()]
        by_name = dict(zip(result_names, result, strict=True))
        out = np.asarray(by_name["outputs"], dtype=np.float32).reshape(-1)
        if out.size != 18452 or not np.isfinite(out).all():
            raise SystemExit(f"invalid CPU policy output frame {i}")
        np.save(OUTDIR / f"out_link_{i}.npy", out)
        outputs.append(out)
        state = {"state_desire_q": by_name["next_state_desire_q"], "state_feat_q": by_name["next_state_feat_q"]}
        checks.append({"frame": i, "raw_bytes": raw.stat().st_size, "hidden_state_shape": list(view.shape), "packed_values": packed.size, "all_finite": True})
    refs = INPUTS
    for i in range(NFRAMES):
        shutil.copy2(refs / f"out_ref_{i}.npy", OUTDIR / f"out_ref_{i}.npy")
    shutil.copy2(refs / "spec.json", OUTDIR / "spec.json")
    (OUTDIR / "input_validation.json").write_text(json.dumps({"frames": checks, "state_initialization": "zero; recursively updated only from this CPU policy", "phone_hidden_state_slice": [2066, 18450], "phone_hidden_state_reshape": list(VIEW_SHAPE)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(outputs), "outdir": str(OUTDIR), "hidden_state_shape": list(VIEW_SHAPE), "all_finite": True}))

if __name__ == "__main__":
    main()
