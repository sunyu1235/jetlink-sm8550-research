#!/usr/bin/env python3
"""Inspect a LiteRT flatbuffer prefix and prepare (never apply) an LN patch plan."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import flatbuffers

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "outputs").is_dir())
PREFIX = ROOT / "outputs" / "jetlink-gelu-optimization-20261101" / "partition-tuning-20261109" / "litert-metadata-prefix.bin"
PLAN = HERE / "gpu-epsilon-sqrt-patch.json"
PATCHES = HERE / "gpu-epsilon-sqrt-patches.bin"
PREFIX_LIMIT = 8 * 1024 * 1024
PREFIX_BYTES = b""
MODEL_SIZE = 793_980_992
FP16_MIN_NORMAL = 2.0 ** -14


def table(buf: bytes, pos: int) -> flatbuffers.table.Table:
  return flatbuffers.table.Table(buf, pos)


def scalar(t, slot: int, fmt: str, default=0):
  off = t.Offset(4 + 2 * slot)
  return default if off == 0 else struct.unpack_from(fmt, t.Bytes, t.Pos + off)[0]


def vec(t, slot: int, fmt: str | None = None):
  off = t.Offset(4 + 2 * slot)
  if not off:
    return [], []
  start = t.Vector(off)
  count = t.VectorLen(off)
  if fmt is None:
    return [t.Indirect(start + 4 * i) for i in range(count)], [start + 4 * i for i in range(count)]
  size = struct.calcsize(fmt)
  return ([struct.unpack_from(fmt, t.Bytes, start + size * i)[0] for i in range(count)],
          [start + size * i for i in range(count)])


def string(t, slot: int) -> str:
  off = t.Offset(4 + 2 * slot)
  return "" if not off else t.String(t.Pos + off).decode("utf-8", "replace")


def vector_tables(buf: bytes, t, slot: int):
  positions, _ = vec(t, slot)
  return [table(buf, p) for p in positions]


def patch(old: bytes, offset: int, new: bytes, label: str):
  if offset < 0 or offset + len(old) > PREFIX_LIMIT:
    raise ValueError(f"{label}: byte range lies beyond captured prefix")
  actual = PREFIX_BYTES[offset:offset + len(old)]
  if actual != old:
    raise ValueError(f"{label}: expected old bytes {old.hex()}, found {actual.hex()}")
  return {"label": label, "byte_offset": offset, "old_hex": old.hex(), "new_hex": new.hex()}


def main():
  global PREFIX_BYTES
  data = PREFIX.read_bytes()
  PREFIX_BYTES = data
  if len(data) != PREFIX_LIMIT:
    raise SystemExit(f"refusing: expected {PREFIX_LIMIT}-byte metadata prefix, got {len(data)}")
  if data[4:8] != b"TFL3":
    raise SystemExit("refusing: missing TFL3 identifier")
  root = table(data, struct.unpack_from("<I", data, 0)[0])
  opcodes = vector_tables(data, root, 1)
  subgraphs = vector_tables(data, root, 2)
  buffers = vector_tables(data, root, 4)
  codes = []
  for code in opcodes:
    # BuiltinCode field 3 is authoritative for schema v3; slot 0 is deprecated.
    codes.append(scalar(code, 3, "<i", scalar(code, 0, "<b", 0)))

  all_plans = []
  chains_seen = 0
  candidates = []
  for sg_i, sg in enumerate(subgraphs):
    tensors = vector_tables(data, sg, 0)
    operators = vector_tables(data, sg, 3)
    names = [string(t, 3) for t in tensors]
    producer = {}
    consumers = {i: [] for i in range(len(tensors))}
    parsed_ops = []
    for oi, op in enumerate(operators):
      opcode_i = scalar(op, 0, "<i")
      code = codes[opcode_i]
      ins, in_offsets = vec(op, 1, "<i")
      outs, out_offsets = vec(op, 2, "<i")
      parsed_ops.append({"index": oi, "code": code, "inputs": ins, "input_offsets": in_offsets,
                         "outputs": outs, "output_offsets": out_offsets})
      for out in outs:
        if out >= 0:
          producer[out] = oi
      for at, inp in enumerate(ins):
        if inp >= 0:
          consumers.setdefault(inp, []).append((oi, at))

    for eps_i, eps_name in enumerate(names):
      if "__eps" not in eps_name:
        continue
      eps_p = producer.get(eps_i)
      if eps_p is None:
        continue
      eps_op = parsed_ops[eps_p]
      # Candidate: scaled epsilon MUL(r2, epsilon), where r2 is MUL(r, r).
      if eps_op["code"] != 18 or len(eps_op["inputs"]) != 2:
        continue
      for r2_at, r2_i in enumerate(eps_op["inputs"]):
        if r2_i < 0 or names[r2_i].find("__r2") < 0:
          continue
        r2_p = producer.get(r2_i)
        if r2_p is None:
          continue
        r2_op = parsed_ops[r2_p]
        if r2_op["code"] != 18 or len(r2_op["inputs"]) != 2 or r2_op["inputs"][0] != r2_op["inputs"][1]:
          continue
        epsilon_i = eps_op["inputs"][1 - r2_at]
        if epsilon_i < 0:
          continue
        chains_seen += 1
        eps_tensor = tensors[epsilon_i]
        eps_buf_i = scalar(eps_tensor, 2, "<i")
        eps_val = None
        scalar_data_offset = None
        reason = None
        eps_shape, _ = vec(eps_tensor, 0, "<i")
        r_tensor_i = r2_op["inputs"][0]
        r_shape, _ = vec(tensors[r_tensor_i], 0, "<i")
        r2_shape, _ = vec(tensors[r2_i], 0, "<i")
        scaled_shape, _ = vec(tensors[eps_i], 0, "<i")
        if scalar(eps_tensor, 1, "<b") != 0:
          reason = "epsilon tensor is not FLOAT32"
        elif eps_shape not in ([], [1]):
          reason = f"epsilon tensor shape is not scalar/[1]: {eps_shape}"
        elif r_shape != r2_shape or scaled_shape != r2_shape:
          reason = f"r/r2/scaled-epsilon shapes differ: {r_shape}, {r2_shape}, {scaled_shape}"
        elif r_shape != vec(tensors[r_tensor_i], 0, "<i")[0]:
          reason = "invalid input tensor shape"
        elif eps_buf_i >= len(buffers):
          reason = "epsilon buffer index out of range"
        else:
          buf_table = buffers[eps_buf_i]
          content, content_offsets = vec(buf_table, 0, "<B")
          external_offset = scalar(buf_table, 1, "<Q")
          external_size = scalar(buf_table, 2, "<Q")
          if external_offset > 1 or external_size:
            reason = "epsilon constant is stored externally; prefix cannot safely patch it"
          elif len(content) != 4 or content_offsets[0] + 4 > len(data):
            reason = "epsilon inline buffer is absent or outside captured prefix"
          else:
            scalar_data_offset = content_offsets[0]
            eps_val = struct.unpack_from("<f", data, scalar_data_offset)[0]
            if not math.isfinite(eps_val) or eps_val <= 0:
              reason = f"epsilon must be finite positive, got {eps_val!r}"
            elif math.sqrt(eps_val) < FP16_MIN_NORMAL:
              reason = f"sqrt(epsilon) {math.sqrt(eps_val)!r} is below FP16 min normal"
        # `r2` must have no consumer except epsilon MUL, and epsilon must be used
        # only by this candidate chain. These constraints make index edits local.
        if consumers.get(r2_i) != [(eps_p, r2_at)]:
          reason = reason or f"r2 tensor consumers differ: {consumers.get(r2_i)}"
        if consumers.get(epsilon_i) != [(eps_p, 1 - r2_at)]:
          reason = reason or f"epsilon tensor consumers differ: {consumers.get(epsilon_i)}"
        candidates.append({"subgraph": sg_i, "epsilon_tensor": epsilon_i, "epsilon_name": names[epsilon_i],
                           "epsilon_value": eps_val, "r2_tensor": r2_i, "r2_name": names[r2_i],
                           "eps_output": eps_i, "eps_name": eps_name, "r2_operator": r2_p,
                           "eps_operator": eps_p, "r2_input_offsets": r2_op["input_offsets"],
                           "eps_input_offsets": eps_op["input_offsets"], "epsilon_data_offset": scalar_data_offset,
                           "reason": reason})

  for item in candidates:
    if item["reason"]:
      continue
    eps_idx = item["epsilon_tensor"]
    r2_at = next(i for i, off in enumerate(item["eps_input_offsets"])
                 if struct.unpack_from("<i", data, off)[0] == item["r2_tensor"])
    # r2 := r * epsilon; eps_scaled := r2 * r2; epsilon := sqrt(epsilon).
    r2_input = item["r2_input_offsets"][1]
    eps_inputs = item["eps_input_offsets"]
    edits = [
      patch(struct.pack("<i", struct.unpack_from("<i", data, r2_input)[0]), r2_input,
            struct.pack("<i", eps_idx), f"sg{item['subgraph']}:{item['r2_name']} MUL second input -> {item['epsilon_name']}"),
      patch(struct.pack("<i", struct.unpack_from("<i", data, eps_inputs[1-r2_at])[0]), eps_inputs[1-r2_at],
            struct.pack("<i", item["r2_tensor"]), f"sg{item['subgraph']}:{item['eps_name']} MUL other input -> {item['r2_name']}"),
      patch(struct.pack("<f", item["epsilon_value"]), item["epsilon_data_offset"],
            struct.pack("<f", math.sqrt(item["epsilon_value"])), f"sg{item['subgraph']}:{item['epsilon_name']} constant -> sqrt(epsilon)"),
    ]
    all_plans.append({"subgraph": item["subgraph"], "epsilon_tensor": item["epsilon_name"],
                      "r2_tensor": item["r2_name"], "scaled_epsilon_tensor": item["eps_name"],
                      "r2_operator": item["r2_operator"], "epsilon_operator": item["eps_operator"],
      "epsilon_original": item["epsilon_value"], "epsilon_sqrt": math.sqrt(item["epsilon_value"]),
                      "mathematical_identity": "r^2 * epsilon == (r * sqrt(epsilon))^2",
                      "edits": edits})

  # A shared scalar constant may be referenced by multiple LayerNorms. Emit
  # one patch record per unique byte range, requiring identical old/new bytes.
  unique_edits = {}
  for chain in all_plans:
    for edit in chain["edits"]:
      key = edit["byte_offset"]
      prior = unique_edits.get(key)
      if prior and (prior["old_hex"], prior["new_hex"]) != (edit["old_hex"], edit["new_hex"]):
        raise SystemExit(f"conflicting edits at byte offset {key}")
      unique_edits[key] = edit
  records = sorted(unique_edits.values(), key=lambda x: x["byte_offset"])
  patch_blob = bytearray(b"JLP1" + struct.pack("<IQ", len(records), MODEL_SIZE))
  for edit in records:
    patch_blob += struct.pack("<Q4s4s", edit["byte_offset"], bytes.fromhex(edit["old_hex"]), bytes.fromhex(edit["new_hex"]))
  PATCHES.write_bytes(patch_blob)

  result = {"source_prefix": str(PREFIX), "prefix_bytes": len(data), "model_size": MODEL_SIZE,
            "subgraphs": len(subgraphs),
            "candidate_chains_found": chains_seen, "safe_chains": len(all_plans),
            "patch_records_unique": len(records), "patch_binary": str(PATCHES),
            "rejected_candidates": [{k: v for k, v in c.items() if k not in {"r2_input_offsets", "eps_input_offsets"}}
                                     for c in candidates if c["reason"]],
            "plans": all_plans,
            "status": "ready" if chains_seen == 103 and len(all_plans) == 103 else "refused",
            "caveat": "Algebraic identity is exact over reals; LiteRT GPU compiler reassociation, runtime behavior, numerical error, and overflow still require validation."}
  PLAN.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({"candidate_chains_found": chains_seen, "safe_chains": len(all_plans),
                    "rejected": len(result["rejected_candidates"]), "plan": str(PLAN)}, indent=2))
  if chains_seen != 103 or len(all_plans) != 103 or result["rejected_candidates"]:
    PATCHES.unlink(missing_ok=True)
    raise SystemExit("refused: not all 103 candidate chains passed validation; no binary patch emitted")


if __name__ == "__main__":
  main()
