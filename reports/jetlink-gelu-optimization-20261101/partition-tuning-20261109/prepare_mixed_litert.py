"""Prepare a policy-only metadata patch for the original full LiteRT model.

Reuse original tensor/buffer/operator tables, compact only the operators vector,
and merge the validated square-root-epsilon edits against original old bytes.
No model weights are read or copied and no device is accessed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import struct
from pathlib import Path

import prepare_gpu_epsilon_patch as fb


HERE = Path(__file__).resolve().parent
INPUTS = [1644, 1, 2, 3, 5, 6]
OUTPUTS = [3785, 20, 3792]


def fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def model_parts(data):
    if data[4:8] != b"TFL3":
        raise ValueError("missing TFL3 model identifier")
    root = fb.table(data, struct.unpack_from("<I", data, 0)[0])
    subgraphs = fb.vector_tables(data, root, 2)
    if len(subgraphs) != 1:
        raise ValueError("only the known single-subgraph model is supported")
    return root, subgraphs[0], fb.vector_tables(data, subgraphs[0], 0), fb.vector_tables(data, root, 4)


def parse_ops(data, sg):
    positions, entries = fb.vec(sg, 3)
    ops = []
    producer = {}
    for index, position in enumerate(positions):
        op = fb.table(data, position)
        inputs, input_offsets = fb.vec(op, 1, "<i")
        outputs, _ = fb.vec(op, 2, "<i")
        ops.append({"index": index, "position": position, "inputs": inputs, "input_offsets": input_offsets,
                    "outputs": outputs, "opcode_index": fb.scalar(op, 0, "<I")})
        for tensor in outputs:
            if tensor < 0:
                continue
            if tensor in producer:
                raise ValueError(f"duplicate tensor producer: {tensor}")
            producer[tensor] = index
    return ops, producer, entries


def closure(data, tensors, buffers, ops, producer):
    used, reached, constants, leaves = set(), set(), set(), set()
    pending = list(OUTPUTS)
    while pending:
        tensor = pending.pop()
        if tensor < 0 or tensor in reached:
            continue
        if tensor >= len(tensors):
            raise ValueError("tensor index out of bounds")
        reached.add(tensor)
        if tensor in INPUTS:
            leaves.add(tensor)
            continue
        buffer_index = fb.scalar(tensors[tensor], 2, "<I")
        if buffer_index >= len(buffers):
            raise ValueError("tensor buffer index out of bounds")
        buffer = buffers[buffer_index]
        data_field = buffer.Offset(4)
        inline_bytes = buffer.VectorLen(data_field) if data_field else 0
        external_bytes = fb.scalar(buffer, 2, "<Q")
        if buffer_index and (inline_bytes or external_bytes):
            constants.add(tensor)
            continue
        if tensor not in producer:
            raise ValueError(f"nonconstant, noninput tensor without producer: {tensor}:{fb.string(tensors[tensor], 3)}")
        index = producer[tensor]
        if index not in used:
            used.add(index)
            pending.extend(ops[index]["inputs"])
    if reached.intersection({0, 4}):
        raise ValueError("policy still depends on new_img/state_img_q")
    # Validate forward execution order independently of the backwards walk.
    available = set(INPUTS) | constants
    for index in sorted(used):
        missing = set(ops[index]["inputs"]) - available - {-1}
        if missing:
            raise ValueError(f"operator {index} has unresolved forward inputs: {sorted(missing)}")
        available.update(ops[index]["outputs"])
    if not set(OUTPUTS).issubset(available):
        raise ValueError("policy outputs are not available")
    return sorted(used), sorted(reached), sorted(constants), sorted(leaves)


def main() -> None:
    output_path = HERE / "mixed-litert-policy-patches.bin"
    manifest_path = HERE / "mixed-litert-policy-manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite mixed LiteRT patch evidence")
    original = fb.PREFIX.read_bytes()
    if len(original) != fb.PREFIX_LIMIT:
        raise ValueError("unexpected captured metadata prefix size")
    root, sg, tensors, buffers = model_parts(original)
    old_inputs, input_entries = fb.vec(sg, 1, "<i")
    old_outputs, output_entries = fb.vec(sg, 2, "<i")
    if old_inputs != list(range(7)) or old_outputs != [3785, 3793, 20, 3792]:
        raise ValueError("original full-model public interface differs")
    if (fb.string(tensors[1644], 3) != "view_39" or fb.scalar(tensors[1644], 1, "<b") != 0
            or fb.vec(tensors[1644], 0, "<i")[0] != [1, 32, 512]):
        raise ValueError("known feature seam differs")
    ops, producer, op_entries = parse_ops(original, sg)
    if len(ops) != 2919:
        raise ValueError("unexpected original operator count")
    edits = []

    def edit(offset: int, new: bytes, label: str, group: str) -> None:
        old = original[offset:offset + 4]
        if len(old) != 4 or len(new) != 4 or not 0 <= offset <= len(original) - 4:
            raise ValueError("patch must lie inside the captured prefix and be four bytes")
        if old != new:
            edits.append({"label": label, "group": group, "byte_offset": offset,
                          "old_hex": old.hex(), "new_hex": new.hex()})

    # Conversion folded view_39 out of two reshape-only branches. Redirect
    # those aliases to the same 16384 feature values before finding closure.
    alias_rewrites = []
    for op_index, output_tensor in [(1440, 1646), (2912, 3782)]:
        op = ops[op_index]
        if op["inputs"][0] != 1643 or op["outputs"] != [output_tensor]:
            raise ValueError("known feature reshape alias differs")
        opcode = fb.vector_tables(original, root, 1)[op["opcode_index"]]
        if fb.scalar(opcode, 3, "<i", fb.scalar(opcode, 0, "<b")) != 22:
            raise ValueError("feature alias is not a plain RESHAPE")
        if fb.vec(tensors[output_tensor], 0, "<i")[0] not in ([1, 1, 32, 512], [1, 16384]):
            raise ValueError("feature alias element count differs")
        offset = op["input_offsets"][0]
        edit(offset, struct.pack("<i", 1644), f"operator{op_index} feature reshape input 1643->1644", "feature_alias")
        op["inputs"][0] = 1644
        alias_rewrites.append({"operator": op_index, "output_tensor": output_tensor,
                               "output_name": fb.string(tensors[output_tensor], 3), "old_input": 1643, "new_input": 1644})
    kept, reached, constants, leaves = closure(original, tensors, buffers, ops, producer)
    for slot, names, entries, label in [(1, INPUTS, input_entries, "inputs"), (2, OUTPUTS, output_entries, "outputs")]:
        edit(entries[0] - 4, struct.pack("<I", len(names)), f"subgraph0 {label} vector length", "interface")
        for index, tensor in enumerate(names):
            edit(entries[index], struct.pack("<i", tensor), f"subgraph0 {label}[{index}] tensor={tensor}", "interface")
    edit(op_entries[0] - 4, struct.pack("<I", len(kept)), "subgraph0 operators vector length", "operators_vector")
    minimum_uoffset = None
    for new_index, old_index in enumerate(kept):
        entry_position = op_entries[new_index]
        relative = ops[old_index]["position"] - entry_position
        if not 0 < relative <= 0xffffffff:
            raise ValueError("compacted operator uoffset must remain positive uint32")
        minimum_uoffset = relative if minimum_uoffset is None else min(minimum_uoffset, relative)
        edit(entry_position, struct.pack("<I", relative), f"operators[{new_index}] -> original operator {old_index}", "operators_vector")
    signatures, _ = fb.vec(root, 7)
    if len(signatures) != 1:
        raise ValueError("expected exactly one serving_default signature")
    signature = fb.table(original, signatures[0])
    if fb.string(signature, 2) != "serving_default" or fb.scalar(signature, 3, "<I") != 0:
        raise ValueError("original serving_default signature differs")
    signature_inputs = fb.vector_tables(original, signature, 0)
    signature_outputs = fb.vector_tables(original, signature, 1)
    signature_input_by_index = {fb.scalar(item, 1, "<I"): item for item in signature_inputs}
    signature_output_by_index = {fb.scalar(item, 1, "<I"): item for item in signature_outputs}
    if list(signature_input_by_index) != old_inputs or list(signature_output_by_index) != old_outputs:
        raise ValueError("original signature and subgraph tensor ordering differ")
    # Keep serving_default, required by the existing official LiteRT runner.
    # Reuse the seven-byte new_img TensorMap name allocation as view_39.
    feature_map = signature_input_by_index[0]
    if fb.string(feature_map, 0) != "new_img":
        raise ValueError("original feature TensorMap name differs")
    index_field = feature_map.Offset(6)
    if not index_field:
        raise ValueError("feature TensorMap lacks an allocated tensor_index field")
    edit(feature_map.Pos + index_field, struct.pack("<I", 1644), "signature feature tensor_index 0->1644", "signatures")
    name_field = feature_map.Offset(4)
    name_position = feature_map.Indirect(feature_map.Pos + name_field)
    if original[name_position:name_position + 12] != struct.pack("<I", 7) + b"new_img\0":
        raise ValueError("feature TensorMap name storage is not the expected seven-byte string")
    replacement_name = b"view_39\0"
    for part in (0, 4):
        edit(name_position + 4 + part, replacement_name[part:part + 4],
             f"signature feature name new_img->view_39 bytes{part}-{part + 3}", "signatures")
    for slot, selected, label in [
            (0, [signature_input_by_index[index].Pos for index in [0, 1, 2, 3, 5, 6]], "inputs"),
            (1, [signature_output_by_index[index].Pos for index in OUTPUTS], "outputs")]:
        _, entries = fb.vec(signature, slot)
        edit(entries[0] - 4, struct.pack("<I", len(selected)), f"signature {label} vector length", "signatures")
        for index, table_position in enumerate(selected):
            relative = table_position - entries[index]
            if not 0 < relative <= 0xffffffff:
                raise ValueError("signature TensorMap uoffset must remain positive uint32")
            edit(entries[index], struct.pack("<I", relative), f"signature {label}[{index}] TensorMap pointer", "signatures")
    epsilon_path = HERE / "gpu-epsilon-sqrt-patches.bin"
    epsilon_blob = epsilon_path.read_bytes()
    if epsilon_blob[:4] != b"JLP1":
        raise ValueError("epsilon patch magic differs")
    epsilon_count, model_size = struct.unpack_from("<IQ", epsilon_blob, 4)
    if epsilon_count != 309 or model_size != fb.MODEL_SIZE or len(epsilon_blob) != 16 + 16 * epsilon_count:
        raise ValueError("epsilon patch count/model size differs")
    for index in range(epsilon_count):
        offset, old, new = struct.unpack_from("<Q4s4s", epsilon_blob, 16 + index * 16)
        if original[offset:offset + 4] != old:
            raise ValueError(f"epsilon record {index} old bytes differ from original prefix")
        edit(offset, new, f"existing sqrt-epsilon record {index}", "sqrt_epsilon")
    unique = {}
    for item in edits:
        offset = item["byte_offset"]
        if offset in unique:
            raise ValueError(f"overlapping patch offset {offset}; groups are required to be disjoint")
        unique[offset] = item
    records = sorted(unique.values(), key=lambda item: item["byte_offset"])
    for previous, current in zip(records, records[1:]):
        if previous["byte_offset"] + 4 > current["byte_offset"]:
            raise ValueError("patch ranges overlap")
    if not 0 < len(records) <= 10000:
        raise ValueError("patch record count exceeds existing C apply limit")
    patched = bytearray(original)
    blob = bytearray(b"JLP1" + struct.pack("<IQ", len(records), fb.MODEL_SIZE))
    changed_positions = set()
    for item in records:
        offset = item["byte_offset"]
        old, new = bytes.fromhex(item["old_hex"]), bytes.fromhex(item["new_hex"])
        if original[offset:offset + 4] != old or old == new:
            raise ValueError("invalid per-record old/new diff")
        patched[offset:offset + 4] = new
        blob += struct.pack("<Q4s4s", offset, old, new)
        changed_positions.update(offset + i for i in range(4) if old[i] != new[i])
    actual_changed = {index for index, (old, new) in enumerate(zip(original, patched)) if old != new}
    if actual_changed != changed_positions:
        raise ValueError("prefix byte diff contains unrecorded changes")
    new_root, new_sg, new_tensors, new_buffers = model_parts(patched)
    if fb.vec(new_sg, 1, "<i")[0] != INPUTS or fb.vec(new_sg, 2, "<i")[0] != OUTPUTS:
        raise ValueError("readback public interface differs")
    new_signatures = fb.vector_tables(patched, new_root, 7)
    if len(new_signatures) != 1 or fb.string(new_signatures[0], 2) != "serving_default" or fb.scalar(new_signatures[0], 3, "<I") != 0:
        raise ValueError("readback serving_default signature differs")
    readback_signature_inputs = [(fb.string(item, 0), fb.scalar(item, 1, "<I"))
                                for item in fb.vector_tables(patched, new_signatures[0], 0)]
    readback_signature_outputs = [(fb.string(item, 0), fb.scalar(item, 1, "<I"))
                                 for item in fb.vector_tables(patched, new_signatures[0], 1)]
    expected_signature_inputs = [(fb.string(tensors[index], 3), index) for index in INPUTS]
    expected_signature_outputs = [(fb.string(tensors[index], 3), index) for index in OUTPUTS]
    if readback_signature_inputs != expected_signature_inputs or readback_signature_outputs != expected_signature_outputs:
        raise ValueError("readback signature TensorMap names/indices differ")
    new_ops, new_producer, _ = parse_ops(patched, new_sg)
    if [item["position"] for item in new_ops] != [ops[index]["position"] for index in kept]:
        raise ValueError("readback compacted operator table addresses differ")
    checked_kept, checked_reached, _, checked_leaves = closure(patched, new_tensors, new_buffers, new_ops, new_producer)
    if checked_kept != list(range(len(kept))) or {0, 4}.intersection(checked_reached):
        raise ValueError("readback graph contains dead ops or image dependencies")
    if fb.vec(new_sg, 0)[0] != fb.vec(sg, 0)[0] or fb.vec(new_root, 4)[0] != fb.vec(root, 4)[0]:
        raise ValueError("tensor/buffer table addresses changed")
    output_path.write_bytes(blob)

    def tensor_info(index):
        return {"index": index, "name": fb.string(new_tensors[index], 3),
                "type": fb.scalar(new_tensors[index], 1, "<b"),
                "shape": fb.vec(new_tensors[index], 0, "<i")[0]}

    manifest = {
        "status": "metadata_patch_and_epsilon_merge_readback_verified_pending_device_execution",
        "source_prefix": fingerprint(fb.PREFIX), "original_model_bytes": fb.MODEL_SIZE,
        "patch_binary": fingerprint(output_path), "existing_epsilon_patch": fingerprint(epsilon_path),
        "device_target_filename": "model-mixed-policy-sqrt-eps.tflite",
        "compatible_existing_apply_program": "gpu_tflite_patch_apply",
        "apply_to": "fresh byte-identical copy of the original full model, never an already epsilon-patched model",
        "original_inputs": old_inputs, "original_outputs": old_outputs,
        "inputs": [tensor_info(index) for index in INPUTS], "outputs": [tensor_info(index) for index in OUTPUTS],
        "feature_alias_rewrites": alias_rewrites,
        "feature_alias_rewrite_identity": "Two RESHAPE inputs use view_39 instead of its same-element-count conv2d_40__nhwc_out source; reshaping preserves the exact same ordered float values",
        "original_operator_count": len(ops), "policy_operator_count": len(kept),
        "kept_original_operator_indices": kept, "minimum_compacted_operator_uoffset": minimum_uoffset,
        "original_signature_defs_count": len(signatures), "patched_signature_defs_count": len(new_signatures),
        "signature_key": "serving_default", "signature_inputs": readback_signature_inputs,
        "signature_outputs": readback_signature_outputs,
        "patch_record_count": len(records), "patch_record_groups": dict(collections.Counter(item["group"] for item in records)),
        "existing_epsilon_records": epsilon_count, "all_patch_ranges_disjoint": True,
        "all_original_old_bytes_checked": True, "all_per_record_old_new_bytes_differ": True,
        "changed_byte_count": len(actual_changed), "full_prefix_diff_exactly_matches_records": True,
        "metadata_readback": "passed", "closure_and_forward_topology_check": "passed before and after epsilon patch merge",
        "policy_reached_input_indices": checked_leaves, "policy_has_no_new_img_or_state_img_dependency": True,
        "tensors_and_buffers_table_addresses_unchanged": True,
        "buffer_data_patch_scope": "Only the existing epsilon scalar corrections; no weights copied or converted",
        "device_access": "none", "patch_records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"patch_binary": manifest["patch_binary"], "record_count": len(records),
                      "groups": manifest["patch_record_groups"], "policy_operators": len(kept),
                      "inputs": manifest["inputs"], "outputs": manifest["outputs"],
                      "original_signatures": len(signatures), "patched_signatures": len(new_signatures),
                      "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
