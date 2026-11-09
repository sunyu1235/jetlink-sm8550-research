"""Replace only the final attention EPContext, retaining the full model interface."""
from pathlib import Path
import copy
import hashlib
import json
import onnx

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/jetlink-gelu-optimization-20261101/partition-tuning-20261109'
SOURCE = OUT / 'whole-context-shared-all-boundaries.onnx'
PARTITION = OUT / 'last-query-compile-a/last-query-ctx.onnx'
DEST = OUT / 'whole-context-last-query-shared-all.onnx'
BIN_NAME = 'last-query-20261109-soc43_qnn.bin'

def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    model = onnx.load(SOURCE, load_external_data=False)
    part = onnx.load(PARTITION, load_external_data=False)
    if len(part.graph.node) != 1 or part.graph.node[0].op_type != 'EPContext':
        raise ValueError('Expected exactly one fully compiled QNN partition')
    replacement = copy.deepcopy(part.graph.node[0])
    targets = [n for n in model.graph.node if n.name.endswith('_131_41')]
    if len(targets) != 1:
        raise ValueError('Expected one source final-attention partition')
    target = targets[0]
    original_partition_name = target.name
    if set(target.input) != set(replacement.input) or set(target.output) != set(replacement.output):
        raise ValueError('Partition boundary names changed')
    for attr in replacement.attribute:
        if attr.name == 'ep_cache_context':
            attr.s = BIN_NAME.encode()
    if any(n.name == replacement.name for n in model.graph.node if n is not target):
        raise ValueError('Replacement node name collision')
    original = [n.SerializeToString() for n in model.graph.node if n.name != target.name]
    for i, node in enumerate(model.graph.node):
        if node.name == target.name:
            model.graph.node[i].CopyFrom(replacement)
            break
    if original != [n.SerializeToString() for n in model.graph.node if n.name != replacement.name]:
        raise ValueError('Unrelated nodes changed')
    onnx.checker.check_model(model)
    onnx.save(model, DEST)
    manifest = {
        'status': 'ready_for_full_model_parity', 'wrapper': str(DEST),
        'wrapper_sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
        'original_partition': original_partition_name, 'replacement_partition': replacement.name,
        'new_cache_basename': BIN_NAME, 'source_wrapper': str(SOURCE),
        'original_interface_and_all_other_nodes_preserved': True,
        'boundary_prefix': 'Rows 0..286 of the replacement outputs are intentionally zero; final consumers select only row287.',
    }
    (REPORT / 'last-query-combined-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
