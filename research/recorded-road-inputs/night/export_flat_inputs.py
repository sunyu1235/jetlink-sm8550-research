#!/usr/bin/env python3
"""Export a generated road input pack to Jetlink verify_parity file conventions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, default=Path(__file__).parent/'inputs320')
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    assert spec['sha256'] == '6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21'
    assert spec['input_shapes']['new_img'] == [2, 6, 128, 256]
    assert spec['input_shapes']['desire'] == [8]
    assert spec['input_shapes']['traffic_convention'] == [1, 2]
    assert spec['input_shapes']['action_t'] == [1, 2]
    # jetlink.spec.ModelSpec.packed_shapes: desire, traffic_convention, action_t.
    # Deliberately do not rely on JSON input_shapes ordering.
    image = np.load(args.inputs/'new_img.npy', mmap_mode='r')
    parts = [np.load(args.inputs/f'{n}.npy').reshape(len(image), -1)
             for n in ['desire', 'traffic_convention', 'action_t']]
    packed = np.concatenate(parts, axis=1).astype(np.float32)
    assert packed.shape == (len(image), 12) and image.dtype == np.uint8
    for i in range(len(image)):
        np.save(args.inputs/f'in_warped_{i}.npy', image[i])
        np.save(args.inputs/f'in_packed_{i}.npy', packed[i])
    if args.spec.resolve() != (args.inputs/'spec.json').resolve():
        shutil.copyfile(args.spec, args.inputs/'spec.json')
    print(json.dumps({'frames': len(image), 'packed_order': ['desire', 'traffic_convention', 'action_t'],
                      'spec_sha256': hashlib.sha256(args.spec.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
