"""Create a source/evidence-only archive after the combined one-frame gate.

The explicit allowlist excludes generated models, weights, SDK code and binaries.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile


FILES = [
    'README.md', 'candidate.json', 'baseline-audit.json', 'model-size.json',
    'extract_region.py', 'prepare_reference.py', 'check_remainder.py',
    'run_native_region.sh', 'build_model.py', 'test_build_model.py',
    'test_extract_region.py', 'tests.log', 'model-tests.log',
    'repack_remainder.py', 'test_repack_remainder.py', 'repacked/evidence.json',
    'repacked/tests.log', 'repacked/sha256sums.txt', 'final-tests.log', 'final-validation.json',
    'bridge/bridge.cpp', 'bridge/build.sh', 'bridge/test_bridge.py',
    'bridge/test_contract.cpp', 'bridge/test_full_frame.py',
    'bridge/run_full_frame.sh', 'bridge/README.md', 'bridge/negative-tests.json',
    'bridge/model/manifest.json', 'bridge/full-frame-check/result.json',
    'bridge/full-frame-check/provenance.json', 'bridge/vector-full-frame-check/result.json',
    'bridge/vector-full-frame-check/provenance.json',
    'bridge/vector-full-frame-check/cpu-baseline-check/evidence.json',
    'bridge/vector-full-frame-check/cpu-baseline-check/checker.log',
    'bridge/contract-tests.log', 'bridge/negative-tests.log',
    'vision/manifest.json', 'vision/config.json', 'vision/extensions.json',
    'vision/night0-run-result.json', 'native-first-frame/result.json',
    'native-first-frame/checker.log', 'reference-night0-verified/reference.json',
    'cpu-cut-control-v2/result.json', 'cpu-cut-control-v2/checker.log',
    'preparation/README.md', 'preparation/Gelu.xml',
    'preparation/map_model.py', 'preparation/promote_images.py',
    'preparation/convert_preserve_arithmetic.py', 'preparation/repair_preserved_bias.py',
    'preparation/repair-check.log', 'preparation/repair-check-hashes.txt',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--polynomial-archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.stage.exists() or args.archive.exists():
        raise FileExistsError('Stage and archive must be fresh')
    gate = json.loads((root/'bridge/vector-full-frame-check/result.json').read_text())
    if set(gate['outputs']) != {'outputs','next_state_img_q','next_state_desire_q','next_state_feat_q'} or gate['native_invocations'] != 1 or not all(v['exact_bytes'] and not v['nonfinite'] for v in gate['outputs'].values()):
        raise ValueError('Combined functional gate has not passed')
    checker = json.loads((root/'bridge/vector-full-frame-check/cpu-baseline-check/evidence.json').read_text())
    if checker['exit_code'] != 0 or not checker['equal_full_lengths'] or not checker['finite_inputs']:
        raise ValueError('Original full-head checker has not passed')
    dest = args.stage/'research/aggressive-vision-fusion'
    for name in FILES + ['package_source.py']:
        source = root/name
        if not source.is_file() or source.is_symlink():
            raise ValueError('Missing/non-regular allowlisted source: ' + name)
        target = dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Relocate the companion reference for the deliverable's repository layout.
    readme = dest/'README.md'
    readme.write_text(readme.read_text().replace('../jetlink-gelu-polynomial-20261110', '../../experiments/gelu-polynomial'))
    poly = args.stage/'experiments/gelu-polynomial'
    with tarfile.open(args.polynomial_archive) as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not member.isfile() or not parts or parts[0] != 'jetlink-gelu-polynomial' or '..' in parts or Path(member.name).is_absolute():
                raise ValueError('Unexpected polynomial archive entry')
            target = poly.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out)
    manifest = []
    forbidden = {'.so', '.o', '.elf', '.bin', '.onnx', '.npy', '.npz', '.raw', '.weights'}
    for path in sorted(args.stage.rglob('*')):
        if path.is_dir():
            continue
        if not path.is_file() or path.is_symlink() or path.suffix in forbidden or path.name == 'LocalLicensedInterface.cpp':
            raise ValueError('Forbidden artifact in source package: ' + str(path))
        manifest.append(dict(path=str(path.relative_to(args.stage)), bytes=path.stat().st_size, sha256=sha(path)))
    (args.stage/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(
        scope='Authored source and functional evidence; no device or recurrent acceptance',
        files=manifest), indent=2))
    with tarfile.open(args.archive, 'w:gz') as archive:
        for path in sorted(args.stage.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(args.stage)), recursive=False)
    print(json.dumps(dict(archive=str(args.archive),sha256=sha(args.archive),
                         bytes=args.archive.stat().st_size,files=len(manifest)+1),indent=2))


if __name__ == '__main__':
    main()
