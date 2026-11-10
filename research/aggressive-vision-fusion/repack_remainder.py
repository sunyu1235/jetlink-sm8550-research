#!/usr/bin/env python3
"""Losslessly prune unused root initializers and stream-compact live external spans.

No graph rewriting or inference. An external symlink escaping the source directory
requires explicit --allow-external-file with its resolved target. Output must not exist.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import onnx
from onnx import TensorProto, AttributeProto

# Fixed-width ONNX numeric types, bits per element. Unsupported encodings fail closed.
BITS = {1:32, 2:8, 3:8, 4:16, 5:16, 6:32, 7:64, 9:8, 10:16,
        11:64, 12:32, 13:64, 14:64, 15:128, 16:16,
        17:8, 18:8, 19:8, 20:8, 21:4, 22:4, 23:4, 24:8, 25:2, 26:2}
PAYLOAD_FIELDS = ('raw_data','float_data','int32_data','string_data','int64_data','double_data','uint64_data')

def digest_span(path, offset, length, dest=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while length:
            data = f.read(min(length, 1024*1024))
            if not data:
                raise ValueError('Truncated external tensor')
            h.update(data)
            if dest is not None:
                dest.write(data)
            length -= len(data)
    return h.hexdigest()

def shape_bytes(t):
    if t.data_type not in BITS or any(d < 0 for d in t.dims):
        raise ValueError('Unsupported tensor encoding or dimensions: ' + t.name)
    return (math.prod(t.dims) * BITS[t.data_type] + 7)//8

def external_span(t, source_dir, allowed):
    entries = [(x.key, x.value) for x in t.external_data]
    data = dict(entries)
    if len(data) != len(entries) or set(data) - {'location','offset','length','checksum'}:
        raise ValueError('Unsupported or duplicate external metadata: ' + t.name)
    location = data.get('location', '')
    p = PurePosixPath(location)
    if not location or '\\' in location or ':' in location or p.is_absolute() or '..' in p.parts or location != str(p):
        raise ValueError('Unsafe external location: ' + location)
    path = (source_dir / location).resolve(strict=True)
    if not path.is_file() or (not path.is_relative_to(source_dir) and path not in allowed):
        raise ValueError('External path escapes source directory: ' + location)
    if any(getattr(t, field) for field in PAYLOAD_FIELDS):
        raise ValueError('Mixed inline/external payload: ' + t.name)
    if t.HasField('segment'):
        raise ValueError('Segmented tensor unsupported')
    expected = shape_bytes(t)
    offset_s, length_s = data.get('offset','0'), data.get('length')
    if not offset_s.isdecimal() or (length_s is not None and not length_s.isdecimal()):
        raise ValueError('Invalid external range')
    offset = int(offset_s)
    length = int(length_s) if length_s is not None else path.stat().st_size-offset
    if length != expected or offset + length > path.stat().st_size:
        raise ValueError('Inconsistent external length: ' + t.name)
    return path, offset, length

def graph_without_initializers(m):
    copy = onnx.ModelProto()
    copy.CopyFrom(m)
    del copy.graph.initializer[:]
    return copy.SerializeToString(deterministic=True)

def validate_structure(m):
    if m.training_info or m.graph.sparse_initializer:
        raise ValueError('Functions, training, or sparse initializers unsupported')
    for f in m.functions:
        if (f.domain != 'org.tinygrad' or f.name != 'Contiguous' or f.overload
                or len(f.input) != 1 or len(f.output) != 1 or len(f.node) != 1
                or f.attribute or f.attribute_proto):
            raise ValueError('Unsupported function definition')
        n = f.node[0]
        if (n.op_type != 'Identity' or n.domain or n.overload or n.attribute
                or list(n.input) != list(f.input) or list(n.output) != list(f.output)):
            raise ValueError('Unsupported Contiguous function body')
        if len(f.opset_import) != 1 or f.opset_import[0].domain or f.opset_import[0].version != 20:
            raise ValueError('Unsupported Contiguous function opset')
    for n in m.graph.node:
        for a in n.attribute:
            if a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS, AttributeProto.SPARSE_TENSOR, AttributeProto.SPARSE_TENSORS):
                raise ValueError('Subgraphs or sparse attributes unsupported')
            ts = [a.t] if a.type == AttributeProto.TENSOR else a.tensors
            for t in ts:
                if t.data_location == TensorProto.EXTERNAL or t.external_data:
                    raise ValueError('External attribute tensors unsupported')
    names = [t.name for t in m.graph.initializer]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate initializer name')

def repack(source, output, context=None, allow_external_files=()):
    source, output = Path(source).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(str(output))
    allowed = {Path(p).resolve(strict=True) for p in allow_external_files}
    m = onnx.load(source, load_external_data=False)
    validate_structure(m)
    used = {s for n in m.graph.node for s in n.input} | {v.name for v in m.graph.input} | {v.name for v in m.graph.output}
    kept, removed, spans = [], [], {}
    # Validate even discarded tensor metadata, so malformed sources fail closed.
    for t in m.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL:
            spans[t.name] = external_span(t, source.parent, allowed)
        elif t.external_data:
            raise ValueError('External metadata without EXTERNAL location')
        (kept if t.name in used else removed).append(t)
    files = {source} | {v[0] for v in spans.values()}
    signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.repack-', dir=output.parent))
    try:
        result = onnx.ModelProto()
        result.CopyFrom(m)
        del result.graph.initializer[:]
        records = []
        with (stage/'remainder.weights').open('xb') as out:
            for original in kept:
                t = result.graph.initializer.add()
                t.CopyFrom(original)
                if t.name in spans:
                    path, offset, length = spans[t.name]
                    new_offset = out.tell()
                    old_hash = digest_span(path, offset, length, out)
                    del t.external_data[:]
                    for key, value in [('location','remainder.weights'),('offset',str(new_offset)),('length',str(length))]:
                        e=t.external_data.add();e.key=key;e.value=value
                    records.append(dict(name=t.name, bytes=length, source_sha256=old_hash, output_offset=new_offset, external=True))
                else:
                    h=hashlib.sha256(t.SerializeToString(deterministic=True)).hexdigest()
                    records.append(dict(name=t.name, source_sha256=h, external=False))
        target = stage/'vision-native.onnx'
        target.write_bytes(result.SerializeToString())
        check = onnx.load(target, load_external_data=False)
        assert graph_without_initializers(m) == graph_without_initializers(check), 'Non-initializer model content changed'
        for t, record in zip(check.graph.initializer, records):
            if record['external']:
                h = digest_span(stage/'remainder.weights', record['output_offset'], record['bytes'])
                before = onnx.TensorProto(); before.CopyFrom(next(v for v in kept if v.name==t.name)); del before.external_data[:]
                after = onnx.TensorProto(); after.CopyFrom(t); del after.external_data[:]
                assert before == after, 'Tensor metadata changed'
            else:
                h=hashlib.sha256(t.SerializeToString(deterministic=True)).hexdigest()
            assert h==record['source_sha256'], 'Tensor bytes changed'
            record['output_sha256']=h
        onnx.checker.check_model(str(target))
        for p, sig in signatures.items():
            assert (p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino)==sig, 'Source changed during repacking'
        sizes = {'onnx_bytes':target.stat().st_size,'weights_bytes':(stage/'remainder.weights').stat().st_size}
        if context is not None:
            context=Path(context).resolve(strict=True)
            sizes['existing_context_bytes']=context.stat().st_size
            sizes['complete_model_data_bytes']=sum(sizes.values())
        report=dict(source=str(source), kept_initializers=len(kept), removed_initializers=len(removed), removed_names=[t.name for t in removed],
                    graph_and_all_noninitializer_model_fields_exact=True, all_live_tensor_hashes_equal=True, onnx_checker='passed', sizes=sizes, tensors=records,
                    preserved_functions=[dict(domain=f.domain,name=f.name,sha256=hashlib.sha256(f.SerializeToString(deterministic=True)).hexdigest(),root_call_count=sum(n.domain==f.domain and n.op_type==f.name for n in m.graph.node)) for f in m.functions],
                    scope='Model data only; excludes runtime libraries. No inference or performance claim.')
        (stage/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
        # Atomic directory creation without overwriting an existing destination.
        output.mkdir()
        for p in stage.iterdir():
            p.rename(output/p.name)
        stage.rmdir()
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('output');p.add_argument('--context');p.add_argument('--allow-external-file',action='append',default=[])
    a=p.parse_args()
    r=repack(a.source,a.output,a.context,a.allow_external_file)
    print(json.dumps({k:v for k,v in r.items() if k not in ('tensors','removed_names')},indent=2))
