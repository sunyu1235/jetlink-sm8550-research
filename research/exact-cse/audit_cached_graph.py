#!/usr/bin/env python3
"""Map exact-source aliases to an accepted Android cache without editing it.

EPContext internals remain opaque. Seeing an output name at a QNN boundary
cannot prove that its MatMul executes separately inside the compiled context.
"""
import argparse
import collections
import json
from pathlib import Path
import onnx
from exact_cse import rewrite, sha256_file


def audit(source, cached, source_directory, allow_exact=False):
    _, cse = rewrite(source, source_directory, allow_exact)
    producers = {name: node for node in cached.graph.node for name in node.output}
    results = []
    for removed in cse['removed_nodes']:
        if removed['op_type'] not in ('MatMul', 'Gelu', 'ExactGelu', 'ExactGelu32', 'ExactGeluTanh', 'ExactGeluTanh32'):
            continue
        rows = []
        for alias, canonical in zip(removed['outputs'], removed['replacement_outputs']):
            a, b = producers.get(alias), producers.get(canonical)
            explicit = (a is not None and b is not None and a is not b and
                        a.op_type != 'EPContext' and b.op_type != 'EPContext')
            def describe(n):
                return None if n is None else {'name': n.name, 'op_type': n.op_type, 'domain': n.domain}
            rows.append({'alias': alias, 'canonical': canonical,
                         'alias_producer': describe(a), 'canonical_producer': describe(b),
                         'finding': 'two_explicit_source_boundary_operators' if explicit else 'compiled_internal_execution_unknown'})
        results.append({'source_operation': removed['op_type'], 'source_node': removed['name'], 'values': rows})
    return {'status': 'READ_ONLY_CACHE_AUDIT_NOT_DEVICE_EXECUTION',
            'cache_operations': dict(collections.Counter(n.domain + ':' + n.op_type for n in cached.graph.node)),
            'source_removed_operations': cse['removed_operations'], 'source_alias_locations': results,
            'qnn_internal_matmul_execution': 'UNKNOWN: requires compiler graph dump or target operator trace',
            'phone_speed_measured': False, 'numerical_160_run': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--cached', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--allow-jetlink-exact-gelu', action='store_true')
    a = p.parse_args()
    if a.report.exists():
        p.error('Refusing to overwrite report')
    report = audit(onnx.load(a.source, load_external_data=False),
                   onnx.load(a.cached, load_external_data=False), a.source.parent, a.allow_jetlink_exact_gelu)
    report.update(source=str(a.source), source_sha256=sha256_file(a.source),
                  cached=str(a.cached), cached_sha256=sha256_file(a.cached))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
