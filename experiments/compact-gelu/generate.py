"""Generate a compact *exact* map; no calibration or approximation involved."""
import argparse, hashlib, json, pathlib, struct
p=argparse.ArgumentParser(); p.add_argument('reference',type=pathlib.Path); p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('.')); a=p.parse_args()
raw=a.reference.read_bytes()
if len(raw)!=262144: raise ValueError('Expected both complete 65536-entry uint16 LE tables')
a.out.mkdir(parents=True,exist_ok=True)
values=struct.unpack('<131072H',raw)
IDENTITY,ZERO,NEGZERO=65535,65534,65533
results=[]
for shift in range(3,10):
    n=1<<shift; pages=[]; bank=[]; indices={}; tags=[]
    for mode in range(2):
        mode_tags=[]
        for start in range(0,65536,n):
            page=values[mode*65536+start:mode*65536+start+n]
            if page==tuple(range(start,start+n)): tag=IDENTITY
            elif all(v==0 for v in page): tag=ZERO
            elif all(v==32768 for v in page): tag=NEGZERO
            else:
                if page not in indices: indices[page]=len(bank); bank.append(page)
                tag=indices[page]
            mode_tags.append(tag)
        tags.append(mode_tags)
    size=2*(sum(map(len,tags))+len(bank)*n)
    results.append(dict(shift=shift,page_entries=n,unique_pages=len(bank),bytes=size,tags=tags,bank=bank))
best=min(results,key=lambda r:r['bytes']); shift=best['shift']; n=1<<shift
header=['/* Generated exact bit maps; regenerate with generate.py. */','#ifndef JL_COMPACT_GELU_TABLES_H','#define JL_COMPACT_GELU_TABLES_H','#include <stdint.h>',f'#define JL_GELU_PAGE_SHIFT {shift}',f'#define JL_GELU_PAGE_MASK {n-1}',f'static const uint16_t jl_gelu_pages[2][{65536//n}] = {{']
for tags in best['tags']:
    header.append('{'+','.join(map(str,tags))+'},')
header += ['};',f'static const uint16_t jl_gelu_bank[{len(best["bank"])}][{n}] = {{']
for page in best['bank']: header.append('{'+','.join(map(str,page))+'},')
header += ['};','#endif']
(a.out/'gelu_tables.h').write_text('\n'.join(header)+'\n')
report={k:v for k,v in best.items() if k not in ('tags','bank')}
report.update(reference_sha256=hashlib.sha256(raw).hexdigest(),full_two_table_bytes=262144,reduction_percent=100*(1-best['bytes']/262144),candidates=[{k:v for k,v in r.items() if k not in ('tags','bank')} for r in results],note='Storage only. Not a speed measurement. NaN payloads and signed zeros preserve host reference bit patterns; verify against Android-generated tables before deployment.')
(a.out/'storage-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
