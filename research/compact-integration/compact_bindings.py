"""Strict, zero-padding-free ABI between compact last-query partition and tails.
A name alias is not a device zero-copy guarantee. Arrays are borrowed unchanged.
"""
import numpy as np
ORDER=('add_52','add_72','add_88','linear_110','linear_151','linear_180')
SHAPES={n:(1,1,512 if n.startswith('add_') else 1024) for n in ORDER}
ALIASES={n:n+'__last_query' for n in ORDER}

def validate_metadata(upstream_outputs,downstream_inputs):
    up={v.name:v for v in upstream_outputs};down={v.name:v for v in downstream_inputs}
    if set(up)!=set(ALIASES.values()) or set(down)!=set(ORDER):raise ValueError('incompatible partition names: do not mix old cached contexts')
    for n in ORDER:
        for v in (up[ALIASES[n]],down[n]):
            if tuple(v.shape)!=SHAPES[n] or v.type!='tensor(float16)':raise ValueError(f'incompatible compact ABI for {n}')

def bind(outputs):
    if set(outputs)!=set(ALIASES.values()):raise ValueError('unexpected upstream boundary names')
    feed={}
    for n in ORDER:
        a=outputs[ALIASES[n]]
        if not isinstance(a,np.ndarray) or a.dtype!=np.float16 or a.shape!=SHAPES[n]:raise ValueError(f'expected FP16 {SHAPES[n]} for {n}')
        if not a.flags.c_contiguous:raise ValueError('contiguous storage required; no silent boundary copy')
        if not np.isfinite(a).all():raise ValueError('nonfinite boundary')
        feed[n]=a
    assert sum(a.nbytes for a in feed.values())==9216
    return feed

def execute(upstream,downstream,inputs):
    validate_metadata(upstream.get_outputs(),downstream.get_inputs())
    values=upstream.run(None,inputs)
    feed=bind(dict(zip((v.name for v in upstream.get_outputs()),values,strict=True)))
    return dict(zip((v.name for v in downstream.get_outputs()),downstream.run(None,feed),strict=True))
