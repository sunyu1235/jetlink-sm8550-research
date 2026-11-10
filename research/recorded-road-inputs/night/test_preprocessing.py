#!/usr/bin/env python3
"""Lightweight deterministic regression checks for local road preprocessing."""
import importlib.util
from pathlib import Path
import numpy as np
from preprocess_road import WIDTH, HEIGHT, MODEL_W, MODEL_H, sampling_indices, prepare_i420, maps_for, transforms


def test_packing_and_identity():
    n = WIDTH*HEIGHT
    yy, xx = np.indices((HEIGHT,WIDTH))
    y = ((yy*3+xx*7)%256).astype(np.uint8)
    u = np.full((HEIGHT//2,WIDTH//2),65,np.uint8)
    v = np.full_like(u,175)
    raw = y.tobytes()+u.tobytes()+v.tobytes()
    assert len(raw)==n*3//2
    out=prepare_i420(raw,maps_for(np.eye(3,dtype=np.float32)))
    assert out.shape==(6,128,256) and out.dtype==np.uint8
    expected=[y[:MODEL_H:2,:MODEL_W:2],y[1:MODEL_H:2,:MODEL_W:2],
              y[:MODEL_H:2,1:MODEL_W:2],y[1:MODEL_H:2,1:MODEL_W:2],
              u[:128,:256],v[:128,:256]]
    np.testing.assert_array_equal(out,np.stack(expected))


def test_rounding_and_clamping():
    m=np.eye(3,dtype=np.float32);m[0,2]=.5;m[1,2]=-.5
    yi,xi=sampling_indices(m,(5,2),(2,5))
    np.testing.assert_array_equal(xi[0],[0,2,2,4,4])
    np.testing.assert_array_equal(yi,np.zeros((2,5),np.int32))
    m[0,2]=-100;m[1,2]=100
    yi,xi=sampling_indices(m,(5,2),(2,5))
    assert (xi==0).all() and (yi==1).all()


def test_upstream_geometry():
    root=Path(__file__).parent
    # Load immutable original Light Speed-era Python source directly, no device libraries.
    p=root/'source/openpilot/common/transformations/transformations.py'
    spec=importlib.util.spec_from_file_location('upstream_transforms',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rpy=np.array([-.000017203474271809682,-.029515165835618973,-.00638499204069376],np.float32)
    view=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    source=[np.array([[1141.5,0,672],[0,1141.5,380],[0,0,1.]]),
            np.array([[425.25,0,672],[0,425.25,380],[0,0,1.]])]
    targets=[np.array([[910.,0,256],[0,910.,47.6],[0,0,1.]]),
             np.array([[455.,0,256],[0,455.,151.8],[0,0,1.]])]
    expected=np.stack([(k@view@module.euler2rot_single(rpy)@np.linalg.inv(t@view)).astype(np.float32)
                       for k,t in zip(source,targets)])
    np.testing.assert_array_equal(transforms(rpy),expected)


if __name__=='__main__':
    for fn in [test_packing_and_identity,test_rounding_and_clamping,test_upstream_geometry]:
        fn();print(fn.__name__,'PASS')
