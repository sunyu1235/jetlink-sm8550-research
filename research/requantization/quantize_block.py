#!/usr/bin/env python3
"""Actual ONNX QDQ block pilot. CPU correctness diagnostics, never HTP speed claims.
Inputs: original FP16/FP32 block and NPZ x_train[N,*inputshape], x_validation[M,*inputshape].
Quantization calibration only uses x_train. Conversion drift is measured separately.
"""
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import QuantType, CalibrationDataReader, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config


def promote(model):
    m=copy.deepcopy(model)
    for t in m.graph.initializer:
        if t.data_type==TensorProto.FLOAT16:t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32),t.name))
    for v in list(m.graph.input)+list(m.graph.output)+list(m.graph.value_info):
        if v.type.tensor_type.elem_type==TensorProto.FLOAT16:v.type.tensor_type.elem_type=TensorProto.FLOAT
    for n in m.graph.node:
        for a in n.attribute:
            if n.op_type=='Cast' and a.name=='to' and a.i==TensorProto.FLOAT16:a.i=TensorProto.FLOAT
            if a.type==onnx.AttributeProto.TENSOR and a.t.data_type==TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32),a.t.name))
    onnx.checker.check_model(m)
    return m

class Reader(CalibrationDataReader):
    def __init__(self,name,x):self.name=name;self.x=x;self.rewind()
    def get_next(self):
        try:return {self.name:next(self.it).astype(np.float32)}
        except StopIteration:return None
    def rewind(self):self.it=iter(self.x)

def session(path):
    o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;o.log_severity_level=3
    return ort.InferenceSession(str(path),o,providers=['CPUExecutionProvider'])

def run(s,x):
    i=s.get_inputs()[0];dt=np.float16 if i.type=='tensor(float16)' else np.float32
    return np.stack([s.run(None,{i.name:v.astype(dt)})[0] for v in x])

def metrics(y,ref):
    d=y.astype(np.float64)-ref.astype(np.float64)
    return {'finite':bool(np.isfinite(y).all()),'max_abs':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d))),
            'normalized_rmse':float(np.sqrt(np.mean(d*d))/max(float(ref.astype(np.float64).std()),1e-20)),
            'correlation':float(np.corrcoef(y.reshape(-1),ref.reshape(-1))[0,1])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exclude-node',action='append',default=[]);p.add_argument('--a16-only',action='store_true');p.add_argument('--block',required=True,type=Path);p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.data,allow_pickle=False) as d:xtrain=d['x_train'];xval=d['x_validation']
    if not len(xtrain) or not len(xval) or not (np.isfinite(xtrain).all() and np.isfinite(xval).all()):raise ValueError('invalid data')
    original=onnx.load(a.block)
    if len(original.graph.input)!=1 or len(original.graph.output)!=1:raise ValueError('one-input one-output block required')
    s=session(a.block);shape=tuple(s.get_inputs()[0].shape)
    if tuple(xtrain.shape[1:])!=shape or tuple(xval.shape[1:])!=shape:raise ValueError(f'expected frames shaped {shape}; got {xtrain.shape}, {xval.shape}')
    ref=run(s,xval);del s
    fp32path=a.output/'block-fp32.onnx';onnx.save(promote(original),fp32path)
    s=session(fp32path);fp32=run(s,xval);del s
    report={'status':'REAL_BLOCK_QDQ_CPU_ONLY_NOT_WHOLE_MODEL_OR_HTP',
            'source_model_sha256':hashlib.sha256(a.block.read_bytes()).hexdigest(),
            'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'onnx_version':onnx.__version__,'ort_version':ort.__version__,
            'excluded_nodes':a.exclude_node,'calibration_frames':len(xtrain),'validation_frames':len(xval),
            'fp32_conversion_vs_original':metrics(fp32,ref),'cases':[]}
    for activation_type in ((QuantType.QUInt16,) if a.a16_only else (QuantType.QUInt16,QuantType.QUInt8)):
        for per_channel in ((False,) if a.a16_only else (False,True)):
            name=f'{activation_type.name}-w8-pc{int(per_channel)}';path=a.output/f'{name}.onnx'
            result={'name':name}
            try:
                reader=Reader(original.graph.input[0].name,xtrain)
                config=get_qnn_qdq_config(fp32path,reader,activation_type=activation_type,weight_type=QuantType.QUInt8,
                    per_channel=per_channel,nodes_to_exclude=a.exclude_node,op_types_to_quantize=['MatMul','Gemm'],calibration_providers=['CPUExecutionProvider'])
                quantize(fp32path,path,config)
                onnx.checker.check_model(str(path)); qm=onnx.load(path)
                result['weight_encoding_scale_shapes']={t.name:list(t.dims) for t in qm.graph.initializer if t.name in ('val_645_scale','val_647_scale')}
                result['per_channel_requested']=per_channel
                result['per_channel_note']='QNN MatMul forces per-tensor in this ORT version; see actual scale shapes.'
                s=session(path);actual=run(s,xval);del s
                result.update({'vs_original':metrics(actual,ref),'vs_fp32':metrics(actual,fp32),'bytes':path.stat().st_size,
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'executed':True})
            except Exception as e:result.update({'executed':False,'error':str(e),'error_type':type(e).__name__})
            report['cases'].append(result)
            (a.output/'summary.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
    np.save(a.output/'original_validation.npy',ref);np.save(a.output/'fp32_validation.npy',fp32)
if __name__=='__main__':main()
