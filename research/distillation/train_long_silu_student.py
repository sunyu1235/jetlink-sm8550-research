"""Train both affine layers of a real ReLU MLP using independent long synthetic histories.
Never use original acceptance targets. This is offline block training, not a road model.
"""
import json,time
from pathlib import Path
import numpy as np
from fit_relu_student import fit
p=Path('/workspace/shared/jetlink-distillation/long-silu-student');p.mkdir(exist_ok=True)
z=np.load('/workspace/shared/jetlink-long-calibration/sequences.npz');w=np.load('/workspace/shared/jetlink-distillation/real-block/weights.npz')
x=z['x_train'].reshape(-1,512).astype(np.float32);y=z['y_train'].reshape(-1,512).astype(np.float32)
xv=z['x_validation'].reshape(-1,512).astype(np.float32);yv=z['y_validation'].reshape(-1,512).astype(np.float32)
rng=np.random.default_rng(888);ix=rng.choice(len(x),12288,replace=False)
t=time.monotonic()
beta=float(np.float16(1.702))
def sigmoid(a): return 1/(1+np.exp(-np.clip(a,-40,40)))
def activation(a): return a*sigmoid(beta*a)
w1=w['w1'].astype(np.float64);b1=w['b1'].astype(np.float64)
h=activation(x[ix].astype(np.float64)@w1+b1);mean=h.mean(0);hs=np.maximum(h.std(0),1e-12);z=(h-mean)/hs;yc=y[ix].astype(np.float64);ym=yc.mean(0);gram=z.T@z/len(z);gram.flat[::1025]+=1e-5
coef=np.linalg.solve(gram,z.T@(yc-ym)/len(z));w2=coef/hs[:,None];b2=ym-mean@w2
s={k:v.astype(np.float32) for k,v in dict(w1=w1,b1=b1,w2=w2,b2=b2).items()}
del h,z,yc,gram,coef
print('ridge',time.monotonic()-t,flush=True)
scale=np.maximum(y.std(0),np.median(y.std(0))*.1)

def evaluate(s):
 # Match FP16 storage, use FP32 computation as a screening metric only.
 q={k:v.astype(np.float16).astype(np.float32) for k,v in s.items()}
 loss=se=0.;mx=0.
 for i in range(0,len(xv),1024):
  error=activation(xv[i:i+1024]@q['w1']+q['b1'])@q['w2']+q['b2']-yv[i:i+1024]
  se+=float(np.sum(error.astype(np.float64)**2));loss+=float(np.sum((error/scale).astype(np.float64)**2));mx=max(mx,float(abs(error).max()))
 return {'scaled_mse':loss/yv.size,'rmse':(se/yv.size)**.5,'max_abs':mx}

mom={k:np.zeros_like(v) for k,v in s.items()};var={k:np.zeros_like(v) for k,v in s.items()};history=[]
best=evaluate(s);np.savez(p/'student-best.npz',**{k:v.astype(np.float16) for k,v in s.items()});np.savez(p/'student-ridge.npz',**{k:v.astype(np.float16) for k,v in s.items()});history.append({'step':0,**best});print(0,best,flush=True)
for step in range(1,1001):
 idx=rng.integers(len(x),size=512);a=x[idx];target=y[idx];pre=a@s['w1']+s['b1'];sig=sigmoid(beta*pre);h=pre*sig;error=h@s['w2']+s['b2']-target
 dy=2*error/(scale*scale*error.size);dh=(dy@s['w2'].T)*(sig+beta*pre*sig*(1-sig))
 grads={'w2':h.T@dy,'b2':dy.sum(0),'w1':a.T@dh,'b1':dh.sum(0)}
 lr=2e-5*(.1+.9*.5*(1+np.cos(np.pi*step/1000)))
 for k,g in grads.items():
  mom[k]*=.9;mom[k]+=.1*g;var[k]*=.999;var[k]+=.001*g*g
  s[k]-=lr*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
 if step%200==0:
  e=evaluate(s);history.append({'step':step,**e});print(step,e,flush=True)
  if e['scaled_mse']<best['scaled_mse']:
   best=e;np.savez(p/'student-best.npz',**{k:v.astype(np.float16) for k,v in s.items()})
(p/'training.json').write_text(json.dumps({'status':'REAL_BLOCK_TWO_LAYER_TRAINING_SYNTHETIC_LONG_HISTORY','train_seed':713,'validation_seed':1777,'frames_each':160,'steps':1000,'batch':512,'hidden_width':1024,'activation':'x*sigmoid(1.7021484375*x)','original_acceptance_data_used':False,'metric':'FP16 stored weights FP32 CPU operations, not device','history':history,'total_seconds':time.monotonic()-t},indent=2))
