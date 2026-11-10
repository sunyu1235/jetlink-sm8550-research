"""Distill real MLPs independently. Selection uses only synthetic validation seed1777."""
import argparse,json,time
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('blocks',nargs='+');ap.add_argument('--steps',type=int,default=1000);args=ap.parse_args()
root=Path(__file__).resolve().parent
beta=float(np.float16(1.702))
def sigmoid(a):return 1/(1+np.exp(-np.clip(a,-40,40)))
def activation(a):return a*sigmoid(beta*a)
def load(d,split,key):
 files=sorted(d.glob(f'{split}_???_{key}.npy'))
 if len(files)!=160:raise ValueError(f'{d} {split} {key}: require160 frames, got{len(files)}')
 return np.concatenate([np.load(f,allow_pickle=False) for f in files]).astype(np.float32)
for name in args.blocks:
 d=root/name
 if (d/'training.json').exists():raise FileExistsError('refuse to overwrite completed training')
 x,y,xv,yv=[load(d,*pair) for pair in [('train','x'),('train','y'),('validation','x'),('validation','y')]]
 teacher=np.load(d/'teacher-weights.npz',allow_pickle=False)
 last_query_only=name in ['gelu_44','gelu_62']
 last_indices=np.arange(160)*96+95
 if last_query_only:
  xv=xv.reshape(160,288,512)[:,287,:].copy();yv=yv.reshape(160,288,512)[:,287,:].copy()
 scale=np.maximum(y.std(0),max(float(np.median(y.std(0)))*.1,1e-6));rng=np.random.default_rng(888+int(name.split('_')[-1]));start=time.monotonic()
 def evaluate(s):
  q={k:v.astype(np.float16).astype(np.float32) for k,v in s.items()}
  if not all(np.isfinite(v).all() for v in q.values()):raise ValueError('FP16 weight overflow')
  loss=se=0.;mx=0.
  for i in range(0,len(xv),1024):
   e=activation(xv[i:i+1024]@q['w1']+q['b1'])@q['w2']+q['b2']-yv[i:i+1024]
   se+=float(np.sum(e.astype(np.float64)**2));loss+=float(np.sum((e/scale).astype(np.float64)**2));mx=max(mx,float(abs(e).max()))
  return dict(scaled_mse=loss/yv.size,rmse=(se/yv.size)**.5,max_abs=mx)
 def save(s):np.savez(d/'student-best.npz',**{k:v.astype(np.float16) for k,v in s.items()})
 raw={k:teacher[k].astype(np.float32) for k in ['w1','b1','w2','b2']};best=evaluate(raw);save(raw);history=[dict(step='untrained_swish',**best)];best_step='untrained_swish'
 ix=(np.concatenate([rng.choice(last_indices,6144,replace=True),rng.choice(len(x),6144,replace=False)]) if last_query_only else rng.choice(len(x),min(12288,len(x)),replace=False));w1=raw['w1'].astype(np.float64);b1=raw['b1'].astype(np.float64);h=activation(x[ix].astype(np.float64)@w1+b1);mean=h.mean(0);hs=np.maximum(h.std(0),1e-12);z=(h-mean)/hs;yc=y[ix].astype(np.float64)-(h@raw['w2']+raw['b2']);ym=yc.mean(0);gram=z.T@z/len(z);gram.flat[::1025]+=1e-5;coef=np.linalg.solve(gram,z.T@(yc-ym)/len(z));delta=coef/hs[:,None];w2=raw['w2']+delta;b2=raw['b2']+ym-mean@delta
 s={k:v.astype(np.float32) for k,v in dict(w1=w1,b1=b1,w2=w2,b2=b2).items()};del h,z,yc,gram,coef
 e=evaluate(s);history.append(dict(step=0,**e));print(name,0,e,flush=True)
 if e['scaled_mse']<best['scaled_mse']:best=e;best_step=0;save(s)
 mom={k:np.zeros_like(v) for k,v in s.items()};var={k:np.zeros_like(v) for k,v in s.items()}
 for step in range(1,args.steps+1):
  idx=(np.concatenate([rng.choice(last_indices,256),rng.integers(len(x),size=256)]) if last_query_only else rng.integers(len(x),size=512));a=x[idx];pre=a@s['w1']+s['b1'];sig=sigmoid(beta*pre);h=pre*sig;error=h@s['w2']+s['b2']-y[idx];dy=2*error/(scale*scale*error.size);dh=(dy@s['w2'].T)*(sig+beta*pre*sig*(1-sig));grads={'w2':h.T@dy,'b2':dy.sum(0),'w1':a.T@dh,'b1':dh.sum(0)};lr=2e-5*(.1+.9*.5*(1+np.cos(np.pi*step/args.steps)))
  for k,g in grads.items():
   mom[k]*=.9;mom[k]+=.1*g;var[k]*=.999;var[k]+=.001*g*g;s[k]-=lr*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
  if step%200==0 or step==args.steps:
   e=evaluate(s);history.append(dict(step=step,**e));print(name,step,e,flush=True)
   if e['scaled_mse']<best['scaled_mse']:best=e;best_step=step;save(s)
 (d/'training.json').write_text(json.dumps(dict(block=name,train_seed=713,validation_seed=1777,train_frames=160,validation_frames=160,train_tokens_per_frame=96,validation_tokens_per_frame=(1 if last_query_only else 288),last_query_only_observable=last_query_only,train_last_token_sampling_fraction=(0.5 if last_query_only else None),steps=args.steps,batch=512,activation_beta=beta,ridge_prior='fit residual around original teacher w2/b2',original_acceptance_or_road_data_used=False,selected_step=best_step,history=history,seconds=time.monotonic()-start),indent=2))
 del x,y,xv,yv,s,mom,var,raw
