"""Read-only value-based structural signatures; equality is not device validation."""
import hashlib,json,sys
from pathlib import Path
import onnx
sys.setrecursionlimit(10000)
base=Path('/workspace/shared/jetlink-distillation/real-block');m=onnx.load(base/'teacher-capture.onnx',load_external_data=False)
w={t.name:t for t in m.graph.initializer};p={o:(n,j) for n in m.graph.node for j,o in enumerate(n.output)};ins={x.name for x in m.graph.input};cache={};regions={}
def H(parts):
 h=hashlib.sha256()
 for a in parts:
  a=a if isinstance(a,bytes) else str(a).encode();h.update(len(a).to_bytes(8,'little'));h.update(a)
 return h.hexdigest()
def sig(name):
 if name in cache:return cache[name]
 if name in ins:s=H(['input',name])
 elif name in w:
  t=w[name];attrs={x.key:x.value for x in t.external_data}
  if 'location' in attrs:
   key=(attrs['location'],int(attrs.get('offset',0)),int(attrs['length']))
   if key not in regions:
    h=hashlib.sha256()
    with (base/key[0]).open('rb') as f:
     f.seek(key[1]);left=key[2]
     while left:
      b=f.read(min(left,1024*1024));assert b;h.update(b);left-=len(b)
    regions[key]=h.hexdigest()
   content=regions[key]
  else:
   c=onnx.TensorProto();c.CopyFrom(t);c.name='';content=hashlib.sha256(c.SerializeToString()).hexdigest()
  s=H(['weight',t.data_type,list(t.dims),content])
 elif not name:s=H(['missing_optional'])
 else:
  n,j=p[name];s=H([n.domain,n.op_type,j,*[a.SerializeToString() for a in n.attribute],*[sig(i) for i in n.input]])
 cache[name]=s;return s
if __name__=='__main__':
 pairs=[('layer_norm_49','layer_norm_72'),('linear_99','linear_140'),('layer_norm_53','layer_norm_76'),('layer_norm_57','layer_norm_80'),('layer_norm_61','layer_norm_84')]
 r=[{'a':a,'b':b,'same_value_graph':sig(a)==sig(b),'a_signature':sig(a),'b_signature':sig(b)} for a,b in pairs]
 Path(__file__).with_name('sharing-audit.json').write_text(json.dumps(r,indent=2));print(r)
