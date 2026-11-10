"""Compare fixed QDQ block candidates on held-out long-history activations."""
import argparse,json
from pathlib import Path
import numpy as np
from quantize_block import session,run,metrics
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--candidate',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=np.load(a.data,allow_pickle=False);x=d['x_validation'];y=d['y_validation'];report={'scope':'held-out synthetic block only, not whole-model parity','seed':1777,'frames':len(x),'cases':[]}
for c in a.candidate:
 s=session(Path(c));z=run(s,x);del s
 case={'candidate':c,'overall':metrics(z,y),'bins':[{'frames':[i,min(i+32,len(x))-1],**metrics(z[i:i+32],y[i:i+32])} for i in range(0,len(x),32)]};report['cases'].append(case)
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(case),flush=True)
