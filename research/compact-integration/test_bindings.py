import unittest
from types import SimpleNamespace as V
import numpy as np
from compact_bindings import ORDER,SHAPES,ALIASES,bind,validate_metadata
class Tests(unittest.TestCase):
 def fresh(self):return {ALIASES[n]:np.zeros(SHAPES[n],np.float16) for n in ORDER}
 def test_borrow_no_pad(self):
  x=self.fresh();y=bind(x);self.assertEqual(sum(v.nbytes for v in y.values()),9216)
  for n in ORDER:self.assertIs(y[n],x[ALIASES[n]])
 def test_legacy_shape_rejected(self):
  x=self.fresh();x[ALIASES[ORDER[0]]]=np.zeros((1,288,512),np.float16)
  with self.assertRaises(ValueError):bind(x)
 def test_wrong_dtype(self):
  x=self.fresh();x[ALIASES[ORDER[0]]]=x[ALIASES[ORDER[0]]].astype(np.float32)
  with self.assertRaises(ValueError):bind(x)
 def test_nonfinite(self):
  x=self.fresh();x[ALIASES[ORDER[0]]][0,0,0]=np.nan
  with self.assertRaises(ValueError):bind(x)
 def test_noncontiguous(self):
  x=self.fresh();x[ALIASES[ORDER[0]]]=np.zeros((1,1,1024),np.float16)[...,::2]
  with self.assertRaises(ValueError):bind(x)
 def test_names(self):
  x=self.fresh();x['other']=x.pop(ALIASES[ORDER[0]])
  with self.assertRaises(ValueError):bind(x)
 def test_session_contract(self):
  up=[V(name=ALIASES[n],shape=list(SHAPES[n]),type='tensor(float16)') for n in ORDER];dn=[V(name=n,shape=list(SHAPES[n]),type='tensor(float16)') for n in ORDER];validate_metadata(up,dn);up[0].shape[1]=288
  with self.assertRaises(ValueError):validate_metadata(up,dn)
if __name__=='__main__':unittest.main()
