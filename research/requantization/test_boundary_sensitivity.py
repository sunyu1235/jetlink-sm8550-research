import unittest
import numpy as np
from boundary_sensitivity import calibrate, qdq, groups, parity, ROOT

class QuantizationTests(unittest.TestCase):
    def test_zero_exact(self):
        for bits in (8,12,16):
            e=calibrate([-7,0,3],bits)
            y,c=qdq([0],e)
            self.assertEqual(float(y[0]),0)
            self.assertEqual(c,0)
    def test_roundtrip_error_bound(self):
        x=np.linspace(-7,3,10001)
        for bits in (8,12,16):
            e=calibrate(x,bits)
            y,c=qdq(x,e)
            self.assertEqual(c,0)
            self.assertLessEqual(np.max(np.abs(y-x)),e['scale']/2+1e-6)
    def test_clipping(self):
        e=calibrate([-1,1],8)
        _,c=qdq([-10,0,10],e)
        self.assertEqual(c,2)
    def test_all_zero(self):
        e=calibrate(np.zeros(10),8)
        y,c=qdq(np.zeros(10),e)
        self.assertTrue(np.array_equal(y,np.zeros(10)))
        self.assertEqual(c,0)
    def test_bad_calibration(self):
        for x in ([],[np.nan],[np.inf]):
            with self.assertRaises(ValueError):calibrate(x,8)
        with self.assertRaises(ValueError):calibrate([0],7)
        with self.assertRaises(ValueError):calibrate([0],8,.5)
    def test_nonfinite_evaluation(self):
        with self.assertRaises(ValueError):qdq([np.nan],calibrate([0],8))
    def test_groups_cover_exactly(self):
        s=parity.ModelSpec.load(ROOT/'reports/jetlink-gelu-optimization-20261101/partition-tuning-20261109/last-query-full160/compare/spec.json')
        for g in ('tensor','head','column'):
            v=np.concatenate(list(groups(s,18452,g).values()))
            self.assertTrue(np.array_equal(np.sort(v),np.arange(18452)))

if __name__=='__main__':unittest.main()
