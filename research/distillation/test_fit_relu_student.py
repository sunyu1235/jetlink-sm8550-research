import unittest
import numpy as np
from fit_relu_student import fit, predict, errors

class FitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.x = rng.normal(size=(400, 8))
        self.xv = rng.normal(size=(100, 8))
        self.w = rng.normal(size=(8, 12))
        self.b = rng.normal(size=12)
        self.w2 = rng.normal(size=(12, 4))
        self.b2 = rng.normal(size=4)
        self.teacher = lambda x: np.maximum(x @ self.w + self.b, 0) @ self.w2 + self.b2

    def test_recovers_representable_teacher(self):
        s = fit(self.x, self.teacher(self.x), self.w, self.b, 12, 1e-10)
        self.assertLess(errors(predict(s, self.xv), self.teacher(self.xv))['max_abs'], 1e-6)

    def test_compression_and_determinism(self):
        s = fit(self.x, self.teacher(self.x), self.w, self.b, 6)
        t = fit(self.x, self.teacher(self.x), self.w, self.b, 6)
        self.assertEqual(s['w1'].shape, (8, 6))
        self.assertEqual(s['w2'].shape, (6, 4))
        for k in s: np.testing.assert_array_equal(s[k], t[k])

    def test_constant_inputs(self):
        s = fit(np.zeros((50, 8)), np.ones((50, 4)), self.w, self.b, 6)
        np.testing.assert_allclose(predict(s, np.zeros((5, 8))), np.ones((5, 4)))

    def test_rejects_bad_inputs(self):
        for width, ridge in [(0, 1e-4), (13, 1e-4), (6, 0), (6, float('nan'))]:
            with self.assertRaises(ValueError): fit(self.x, self.teacher(self.x), self.w, self.b, width, ridge)
        bad = self.x.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): fit(bad, self.teacher(self.x), self.w, self.b, 6)

if __name__ == '__main__': unittest.main()
