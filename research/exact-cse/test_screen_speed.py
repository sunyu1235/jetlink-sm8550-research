import csv
from pathlib import Path
import tempfile
import unittest
from screen_speed import timing, summarize


class ScreenTest(unittest.TestCase):
    def row(self, mean):
        return {'mean_excluding_first_ms': mean}

    def test_gain_vs_both_baselines_needed(self):
        r = summarize(self.row(52), self.row(50), self.row(52.2), [25, 25.5, 25])
        self.assertEqual(r['screen'], 'PROMISING_REQUIRE_REVIEW_BEFORE_160')
        self.assertFalse(r['numerical_160_run'])

    def test_drift_or_thermal_confounds_rejected(self):
        for times, temperatures in [((52, 50, 49), [25, 25, 25]), ((52, 50, 52), [25, 30, 25]), ((52, 51.9, 52), [25, 25, 25])]:
            r = summarize(*(self.row(t) for t in times), temperatures)
            self.assertEqual(r['screen'], 'NOT_PROMISING_OR_CONFOUNDED')

    def test_invalid_csvs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'timing.csv'
            p.write_text('frame,execute_ms,nonfinite\n0,52,0\n1,nan,0\n')
            with self.assertRaises(ValueError): timing(p, 2)
            p.write_text('frame,execute_ms,nonfinite\n0,52,0\n2,51,0\n')
            with self.assertRaises(ValueError): timing(p, 2)
            p.write_text('frame,execute_ms,nonfinite\n0,52,0\n1,51,1\n')
            with self.assertRaises(ValueError): timing(p, 2)


if __name__ == '__main__':
    unittest.main()
