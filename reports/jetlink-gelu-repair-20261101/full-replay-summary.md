# ExactGelu full replay comparison

- Frames: 16; raw outputs all exactly 73,808 bytes (18,452 float32 values) and finite.
- Original `verify_parity.py` exit code: 1 (thresholds unchanged).
- Global: MAE 0.0064600774, RMSE 0.062199325, max abs 3.8750076, correlation 0.99989907.
- Failed heads: FAIL: 3 slice(s) below corr 0.999 or past their error bound, whole or in a column: lane_lines pooled, lead pooled, plan pooled

## Per-head summary

| Head | Corr | MAE | Max abs | RMSE |
|---|---:|---:|---:|---:|
| desire_pred | 0.999719 | 0.0698388 | 0.21875 | 0.0861185 |
| meta | 0.999709 | 0.0534168 | 0.273437 | 0.0685461 |
| plan | 0.999936 | 0.0233855 | 1.42188 | 0.0699635 |
| wide_from_device_euler | 0.999956 | 0.0210761 | 0.125 | 0.0357928 |
| lead | 0.999867 | 0.262719 | 3.87501 | 0.650426 |
| lane_lines_prob | 0.998575 | 0.122297 | 0.53125 | 0.160881 |
| road_transform | 0.999993 | 0.0117742 | 0.0566409 | 0.018685 |
| action | 0.999897 | 0.0168303 | 0.0937502 | 0.0247904 |
| road_edges | 0.999774 | 0.0313475 | 0.378907 | 0.0544868 |
| lane_lines | 0.999713 | 0.0278163 | 0.324219 | 0.0475861 |
| pad | n/a | 0 | 0 | 0 |
| desire_state | 0.999690 | 0.462191 | 1.32813 | 0.562209 |
| pose | 0.999976 | 0.00943036 | 0.127441 | 0.0204099 |
| lead_prob | 0.998812 | 0.308395 | 0.509765 | 0.327652 |
| hidden_state | 0.999999 | 0.00146688 | 0.023437 | 0.00204207 |

## QNN context node counts

- EPContext nodes: 37; non-EPContext nodes: 36.
- `jetlink.precision::ExactGelu`: 36

Raw comparator stdout and exact exit code are in this directory.
