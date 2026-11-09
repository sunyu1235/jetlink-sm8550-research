# Exact GELU and tanh GELU full replay comparison

- Parity result: PASS for all 16 frames, slices and columns; original thresholds unchanged (exit code 0).
- Raw per-frame files: 73,808 bytes each, 18,452 float32 values, all finite.
- Global: MAE 0.0018483124, RMSE 0.012868829, max abs 1.3750076, correlation 0.99999522.

| Head | Corr | MAE | Max abs | RMSE |
|---|---:|---:|---:|---:|
| desire_pred | 1.000000 | 0.00198212 | 0.00781345 | 0.00277724 |
| meta | 1.000000 | 0.00180948 | 0.011719 | 0.00279814 |
| plan | 0.999999 | 0.00287784 | 0.249996 | 0.00856743 |
| wide_from_device_euler | 1.000000 | 0.000369866 | 0.00390673 | 0.00105491 |
| lead | 0.999993 | 0.0407214 | 1.37501 | 0.141857 |
| lane_lines_prob | 0.999998 | 0.00421269 | 0.0234365 | 0.00555845 |
| road_transform | 1.000000 | 0.000348159 | 0.00390673 | 0.00108729 |
| action | 0.999998 | 0.00237177 | 0.00993347 | 0.00349196 |
| road_edges | 0.999999 | 0.00134773 | 0.0195308 | 0.00223603 |
| lane_lines | 1.000000 | 0.000882976 | 0.0178223 | 0.00167654 |
| pad | n/a | 0 | 0 | 0 |
| desire_state | 0.999998 | 0.0287478 | 0.0937481 | 0.0359691 |
| pose | 1.000000 | 0.000931012 | 0.00878918 | 0.00192638 |
| lead_prob | 0.999984 | 0.0276689 | 0.0703115 | 0.0306979 |
| hidden_state | 0.999999 | 0.00146688 | 0.023437 | 0.00204207 |

## Context model placement

- 45 EPContext nodes; 63 non-EPContext nodes.
- `jetlink.precision::ExactGelu`: 36
- `jetlink.precision::ExactGeluTanh`: 27

Comparator stdout and exact exit code are in `both-full-replay-compare`.
