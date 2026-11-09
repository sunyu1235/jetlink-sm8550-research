# Phone vision output + PC FP32 policy replay

- Result: original parity comparator passed all 16 frames and all slices/columns (exit code 0; thresholds unchanged).
- CPU policy inputs: phone hidden state slice `[2066:18450]`, reshaped to `[1,32,512]`; `desire`, `traffic_convention`, `action_t` from the same `parity-varied` frames. Policy recurrent states began at zero and recursively used only its own previous FP32 state outputs.
- Compared against `parity-varied/out_ref_i.npy`, which independently recurs the original whole model state.
- Global: MAE 0.0014510526, RMSE 0.003022398, max abs 0.3066864, correlation 0.99999973.

| Head | Corr | MAE | Max abs | RMSE |
|---|---:|---:|---:|---:|
| desire_pred | 1.000000 | 0.00132152 | 0.00564671 | 0.00170047 |
| meta | 1.000000 | 0.00125053 | 0.00652933 | 0.00162141 |
| plan | 1.000000 | 0.000971473 | 0.0614624 | 0.00322275 |
| wide_from_device_euler | 1.000000 | 0.000341754 | 0.0018816 | 0.000610196 |
| lead | 1.000000 | 0.0069081 | 0.306686 | 0.0247062 |
| lane_lines_prob | 1.000000 | 0.0021101 | 0.00769806 | 0.00259554 |
| road_transform | 1.000000 | 0.000474474 | 0.00292683 | 0.000808776 |
| action | 1.000000 | 0.000501101 | 0.0028739 | 0.000855932 |
| road_edges | 1.000000 | 0.000826027 | 0.00661993 | 0.00114323 |
| lane_lines | 1.000000 | 0.000651252 | 0.00624561 | 0.000913372 |
| pad | n/a | 0 | 0 | 0 |
| desire_state | 1.000000 | 0.00790463 | 0.0318832 | 0.0103447 |
| pose | 1.000000 | 0.000766526 | 0.00989413 | 0.00151473 |
| lead_prob | 0.999998 | 0.00427604 | 0.0126553 | 0.00524953 |
| hidden_state | 0.999999 | 0.00146688 | 0.023437 | 0.00204207 |
