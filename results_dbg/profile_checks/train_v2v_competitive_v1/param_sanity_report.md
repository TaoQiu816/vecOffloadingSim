# Parameter Sanity Report

## 1) 计算时延量级

- comp_time_vehicle = C / f_vehicle, C∈[5e+07,2e+08], f∈[1e+09,3e+09]
  - range: [0.01667, 0.2] s
- comp_time_rsu = C / F_RSU, F_RSU=6e+09
  - range: [0.008333, 0.03333] s

## 2) 传输时延量级

- tx_time_v2i = D / (BW_V2I / users)
  - users=1: [0.06667, 0.2] s
  - users=6: [0.4, 1.2] s
  - users=12: [0.8, 2.4] s
- tx_time_v2v = D / R_v2v_typ
  - R_v2v_typ ≈ 3.5e+08 bps
  - range: [0.00286, 0.00858] s

## 3) 量级对比结论

- mid comp_time_vehicle ≈ 0.0625s, comp_time_rsu ≈ 0.02083s
- mid tx_time_v2i (users=6) ≈ 0.8s, mid tx_time_v2v ≈ 0.00572s
- 若 comp_time_rsu << comp_time_vehicle 且 tx_time_v2i 不高，RSU 将在成本上占优。

## 4) 让 V2V 有机会成为 argmin 的必要条件

- 在部分状态满足: (tx + wait + comp)_v2v < (tx + wait + comp)_rsu。
- 这通常要求：V2V 链路速率/距离优势 + 邻居排队不高 + RSU 排队或算力相对变弱。
