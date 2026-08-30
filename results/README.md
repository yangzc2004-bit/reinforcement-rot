# results/

二进制结果文件（.npy 数据、.png 相图）体积较大，不进 git，保存在本地 `output/` 目录：

- `l2_phase_results.npy`：第一版相图扫描（35 格 × 16 seeds）
- `l2_verify_results.npy`：复验扫描（Task A 加密相图 1152 runs + Task B 参数扰动 648 runs）
- `L2_信息素相图.png` / `L2_相图复验.png`：相图可视化

重新生成：`python experiments/sweep_phase.py` 与 `python experiments/verify_robust.py`。
