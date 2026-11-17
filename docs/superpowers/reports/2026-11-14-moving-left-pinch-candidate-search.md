# 使用实际闭合左手数据的独立候选搜索

状态：CPU工具和来源合同通过；本次候选几何**失败**，不得进入原生或Teacher。新原生0，PPO0。没有扩大搜索预算或重跑原生。

## 修改与数据来源

新增 `scripts/solve_dh116_measured_left_pinch.py`，复用既有非线性FK/SDF及有界优化器，不改原控制器。`load_packet` 使用已资格化短诊断的完整232行，只从同一episode SDK211–234提取相关场景，要求实际左手手指动作非零；`validate_packet` 拒绝保持不动的左手标签、跨episode或非有限读回。

实际来源回执SHA `04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3`，物理账本SHA `9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4`。原始数据只读，不复制进Git。

搜索保持原手指闭合增量，参数限于独立Reset/终点提议，复用原CPU候选范围（平移±2mm、旋转向量各±0.03rad、前三主动关节增量最大±0.04rad并受实际耦合余量约束）。这不更改执行器限制或运行时动作幅度。所有历史实测误差迁移到新提议都标为未资格化预测，不能作为实际接触。

## 实际运行

- 原始实际杆场景下，初始最坏约束余量为-0.127142mm（左食指区域指标转换得到的规划余量；**不是穿透深度**）。
- 最多600次查询/60秒搜索。本次实际600次、43.8516s，达到查询预算，未重试。
- 包含来源读取、基线24场景和全场景验算，共86.7492s CPU墙钟。
- 最优候选仍未通过：最坏余量-0.011208mm。

| 未通过的独立约束 | 边界 | 余量mm |
|---|---:|---:|
| 原source finger13桌面净空 | 232 | -0.011208 |
| 左食指原有效区域 | 212 | -0.011087 |
| 原source finger13桌面净空 | 229 | -0.010967 |
| 名义source finger13桌面净空 | nominal | -0.010552 |
| 非目标source finger13对杆间距 | 211 | -0.010433 |
| 名义拇指原有效区域 | nominal | -0.009665 |

这些是相应原约束的剩余余量；负值不能因数值小而放行。桌面项相对既有安全净空，并不等于已发生0.011mm物理穿透。范围内找到的候选尚不能同时兼顾区域、非目标杆碰撞和原桌面净空。没有证明范围外可行，也没有证明无解。

结果SHA `596a7c94b33ea9ab3c1472cddb45152c17a52ff7d1d5cdceeafa99cc62ec0fb5`。全部600次查询与失败候选留在Ubuntu私有目录 `projected-pinch-20261114/moving-left-search-01/`。源码SHA `4cc7684ef6152ff1697a669306960fb27d21528c2f25e963a313f817c90d03ec`。

## 测试与资格

新增 `tests/test_measured_left_pinch_candidates.py`。先失败1 failed/4 errors（模块未实现）；实现后5 passed、0 failed，42.94s。日志SHA `4a4f887670dfae98d5050afbdf68fae425d6346a1530b402554d2300302b0ae4`。集成回归同时覆盖此前同钟几何分解，计数写入Registry。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
  bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_measured_left_pinch_candidates.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
  bash scripts/python.sh scripts/solve_dh116_measured_left_pinch.py --output NEW_PRIVATE_DIRECTORY
```

不生成新原生配方；本结果不能作为继续本版本原生的理由。继续独立R0稳定性测量及Teacher环境准备。下一几何工作须处理已识别的桌面/非目标指节/指腹区域约束冲突并提出明确改动，完整路径及实际原生通过前不授予接触资格。Pinch、Tripod、Power分别保留准入状态。
