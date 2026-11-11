# 371D 状态监督残差和校准证据完整性

Ubuntu CPU 验证以 `5a7eeb41c3d2f4efed98541ad76c375f9a1ee1d2` 为基线。

- `_prediction_loss()` 拒绝由有限输入溢出的非有限损失或残差。已有掩码仍用 NA 排除没有监督的坐标。溢出无法被评估过程当作缺标签跳过。
- `calibrate_residuals()` 验证名义覆盖率和所有绝对残差。拒绝 NaN、Inf 和负残差，即使这些值会被组内最大值掩盖。校准继续使用独立验证组最大残差及原有 split-conformal 秩。
- `_formal_metrics()` 为各指标保留单位、验证组数、测试组数、校准半径、实测测试组覆盖率和无法校准的原因。某坐标只在训练或测试集有监督时，验证组数为 0，校准半径和实测覆盖率为 NA。已有验证数据但组数不足时记录独立原因。

## 验证

新增 5 项 CPU 用例在旧实现全部失败，分别复现 float32 计算溢出、验证集缺某坐标时遗漏报告，以及 3 种无效残差。修复后以下 CPU 回归为 **104 passed**。

```sh
PYTHONPATH=src /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_state_supervision.py \
  tests/test_dh116_stereo_state_multicohort.py \
  tests/test_dh116_stereo_state_training_modes.py \
  tests/test_dh116_stereo_estimator.py \
  tests/test_dh116_stereo_runtime_task_state.py \
  tests/test_stereo_privileged_state_contract_audit.py
```

本机私有日志位于 `/tmp/dh116-wp2-metrics-20261111/`。RED SHA-256 为 `6402cd85e0b2fc29fa3d9418372abcd38af00a8469996b7e8ce3eafa16b96c8c`。GREEN SHA-256 为 `6acaee0509717fe26480e43aac7ee8baf4ac9c5b8414891b0fc7982ed826d9f0`。已重算 528 个受保护源文件和两份历史 Checkpoint 的哈希。

本次原生实验和正式训练数量均为 0。下一步需经批准获得更多同步原生独立初态，然后按各字段的来源资格重跑监督和分组校准。既有两组候选只支撑 Center XYZ 小规模诊断。
