# 现有 solid_cylinder 的刚体状态监督

基线 `75d337e8c2c5c7a21d60e96992b41df2ef7cc185`。本包沿用现有环境、模型结构和 371D 字段合同。

## 数据来源与监督范围

- `read_candidate` 从已绑定 SHA 的原始物理账本读取当前策略边界的 SDK 读回。刚体根速度验证要求 `solid_cylinder` 任务身份及摘要与同回合原生头一致，同时匹配 env、episode、reset、物理 tick、时刻和相机标定。
- `_rigid_motion_observed` 使用同回合连续的两帧有效双目观测和各自已核对的可见目标支持。控制步相差 1、物理步相差 8、时间相差 1/60 秒，且上一执行边界闭合。根速度目标来自同钟 `formal_object_velocity`。材料 track ID 不参与此目标的资格判断。
- 可用线速度维度为原 68–70，世界轴单位 m/s，原归一化除数 0.1。误差报告可读回 MAE、RMSE、P95 的原始物理单位。61–63 继续使用物体根中心减左右腕原点均值，再除以 0.05 m。姿态和角速度的数值读回不解除圆柱对称性造成的可观测性限制。
- 物体姿态 64–67、角速度 71–73、柔性形变、接触、滑移和 Force Proxy 的候选资格在当前回放中仍为空。原生参考和 SDK 读回仅参与监督核验，Actor 输入与部署门槛沿用现有路径。

## 已执行验证

只读回放此前原生候选与原始账本，候选 SHA `4da4f234db2ff77c505d100366bcf30b5c9b2d007f48cfd5b132c1e5a0f8cd70`，账本 SHA `da81d600f58864b059cde81131353ce6384be5e84f53bc6446a94bb7b00c6929`。8 行候选、2 个独立初态与来源组。61–63 各有 6 行可用监督，68–70 各从 0 增至 6 行。该批目标接近静止，最大的线速度分量约为 4.96e-8 m/s。正式划分状态为 `blocked_insufficient_independent_groups`，仍需要独立的动态初态和持出组。

在 Ubuntu 运行相关 CPU 回归，72 项通过，0 失败。测试覆盖同钟绑定、reset/终止排除、原始单位校验、没有材料 track 的连续观测、圆柱角速度保持屏蔽，以及冷重置、独立组划分和运行时状态输入隔离。新增测试在实现前出现 3 项预期失败。最终日志 SHA256 为 `6ec8e44673c9482ac6830b8b889098ad36e9b819b5589356a6ffa6e3d66399a5`。

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_state_training_modes.py \
  tests/test_dh116_stereo_state_supervision.py \
  tests/test_dh116_stereo_cold_candidate_admission.py \
  tests/test_dh116_stereo_state_multicohort.py \
  tests/test_dh116_stereo_runtime_task_state.py
```

本包仅核验标签入口和 CPU 回归。历史数据保留原件，未启动 Isaac、采集或训练。528 个保护文件及两个历史 Checkpoint 核验通过。
