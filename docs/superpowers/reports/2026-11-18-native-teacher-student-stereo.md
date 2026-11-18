# Teacher 与 Student 固定双目动作前对齐

## 已接入

新版无相机 Teacher 的 `NativeTeacherCollector.attach_student_stereo` 增加显式可选的旁路。它接收同一仿真中已创建的两台固定 Student 相机，并核验当前 Student 标定 SHA。图像采集发生在策略动作采样之前。无相机 PPO 的默认入口、特权输入和更新保持原样。

- 绑定运行、环境、episode、Reset、杆、策略步、物理步及状态 SHA。分别保存 SDK 实际时间与杆规范时间，允许实际 SDK 计数的小数精度差异。
- 输出左右 RGB 原始 PNG、图像 SHA、标定和渲染时钟，使用真实来源的两只掌位、同钟 41 个杆节点，在双掌坐标和两台相机的光学坐标中复算几何。投影落在画面内只说明在视锥范围内。
- 保存 371 槽特权账本中已测得的值和有效掩码。未知字段为 JSON null。真值包用于 Student 监督证据，图像没有进入 Teacher PPO 的 Actor 或 Critic 字段。
- 在原账本用同一个策略 token 和状态 SHA 关联采样决策及实际下发的 24D 动作记录。图像时间和物理时间仍以原始字段保存，墙钟耗尽后窗口立即停止并保留所读帧。
- 正常关闭时先释放双目读回，再释放 Teacher owner。旁路本阶段只覆盖显式绑定的 episode，后续 episode 需要单独挂接。

## 验证边界

CPU 夹具覆盖世界、掌和相机坐标、不同 episode 与陈旧渲染拒绝、实际 SDK 时间舍入量级、未知账本字段、两个 RGB SHA、动作相同 token、Teacher Actor 输入隔离、过期退出和清理顺序。尚无新 Isaac 图像、真实动作、重投影误差或 Student 稳定性证据。

当前实际 PPO worker 尚待完整 Reset 准入与 L02 的生产装配。相机旁路接口接收同一原生场景已创建的相机。仍需在原生 worker 的独立采集模式进行装配与首帧复验，再收集可用的训练标签。

## CPU 回归

相关采集器、双目读回、身份校验与 Registry 共 **110 项通过，0 失败**。
