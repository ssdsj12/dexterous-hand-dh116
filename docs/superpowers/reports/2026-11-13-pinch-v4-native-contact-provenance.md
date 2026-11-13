# 左 Pinch v4 原生接触与来源复核

冻结试验 `5c0b71fd410718d76890e0e6f597a59970be607b`，GPU 3 单 Isaac，1 环境，2 初始化加 238 测量，0 PPO。外层和 GDB inferior 均自然退出 0，无信号，72.086 秒。全部三份逐步账本各 238 行。退出阶段保留 136 个 stopped threads 的 GDB 证据。原生回执 SHA256 `5731ca089fa1b06afa0d41d27ac70dcbb9853c86a6e48982e068764a48c83e44`。

| 实测量 | 结果 |
| --- | ---: |
| 固定底座 source17 初始最小台面净空 | 1.965288 mm |
| locked5 初始最小台面净空 | 2.937077 mm |
| source17 / locked5 初始最小杆间隙 | 0.592581 / 0.702152 mm |
| 拇指 / 食指原始手杆 pair | 598 / 567 |
| 拇指 / 食指最大单 pair 法向载荷 | 0.268830 / 0.287106 N |
| 两指同时正法向载荷的求解器子步 | 567 |
| 杆与台面 SDF pair | 99,273 |
| 原始力归约独立 FP64 最大残差 | 1.492 µN |

首次手杆接触在 SDK 步 191，求解器子步 2。首次两指同子步载荷在 SDK 步 193，求解器子步 9。有效对向承托与连续材料锚 Slip 仍无合格来源。本轮无双手、Lift/Hold 成功或 Teacher PPO 更新。

## 手与台面来源

`table_contacts.jsonl` 每行报告零。原生日志对同一静态 `/World/PinchTable` 输出 17 条 GPU filter 不支持警告。原始零值仍可重放，物理接触数量与 absence 资格按未知处理。杆和台面的 DeformX pair 单独保留。旧 v3 日志也有同类警告。v4 已发布原生范围仅用于固定底座双指与杆的观测及自然退出。双手和 R0 均不能以此推断没有手与台面接触。

## 生命周期

明确归还 PHYSICS_READY 订阅，字段清除及原 `clear_instance` 都返回成功。所有冻结手/杆 owner 在关闭前后的弱引用均失效。SimulationContext 自身在 app.close 之后仍有两个直接引用，分别为 IsaacLab 构造帧和调用 acquisition lambda 的保留帧。观察到构造帧上一层是 PyTorch CUDA `_lazy_init`。API 归还、Python 引用和进程自然退出是三个独立观测，构造帧的最终持有源及 native release 未核定。

## 复核

私有产物重开和独立解析 SHA256 `f17ec028d74d2c8bc46eb8045603852847b710f4bc2c94c22ecad7e2b1087efe`。原后置审计 SHA256 `ee88bbdb22dc8fa2164f4ad2ce086400e99652d721f44bb32b40c09ae3663519`。38 项 CPU 测试通过，33.76 秒，日志 SHA256 `004dc6695f285bf9b41011ae5719255e97f82cbd265ae2c497f33470f9e61a8d`。复核命令为现有 Ubuntu launcher 下的 `scripts/audit_dh116_pinch_v4_sources.py --receipt <private-receipt> --output <new-output>`，测试为 `test_dh116_pinch_v4_source_provenance`、`test_dh116_pinch_v4_audit`、`test_dh116_pinch_force_precision`。

下一步使用已通过 2.1 mm 原手腕初态筛查的独立双手 Reset，在新的原生范围核对同杆、实际 SDK 关节/目标、动作、接触和关闭。通过短诊断后仍需验证手与台面来源、材料锚和真实奖励。
