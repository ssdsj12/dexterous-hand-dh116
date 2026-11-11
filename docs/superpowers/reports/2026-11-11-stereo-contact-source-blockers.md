# TASK 5 PhysX 接触来源与柔性后端审计

本轮复核了服务器现有 Isaac Sim 4.5 安装中的 `omni.physics.tensors/impl/api.py` 和 `omni/physx/bindings/_physx.pyi`。文件 SHA-256 分别为 `bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b` 与 `b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9`。审计用文本和已保存的原生记录完成。

| 通道 | 已确认的接口和证据 | R2 资格缺口 |
|---|---|---|
| `get_contact_data(dt)` 与 `get_friction_data(dt)` | 参数 `dt` 将 impulse 换算为 force。历史运行传入实际物理子步的 `dt`。原始记录保留点、法向、分离、传感器与 filter 的 buffer 所有权。 | 数值通道按当前配置报告在手传感器侧，world 向量和 N 单位可按 SDK 接口读回。逐接触作用侧、相反符号及具体 Wire collider ID 尚缺独立校验。 |
| `get_full_contact_report()` 与 callback | 安装绑定定义 raw impulse、actor/collider、点和 friction anchor。已保存的 `post_step16_v01` 含 128 次 poll、128 次 callback 回执。 | 全部回执的 pair header、点和 anchor 均为空。无法取得非零逐对冲量、确认单位和方向，或据此证明无接触。 |
| 台面、支架和额外载荷 | 已有辅助观察入口；本轮重读的环境帧缺合格 passive support 覆盖、外部载荷和 attachment 的闭合证明。 | 缺少实际非零被动表面反力、来源完整性、无外力和无附着的联合验证。空或零 force 不能授予该资格。 |

使用独立脚本对保存的 480Hz 原始记录重读，保留 `run_uuid/env_id/episode_uid/physics_step/sdk_step/time/dt` 与每个 slot 的原始内容。未能确认的 partner、raw impulse、支撑和 attachment 保存为 `null`。记录位置为 `artifacts/stereo-reliability-20261111/contact/`，输入均为以前的实际原生采集。

| 保存的运行 | 物理帧 / 环境帧 | 法向 slot / 非零 | 摩擦 slot / 非零 |
|---|---:|---:|---:|
| `post_step16_v01` | 128 / 1,024 | 7,168 / 0 | 7,168 / 0 |
| `values180_v01` | 1,440 / 11,520 | 82,362 / 31,556 | 82,583 / 37,027 |

两组 Tensor buffer 所有权解析均覆盖所列环境帧。上游 overflow absence 的 attestation 均为 0，所有这些环境帧的 passive support 和外部载荷字段仍未知。180 步样本可复查非零力，先前独立报告也确认了同一批记录的时钟与 saved-value replay，仍缺作用侧/符号、逐点 partner ID 与 support 资格。**R2 和 Force Proxy 继续关闭。**

本轮两次有界 Teacher 诊断均使用 `rigid_proxy/solid_cylinder`。现有 `DeformableRodStateProvider` 只读和校验节点状态。`vendor/DeformX` 是既有 PyElastica/co_sim 源树，上一轮查不到可访问的目标 CUDA solver 提交及完整子模块。服务器本轮再查标准 `/usr/local/cuda-12.8/bin/nvcc` 路径，该文件仍缺失。没有可审计的新版求解器二进制或双向耦合记录。所需证据是固定实际求解器构建、同一物理步可复核的杆到手与手到杆反力、分辨内部力与外力，以及可验证的单位、力矩原点、同步与动量收支。相关源码审计见[安装及源码记录](2026-11-10-deformx-server-source-access.md)。
