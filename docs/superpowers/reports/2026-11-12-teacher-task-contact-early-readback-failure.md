# 刚性 Teacher GPU 任务首次 reset 前启用报告读回的失败

2026-11-12 06:22:05–06:22:26 UTC 在 Ubuntu GPU 3 单实例执行现有任务的短探针，冻结源码差异 SHA256 为 `a91d2ffad5597e18169196993adb2e8aa2edf2a06a009b5882dd8123859f82ae`。首次 reset 前将 `/physics/suppressReadback` 设为 false 后，原环境 `GpuWorld.reset_slots → Articulation.reset → wrench_composer.reset` 抛出 `Index array device (cuda:0 does not match data array device (cpu))`。外层退出码 2，采集 **0 个策略步及 0 条物理源记录**。保留了原 launcher log 和 scene summary，成功池与原物理设置保持原合同。

| 原始证据 | SHA256 |
|---|---|
| launcher log | `b01446be810946f6f599ddbf5ec15692fc15815c34f1adcad85ad0905c914367` |
| scene summary | `77e0bf1fbb41fabe4760b9080b68877e6c4759fd2a790756462c912183a2c57d` |

前一次在 reset 后切换设置时，任务 GPU view 正常运行但结构化 headers 为 0，见 [晚开启探针](2026-11-12-teacher-task-contact-probe.md)。该组合证据表明当前设置不能用于原 GPU 任务的既有启动链。独立左右拇指标定仍有真实报告证据。接下来应验证现有 GPU Tensor filter actor 路径和物体侧反向视图对外部支撑的覆盖，不能用 CPU index 重写或修改原物理以跳过失败。

探针恢复采用已运行的 post-reset 默认行为。已知失败的 pre-reset 设定需要显式选择。新增逐环境动量比较保持 contact 冲量和 friction anchor 两路独立，分别比较作用侧假设，输出仍为未资格化。回归 74 项通过，0 失败，覆盖模式显式选择、actor 顺序、空报告残差与原关闭合同。新版动量记录待独立原生复验。
