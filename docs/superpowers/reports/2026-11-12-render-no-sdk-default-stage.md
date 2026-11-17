# C V0 SDK 默认 stage 隔离

调度器实际执行的 C V0 无探针自建相机、渲染产品和 annotator。输出只有 `plan.json` 与 `app_close_returned.json`，记录零捕获和零物理步。实际持有共享实例锁，App.close 返回，外层进程经 43.818 秒以 `-11` 结束。日志 SHA256 为 `8a570ee5dd34b8084d556d94b679ce501c2eeea6e85aabdc85065063dd3e13d8`。其 faulthandler 显示解释器 GC，未取得原生帧。旧 V5 的 GDB 证据定位 AppWindow/Carb，不能用来给本轮 V0 指定同一对象。

## 确认的 SDK 默认行为

安装的 Isaac Sim 4.5 `simulation_app.py` 默认 `create_new_stage=True`，当前 V0 未覆盖该项。SDK 构造器调用 `utils.create_new_stage()`，该函数执行 `omni.usd.get_context().new_stage()`。本轮 Kit 日志还记录了默认视口的 HydraTexture 及其 `/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0` 路径。由此区分了无探针产品与无 SDK 默认 stage 或视口。实际初始化包含后者。

本次新增 `--no-sdk-default-stage`。V0–V3 显式传给 SimulationApp 的配置只增加 `create_new_stage=False`，其余 experience、应用关闭和 Orchestrator 设置照旧。该参数同时使 SDK 跳过等待活动视口的分支。此后 SDK 仍运行通用应用更新和 UI 准备。探针用布尔值读回默认 stage 是否存在，并检查实际配置值。它不保存 SDK stage 包装引用。若仍有默认 stage，结果失败且保留读回。V1–V3 仍可在验证之后自建独立 stage。V4 启动器不接受该开关。

## CPU 与下一轮命令

Ubuntu CPU 回归 21 项通过，新增回归验证单一配置差异。带开关的 V0–V3 CPU 预检通过，V4 拒绝此开关。本轮未启动 Isaac，下一轮外层退出与实际 stage 读回仍待验证。

| 证据 | SHA256 |
|---|---|
| Ubuntu CPU 回归 | `7a1e06dd573b016bbca2e8aa79ab2e1078cd66ab145c6a78c79ad4f6b440107a` |
| V0 CPU 预检 | `f971005068f7dbde0234d21dd36b396e1bb33b8bd1ded06a380ea2ada17e00b7` |
| 新 V0 调度配方 | `f36a600071b1f4412cf772edbb562a44b2cc704565970b6fb271ad1bd6d7ae60` |

调度配方 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/f6-f7-lifecycle-preflight-GcTvRw/V0-no-sdk-default-stage-command.json` 固定安装的 SDK 和 experience 摘要。产物校验要求 `sdk_default_stage.json` 中的配置和阶段读回、`app_close_returned.json`，以及调度器记录实际外层退出 0。旧 V0 失败保留，复试只使用新配置。
