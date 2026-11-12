# F6 headless rendering experience 对照

当前 Isaac Sim 4.5 默认 `isaacsim.exp.base.python.kit` 引入 `isaacsim.exp.base` 的应用菜单、浏览器、属性面板和视口菜单。已安装的 IsaacLab 4.5 camera headless experience 引入最小 headless、Replicator、材质库和 RTX 视口。它经 `omni.ui` 仍有 `omni.appwindow` 依赖。本轮测试缩小窗口对象来源的可能性，维持原生采集和应用关闭代码。

使用 `/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit`，SHA256 `978bd89e90f2ca00b32805a70dd0a15c5b8b48edeb42cd042f8eb679d33fa4ea`。派生入口仅在显式传入此文件时选择该 experience。启动参数保持有效 Orchestrator 为 true、首选项持久化为 false，与默认工作流一致。安装的 Replicator 在 `Orchestrator/enabled=false` 时会跳过 tick graph，启动后读回必须显示有效 capture。每轮使用同一 prepared manifest、seed 42、frame 0、1 个相机、1×1 累积、`fast_shutdown=False` 和自然退出。未修改 Isaac 安装、上游生成器或资产。

## Drop variant 0 对照

`f6-native-minimal-experience-KzMrXf` 从 09:21:37 UTC 至 09:22:21 UTC。Replicator 1.11.35、Orchestrator true、capture ready、首选项持久化 false。默认运行记录 300 个扩展启动条目，headless camera 记录 145 个。GUI、菜单、窗口和视口名字匹配中有 26 个默认独有条目。两个 experience 均加载 `omni.appwindow-1.1.9`。

读取了 1024×1024 RGB、uint32 Native ID、径向距离和 composed stage。两个真实线缆 ID 均被捕获，798,833 个有限正深度像素。独立 OpenUSD 读回 Z-up、1 m/单位、0 条未解析依赖。App.close 返回后，进程在解释器 GC 时仍 SIGSEGV，外层退出 139。缩减窗口依赖后仍出现同类退出症状，这一轮未在调试器下确认具体 C++ 帧。

| 证据 | SHA256 |
|---|---|
| Drop composed stage | `c0e6cf387dd45a02a87052fee9e9598a44bf79eff0b0e1f4f87fffd7f938eccb` |
| Drop launcher | `1d5a9a74c1baa3b1b7827174a3c6edc035f71600dd1f273da85b0727b266e507` |
| Drop 独立核验 | `f090ec94d222babf88feeda5a67f572d1fd205c0ffc9e8e431214b2dcff33659` |
| Ubuntu CPU 回归 20 通过、0 失败 | `e3a9f6b7865365dd7044232505d11dbda1f629eec5ce23bb2f73a2aabfd77d4c` |

复现时在 [F6 原生冒烟命令](2026-11-12-deformx-render-smoke-f6.md#复现) 中加入 `--headless-rendering-experience "$kit"`。不传该参数时保留原始 SimulationApp 行为。CPU 测试分别核验固定摘要、Orchestrator 和默认配置。派生工作包当前 `native_render_verified=false`。

## 单独授权的 hang variant 3 资产诊断

在 drop 进程退出且 GPU 释放后，单独运行 hang。目录 `f6-hang-artifact-8Ullmn`，09:23:38–09:24:20 UTC。入口和 experience 文件 SHA 与 drop 逐字节相同。只运行 variant 3、frame 0，一次，无自动重试。该轮单独用于验证 hang 渲染资产。

| 核验项 | 实测 |
|---|---|
| 采集产物 | 1024×1024 RGB、径向距离、uint32 Native ID 和映射、composed stage |
| 可见线缆与距离 | 两个线缆 ID，27,937 个有限正径向距离像素 |
| 原生合成场景 | Z-up、1 m/单位、一个相机、两条 2,652 顶点线缆、有效 DomeLight |
| 独立 OpenUSD 重读 | 0 条未解析依赖 |
| 外层生命周期 | App.close 返回后解释器 GC 发生 SIGSEGV，退出 139 |
| Composed stage SHA256 | `93a7630a6e5b503cd4307aaafff1b2bd5d45d528060c0b47c200855433377b2f` |
| Launcher SHA256 | `c18537b09cd2d02300ae634846ea0ab53dec3cae89511d36c3e6ac12e4974553` |
| 独立核验 JSON SHA256 | `b86066e677ddb57387317ba1faf974935470c1ee311317492c29ca207a86910e` |

Drop 和 hang 均有实际原生渲染文件，两个应用的解释器退出仍失败。保留原始证据及 `native_render_verified=false`。这些证据只用于 F6 资产，柔性物理、逐指接触与抓取资格由各自工作包验证。
