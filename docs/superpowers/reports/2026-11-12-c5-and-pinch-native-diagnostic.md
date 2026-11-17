# DH116 C5 与自由杆左手 Pinch 单次诊断

两个批准配方在 GPU 3 串行执行，各启动一次。C5 使用修正后的 `c9a7e7…`，`94ffe…` 没有启动。单实例锁均有实际观测。两个进程已退出，未触发墙钟超时。实验没有产生完整 episode 或 PPO 更新。

| 配方 | 实际物理步 | 外层退出 | 用时 | 原生资格 |
|---|---:|---:|---:|---|
| C5 生命周期 | 0 | 1 | 89.258 s | 未通过 |
| 自由杆左手 Pinch | 3（2 初始化、1 测量） | 255，inferior SIGSEGV 11 | 33.042 s | 未通过 |

## C5

一次显式 App update 后，telemetry 模块存在，但 hook 容器为 `None`。探针按前置条件失败，`owner_release_applied=false`。`app.close()` 返回，Python 退出期间记录了 SIGSEGV。因为本次没有执行 holder 释放，无法用它判断释放是否能修复退出。

旧采集脚本把 `gdb.parameter('args')` 当作可读回的 `--args`。实际 GDB 读回为空，正常 finalizer 与 fatal 捕获均在找 `--output` 时抛出 `ValueError`。原回执的 `debugger_backtrace_captured=true` 来自打印的 fatal 标记，未生成完整 GDB 栈产物，inferior 退出状态也没有通过独立解析。保留三份 `/proc` 映射及原始日志，C5 本次崩溃的模块首帧仍未知。

修复将调度器 `launch_argv.json` 中的规范绝对输出路径绑定到 GDB 命令文件。只有采集成功才打印 GDB 完整 fatal 栈标记。在真实 GDB 的纯 CPU Python 夹具中，旧实现的正常／SIGSEGV 两例均失败。修复后两例均保存映射和栈，且可核对退出码／信号。相关回归 47 passed、0 failed、0 skipped。修复后的指纹只获 CPU 验证。当前 C5 运行配置因缺少目标 holder 暂停原生验证。

## 自由杆左手 Pinch

首个测量步触发已有 `mimic_error` 保护。原始 24D 动作、杆节点接触力、各 pad 与桌面求解器 wrench、施加到手的杆 wrench 均为零。SDK 关节位置最大单步变化约 0.480955 rad，读回角速度绝对峰值约 85.435 rad/s。只产生一行测量账本，无法评估持续接触、反作用力交换或抓取。

用原 USD 的指腹三角网格和测量前记录的真实 link 位姿重建，核对每个指腹的几何 SHA。四个指腹网格均有顶点位于桌面盒体内。各网格最低顶点相对桌面上表面的有符号高度依次为拇指 +2.401 mm、食指 −0.039 mm、中指 −17.058 mm、无名指 −41.153 mm、小指 −63.341 mm。这是源网格静态相交证据。实际 PhysX 接触对、求解器冲量及关节名顺序未保存，尚不能把单步跳变归因到某个接触对或关节。

退出阶段另有独立故障。`app.close()` 返回后，GDB 记录 SIGSEGV 11，并保存栈、线程栈、共享库和进程映射。首帧位于 `libomni.physx.tensors.plugin.so` 文件偏移 `0x59718`，随后经过 `_physicsTensors` 与 `_carb`。需要分别排查候选起始几何与 tensor 视图释放链。候选重建应先证明与桌面没有初始相交，并记录实际关节顺序和物理接触对。原有关节和材料保护未变。

## 发布与证据

- C5 原生结果提交 `1e2ab725cdbccdea7e6210fa4dfe29caeac27420`，相关 CPU 回归 45 项通过。
- Pinch 原生结果提交 `5c5482e816f553e8691e52f7bb6dcf83d5096974`，相关 CPU 回归 39 项通过。
- 只读审计 SHA256 `ca49b493330aebd6f0a933340c39c08493e22468177dd0bb1c2dba98bc6f6095`。
- GDB 测试失败日志 SHA256 `0c2948663c8008b4a26fe9abf5d8dd7a6c7678e59001db4cb6b0c82ad8e02045`，通过日志 SHA256 `a8a4097bc7aa0fd53df7649dd32cf731b5ac80c8907ceac64577ea8712ff39ec`。
- 两个原生单次预算均已消费。P4 的真实 Reset／371D／R0 闭环、三种独立 Teacher 及 Stereo Student 的原生准入仍未完成。

详见 `docs/contracts/dh116_c5_pinch_native_outcome.json` 的原生回执、实际命令与恢复状态。原始日志和映射仍保存在 Ubuntu 私有证据目录。
