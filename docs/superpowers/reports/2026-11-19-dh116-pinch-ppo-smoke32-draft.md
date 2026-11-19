# DH116 Pinch 首次 Teacher PPO 诊断配方

Antigravity 在 `b88e8215f1108c7e040869f7d21fdf28ef308104` 完成两轮只读复核。共同核对了真实 worker/调度器源码 SHA，并确认 `_cpu_reference` 独立构造 CPU 模型与 Adam 审计器。Codex 通过 SSH 单独读取并校验了 Antigravity 只读角色无法打开的配置和历史 Reset 产物。

本次绑定 11 月 18 日相同 Pinch 原生 Reset 的自然退出 0、41 个材料节点和原始数值回执。原观测来源、PPO、奖励、动作、物理和模型配置保持。新增的源码闭包包括 actual provider、入口、现有 worker 审计器和已安装的无相机 .kit／GDB。在 CPU 生成独立训练配置、core 与 launch companion，按真实字节重算语义、装置、Reset、目标和奖励 SHA。冻结配置的 authorization 与 resource_lock 为空，只有经实际调度的活进程才可取得租约。

已登记 `teacher.pinch-ppo-fresh-smoke32-v1` 为 pending。原生诊断预算为 1 环境、32 个控制步、2 个初始化 SDK 步和 256 个测量 SDK 步，最多 3072 个 DeformX 内部子步、1 次 PPO 更新、180 秒内层和 240 秒外层。设备身份沿用现有契约的物理 GPU 3、mask 3、逻辑 cuda:0；最小显存 4096 MiB，最小磁盘 1 GiB。使用 GDB 记录崩溃，无自动重试。该预算是待批准诊断范围。独立无授权计划确认状态为 `waiting_approval`，未列入 native_ready 或 native_qualified。现有 Isaac 单实例锁仍由其他运行持有。

| 指纹 | SHA-256 |
|---|---|
| 命令 | `7518233086782363ff13e7c77a4274b8f0c2a43f90a6ce0ac9eed5a12487594c` |
| 代码和配置 | `c59fa50a6f4b2d90a50e1b0af96e3db671fec91c8fbc444e15fa540f6ac8f640` |
| Actual companion | `fe665ff96856a8fc6ca6864e6bc39ee36b61420cd6b458840b8361bddd4309fb` |
| 调度 attempt | `cde13f53264118f4e83f1e2d58c9b4c49068c78a95e07b12a5dd0a7ce7d3062e` |

CPU 联合回归 189 项通过，新增 4 项。使用 `scripts/python.sh` 在 Ubuntu 独立执行配方冻结，真实后端、Reset、奖励、语义和 CPU 模型审计均通过。私有数据、训练配置产物和 Reset 原始文件留在 Ubuntu，仅提交源码和配方元数据。本轮实际新增原生步数和 PPO 更新均为 0。后续需要准确预算授权、单实例资源释放、真实输出及退出审计。完整抓取成功与 Teacher 晋级仍按原测量标准另行验证。
