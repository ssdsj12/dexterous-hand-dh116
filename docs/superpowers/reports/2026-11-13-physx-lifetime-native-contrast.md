# PhysX 生命周期两格原生结果与保留引用

两格按相同场景串行各运行一次。SDK 实测 0→2 初始化步，未推进杆求解器、未采集接触、未进行 PPO。两格都记录了 STOP 后 SimulationManager view 已清空及 app.close 返回。自然退出资格仍失败。

| 清理 | 秒 | 初始化 / 测量 / PPO | 外层退出 | inferior 自然退出 | 原始回执 SHA256 |
| --- | ---: | --- | ---: | --- | --- |
| registry | 55.81 | 2 / 0 / 0 | 1 | 未取得 | b3c1ae0d41bc568fcee80bd5085315ed4e270a3ab19741ca793a44ad47a316c4 |
| explicit | 57.61 | 2 / 0 / 0 | 1 | 未取得 | cc9f9ce4653055bcba072095a5c46a19cf17022e9e4a85e44a708a2e11dbaee8 |

## 已证实的 GDB 观察失败

两格均在 `Py_FinalizeEx` 取得主线程栈、寄存器、指令和完整进程映射。断点回调内 `thread apply all bt 20` 报 `Selected thread is running.`，随后外层 `continue` 同样失败。外层退出 1 来自 GDB 命令失败。没有合格的 inferior 状态，也不能由此认定再次 SIGSEGV 或自然退出成功。证据原件保留于各私有原生目录。

| 清理 | 运行日志 SHA256 | finalize 映射 SHA256 |
| --- | --- | --- |
| registry | 633ed561ef21e7943ffa70ee5fce0441d935a8292530a49be9bff1129a9a95fc | 2c9f2e5047a7a19fdcca3cf9085a5dcd6630e743f39423102b01e2088afca28d |
| explicit | c0bff4b875d9b26f6fb17bd2233f2450b7b4d7395ef08f7d403ae49388cc286a | 861e81d580e70d8c5193f153af019d5b6bb314d48ae996bcff5ac53dd9a94f42 |

## 释放边界

两格的仿真对象及八个监测资源均在关闭前后仍存活。asset/manager view 弱引用指向同一对象；管理器清空引用并不消除 asset 的引用。以下为关闭前可见的 Python 直接引用数，不能代表 native 引用数。

| 资源 | registry | explicit |
| --- | ---: | ---: |
| asset_sim_view | 5 | 3 |
| hand_view | 3 | 1 |
| composer | 4 | 2 |
| provider | 3 | 1 |
| runtime | 4 | 2 |
| rope | 3 | 1 |

explicit 删除了 acquire_graph 本地 holder，仍观察到已返回的 build_native frame（acquisition.py:146）持有 provider、runtime、composer 及根 view。provider 的绑定方法保留 runtime，runtime 保留 RopeBatch。另有 hand._permanent_wrench_composer 和 _physics_sim_view。当前证据未追到持有已返回 frame 的最外层对象，也未证明哪个引用导致旧 PhysX 崩溃。

两格指纹的预算均已消耗。下一步先用有后台线程的 CPU GDB 夹具验证将取证移动到完整停机后的控制循环，并用独立 CPU 夹具核实构造 frame 作用域释放。只有新配方取得对应授权后才进行原生复验。Pinch、Tripod、Power 的自然退出门继续关闭。
