# F3 DeformX–PhysX 关闭隔离与完整桥接复验

冻结基线 `f008e6eefa3bb645225350d2b289899d7c9cf836`。所有试验在 Ubuntu RTX 4090 GPU 3 顺序执行，采用真实 `ropesim` backend `96b8514c`、F1 cp310/SM89 扩展、Isaac Sim 4.5.0.0 与 Isaac Lab 0.54.4。以 `scripts/python.sh` 启动，每组独立进程和私有输出。外层 Bash 保存子进程真实退出码。未调用强制退出或发送终止信号。

## A–E 结果

| 组 | 本组新增对象或行为 | 物理记录 | 自然外层退出 |
|---|---|---:|---:|
| A | Isaac SimulationContext、显式 PhysX Tensor SimulationView 和刚体 view | 10 步 | 0 |
| B | A 加载固定 `ropesim` 扩展，未创建 solver | 10 步 | 0 |
| C | B 加入一根实际 CUDA RopeBatch，独立推进杆 | 10 PhysX 步、10 杆推进 | 0 |
| D | 建立实际 `RopeColliders` 和刚体资产，姿态传递，反作用力关闭 | 10 步 | 0 |
| E | 原单刚体冷复位对照和开启反作用力，逐步记录 wrench | 240 步 | 0 |
| E2 | 全新进程独立重启，重复完整 E | 240 步 | 0 |

A/B 两份刚体账本逐字节相同。E/E2 的 240 行账本均与旧的数值通过、关闭失败的 F3 账本逐字节一致。关闭盒侧反作用力时盒位姿保持，启用后盒沿 X 移动 83.284020 mm。线冲量残差为 1.452041e−8 N·s，角冲量残差为 4.963053e−9 N·m·s。两次应用 close 返回、所有受跟踪的刚体、solver 和 tensor view 在应用关闭前释放、外层退出码 0。旧独立 PhysX 拇指标定的关闭资格保留在自己的范围内，本表为真实 DeformX 单刚体桥接证据。未验证 DH116 手或抓取。

## 资源所有权和释放证据

| 资源 | 创建和所有者 | 释放与观测 |
|---|---|---|
| A–C 独立 GPU 刚体 view | probe 从显式 SimulationView 创建 | 先丢弃刚体 view，调用显式 SimulationView.invalidate()，记录 SDK is_valid 和弱引用状态 |
| D/E 资产刚体 view | RigidObject，资产注册 STOP 回调 | 独占 composer 清理后完成场景引用退出，STOP 后资产及 view 弱引用已消失 |
| WrenchComposer | 单刚体 RigidObject，独占写入 | 每次世界系写前 reset 刷新 link pose，阶段末 reset 清空最后 wrench，STOP 后弱引用消失 |
| RopeColliders | 阶段内 probe，持有其 RopeBatch 和碰撞体 GPU pose 缓冲 | 账本 fsync 和 CUDA 同步后先释放桥接对象引用，再释放 RopeBatch |
| RopeBatch 与视图 | probe。节点与接触力临时零拷贝视图由 solver 持有底层缓冲 | 每步拷贝到主机记录，阶段末同步和作用域回收，弱引用消失；没有把 Python 释放当成静态 CUDA 内存字节计量 |
| SimulationManager view | Isaac SDK，独立 STOP 回调 | 清除本轮 Lab SimulationContext 的 STOP 钩子后调度 SDK STOP，验证 manager getter 为空、弱引用消失 |
| SimulationContext | 本 probe 和 SDK 运行时 | `clear_instance()` 清 singleton 与回调，probe 放弃本地引用。部分观察点仍有 SDK Python 引用，应用关闭返回及外层 0 另行验证 |
| CUDA stream／扩展静态对象 | Torch、ropesim 和 Isaac 插件内部所有 | probe 不持有手动创建的 stream 或 event；阶段末 device synchronize。扩展静态对象只记录进程生命周期，无伪造的公开析构回执 |

所有权账本区分弱引用失活、SDK view 失效、回调返回和外层退出。旧 GDB 仍证实历史 SIGSEGV 位于 PhysX tensor 插件经 `_physicsTensors` 进入 `Py_FinalizeEx`。当前重新执行旧基线入口 `Eold`，同一硬件、同样 240 行数据仍退出 139。修复后 A/B/C 可稳定关闭，扩展 import 和无桥接 CUDA 杆不会单独重现崩溃。D 已能创建真实碰撞桥接并关闭，E 的完整路径采用已验证的生命周期顺序。

额外单因素对照均执行 240 步、得到相同账本和外层 0。去掉 Kit 覆盖参数后成功。分别在该配置上仅去掉阶段末 composer reset、仅去掉 STOP 后额外 GC、仅保留调用者的 SimulationContext 变量、仅去掉阶段末显式 CUDA synchronize，四组都成功。证据定位到旧完整桥接的 Python／PhysX tensor 关闭组合，尚不能归因于某一个被移除的单项。保留已通过的整体顺序，未调用 SDK 私有清理函数或修改第三方源码。

## 命令与校验

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES=3
build="$HOME/.local/share/dh116-autonomous-3c80166/evidence/flexible-f1-20261112-883RS4"
export LD_LIBRARY_PATH="$build/sdk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash scripts/python.sh scripts/verify_dh116_deformx_bridge_teardown.py \
  --group E --headless --device cuda:0 --ropesim-python "$build/build-attempt3/python" \
  --output "$fresh_private_output"
```

A–E 实验冻结源码 SHA 清单保存在私有目录 `flexible-f3-teardown-20261112-YdD4Ss/source.sha256`。完整桥接试验前后未修改任何物理参数。试验后将原刚体 CLI 复用同一 E 关闭流程，保留其步数参数。新增调用路由通过 CPU 测试，实际原生资格对应上表的独立 A–E CLI。

| 证据 | SHA256 |
|---|---|
| A/B 10 行物理账本 | `35b702af5ed07b0312ec62aafe3b5111ecb7bf45a8a65f0fd207a3ac0b51f25d` |
| C 物理账本 | `93d174928228c71afa80d0333c516955075b58effb6fafaba804abaecae191fe` |
| D 物理账本 | `6d0e4956c916e416377f3afd8e9a8d95394ef8c344b47768f7a56686d9ff452e` |
| E/E2/旧 E 240 行物理账本 | `ae3934bd61f2a98ef8d267f102c1c2a48e17326dd5372294d171a1908d97e206` |
| E/E2 物理结果 | `5bb37ac255843f101dfad7bd8911d39420df2201ac20cec22729378ef898896a` |
| E 所有权 | `174f4939501d9c51761873b507e612298faf278bae0e902ef21d592e50e9b779` |
| E2 所有权 | `4ade870e20bd427f999e09b9204d38f2374f096297599f6dd009a6886c7d3a50` |
| E 启动日志 | `5a9375b655348e3ae71367400d1743e576f0fa52a044f72067c6124157a5acbf` |
| E2 启动日志 | `310a258e9c9f3d53c22e8ddd11e7ca28a8ad951a7b90c53627a234205837dfa5` |
| 旧 E 重现日志 | `921b52e3e2d9349cc43234ff7de1a7c3e9b6c7cbdfd0246126c76708999e543b` |

Ubuntu CPU 复核 34 项通过。528 个受保护文件与 2 个历史 Checkpoint 哈希不变。F3 单刚体原生生命周期通过，进一步 DH116 cooked hull、单指和双手验证使用独立资格。
