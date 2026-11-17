# 左 Pinch v3 原生失败与账本重放

新入口避开 Kit 启动前的 USD 导入，实际通过 SimulationContext 创建。该单次 GPU3 试验完成 **240 SDK 步（2 初始化、238 测量）**，DeformX 每测量步 12 子步。无 PPO 更新。三份逐步账本均完整 238 行，独立重算通过原始几何、动作、目标、接触与时钟一致性。该配方已消耗。

## 原生观察

- 初始 17 个原手碰撞包络和 5 个锁定 hull 无桌面重叠。全手最小源净空 **2.000861 mm**。拇指、食指、中指、无名指的锁定 hull 净空分别为 **2.987793、3.079007、38.878852、56.375189 mm**，对应初始桌面穿透均为 0。stored cooking 缺项依然明确保留。
- **103,976** 个原生 solver pair 全部来自杆与桌面。左手接触行 **0/238**，两必要指腹同时受力行 **0/238**，PhysX 手桌接触 **0**。杆从约 0.308 m 落到轴高 0.305 m。
- 实测末端 link pose 与同钟杆节点的锁定 hull 几何间隙，测量步 208 拇指 **94.783 µm**、食指 **100.009 µm**，步 238 降至 **4.708 µm / 5.368 µm**。第 208 步双指关节跟踪各约差 0.0015 rad。几何重放支持闭合刚好停在接触外，SDF 采样和实际接触仍须以原生 pair 判定。
- `app.close` 实际返回，手、两份根 view 别名、关节 view、composer、runtime、provider、rope、contact view 的弱引用已消失。SimulationContext 引用仍存活。GDB 完成 136 线程的正常 finalization 捕获。外层与 inferior **均为 1，无信号**。本次没有自然退出 0 资格。
- 触发退出 1 的异常发生于全部账本写完后。producer 对 41 个 float32 节点先按 FP32 求和，旧 validator 从 JSON 重建为 FP64 求和，仍使用同一个 1e−9 N 一致性容差。**208/238** 行发生精度差。所有 producer 读数可按实际 dtype 重现，独立 FP64 物理残差最大 **1.491979e−6 N**，低于原 **1e−3 N** 守恒门。没有将这一读回差异解释为非零手指力。

原始失败仍为 failed。新工具只重放原始文件，记录来源精度和独立物理残差，后续候选可采用同一 FP64 归约写入和校验。后续须独立证明接触、生命周期和两手同杆。

## 测试和证据

相关 **31 CPU passed**（新增两项），包含非有限和真实高残差拒绝。脚本使用 Ubuntu `bash scripts/python.sh scripts/audit_dh116_pinch_v3_evidence.py --receipt <实际调度回执> --output <独立新文件>`，空 CUDA_VISIBLE_DEVICES，2 线程，120 秒外层时限。独立重放 0 新物理步和训练更新。

- 原生回执 SHA256 `fe0aa842d11ba118e3151e3a80ba58f7d60507548fade93b0dab88f6a300681f`。
- 外层证据 SHA256 `1e1caf1e0ada1dbf71ccdb6c5845280f39ba15adf12497c119a09c16d90d7b40`。
- 重放结果 SHA256 `84bab3fb94fadd217f5c2418acea0a340205aeccff54e7edf436d82966fc126d`。
- CPU 日志 SHA256 `3921cd5454ed1876d13e4cd52bb71636af2bea79330081595d0b71fa370b9815`。

全部原件和调试映射留在 Ubuntu 私有目录，原 30 mm / 5 s / 0.5 mm、动作、材料、模型及历史 Checkpoint 保留。
