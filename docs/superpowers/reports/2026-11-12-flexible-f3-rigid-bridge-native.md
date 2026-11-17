# F3 DeformX 与单 PhysX 刚体原生双向传递

## 结论与配置

在 Ubuntu RTX 4090 GPU 3 上，使用真实 `ropesim` backend `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`、F1 cp310/SM89 扩展、Isaac Sim 4.5.0.0、Isaac Lab 0.54.4 和 PyTorch 2.7.0+cu128。原生单刚体双向传力、反力对应、线动量和角动量的数值验证通过。外层自然退出仍失败，F3 生命周期资格保持 blocked。此次未使用 DH116 手、原始任务资产、外部支撑、关节或抓取策略。

原生探针运行在一个独立 Isaac 实例和一个场景，0.5 kg 无重力自由盒、一个 21 节点 CUDA 杆、每个 480 Hz PhysX 步交换一次，杆每次交换 12 个子步。盒子在两组都保持 PhysX dynamic。同一个真实 `RopeColliders` 在 control 组设 `ColliderSpec.dynamic=False`，仅向杆传递碰撞体姿态。在 dynamic 组启用反作用力。两组冷复位到同一测得刚体位置、杆端位置和零速度。根部只对杆施加规定的轴向推进，盒子只接收已登记的桥接 wrench。起点横向与垂向各偏置 25 mm，能够审计非零扭矩。

先用上一杆步的世界系 COM wrench 推进 PhysX，再读 link 与 COM 世界位姿、同步碰撞体姿态并推进杆，最后为下一 PhysX 步暂存反作用力。记录与引擎一致的真实 composer 缓冲区，逐步验证一交换时延。杆侧报告真实逐节点世界系接触力和合力。杆侧力矩是用步末节点位置与步平均接触力作叉乘的估计，不能替代每子步精确积分。盒侧报告杆对盒的真实世界系合力、link 扭矩与 COM 扭矩。

## 240 步实测

| 指标 | 关闭盒侧反作用力 | 开启盒侧反作用力 |
|---|---:|---:|
| 样本数 | 120 | 120 |
| 首个接触交换步 | 46 | 46 |
| 有接触交换数 | 75 | 8 |
| 杆对盒峰值合力 | 175.701337 N | 29.181761 N |
| 杆对盒峰值 COM 扭矩 | 6.189831 N·m | 1.064906 N·m |
| 杆侧与盒侧世界系合力之和最大范数 | 0 N | 0 N |
| 盒子 X 位移 | 0 mm | +83.284020 mm |
| X 方向盒侧实收冲量 | 0 N·s | 0.284672603 N·s |
| X 方向实测动量变化 | 0 kg·m/s | 0.284672588 kg·m/s |
| Y/Z 实收角冲量 | 0/0 N·m·s | −0.007392248/−0.007392079 N·m·s |
| 全程线冲量残差范数 | 0 N·s | 1.452041e−8 N·s |
| 全程角冲量残差范数 | 0 N·m·s | 4.963053e−9 N·m·s |

SDK 物理步 3–242 连续。两组在第 46 交换步开始接触，反作用力组真实刚体发生平移和旋转且改变后续接触历程。另一组刚体位置和动量逐步为零变化。物理账本 240 行已 flush/fsync，两个独立完整物理运行的账本 SHA256 相同。峰值为校验环境的轴向冲击与刚体约束响应，不用于 DH116 力阈值或软杆稳定性判断。

## 兼容性失败与恢复

1. 首次原生记录中，Isaac Lab 0.54.4 的 permanent wrench composer 在连续 `set_forces_and_torques` 调用间保留旧 link quaternion。动态盒开始转动后，在第 48 步实际世界系暂存力与上一杆步相比已有约 5.61e−4 N 横向误差。默认 fast shutdown 还会在 Python close 结束标记前退出。该次外层 0 不能视为自然关闭验证通过。
2. 探针在每次桥接写前调用本实验独占 composer 的公开 `reset()`，刷新 link pose，保留上游姿态传递和 wrench 公式。恢复后严格一交换对齐检查通过。共享外力的场景必须独立划分 composer 的所有权。该实现无权清除其他系统的 wrench。
3. 禁用 fast shutdown 后，保存了短暂出现的 JSON 序列化失败及全部物理数据。将 NumPy 计数转换为 Python 整数后，结果与 close-returned 标记均完整。
4. 即使在 app.close 前先清理本轮独占 SimulationContext，再通过 STOP 使 SimulationManager tensor view 释放，普通外层仍返回 139。独立 GDB 复现的 SIGSEGV 位置为 `libomni.physx.tensors.plugin.so`、`_physicsTensors` 到 `Py_FinalizeEx`。这是 PhysX tensor Python/native 生命周期阻断。未调用强制退出，也未改写外层退出码。现有证据不能将双向传力演示升为可生产运行的 DH116 桥接。

## 运行与证据

运行入口 `bash scripts/python.sh scripts/verify_dh116_deformx_rigid_bridge.py`，沿用已锁定的原环境。启动器设置 `CUDA_VISIBLE_DEVICES=3`，使用 F1 独立 SDK 的 lib64 和明确模块目录，执行时核对扩展、桥接源码 SHA，采用 `--headless --device cuda:0 --steps-per-phase 120` 与独立输出路径。native 脚本使用项目同一实例锁。最终一次 08:01:51–08:02:08 UTC，外层 139，未覆盖失败记录或原数据。

| 证据 | SHA256 |
|---|---|
| 探针源码 | `3745d4c8c897dcd606cb79090f4f27902ccdc69944e706d397d30076e930e69c` |
| F1 cp310 扩展 | `c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53` |
| upstream Isaac bridge 源码 | `7a593ec2b56f4115e3c04ebe5cd6969294c7047ec8c577012bfcb3c1bf649eae` |
| 240 行 PhysX／CUDA 账本 | `ae3934bd61f2a98ef8d267f102c1c2a48e17326dd5372294d171a1908d97e206` |
| 物理结果 | `f972f25b56aef41b9bc2b3e679a2d70828aa6a136128ce4dce5eee84c034e0e1` |
| 关闭回执 | `d9d86d4d4c45d04345fa8b4ba6ff5a6cb49dd92e8bd4bcb32ce3eec4108d51d6` |
| SimulationManager view 释放回执 | `fae53100c9114aa1143a39d97c46a431917d99ebf72a2dfff893ad76fc9ffbdd` |
| 最终启动器日志 | `cdc5411bc7028ec302c9b1b65b3f8e8e9592fea374012d98fceec304f630f81d` |
| GDB 栈 | `bd8451b48d6f22f6f3f2158c4417c9df327bf863b6357fbd5f66ce838847c099` |

原始证据保留在 Ubuntu 私有目录 `flexible-f3-rigid-native-20261112-m73Abz` 和 `flexible-f3-rigid-native-gdb-20261112-8YQb3q`。CPU 复核 31 通过，0 失败。F4 的逐手 collider 资格与完整闭环仍需独立实测。
