# 双手 Reset 到控制步的物理链路

`DualHandPhysicalEpisode` 接受已初始化的双手、共享杆和几何检查回调。它依次运行原地 Reset、同钟杆状态快照、双手 link 与 COM 读回、现有手腕净空投影，以及每个 60 Hz 动作的八次 480 Hz 交换。初始世界目标使用含偏移的实际 Reset 目标，避免在第一步重新减掉初始掌高。每次交换把动作、真实 SDK 目标、关节读回和同一杆边界传给调用方的证据写入器。

该入口在已有证据链上增加以下行为。

- 记录 Reset 身份、不可用的 Reset 接触和原始碰撞包络净空。净空不足时拒绝第一步动作。
- 原有保护触发时先传递该物理样本，再结束本 episode。失败后不得重放已推进的 PhysX 或 DeformX。调用方负责真正持久化原始证据和释放 SDK。
- 完整步预算不足时拒绝执行。未提供奖励或 PPO 入口，所有资格标志保留未通过状态。

## 测试

Ubuntu 相关八个文件 **96 passed**。新增四项验证实际组件的串联、零步 Reset 后两个完整控制窗口、初始穿透拒绝、关节终止样本和证据写入故障。早期两个失败来自测试 composer 只支持整手写入，已将替身扩展为支持安装 SDK 的 body_ids 行为。新测试同样保留完整体数组。

命令 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试文件为 `test_dh116_dual_hand_episode`、`test_dh116_dual_hand_clock_handoff`、`test_dual_hand_reset`、`test_dh116_flexible_rod_boundary`、`test_dh116_flexible_physical_exchange`、`test_dh116_flexible_live_geometry`、`test_dual_hand_rod_runtime`、`test_dh116_deformable_bimanual_step`。

CPU 日志 SHA256 `66860c9075c3209b242c5476c2886a01f091a0175af091b0f86314d317649fe4`，原始日志保存在 Ubuntu 私有证据目录。0 个原生物理步，0 次 PPO 更新。仍需在实际冻结场景验证全手碰撞、Reset、接触、真实逐字段状态和 R0。
