# 当前源码 Pinch 数值 Reset 与 Teacher PPO 入口预检

独立无相机 Reset worker 复用现有双手和自由 DeformX 杆场景、锁定 SDF、数值 Reset 读取器和 R06 关闭。它在一次实际同钟 Reset 的前后读取 41 节点位置、速度和 XYZW 四元数，复核同一 solver 的 11 槽碰撞绑定。保留原来的净空、几何和关节判据。后处理独立读取外层退出回执、冻结配方 SHA、各产物 SHA，并调用原 `load_reset_manifest()` 验证实际 manifest。静态预检在认证过独立 Reset 后才停止检查旧 R08，旧回执路径继续核对原 SHA。

## 实测结果

1. 首次尝试在 App 前拒绝，原因是新 worker 继承的配方只锁旧预览二进制。随后冻结实际 mesh-observation 后端及完整 BackendProfile inventory。没有物理步或求解器步。
2. 第一轮原生在 GPU 2、单实例中完成 2 个 SDK 初始化步、1 次数值 Reset 和双手净空。清理前后相同的两个数值样本已写出。STOP 回调尚未提交，SimulationManager 和共享 view 在关闭前仍存活。外层退出码 1，未封存。
3. 修正后单独执行一次。复用现有已核 SDK 的 Kit stat cache 释放和 `Timeline.commit()`，STOP 得到实际 view 释放；R06 报 normal close，外层自然退出码 **0**，用时 42.43 秒。完成的两个 41 节点数值样本逐字节相等。SDK 到 step 2，控制、求解器、Teacher 查询、RGB 和 PPO 各为零。

新 Reset manifest SHA 为 `110f6dbcca61f36111ada20c47dbf698fab72fe2cef4d8e36c866690b4531daa`。使用实际 manifest、`privileged_deformable_teacher_pinch_v1.json` 和 `approach_wiring_smoke_v1.json` 运行 `train_dh116_native_teacher_ppo.py --preflight`。Reset、模型、奖励、后端及适用的 CPU 读者静态核验均通过。尚缺冻结的 32 步配方、实际 worker provider、产物审计和真实租约，所以 PPO 更新仍为 0。旧碰撞体全量严格报告 0/34 仍是独立结果。

## CPU 与证据

专项 Reset/构建器 43 项通过，定向来源与 STOP 7 项通过，套件间有重叠。更宽的初次回归 99 通过、1 失败、3 错误，均出现在旧 R08 的文件字节锁，保留失败证据。产物与冻结 SHA 可见 [readiness](../../contracts/dh116_teacher_fresh_numeric_reset_v1.json)。服务器日志、原始数组和 manifest 没有打包发布。下一步复用这份 manifest 连接原 PPO 工厂和实际 worker，执行已审方案的 1 环境、32 控制步、1 PPO 更新有界诊断。
