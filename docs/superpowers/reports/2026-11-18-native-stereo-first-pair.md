# Pinch 首帧原生双目与 2 ms 门控

本轮采用动作前源检查和 2 ms 样本可用性门控。原生试验只使用已冻结的单环境 Pinch 推理预览。冻结 SHA 为 `e5e1a1e4a5d1c7b37d21a3dd204c6a4f43579ce73fee8fb39da722832dcd4c50`，GPU 2，最多 2 个 SDK 初始化步后再执行 1 个控制步，零 PPO。此次为旧冻结 67D Teacher 推理路径上的原生时钟验证。新版 C04 采集器的 371 槽动作前旁路增加相同的时间门，但尚无该 worker 的原生运行。

## Reset 源核对

Teacher 推理脚本先读取 `pose_reference.json` 并构建场景，随后执行 `sim.reset()`。新版 Pinch builder 同样是在读配置和装配后才执行 SDK Reset。旧尺度的推理路径随后调用原 `reset_all()` 写入 episode seed。本次记录了调用栈及该调用前后 SDK 时间，均为第 2 步和 0.004166666884 秒，调用无额外 SDK 步。

历史 Reset 证据拒绝发生在 `retained_reset_source` 的源文件字节校验。历史 builder SHA 是 `5b190d1896647773c2f8895e73224f387d1dd17a565a44c4e2212db8f477fe17`，本轮 builder SHA 是 `bbbb6a65ba91932e73bd94c13b723c2fb910103cb445963924a12e374c81b2d6`。历史证据的 `/home/xk` 入口经别名解析，实际文件 SHA 与封存数据相同，历史规范路径检查仍拒绝该入口。当前同一批次的数值 Reset 来源仍待验证。

## 原生结果

| 检查项 | 结果 |
|---|---|
| 原生 RGB 读取 | Warp 1.5.0 的 `native_owner_false` 路径生效，两张 640×480 RGB 已保存，SHA 不同。 |
| 双目配对 | 左右 raw frame 和 render epoch 都为 20。 |
| 模拟时间 | 两路 IsaacReadTimes 的 `simulationTime` 与 `simulationTimeMonotonic` 都为 0，SDK 为 0.004166666884 s。差值 `RGB − SDK = −4.166667 ms`。 |
| 样本状态 | `student_sync_usable=false`，图片和 raw 证据保留。实际动作、求解器步和 PPO 更新均为 0。 |
| 几何 | 对从原生载入的 PNG 与标定回执做离线复算，通过尺寸、SHA、相机和双掌坐标一致性。缺独立人工标注，2 px 重投影结果仍为空。 |
| 退出 | STOP 后 SimulationManager 视图为空，`app.close()` 返回。已登记的 SDK/相机 Python 包装对象的弱引用都失效。约 48.13 秒后外层退出为 `-11`。 |

同一渲染帧中的 raw `swhFrameNumber=0`、`externalTimeOfSimNs=-1` 也保留在证据中。一次帧的观测支持相机时间与 SDK 时钟不一致。还需定位时钟来源，才能区分冻结渲染时间映射、滞后以及其他原因。当前证据未定位进程结束阶段的 C++ 调用栈。

## 数据准入

C04 记录图像与 SDK 的有符号时间差，`abs(delta) < 2 ms` 时只通过时间门。超门槛的样本保存诊断帧和拒绝原因，缺少 Student 实际下发动作标签。Teacher 的特权 PPO 路径仍独立。原生首帧诊断在超门槛时也会保存原始两图并在动作采样前退出。

静止同一场景的重复帧可用于测采集稳定性。Student 的训练样本还需独立物理状态和有效动作来源。随机扰动、长采集、BC 和 PPO 本轮未启动。碰撞拓扑原生门仍以此前 0/34 的实测结果为准；本轮没有再做碰撞拓扑测试。

## CPU 回归

双目旁路、控制窗口、首帧、时钟门和 Registry 联合回归 **128 通过，0 失败**。
