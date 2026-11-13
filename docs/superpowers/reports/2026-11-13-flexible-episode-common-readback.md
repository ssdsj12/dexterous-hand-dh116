# 柔性双手动作前同钟状态读回

`DualHandPhysicalEpisode.pre_action_state()` 复用已审计的 `capture_mounted_hand` 和 `adapt_mounted_hands`。在 Reset 后和完整的 60 Hz 控制步之间，检查实际 SDK 步、时间、rod tick、episode/reset、手部路径及测量缓存。把实测关节和掌位子集与同一边界的杆数据一起返回。原始观测继续使用不含初始高度偏移的 authored root seed；物理动作的初始目标使用已含偏移的 Reset pose。

在 SDK 形状的 CPU 替身上，原 371D 中 **66** 项来自 B00/B01/B15/B34/B35/B36。其余 **305** 项原值保留 NaN、valid=false；包含刚性物体特有维度、接触和任务状态。没有用刚体近似、零值或 rod 均值填补。原 Teacher 准入仍关闭。状态副本可独立保存，保护终止后拒绝新动作前观测。

## 回归

新增测试在实现前复现 2 个缺入口失败。最终 7 个文件 **100 passed**，覆盖 Reset/控制步标识、旧缓存拒绝、时钟漂移、返回数组所有权、原始掌高基准和终止后的读回拒绝。

Ubuntu `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试为 `test_dh116_flexible_episode_common`、`test_dh116_dual_hand_episode`、`test_dh116_teacher_mounted_wrist_source`、`test_dh116_teacher_preaction_readback`、`test_dual_hand_reset`、`test_dh116_flexible_rod_boundary`、`test_dh116_dual_hand_clock_handoff`。

CPU 日志 SHA256 `8f5688b041f73e12eaf99576ed89cbe706e2cb8cba84eca26d3af2bdc9d5a79a`。原生物理步 0，PPO 更新 0。下一步需要从柔性运行的实际控制器内存和物理接触获取其余可用字段，并明确不可移植字段。
