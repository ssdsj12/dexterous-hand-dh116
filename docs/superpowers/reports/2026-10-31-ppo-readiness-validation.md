# PPO训练入口验证（2026-10-31）

当前可以进行原生Isaac PPO训练。实际小规模运行已完成16环境、16步rollout、2次优化更新，共512条transition，正常退出并保存模型及TensorBoard事件。本次是流程验证，不是已学会lift，也没有启动长训练。

## 与最新目标对齐

- 显式24维surface模式继续由策略探索12个主动手指关节与双腕各6维外部位姿；物理从动联动和原执行器限幅保留。
- 默认取消该模式额外的软件分力目标调整与主关节回退；动作由策略负责，分面/分指均力作为软奖励，不作为成功硬门。教师的保持关节指令、抬升脚本没有注入PPO。
- 成功按真实杆底面相对支撑初态净升至少30 mm，在有效对向指腹承力下保持5 s；保持窗口三点相对双腕滑移不超过0.5 mm。控制周期使用8个480 Hz子步的原生均值；原始子步仍检查高度、速度、深度和状态。没有增加瞬时力或分力精度门槛。
- surface回合预算从15 s改为30 s，覆盖已验证原生示范所需的21.33 s；旧模式保持原默认值。
- 增加显式`--native-sensor-batch`训练读回选项，复用已验证的原生采样路径；不修改物体、接触或力值。

## 证据

167项定向测试通过，覆盖成功计时、局部reset、分力偏差软奖励、动作目标所有权、回合时限、批量读回和旧模式回归。独立只读审查未发现新增阻塞问题。

初次v1在训练和保存完成后，Isaac关闭时进入安装版IsaacLab的`_app_control_on_stop_handle_fn`持续render循环；用中断信号释放了该已完成测试。随后按公开清理接口，在请求quit前执行`sim.clear_all_callbacks()`和`sim.clear_instance()`，解除timeline订阅。新增回调反例先失败后通过，终止状态/入口相关19项通过。v2重新完整运行两次更新，SSH实测退出码0，无需中断，进程已结束；v1原始记录保留。

将AK成功示范的全部10,240个原生帧输入PPO成功判据，保持时长为`[0, 5.016666666666654]` s，成功事件为`[0, 1]`，与原独立审计一致；未按手总力或分指精度否决真实提升。

实际入口：

```bash
cd /root/hand
DH116_NUM_THREADS=2 bash scripts/python.sh scripts/smoke_dh116_surface_ppo.py \
  --output /root/hand/runs/surface_ppo_ready_20261031_v2
```

本次使用脚本默认16环境、16步rollout、2更新。实际观测351维、动作24维。最终v2两次采样约8.97/9.16 s，优化分别0.59/0.38 s；平均单步奖励0.05600/0.05536，损失有限。最终状态`training_budget_complete`，`iteration=2`，`timesteps=512`，`process_exit_code=0`，恢复/丢弃transition均为0。`tensor_ppo.pt`为1,570,116字节，保存policy与optimizer，递归检查全部张量有限；TensorBoard事件文件存在。

证据位于：

- `/root/hand/runs/surface_ppo_ready_20261031_v2/`：最终checkpoint、配置、训练contract、逐次更新和TensorBoard。
- `/root/hand/runs/deployment/lift_continuation_20261031/ppo_readiness_summary_v2.json`：最终紧凑核验结果。
- `/root/hand/runs/deployment/lift_continuation_20261031/ppo_native_demo_replay.json`：AK原生轨迹独立回放结果。
- `/root/hand/runs/deployment/surface_sharing_20261031/source_ppo_ready_v2/`：最终运行前冻结源码，v1也保留用于追溯退出修复。

## 适用范围

训练从成功候选的静态预抓姿、原支撑高度重新开始，仍需策略自己抓起圆杆。本次两次更新每环境仅累计0.5333 s，成功数为0，不用于评价收敛或最终成功率。现有AK示范未保存策略观测，不兼容旧156维观测/20维动作BC入口；本次没有教师策略预热。大规模并行数量和长训练吞吐尚需单独验证，不根据16环境结果承诺2048环境可用。
