# Pinch 原生双目时钟热身对照

同一环境、同一固定双目、同一 Reset。仅在两次采集之间执行一个 60 Hz 的零命令控制窗口。手控制器照常写入目标，PhysX 以 480 Hz 推进 8 步，DeformX 每物理步积分 12 个子步。捕获时冻结物理。运行使用 GPU 2 和 Isaac 单实例，使用 640×480 RGB，实际显式渲染 3 次。

| 采集 | SDK 步 | SDK 时间 ms | 左 / 右 RGB 时间 ms | 左右共同渲染编号 | RGB 减 SDK ms |
|---|---:|---:|---:|---:|---:|
| 热身前 | 2 | 4.166667 | 0.000000 / 0.000000 | 20 | -4.166667 |
| 热身后 | 10 | 20.833334 | 16.666668 / 16.666668 | 21 | -4.166667 |

SDK 与 RGB 均推进 16.666668 ms。建立第二个真实渲染后，相机时间从零推进，原始 4.166667 ms 偏差保持不变。两次原始 RGB 时间都与当时的杆求解器时钟数值相同。SDK 在已记录的 episode Reset 前后都是 step 2。实际安装 SDK 的 `SimulationContext._physics_timer_callback_fn` 每个物理步累加 `_current_time`。相机侧读取已安装 `IsaacReadTimes` 的 `simulationTime`，两次元数据的 `externalTimeOfSimNs` 均为 -1。这个结果支持进一步核查初始化与 episode 时钟起点。单环境、两个采样点足以检验本次热身假设，尚无法证明其他 Reset 或抓法使用同一偏移。零动作在 SDK step 2 下发，step 3 首次积分。两对图像和独立回执完整保存，均未通过 2 ms 时序门槛。没有视觉训练样本、模型查询、PPO 更新、原生重投影测量或抓取成功数据。

`retained_reset_source()` 对封存的 R08 回执及其源文件逐项核验，检查发生在静态预检。当前 builder SHA 与历史 SHA 不一致，核验继续严格执行。热身实验使用已有独立 Pinch 诊断入口进行初始化，结果不授予当前 PPO Reset 准入。

关闭阶段的顺序与上一轮一致。STOP 后 SimulationManager 返回无物理视图，`app.close()` 返回，探针里全部 Python 弱引用已释放。外层退出码仍为 **-11（SIGSEGV）**，墙钟 51.87 秒。保留关闭记录和动态库映射。本次未获得新的崩溃 C++ 调用栈，仍需定位原生退出故障。

## 验证和恢复点

- CPU 联合回归 131 项通过。命令与日志 SHA 见 [readiness](../../contracts/dh116_native_stereo_clock_warmup_v1.json)。
- 诊断命令 `python -B -m scripts.record_dh116_teacher_stereo_sync --mode pinch --max-control-steps 1 --clock-warmup --output <exclusive-run> --plan <plan>` 冻结预算。执行命令在相同参数上使用 `--record-sync-data --approved-frozen-sha256 <sha>`。本次冻结 SHA 为 `65bdb0cd3b0608c3f4b852205c7afe978b7c48182e8e6d8b724963cc8d084bb4`。
- 服务器证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-stereo-clock-warmup-20261118/native-pinch-clock-warmup`。对照回执、下发边界、PNG、关闭记录和完整性 SHA 可追溯。只执行一次，失败证据保留。
- 下一步独立记录同一 Reset 前后的 SDK、渲染、DeformX 与 episode 参考起点，核对不同初态和三抓法的推进与重置，之后才资格化相机到动作的时序映射。当前 PPO 仍需要新鲜原生 Reset 和实际状态 provider 资格。原生退出单独定位。
