# Pinch 双目时钟第二次 Reset 复验

复用 `record_dh116_teacher_stereo_sync.py`。新增可选 `--clock-reset-recheck`，与原 `--clock-warmup` 一起冻结三对同步 RGB 和两次 episode Reset。GPU 2、无流媒体、一个 Pinch 初态种子、单 Isaac。两个初始化物理步和一个零动作控制窗口共到 SDK step 10，求解器 96 个子步。第二次 Reset 不增加 SDK 物理步。无 Teacher 查询及 PPO 更新。

| 样本 | Reset ID / policy tick | SDK 步 / 时间（ms） | 左右共同渲染帧 | RGB / Solver 时间（ms） | 原始 RGB 减 SDK（ms） |
|---|---|---|---:|---:|---:|
| 初次 Reset 后 | 0 / 0 | 2 / 4.166667 | 20 | 0.000000 | −4.166667 |
| 热身后 | 0 / 1 | 10 / 20.833334 | 21 | 16.666668 | −4.166667 |
| 第二次 Reset 后 | 1 / 0 | 10 / 20.833334 | 22 | 16.666668 | −4.166667 |

第二次 Reset 还原初始物理摘要，产生新 episode、policy tick 和双目渲染编号。RGB、SDK 与 DeformX 保持各自场景累计时钟。按首个 SDK/求解器边界得到的场景起点复算，三帧候选残差均为 0。如果错误地按第二次 Reset 的 SDK 当前时间重新补偏移，就会给第三帧额外引入 16.666668 ms。本次覆盖同一 Pinch 种子内的两次 Reset，尚未覆盖不同初态、三种抓法和新场景重启。保留原始 2 ms 正式门槛，三对记录均为诊断数据，实际动作训练标签为零。

CPU 定向回归 84 项通过。进程已记录 `app.close()` 返回，外层仍为 SIGSEGV（−11），墙钟 52.30 秒。该无流媒体进程未采集新的 C++ 栈，WebRTC 的 livestream 栈不能归因为同一个问题。

冻结 SHA `9323bb4a3bfeed05badcc0aa0b991c602cdb81d2e84083d928b11f7a803a5434`，完整运行和哈希见 [readiness](../../contracts/dh116_native_stereo_second_reset_clock_v1.json)，数据在服务器 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-stereo-reset-recheck-20261118/native-pinch-reset-recheck`。下一次采集资格验证应在不同初态、抓法及新场景分别复核场景起点与实际动作令牌。
