# DH116 Teacher 固定双 RGB WebRTC 预览

## 实现

在既有 `record_dh116_teacher_stereo_sync.py` 中新增显式 `--live-webrtc`。两台固定相机都使用 Student 的 640×480 标定。左右画面在 Isaac 的同一窗口并排显示，并显示同一对图像的 SDK 物理步、共同渲染编号和原始 RGB–SDK 时差。等待接收和留屏阶段逐次检查 SDK 时钟，冻结的动作窗口仍使用原控制和求解器频率。

每对 RGB 使用原始图像 SHA、标定 SHA、episode、Reset 和动作前状态 SHA 绑定冻结 Teacher 的 67D 特权输入，以及实际掌位和 41 个 DeformX 材料节点。原 24D 采样动作与实际控制器下发分别保存。显示和日志重用同一对原生帧。WebRTC 只用于预览，所有该模式样本均标记 `student_training_sample=false`。正式特权 Teacher PPO 仍使用原入口。

## 原生结果

GPU 2 单实例串行，Pinch、每次 1 环境、2 初始化 SDK 步、16 个控制步，合计每次 130 个 SDK 步及 1536 个求解器子步。每次 PPO 更新数为 0。第二次加入短暂首帧等待，Windows 客户端实收到两幅相机画面。画面复核时显示的是回放留屏的最后一对，对应 SDK step 122、renderer frame 578。

两次运行分别读取了 16 对左右共同渲染帧。每帧保存的 RGB SHA、特权状态身份以及同边界动作下发均完整，回放末尾 SDK step 130。两个运行的原始 RGB−SDK 差均为 −4.166667 ms。只在离线复算中使用首个 SDK/求解器边界作为场景起点，两次各 16 对的候选残差均为 0 ms，RGB 与求解器时钟最大差约 0.000162 ms。正式 2 ms 原始时钟门槛仍生效，可用于训练的样本为 0。缺少独立标注，本轮没有像素重投影精度结论。

- 首次运行完成采集，Windows 未在留屏期间连上。外层结束为 SIGSEGV（−11）。
- 第二次运行完成采集和 Windows 双目显示，STOP 回执确认物理视图已释放。外层最终退出回执仍待，关闭超过 180 秒软截止。诊断保留日志，不自动重试或强制结束。
- 原始数据及冻结执行命令保存在 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-live-stereo-20261118T124207Z` 和 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-live-stereo-connect-20261118T124719Z`。原生资格与 Student 数据资格均待验证。

## CPU 回归

相机、采集与 WebRTC 回归 59 项通过，调度器 Registry 55 项通过，合计 114 项通过、0 失败。先复现了直播选项、RGBA 双帧和首帧等待预算缺失。最后整理仅限制新增字段的作用域，直播模式行为保持一致。

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_teacher_stereo_sync.py tests/test_native_teacher_stereo.py \
  tests/test_teacher_stereo_alignment.py tests/test_tiled_stereo_capture.py \
  tests/test_teacher_webrtc_display_gate.py
```
