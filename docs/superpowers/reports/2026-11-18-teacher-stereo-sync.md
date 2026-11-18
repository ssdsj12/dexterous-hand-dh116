# DH116 原生 Teacher 视觉与动作同步

## 记录模式

`scripts/record_dh116_teacher_stereo_sync.py` 的默认操作是冻结计划。`--record-sync-data` 使用精确的计划 SHA 进入原生记录。支持 Pinch、Tripod、Power 的独立同构运行。单实例一次只加载一个抓法和对应的历史 Checkpoint，原相机标定与 67D Teacher 输入保持原样。使用两台固定 640×480 RGB 相机，关闭 WebRTC 和训练。10 秒墙钟采样窗口内至多执行 500 个控制步。原模型的时间特征继续用 480 步归一化，观测记录包含是否处于原策略时间范围。

- `sync_log.jsonl` 关联同一 pre-action 状态 SHA、两相机原始渲染时间、SDK 物理时间、双手目标写入后时间、第一步 SDK 积分后的时间，以及 24D 策略请求和实际控制器输入。控制器写入和 SDK 步完成是软件观测边界，真实关节反应另需测量。
- `sync_log.json` 汇总两类时间。模拟时钟报告原始 RGB 与物理时间差和动作边界偏差。服务器单调时钟报告捕获耗时、RGB 到动作写入和积分的延迟，以及相邻帧读取完成间隔。帧间隔包含读图与核验、推理、求解、PNG 编码和证据写盘。
- 每个已完成控制步保存左右 RGB PNG、SHA 和几何回执。没有独立图像标注时重投影误差为空。2 ms 同步判据只用于时间。2 px 重投影验证、物理抓取和 Teacher 准入分别保留独立证据。

## 首次原生结果

GPU 2 的 Pinch 运行冻结 SHA 为 `32805673886c4239f59567b7bd91955377fde9392e4568f1e7feadb929c8cb0f`。实际完成 2 个 SDK 初始化步。原生相机读取在首帧前报 `unverified Isaac/Warp RGB pointer ABI`，控制、求解器、PPO 和有效双目记录均为 0。没有启动 Tripod 或 Power。保留库映射、错误回执和外层退出回执。

运行时动态库映射包含 Isaac 4.5 自带的 Warp 1.5.0，原适配器仅接受 pip Warp 1.12.1 接口。修正版校验 Replicator 源和所用 Warp 源，已核验的 1.5.0 直接调用 SDK 原生的非拥有指针路径，1.12.1 保留原局部兼容路径。CPU 回归用实际安装的 Warp 1.5.0 CPU buffer 和实际安装的 Replicator 函数复现并验证。修正版同时把记录步数上限与模型原始时间特征分开，允许已存在的 Kit 缓存释放探针锁定实际结束步，原初始化用法仍严格绑定第 2 步。这些修复尚无新的原生回执。

SDK STOP 后 SimulationManager 的物理视图已释放，`app.close()` 返回。外层 49.27 秒退出，退出码为 `-11`（SIGSEGV）。现有证据未定位关闭后的原生调用栈。采样与自然退出均未通过，未重试，未强制结束。真实双目同步、动作延迟、墙钟抖动和重投影误差仍缺数据。

## CPU 验证

联合回归 **610 通过，0 失败**。源码、原运行错误和库映射的证据路径记录在 readiness。

## 重现

```bash
PYTHONPATH=/mnt/mydisk/xk/projects/dh116-teacher-display-owner-20261117:src \
  /mnt/mydisk/xk/envs/dh116/bin/python -B -m scripts.record_dh116_teacher_stereo_sync \
  --mode pinch --max-control-steps 500 --output /evidence/native-pinch --plan /evidence/pinch-plan.json
```

原生执行还需使用同一 CLI 参数和冻结计划绑定的 `--record-sync-data --approved-frozen-sha256`。后续先验证修正版 Pinch 有真实配对帧和动作回执，再按相同规则分别测试 Tripod 和 Power。建议在独立原生进程诊断关闭后的调用栈，保留外层进程退出码。
