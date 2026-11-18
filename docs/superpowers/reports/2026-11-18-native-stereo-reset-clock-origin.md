# 原生 Pinch 双目 Reset 时钟起点复算

本轮离线读取上一轮原生保存的两对帧和同一运行的 `ownership.jsonl`。CLI 校验账本 SHA、同一环境和 episode 身份、Reset 前后边界、连续 SDK 步和渲染编号。候选偏移从账本的 SDK Reset 时刻读取。图像和原始时间戳按原记录保留。

| SDK 步 | 原始 RGB 减 SDK（ms） | 使用 Reset 起点后的候选残差（ms） | 原始 RGB 减杆求解器（ms） |
|---:|---:|---:|---:|
| 2 | −4.166667 | 0.000000 | 0.000000 |
| 10 | −4.166667 | 0.000000 | 0.000000 |

4.166667 ms 等于两个 480 Hz 物理步。两套时钟在同一轮以相同增量推进，结果与一个使用场景累计时间、另一个使用初始化后计时的解释一致。候选的第一帧与 Reset 处于同一边界，第二帧用相同候选偏移核对推进。样本限于一个 Pinch 初态、一次 Reset、两个采样点。要进入采集运行时，仍需用独立 Reset 验证起点和推进，检查跨 episode 是否继承前序偏移。采集路径仍采用原始 2 ms 时序门槛。两个样本均未准入，重投影和动作对齐无新增原生数据。

## 退出与 Reset 来源

- 已安装的 Kit `omni.kit.async_engine` 公开 `run_coroutine`。建议直接导入的 `run_until_complete` 未在安装包中导出。Kit 的更新等待 API 等待下一帧。采集入口生成内存可视 USD，当前调用链未发现显式持久化 USD Layer 的异步写操作。既有正常关闭路径使用 `wait_for_replicator=False`，安装的 `SimulationApp.close` 也确实有可选的 Replicator 等待分支；该分支针对 Replicator，并不能证明 GPU/PhysX 已清空。此前 `app.close()` 返回，外层仍收到 SIGSEGV，关闭故障留为未解决。下一次关闭对照应以原生调用栈、任务所有权和外层自然退出码为证据。[Kit 更新 API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/omni.kit.app/omni.kit.app.IApp.html)
- 在真实工作树读取 R08 封存来源列表，复现 `retained_reset_source()` 对实际文件字节抛出 `ValueError`。共 17 项历史源 SHA 与当前文件不符。首项 `tabletop_dual_pinch_builder.py` 为历史 `5b190d18`、当前 `bbbb6a65`。深拷贝内存配置不会改变文件字节。应保留历史链，并通过当前源码的独立原生 Reset 证据准入。

## 验证

CPU 联合回归 137 项通过，新增 6 项含同一偏移下的时钟漂移、账本篡改、不同运行、错误 Reset 边界及重复 Reset 的检查。命令、日志 SHA、上次原生帧和本轮复算可见 [readiness](../../contracts/dh116_native_stereo_reset_clock_origin_v1.json)。本轮未启动 Isaac、未生成新物理证据、未训练。

离线复算命令使用已有入口 `python -B -m scripts.audit_dh116_teacher_stereo_alignment <帧0回执> <帧1回执> --reset-ledger <同一运行的ownership.jsonl> --reset-ledger-sha256 <封存SHA>`。下一原生批次需要覆盖多个独立 Reset，并分别记录 SDK、RGB 和 DeformX 时钟。
