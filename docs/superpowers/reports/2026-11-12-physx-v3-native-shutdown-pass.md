# PhysX v3 自然退出与全新实例重启验证

2026-11-12 04:27:29–04:28:59 UTC，Ubuntu GPU 3 在 Isaac Sim 4.5.0、Isaac Lab 0.54.4 环境完成两轮隔离左拇指标定。外层验证器退出码为 0。两轮使用单独进程，第二轮在第一轮退出约 0.018 秒后启动，进程 PID 和 episode UID 各不相同。

- 两轮分别写入 240 条 480 Hz 物理子步，SDK 步号均为 3–242。冻结验证器核对了 JSONL、时间连续性、原始文件 SHA256、fsync 完成、原生回调释放和 `SimulationContext` 清理记录。
- 标定脚本在关闭前释放采样句柄。两轮 `app.close()` 均返回，无待处理回调异常。`/app/fastShutdown` 实际读回为 `false`。两轮子进程均自然退出码 0，监督器未发送信号，未记录 SIGTERM 或超时。
- `verify_trial()` 独立复核的两轮证据与 `shutdown_retest_passed.json` 逐字段相同。首轮与重启轮的物理行排除 episode UID 后，240 行均相同。

| 轮次 | 子进程总耗时 | `app.close()` 耗时 | 关闭返回至进程退出 | JSONL SHA256 |
|---|---:|---:|---:|---|
| 首轮 | 75.613 s | 6.729 s | 1.359 s | `ad645bc234f72298171164be456004bf8a76611517af7aa7d15c5c8966a50116` |
| 重启轮 | 14.356 s | 6.199 s | 0.668 s | `3e57bafa1c372d0eed5185b8cada41cb982c2aa09cbc2fd8f8f0d985f85ac003` |

冻结源码清单 SHA256 为 `20cb1c8088ac6e0830a793eb948ce7b2c4a704487f1b0efd11ef87a8c3995717`。验证标记 SHA256 为 `a33ec3726252a3ac08607fc4b0825e02291679a5f7dbf7b6032472b8476c46c3`。原始结果保存在 Ubuntu 的 `/tmp/dh116-stereo-shutdown-20261112-v3-run`。

标定入口和关闭验证器与集成提交 `b2f5d84ff5e3f9d87d5cd24ba55a3f4edb6f8ca6` 逐字节一致。冻结版 `tabletop_stereo_contact_calibration.py` SHA 为 `284f201aa3407114cd1eed127a03b7510993e88dc094f3600f1f1d490db9d806`，集成版 SHA 为 `9eafb16581041ad4d3032c239d0181176ecbf980a9f994a8d96bf018bfd6d730`。定义级比对发现差异位于接触对解析、摘要和归档审计，关闭、采样及回调定义一致。原生结论对应冻结快照，集成版接触来源审计另以只读重放验证。

## 接触来源与 CPU 复核

集成版只读重放核对了两轮 ledger、inventory 和 stage SHA。两轮各有 30 个非零 Tensor 采样步、240 个空结构化 poll 和 0 个匹配到 actor/collider 身份的接触对 header。Tensor 传感器侧动量残差最大值为 `4.3487548e-9 N·s`。接触冲量作用侧、外部支撑和附着仍待验证。R2、Force Proxy 与实际抓取成功各需独立原生证据。本结果通过关闭与重启门槛，后续采集沿用批准的单实例预算。

首轮和重启轮 CPU 来源审计输出 SHA256 分别为 `69919f2e334d08bc749536bb8aa1b51804e105dae7591b2856cd842e73244938` 与 `44c54253cbc1851cbddc581085579b11437a6580b257a663d6ead99084500b02`。Ubuntu 回归命令为 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_shutdown_validation.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_physx_polled_contact_audit.py`，65 项通过，失败 0，日志 SHA256 为 `85c1657d3a5bc0341b8f6cd9ae99592baf163a5e5e72944adb329fa982e9d872`。528 个保护路径和两个历史 Checkpoint 已核对。
