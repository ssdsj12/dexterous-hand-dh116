# PhysX 标定自然退出诊断

2026-11-12 首次试验使用已批准的 Ubuntu GPU 3 单实例左拇指标定。完成 240 个 480 Hz 物理步，SDK 步 3–242，原始 JSONL SHA256 为 `f8caa8d94bcd2f7fefe300ebf94c01b32e12a391caaf062875a6dc8eab487a7e`。摘要记录 fsync 已完成，两个接触回调已释放，所属 `SimulationContext` 已清理。子进程在关闭开始约 0.63 秒后以退出码 0 退出，监督器没有发送信号。缺少 `shutdown_complete.json`，外层验证退出码为 1，第二轮未启动。原始记录保留在 Ubuntu 的 `/tmp/dh116-stereo-shutdown-20261112-run`。

Kit 报告了 296 条文件监视 `errno=28`。服务器可见同 UID inotify watch 数达到 65,536 上限，文件系统仍有 45 GiB 根分区和 3.5 TiB 数据盘空间。两轮历史标定也在同样告警下完成了 240 步。此次未调整系统配额。

已安装 Isaac Sim 4.5.0 的 `SimulationApp` 默认使用 `fast_shutdown=True`，文档字符串说明它可立即结束进程。Isaac Lab 0.54.4 的公开 `AppLauncher` 参数允许明确传入 `fast_shutdown=False`。独立标定入口现在传入该参数，在物理采样前读取实际 `/app/fastShutdown`，并写入 fsync 的 `shutdown_mode.json`。外层验证要求该设置为布尔 `false`，且标记 PID 与实际子进程退出记录一致。完成标记仍在 `app.close()` 返回后写入。首轮全部证据核验通过才启动第二轮。

Ubuntu CPU 回归使用 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_shutdown_validation.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_physx_polled_contact_audit.py`。新增回归在修复前 11 项失败，修复后 61 项通过，0 项失败。修复后日志 SHA256 为 `ead06af028c989cfa0bb400ce5c3c76825caf125bfbbfc2aae661a82e6c150cb`。此提交未运行 Isaac，下一次原生验证需使用独立冻结快照和新输出目录。逐接触力学资格仍待实测。
