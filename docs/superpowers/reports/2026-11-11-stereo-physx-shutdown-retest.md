# PhysX 自然关闭与重启预检

状态为 `native_retest_pending_approval`。本工作包验证了关闭证据检查器、服务器环境和冻结副本。Isaac 自然退出仍待原生复验。

`scripts/verify_dh116_stereo_shutdown.py` 默认只做预检。原生模式顺序启动两次同样的左拇指固定标定，每次 240 个 480 Hz 物理步。首轮只有在完整 JSONL、原始数据 SHA、外层退出码 0、回调句柄释放、所属 SimulationContext 清理及 `shutdown_complete.json` 全部通过后才启动重启轮。两轮必须有独立输出目录及 episode UID。若子进程在 600 秒内没有自然退出，外层写入超时与 PID，终止验证流程，不发送 SIGTERM 或 SIGKILL，也不启动重启轮。超时 PID 仍可能运行，须复查该 PID 后再安排下一次 Isaac 实验。已知 SIGTERM 辅助的历史记录不能作为自然退出证据。

`scripts/calibrate_dh116_stereo_contacts.py` 在 `sampling_complete` 之前对 raw JSONL 执行 flush 和 fsync。两个回调订阅的所有者引用在清理 SimulationContext 之前释放并记录。既有 `SimulationContext.clear_instance()` 和 `App.close()` 顺序保留。退出判定还核对 SDK 物理步连续性、480 Hz 时间增量和 episode 身份。验证器要求子进程在 App 启动后记录同 PID 的 SIGTERM 监测标记，收到可观测 SIGTERM 时写入中断标记并拒绝通过。外层记录自己未发送信号和子进程的真实退出码。执行时须保留监督日志并保持无外部进程干预。

## 已执行检查

- CPU：`python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_shutdown_validation.py tests/test_dh116_stereo_contact_calibration.py`，41 项通过，0 失败，0.76 秒。覆盖截断、时钟跳变、episode 串线、缺失关闭标记、SIGTERM 辅助和外层超时。测试使用合成证据，不计作 Isaac 验证。
- 服务器：在此前测试快照的独立副本中替换两个预检脚本，使用 `/mnt/mydisk/xk/envs/dh116/bin/python` 运行默认预检，退出码 0。Isaac Sim 4.5.0.0、Isaac Lab 0.54.4、左拇指网格和不存在的输出目录均通过。单实例锁可获取，GPU 3 查询未返回计算进程。预检没有启动 Isaac。标定脚本 SHA256 为 `ba040b0eef25940ff1d2f5ada8ef788df4cbfe9293921f4679c35f2f2661c1ca`，验证脚本为 `1017688753844e21928233643b2e24583a77face13e7a484a568ce9dacd4f1dd`。副本的 1,293 个文件 SHA 已校验，清单 SHA256 为 `7e5aee5f71ab3c0808d1818b37fcfa185654f5424abed89f70f9b3f5dca15dce`。

## 待批准的原生命令

批准后先重查单实例锁、GPU 和副本清单 SHA，再执行：

```sh
cd /tmp/dh116-stereo-shutdown-preflight-e1M1IW/snapshot
sha256sum --quiet -c ../shutdown-source.sha256 && \
  /mnt/mydisk/xk/envs/dh116/bin/python scripts/verify_dh116_stereo_shutdown.py \
    --output /tmp/dh116-stereo-shutdown-20261111-run --execute-native
```

只有两轮自然完成后才读取 `shutdown_retest_passed.json`。这个标定不授予逐 contact、双手接触、R2 或 Force Proxy 资格。完整双手接触实验继续等待自然退出通过和独立批准。
