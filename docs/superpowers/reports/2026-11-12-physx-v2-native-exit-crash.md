# PhysX 修正版 v2 原生复验与退出崩溃

2026-11-12 03:04:53 UTC，在 Ubuntu GPU 3 以单实例启动独立左拇指标定。冻结源码清单 SHA256 为 `31c851bc42961b652d277f6e5abb6af002b416b1616512506e9beffd374eb1f4`。真实环境为 Isaac Sim 4.5.0 和 Isaac Lab 0.54.4。

| 验证项 | 首轮原生结果 |
|---|---|
| 物理原始证据 | 240 个连续 480 Hz 子步，SDK 步 3–242，摘要与原始 JSONL 的 SHA 一致 |
| 与此前 v1 对照 | 排除 episode UID 后，240 行所有记录逐行相等 |
| 关闭模式 | Kit 读回 `/app/fastShutdown=false`，标记 PID 与实际子进程一致 |
| 关闭前清理 | 两个回调释放，所属 `SimulationContext` 清除，原始数据 fsync |
| `app.close()` | 6.460 秒后返回，`shutdown_complete.json` 存在，未记录 callback 异常 |
| 进程退出 | 返回后 0.487 秒收到 SIGSEGV（子进程状态 -11），监督器未发送信号，未超时 |
| 最终状态 | 外层验证退出 1，第二轮未启动，原生关闭及重启资格仍未通过 |

原始 JSONL SHA256 为 `b2c5f0a6501b397136cd2ed6b08be66590446b399e94363f8de4c5e2dd5a7164`。监督日志 SHA256 为 `b42b03b6207f4de6bfeb770c9d5f62b1b2dde469f65d64aa71d68d78156ef83b`。数据与关闭标记保存在 Ubuntu 的 `/tmp/dh116-stereo-shutdown-20261112-v2-run/initial`。

未找到 core、minidump 或能定位故障对象的栈。已安装 `SimulationApp.close()` 在返回前卸载 Kit 插件，而标定函数此时仍持有接触和刚体 Tensor view、SimulationView 及其它原生包装对象。这是可独立修正的生命周期顺序问题。现有证据把异常定位到关闭返回后的进程退出阶段，具体触发析构器仍待定位。

物理数据和标记仅来自隔离的刚性标定。逐接触 A/B、作用侧、外部支撑、R1/R2 及 Force Proxy 资格仍待独立原生证据。
