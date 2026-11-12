# A 左拇指 v2 有界调度器配方

`configs/dh116_execution/deformx_thumb_left_lifecycle_v2.json` 固定左侧、一个环境、GPU3、无训练、240 个原生物理账本行。指纹包括入口、动作和生命周期源码、GPU readback、SDF、PhysX parity、固定 F1 扩展及默认无相机经验文件。native 入口继续独立核验原 USD、GPU hull identity 和 SDF 摘要。原生脚本的关节状态、mimic、SDK target、动作边界与两侧接触力检查结果进入 `sample_validation_passed`。调度器读取持久化 artifact，要求左侧身份、锁定 SDF、零接触窗口、样本通过、无失败以及恰好 240 个完整 JSON 行。只有共享锁已观察、时限内自然退出 0 时才继续产物验证。

在临时目录用调度器提交 `709f8a2` 的真实实现做 CPU 预检。源文件和配方通过身份检查。首次左侧运行的 240 行和四类旁证仅用于验收产物结构。另启动纯 CPU 假进程，复制这些证据、持有临时锁后返回 139。真实 scheduler worker 路径正确保留失败，不授予资格。该 fixture 未导入或启动 Isaac。调度器外层验收独立于 `app.close()` 的返回标记。

右侧入口保留原手工外层证据方式，并可使用 `--left-scheduler-receipt` 和从可信调度器取得的 `--left-receipt-sha256`。该入口核对实际 attempt 的路径、intent 与命令/代码配置身份、独立 outer exit 0 和锁证据、launcher 日志 SHA，以及被 receipt 绑定的 240 行账本、结果与关闭标记 SHA。CPU 测试分别拒绝真实外层字段不一致和等行数的账本篡改。未生成右侧原生执行配方。

| CPU 证据 | SHA256 |
|---|---|
| 左配方 | `603856032f7362c908d85cd8b02331b8a503383963b4fa365290dbdc69ea5f07` |
| 临时布局代码与配置指纹，package tests 仅含 thumb | `2477d57d19c91ee28fffdb07f0e76b737e47af4fda5172fb0a297364cc92c01c` |
| 临时 CPU 调度器测试汇总 | `6de45d6b04fd869ec2a252a774060c7264bf810f2804bda6363ae662ff73a356` |
| 13 项 OpenUSD 回归日志 | `de2fa3ff2ddbe9b85b81f5c583fab9b164f4a9902f95150ab6f69a347b58ba37` |

证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-a-thumb-scheduler-preflight-ZtrfZa`。OpenUSD 回归 13 通过，裸解释器拇指与生命周期测试 7 通过。528 个保护文件及 2 个历史 Checkpoint SHA 核验通过。本包没有改 registry、没有启动原生任务。合入后应按实际 registry 的测试集重读最终指纹及批准范围。
