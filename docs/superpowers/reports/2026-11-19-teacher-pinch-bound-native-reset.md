# Pinch 重定位候选原生 Reset

旧构造器固定读取原始 pair。本包把通过 CPU 净空检查的新 pair 路径和 SHA 固定在采集计划中。构造器复核每侧来源与同一 Reset seed。外层自然退出回执、原生 acquisition、最终 manifest 共同绑定此 pair。Teacher Worker 只从已验证的 manifest 选择 pair。新 seed 省略 pair 会被拒绝，历史 retained 的规则单独保留。

| 项目 | 本次结果 |
| --- | --- |
| 新 pair SHA | `eb60404a0f9a889edfd377e1ef59d989e0ca1498ec45e7e636810678a9cad063` |
| 采集计划指纹 | `58cce32c4c1f20cdde6e1b4a6abe1f9806ba485f713b113d81f1ce3d97aa4de8` |
| 实测计数 | 1 环境、2 SDK 初始化步、1 次 Reset、0 控制步、0 solver step、0 PPO |
| 原生测量 | 左右实际 source 和锁定指腹初始几何通过，Clearance guard 通过 |
| 自然退出 | 外层 0，45.08 s，正常关闭，8 项原生读回工件 |
| 自然退出回执 SHA | `63ec28aa67e5be512072a2a838b130bed5899f37ad8cd11ad37b5c648d402ddd` |
| 解析 manifest SHA | `ea45be098f8ab92be6ac0d8ac62063d8189aa56790c866d19bb798a9cd0c12bc` |

`record_dh116_teacher_numeric_reset` 使用成对的 `--pair-proposal` 与 `--pair-sha256`。先冻结，再用同一命令加精确 `--approved-frozen-sha256` 运行新独立输出，随后提供外层回执和 SHA 进行 `--manifest` 封存。本轮留存原始文件于 Ubuntu 私有 evidence。读取先前已封存的结果可以继续验证来源和 SHA，无需再次启动 Isaac。

`prepare_dh116_pinch_ppo_draft` 在新 manifest 上通过实际 CPU 源码预检。独立注册 v7，保持 1 环境、32 控制步、最多 258 SDK 步、3072 求解器子步、1 次更新、内外 180/240 秒。候选指纹 `380a08af61036143ae4d39784a31e1a140aaef1bb4408994cb69584d0cf1ca9f`。该工作包有独立原生执行授权门槛；没有 v7 更新或抓取验证结果。

相关 CPU 启动、证据与调度回归 133 项通过。另一个包含历史 R08 原始回放的套件报告 67 项通过、1 项失败、3 个 setup 错误。失败的首个失配是归档期 builder SHA `5b190d18…` 与本轮基线 `fd429adb…` 不同，其他被归档锁定的源也已更新。保留该失败与归档校验，独立新 manifest 路径通过现行校验。30 mm / 5 s / 0.5 mm 成功门槛继续适用。
