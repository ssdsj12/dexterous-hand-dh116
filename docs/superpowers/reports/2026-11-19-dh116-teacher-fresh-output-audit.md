# DH116 Pinch 首次更新的独立产物审计

代码审查基线为 `822ff0a74d53f4b0cabff74a38a9dbf6706eb66d`。Antigravity 三轮只读复核与 Ubuntu 核验的调度器、证据生产者和旧 C06 审计 SHA 一致。首次建议直接复用旧入口，复问分离两种 scope 后确认了误拒。旧 C06 将原账本 scope 用于匹配奖励 stamp。实际 Pinch `NativeJournal` 使用 `owner_source_unqualified`，`NativePinchSources._stamp` 使用 `unqualified_source`。原采集与奖励范围均保留，实际分支显式核验两者。CPU 参考模型验证 checkpoint 字节、拓扑、Adam 与策略身份。

## 修改

- `freeze_actual_fresh_artifacts` 逐文件单次读回 R06、原始账本、派生文件和检查点。重新构建确定性的派生视图并比较字节。篡改任何一侧、将结束标记写成已验证外层退出，均被拒绝。
- `audit_actual_fresh_outputs` 只接受实际注册的 A-fresh 角色。在原 32 控制步／256测量步的物理关系上额外要求实际 41 材料节点及配方锁定的后端 SHA。调用原模型输入重建，冻结到 checkpoint seal 前的原始账本前缀，并与同一 seal 的 checkpoint SHA 核对。CPU 审计读取同一 checkpoint 字节，核对模型、权重、优化器和更新计数。
- 调度器仅在自己观测到自然退出 0、GDB inferior exit 0、共享锁和真实耗时后进入该审计。现阶段输出可记录已核验的原始来源并保留失败原因。初始 setup 及 episode cap 的独立来源、正式 receipt 重验仍待接入。采用 `actual_setup_and_cap_receipt_audit_pending` 的阻断原因。正式 A/B 晋级、抓取与 Teacher admission 均未授予。旧 v1/v2 验证默认行为保留。

## 验证

新用例覆盖 scope 正负例、旧 C06 兼容、R06/检查点/派生视图被替换、错误的外层退出声明、模型输入前缀缺失或和封存记录不一致。首次重跑旧 CPU Checkpoint 套件时，其历史 R08 夹具被当前 builder 的不同文件 SHA 拒绝。测试改为只读使用已经独立封存的当前 Pinch Reset manifest，再复现 scope 合同缺口。生产 Reset、历史输入和历史索引都未替换。最终相关 CPU 回归 **220 项通过**，日志 SHA-256 为 `ce1fc99367da7a210858014eed4a2a3500b411bde4ed9384b990e6ee3239d24c`。

本轮没有启动新 Isaac。新原生 PPO 更新、原生资格和抓取合格回合均为 0。现有 WebRTC 单实例锁仍占用。下一步完成 setup/cap 证据和正式回执核验，再冻结首轮实际配方。Tripod 与 Power 的候选静态净空和所需反力见同轮报告。
