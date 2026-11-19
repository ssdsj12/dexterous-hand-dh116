# DH116 Pinch 实际 PPO 更新时限证据

Antigravity 的只读复核以 `e86e74bf4428dfe1c477c53125893f6114c69af1` 为基线。第二轮只列出了本地预算。追问后确认 `SessionBudget` 实际使用本地剩余和父绝对截止剩余的较小值。Codex 独立核对后保留原计算。

Worker 用 `wall_readback` 对同一单调时钟单次采样，在首个策略决策前和检查点封存后记两行。记录包含本地时限、session 开始、父绝对截止、剩余时间、实际 SDK 步和 episode cap。记录前核对实际会话、kinematic context 的时限来源和父租约。若更新或封存后预算已耗尽，Worker 保留原失败证据并拒绝记录完成窗口。父调度器用进程内的同一个 `ParentLaunchEnvelope`、冻结配置和原始账本重算时限。验证 SDK 步 2 和 258、同一 episode、顺序和剩余值。一轮普通 CPU fixture 不授予原生状态。现有自然退出、原生回执重验和抓取成功标准独立。

6 项新测试先复现缺口。联合 CPU 回归 186 项通过，含错父截止、过期、只考虑本地上限、错 episode cap 及错顺序。日志 SHA-256 为 `90a0c1fe59d6bdfeec496ac4b32550e5349335b56725ee6c736d14d6ad6586c3`。本轮 0 次 Isaac 新实例、0 次 PPO 更新。尚需正式回执重验和一轮独立冻结的原生执行配方。
