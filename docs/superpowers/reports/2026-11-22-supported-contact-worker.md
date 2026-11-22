# Supported17 Tripod：原生 Worker 接触 profile 与预热账本接线

基线为 `e46b86465db1707aa7b2b17c11934ee1b13108f9`，分支 `11-17`。本包复用现有 Collector、Actor、PPO、Checkpoint 和进程执行器，接通独立 `approach_contact_prelift_v1` 的 worker 合同。CPU 替身替代 SDK、SDF 和真实参数 getter，原 Collector 连续8个策略控制步、原 PPO 一次更新及检查点封存实际运行；**不等于新的原生训练、真实接触通过或抓取成功**。

## 实际修改

| 原文件/函数 | 行为 |
|---|---|
| worker_contract.teacher_profile_limits / validate_profile_rollout / validate_teacher_recipe_v2 | CONTACT 仅允许 Tripod；完整来源闭合包含预热、Context、测量、contact evidence、奖励配置和新测试。独立预热计入总预算，不计 PPO transition。旧 SMOKE 保留。 |
| actual_provider._prepare_source_pack | 精确选择 NativeContactTripodSources，核验真实 rod、双手、参数 getter、owned history 与 SDK3/rod12 同身份；其他抓法不能借用。 |
| actual_update._run_update | 原更新前核验 Reset2→零动作 prime→SDK3 以及真实 runtime3/1/12，更新后核验总步数；原始输入审计绑定 profile，提前终止保留原账本但不封存 clean checkpoint。 |
| worker_artifacts._contact_prime / _physical_journal / _cpu_reference | 独立回放原始零动作协议、24D合同、准确env0、backend、Reset、同 owner receipt 和预算；V3 history 与 prime 摘要绑定，策略预算仍按1..8N。参考模型核验额外来源的真实文件字节。 |
| actual_artifacts._derived_payloads / _audit_fresh_setup / _audit_fresh_budget / _audit_sealed_packs / _audit_actual_outputs | 转发明确 profile；setup2、prime3、首次策略的物理执行终点4；墙钟 before_update 在SDK3，来源审计使用检查点封存前准确的原始 journal prefix。 |
| training.preflight_training | 不再把已实现的 ContactTripod 适配器误报为缺失；缺真实 C06 lease 时 native_execution_ready 仍为 false。 |

测试辅助默认 env7、旧 profile 均保留。新 worker 夹具在任何 Reset 前明确选 env0；真实 getter 对应的 distributed_contact 夹具从源头用 float32，未放松生产 array_equal。原生 Reset 证明 c123cc...只作为已有独立证据使用，未改写历史证明。

## 失败证据与独立审查

先写失败测试，复现旧 profile 拒绝、新 SDK 起点拒绝、缺 prime 预算仍可触发更新、旧预检误报。独立审计又复现攻击者同时修改 prime action protocol 并重签 receipt 摘要时可被接受；增加原协议、env整数身份、backend 和原24D合同核验后负例通过。初次夹具失败包括 env7、缺 owner.backend_profile、float64/f32零动作摘要差异、缺 setup phase 和源 float64 小数差异，均保留失败日志，未改动原始数据。

第一轮18文件回归395通过、9失败：四项测试遇到并行修改源码后的严格 SHA 拒绝；其余五项旧 audit stub 缺新增 profile/endpoint 参数。修正仅测试夹具，并在最终回归期间冻结源码，严格 SHA 校验不降级。原始日志、失败 CPU journal 和 SHA 保留服务器私有证据目录，不上传大日志或模型。

Antigravity 通过指定桌面 SSH reviewer 完成首轮真实源码审查，源码摘要由 Codex 独立核对。无障碍文本有长度上限，另通过滚动截图阅读剩余回复，存储的部分原文与人工阅读笔记明确区分。第二轮因 Individual quota reached 未产生意见，**不算复审通过**；未切换模型、购买额度或伪造回复。最终实现由 Codex 独立复核并运行测试。

## 验证和预算

最终18文件联合CPU回归 **404 passed、0 failed**，644.17s；运行期间源码摘要保持不变。命令、日志SHA和独立失败证据见 readiness JSON。Registry/调度器回归另为 **60 passed、0 failed**，14.31s；两组不累加为新用例数。

新10项测试覆盖8/32控制步预算、错抓法/旧预算/缺来源拒绝、原8步 Collector/PPO/Checkpoint/封存审计、提前触桌无 clean checkpoint、provider绑定、缺 prime 在采样前拒绝、V3/原始协议/摘要重签攻击和SDK3墙钟起点。已有恢复与 Adam/RNG/checkpoint 检查继续回归；**本包未新增 Contact 原生恢复资格**。

| 控制步 N | setup SDK | 独立 prime SDK/solver | 策略 SDK/solver | 总 SDK | 总 solver | 总 solver 子步 |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1 | 64 | 67 | 65 | 780 |
| 32 | 2 | 1 | 256 | 259 | 257 | 3084 |

原 native worker 运行次数、原生 PPO 更新、原生物理/求解器步骤均为0。测试中的物理 transport 与 lifecycle 记录是明确 CPU 夹具，不能当外层自然退出证据。新原生配方尚未冻结，不能复用已消耗 SMOKE 预算。

## 保护和后继

核验528个受保护文件与3个历史Checkpoint。原24D/371D、左右手、固定双RGB、资产/物理/控制器及30mm Lift、5s Hold、0.5mm Slip标准保留；R2/ForceProxy关闭，正式BC/DAgger/Student与成功专家池继续门控。

下一包冻结 CONTACT Tripod 的真实配置、完整来源集合、GPU/墙钟与67/65/780预算，使用当前独立Reset证据，先执行单次有界诊断并逐原始接触账本验证。Pinch需独立 supported17 Reset/Source；Power需独立五指和掌面来源，不能复制Tripod资格。接触成功之后仍需固定腕保持、离架Lift/Hold和固定材料Slip分别通过。
