# C03 独立奖励配置与 CPU 状态结算器

日期：2026-11-15。基线 `3011160cc9df31f9430b167dfd65047845e3e4f5`。分支 `codex/teacher-c03-reward-20261115`。

## 结果与边界

新增两个独立版本配置、纯数学内核、不可变端点来源 schema 和实际运行的 `TrainingReward.reset_from_verified_state / begin / observe / settle_once`。只完成 C03 的 CPU 软件部分：两 profile 的数学和状态结算已通过测试，真实 native source adapter、与原 evaluator 共源的测量绑定、factory/collector/Teacher/PPO 串联仍待 C01/C04。不能据此宣布 C03 的原生来源验收或整个 C03 完成。

所有 endpoint 的 scope 必须是 `cpu_fixture` 或 `unqualified_source`；后者也不是原生资格。engine/packet 的 `native_ready`、`teacher_admission`、`native_qualification` 恒为 false；没有 `qualified=True`、solver/source 占位 shim 或原生准入创建接口。CPU 5 秒 Hold 是 480Hz 合成端点测试，不能当实际抓取、原生 Hold 或真实学习证据。

没有启动 Isaac、GPU、原生物理或训练，没有安装依赖、写入真实模型/训练输出或推送。旧 R0 null 合同、旧训练配置、Teacher/PPO、物理及旧 R1/R2 开关保持原字节。

## 数学与来源合同

配置绑定用户批准的新 DOCX SHA256 `a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9`，状态为 `design_approved_native_unverified`。`load_config(path, *, ppo_gamma=None)` 返回不可变 `RewardConfig`；`config.gamma` 只读取同一 `config.ppo['gamma']`。独立 reward gamma 被拒绝；传入 trainer gamma 时必须相等。配置原始字节 SHA256 和 schema SHA256 进入 packet/context。

- smoke：`Phi=2*p_distance`；成本 `.05*q_command+.10 reward/s`；可信原硬失败 `-40`。contact/persistence/geometry/lift/Hold/slip/velocity/success/额外 support reward 都显式 `profile_excluded`，不要求 full 专用来源。
- smoke 仍读取原 numerical health 和 constraint guard；若 reset 的原 guard bundle 含 support 等失败，其 `original_guard_*` 字段每个端点必须完整保留，并触发原可信失败。不会为 smoke 新增 full support 测量门槛。
- full：`Phi=2*p_d+2*p_c+.5*p_p+p_g+4*p_l+4*p_h`；成本 `q_slip+.05*q_command+.10 reward/s`。源完整的无接触是有效零，不当数值故障。
- d_ref 只在本次 reset 取 `max(d0,.005m)`；目标材料 IDs、required-pad 分母 IDs、reset median/min z、材料及切向锚身份/坐标/初始 tick 在回合中固定。丢失接触后保留锚，重接触不清漂移；同 episode 不允许再次 reset 清理参考。
- 接触持续时间和 Hold 首个合格端点均为 0。5 秒需要 2400 间隔 / 2401 合格端点。Hold 用瞬时高度、原接触/对向、真实承托、离桌、无辅助、原 hold_motion 及原滑移三值判据计算；中断清零，可产生负潜势差。
- 净高度用当前 `min(median_z-reset_median_z,min_z-reset_min_z)`，没有历史最大值。材料/切向 drift+uncertainty 的最大值只在一个纯函数 `conservative_slip` 中计算；`slip_status` 保留合规/违规/unknown 三值。超过 0.5mm 本身不会新增终止条件。
- 本模块消费 sourced anchored drift，不连接原有有状态私有 evaluator reader。与原 evaluator 共源仍未接通；未来 adapter 必须供给同一底层实际测量，不能把 CPU fixture 或 source 字符串当实际执行证明。
- full 初始从未双侧接触时，仅接受 `raw=None, applicable=false` 且有 `no_bilateral_contact_since_reset` 来源依据的滑移 NA，贡献 0；一旦有锚/接触历史就不能回退 NA。应适用但未知、NaN、坏单位、错时、换参考或 guard 丢字段触发 `SourceFailure`，保留 `failed_endpoint`，拒绝 packet/后续采样；collector 必须隔离整份未完成 rollout，不得用 `-40` 伪装有效负例。
- 原健康、约束、Hold motion 独立。可信失败全部保留 raw/reasons，多个失败取最大罚一次：普通 `-40`，full 禁止辅助 `-50`。失败优先于 `+100` success；来源损坏先于事件。事件与 token 不能重复发放。

full `Cmax=2.15/s`，gamma `.999` 时无限折扣成本 `2.15/(60*(1-.999))=35.833333...`；smoke `.15/s/.995` 时为 `.5`。loader 重算并拒绝 `.9995` 对应 `71.666666...` 超过 40 的配置，不修改成功阈值或放松罚界。成功 `.03m/5s/.0005m` 为只读常量。

两 profile 首版 finite task deadline disabled，remaining 为显式 NA。外部 cap 用 truncation；本包不凭未知 deadline 制造 terminal。

## 窗口与输出

`begin` 只消费 owner 对应 PRE 时钟的归一化 `normalized_submitted_command24_preprojection`（24 维、有限、[-1,1]），并在启动前要求至少 8 个实际物理步预算。不是 SDK targets 或 action sample 副本的执行证明。`q_command=mean(((request-previous)/2)^2)`，真实前一 request/token/policy 留在 context。

每个 `observe` 只接收同 env/episode/reset/rod/world 的连续 480Hz 端点；来源 scope、单位与本端点时钟逐项检查。正常窗口必须完整 8 步；可信第 3 子步失败只结算 3/480 秒。没有虚造剩余物理样本。

`R=gamma*endpoint_Phi-Phi_pre-sum(actual_dt*C)+event`。真实终止的 endpoint Phi 为 0，仍保留真实 post Phi、端点和上下文；bootstrap 为 0。truncation 保留真实 Phi 和 shaped V、done=true；buffer 满保留真实 Phi 和 shaped V、done=false、不 reset。`packet.bootstrap(V)` 只返回 `gamma*V`，不会再加 Phi。

packet 保留 token/policy/profile/version/config/schema SHA、真实 duration、pre/post 原始潜势、吸收端点潜势、逐物理步成本与各项 metadata、事件与原始来源、真实 pre/post endpoint/context/terms。term 包含 raw/unit/valid/applicable/source/contribution；context copy-on-write。重复 token、重复 settle 或终止后 observe/begin 被拒绝；引擎没有发动作功能。

## 验证与私有证据

统一 CPU 命令前缀：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest
```

首版提交的相关回归实际执行：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q \
 tests/test_native_teacher_reward.py tests/test_dh116_r0_reward_domain.py \
 tests/test_dh116_r0_applicability.py tests/test_r0_health_separation.py \
 tests/test_dh116_deformable_control_step.py
```

首版实际结果 **134 passed / 0 failed，exit 0**：新 C03 59 项，原 R0/health/control 必要回归 75 项。覆盖 gamma 同源/罚界、折扣 contact loop telescoping、正常负例和 NA/unknown 区分、终止/截断/buffer bootstrap、3/8 可信失败、2400/2401 Hold、下降/中断负进度、失败优先、跨时钟/目标/分母/锚/源防篡改、重复 token 和 malformed schema quarantine。第一轮 GREEN attempt 的唯一失败是测试手算错误：未缩短默认 distance 时 full 最大 Phi 为 11.5；测试误写 13.5 后已纠正，未改奖励数学迎合错误预期。

日志位于私有 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence`，不提交实验日志。实际 RED/GREEN：

| 日志 | 实际结果 | SHA256 |
|---|---|---|
| c03-reward-red-20261115.log | 缺新模块，collection error，exit 2 | a196b3da04f7a6f3eb1071275c88dc264255cad57058fb162f332bea71cdb441 |
| c03-reward-red-assertion-20261115.log | 明确缺模块 assertion，1 failed，exit 1 | 3338fab184c94ffc1b2aef81ae624d165825b73670072f53b1ca81d54e2aecc9 |
| c03-reward-green-attempt-20261115.log | 38 passed / 1 failed，exit 1 | 51faab98edf4609deb771f0d0c2e2fe79c241a21f53279d32116ca3faed2a735 |
| c03-reward-boundaries-red-20261115.log | 新边界 7 failed，exit 1 | 62163a3acc5df9de5d12d10d5c9974526e0e6eb4d32e92a20ac2e7aba9b3041c |
| c03-reward-audit-red-20261115.log | 独立只读审计+范围检查 5 failed，exit 1 | 95d13f27a8be46c8cea39e35282bc2688397d73540c1ba46059089dcbc8bd070 |
| c03-reward-nested-schema-red-20261115.log | nested corrupt schema 4 failed / 1 passed，exit 1 | 086f992947c994c0c32737a0a21791559b8b726013bc6105b5c782904b2bd67f |
| c03-reward-final-regression-20261115.log | 134 passed，exit 0 | 07009a486c3a9ba87cddc8db7783742549bcb8478253f6e8b6eaa971f6257c11 |

## 范围错误及恢复记录

开发中一次只应更新两新配置 schema 的临时脚本误用了 `configs/dh116_training/*.json`，向 80 个旧 tracked 配置追加 schema 字段并重排格式。发现时没有暂存或提交；初始 worktree clean、HEAD 和独占工作目录有已读取证据。该脚本已停止，后续任何新配置处理只能用明确的两新路径；没有保存会再次执行的全目录写脚本。

恢复逐份先确认当前 JSON 精确等于 HEAD 原 JSON 加本次 schema 字段；然后用 `git show HEAD:<明确路径>` 原始 bytes 覆盖本次错误，立即比较恢复 bytes==HEAD。所有路径 resolve 被限制在该专属 tree 的 `configs/dh116_training`，未调用 reset/批量 checkout，也没有写入其他工作树或保留的用户修改。新两配置保留。

私有恢复清单 `c03-restored-own-config-glob-20261115.json` 含 80 明确路径、HEAD/恢复后实测 SHA256 和错误输出 SHA256；错误输出 SHA 明确标注为通过该确定性 JSON transform **重构**，不是伪称事前抓取。清单 SHA256：`e8080709793116f058097109feb949f6b063ec3036911d5ecf4b86dd1902006e`。

恢复后全部 **3044** 个 HEAD tracked blob（含符号链接目标文本）按 Git blob 算法逐个和 `3011160` 的 object ID 精确比较，0 差异。私有清单 `c03-all-tracked-byte-audit-20261115.json` SHA256：`616657f638cf4e628bf5e95f8991629eb11a436ddcff28706b72020b189cbc30`。新增回归保证所有旧训练配置及原任务合同相对基线无 diff；此前错误配置不参与最终回归。

实际使用 `scripts.run_dh116_work_packages.Repository(Path.cwd(), evidence_dir=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence')).verify_protection()` 核对 **528** 受保护源码/config + **2** 历史 Checkpoint，通过；未调用 `verify_local_target`。保护日志 `c03-protection-20261115.log` SHA256：`f2fd6839bad780e97d55f54d04348ec94f7694baf175120ee37c6f5646892746`。最终提交范围只允许计划列出的新增 5 文件。

## 下一步

由 root 独立 SPEC/QUALITY 审查并按 Windows 发布流程集成。C01/C04 再绑定实际 reset/distance/guard、full 接触/锚/健康来源、唯一物理 owner、post_context/next value 与 collector quarantine。真实来源证明、原 evaluator 共源、原生正向、真实 PPO update/save/resume、实际抓取仍未完成；本报告不赋予这些资格或消耗其预算。


## 首次 SPEC 复审后的两项修复

独立 SPEC 审查实际复现两项 CPU 缺口：非空 list/dict/int/bool 可被当作 `SlipAnchor.reference_uid` 接受，可变 UID 会通过已发 packet.endpoint 和 post_context 共享别名；合法 precontact NA 的 early return 又绕开 per-field source locking，允许在一个 episode 内换 NA reader。这些是软件实现缺陷，首版 134 通过不构成两项边界通过。

本次先增加失败测试，再修正实现：

1. 锚 reference UID 必须是原生非空字符串，空白字符串也被拒绝；不 stringify、不靠 shallow freeze 掩盖可变对象。非法 UID 在 observe 触发 `SourceFailure`、保留 failed_endpoint、quarantine 并禁止 settle 出可用 packet。
2. 一并检查同类身份入口：SourceStamp episode/reset/rod/source/scope/frame、request token/policy/representation、target/pad IDs/side、measurement unit/history_reason 都使用原生非空字符串校验，拒绝可变对象借重载比较冒充文本。required-pad side 在 set lookup 前校验，坏 list 不再泄漏 TypeError 绕过 quarantine。
3. 所有合法 NA 在 early return 前也执行同 episode per-field source invariant；第 1、5、8 子步切换 slip NA reader 都被拒绝。真实 `None/applicable=false` 且历史来源不变的 NA 仍贡献 0。smoke 仍不消费 excluded full slip source，即便这些未启用测量 unknown/invalid/换 reader 也不增加其来源门槛。
4. 新测试覆盖非法 UID 拒绝/quarantine，并验证合法请求、anchor reference、target/pad/fields/anchor 容器经外部修改后，已发 packet 的 endpoint/context 不受影响。

这是对原声明的字符串身份和同 episode 来源不变合同的实现修正；schema 布局与 digest 不变，两新 JSON 原始字节也未修改，没有执行任何配置目录 glob 写操作。两配置 SHA256 仍分别为 smoke `b4a74ce7390dbf825b3c132d11cddc40d2edeafa5af2d4ec27c15e20774bcac1`、full `dcbab3aee773d1b36ade049aba7076a2ee72a2e0c3ee40bcbb082d3b117c5d9e`。

独立证据全部新增为 `c03-spec-fix-*`，没有覆盖原 RED/GREEN 或 80 配置事故/恢复证据：

| 日志 | 实际结果 | SHA256 |
|---|---|---|
| c03-spec-fix-red-20261115.log | 8 failed / 10 passed / 59 deselected，exit 1 | 29fee302835f6dba6534d2a9af88156646103b9b9cbbdf3b186f8f0673011e0c |
| c03-spec-fix-metadata-red-20261115.log | 4 failed / 77 deselected，exit 1 | 0d146d2a58dcb3a76979a6696cdf5c2e192312afca8dace2437594b819f38e37 |
| c03-spec-fix-green-20261115.log | 22 passed / 59 deselected，exit 0 | 1740a6ec728f439791df65a5514eadc73f39f755ddbe6f826bc29e5c4964ca8a |
| c03-spec-fix-regression-20261115.log | 158 passed，exit 0 | 79478c916eacd8f5cd8869561e31ee0a800d1893247c0a2aefc3f4767728eba5 |

初始 22 项独立 GREEN 后将 NA 换源测试扩展为第 1/5/8 步；最终同上完整 CPU 回归命令实际 **158 passed / 0 failed**，其中 C03 **83** 项（首版 59 + 本轮 24）、原 R0/health/control **75** 项。独立 command/exit/logSHA 清单 `c03-spec-fix-command-evidence-20261115.json` SHA256：`11e1ca3d9a1572f6954ed1ebf0cd6c3628c8504958a1bba7e74efc5bf9552208`。

再次逐个比对基线 `3011160` 的全部 **3044** tracked blob/符号链接目标文本，0 差异；旧 80 配置逐份仍匹配恢复清单，原恢复证据 SHA256 仍为 `e8080709793116f058097109feb949f6b063ec3036911d5ecf4b86dd1902006e`。新审计 `c03-spec-fix-all-tracked-byte-audit-20261115.json` SHA256：`753f211ad1c492be9d9a18d2b9babee8a3b6f56aa5383b15b9eb6b926e689df2`。

再次按前述 `Repository(...).verify_protection()` 检查 528 受保护源 + 2 Checkpoint，通过；新保护日志 `c03-spec-fix-protection-20261115.log` SHA256：`f2fd6839bad780e97d55f54d04348ec94f7694baf175120ee37c6f5646892746`。本轮只改原 5 新文件集合内的 source/tests/report；两 config 未改，累计提交仍仅原 5 新路径。尚待同一 SPEC reviewer 复审及后续独立 QUALITY，未发布、未运行 native/GPU/训练。actual adapter/evaluator 共源 pending 和所有原生资格 false 的限制不变。


## 首次 QUALITY 复审后的三项修复

QUALITY 基于候选 8685de3d3cb70c1d780bf50d06b8e69a670c9c73 实际复现三项 CPU 边界缺陷：settle_once 的比较兼容可变 token 可进入 packet；10**1000 经 math.isfinite 抛出 OverflowError 后未 quarantine；未消费的附加 Measurement(raw=[1]) 保留外部可变引用。此前 158 项通过不能证明这些边界安全。

本轮按失败测试先行处理：

1. 活跃 window 的 settlement token 在任何比较或记录前必须为原生非空、非全空白 str。list/dict/int/bool/None/自定义比较对象均触发 SourceFailure，保留当前失败端点、quarantine，不能改用合法 token 补发 packet；不 stringify。
2. 数值谓词只接受原生 int/float，拒绝 bool 和可重载转换的数值对象；仅在该有限数值检查中将 OverflowError 视为输入域无效。reset、request、source clock、distance、height、coverage、anchor reference/drift/uncertainty 的超大正负整数与溢出转换对象均触发 SourceFailure 和 quarantine，重送原 tick 也不能恢复窗口。配置 JSON 中超大整数和指数溢出的 gamma 以 ValueError 拒绝；外部 trainer gamma 同样校验。没有 catch-all；专门回归证明实现自身 RuntimeError 仍原样暴露，不伪装成来源失败。
3. 端点中所有**实际保留**的 Measurement（含未消费字段和 smoke 排除字段）执行结构所有权检查。raw 只允许原生有限数值、bool、None，以及递归 tuple 组成的不可变数据；拒绝 list/dict/自定义可变对象和含这些对象的 tuple，不丢弃证据。元数据、source stamp、附带 anchor/reference/pad identity 也必须是已有不可变 schema 可持有的类型。合法 tuple raw 原样保留在 packet。
4. 所有权检查不要求 optional/full 排除来源存在，不对它们新增 valid=True、当前 endpoint clock、指定 unit、reader 锁定、四锚或双侧 pad 门槛。新增正向回归在 smoke 中保留 valid=False、旧 tick、改变 reader 的 tuple 附加测量，以及不完整但不可变的 optional anchor/pad 证据，仍得到正确 smoke 奖励。原先 UID/scope 严格校验和 full NA source lock 的 83 项测试继续通过。

原数学、权重、schema 布局/digest、两个新 JSON 原始 bytes 均未修改。本轮没有任何配置写入，仍分别为 smoke b4a74ce7390dbf825b3c132d11cddc40d2edeafa5af2d4ec27c15e20774bcac1、full dcbab3aee773d1b36ade049aba7076a2ee72a2e0c3ee40bcbb082d3b117c5d9e。

独立证据全部新增为 c03-quality-fix-*，保留先前所有失败、恢复及审计记录：

| 日志 | 实际结果 | SHA256 |
|---|---|---|
| c03-quality-fix-red-20261115.log | 65 failed / 7 passed / 83 deselected，exit 1；均为预期行为缺陷，无导入错误 | 104c79f2823cc5afb14dc810acbf96c920e4d63533ffb8e483cf0498dd393088 |
| c03-quality-fix-green-20261115.log | 第一轮 72 passed / 83 deselected，exit 0 | 13a1b0ff334e4811f3328a6b650ffa639b0360a87c5b22c576d2934abb43ad24 |
| c03-quality-fix-retained-schema-red-20261115.log | 扩展保留证据的元数据/anchor/pad 所有权测试，12 failed / 1 passed / 155 deselected，exit 1 | cdb27957ca5fea1f407902b59aa99591cb79fabf577c7886c4a4f9086c3f5381 |
| c03-quality-fix-regression-20261115.log | 243 passed / 0 failed，exit 0 | 94d960806dd434450e85e0b90ef5a92056d9de9966b2d842a3c21a14bd655793 |
| c03-quality-fix-protection-counts-20261115.log | 528 protected sources + 2 historical Checkpoints 通过，exit 0 | f2fd6839bad780e97d55f54d04348ec94f7694baf175120ee37c6f5646892746 |

完整回归严格使用本报告前述统一 CPU 前缀和同五个测试路径，实际 **243 passed**：C03 **168** 项（前版 83 + 本轮 85），相关旧 R0/health/control **75** 项。c03-quality-fix-command-evidence-20261115.json 保存实际命令、exit code 与 log SHA，SHA256 为 fa5bea49d806de7e9dcbb18927a92364bcbd48bf6bca314e68fc02ccbc0ba2bc。原始日志继续仅保存在私有 evidence 目录，不进入提交。

对基线 3011160 全部 **3044** tracked blob（含 symlink target 文本）再次逐个 Git blob hash 比较，0 差异；新审计 c03-quality-fix-all-tracked-byte-audit-20261115.json SHA256 为 c14c7973b671770e862e854d6b677df22531ee52b696be38835454b2e3f615b7。旧事故恢复清单中的 **80** 旧配置逐份与恢复后的 SHA256 相等，新明细 c03-quality-fix-restored-80-config-byte-audit-20261115.json SHA256 为 c86cc97c95efda242ceda617641f6feb7335f4b6a561fb920d8695f53393d565；原恢复清单 SHA256 仍为 e8080709793116f058097109feb949f6b063ec3036911d5ecf4b86dd1902006e。保护核对继续仅调用 verify_protection()，未调用 verify_local_target()。

本轮仅改 C03 既定五个新文件内的 source/tests/report 三路径，累计相对基线仍只增加原五路径。等待 root 对最终提交重新独立 SPEC/QUALITY 审查和 Windows 发布。未运行 Isaac/native/GPU/训练，未推送；实际 source adapter、与原 evaluator 共源的真实测量、collector/PPO 绑定仍 pending，所有 native/admission 标志保持 false。没有引入 C02 API 或把当前 mapping 当作实际 owner 绑定证明。


## 最终 SPEC 复审：派生溢出与可变 schema 子类

独立 SPEC 对候选 8e8ad4147b3b0ad6c56b5711e3a30259d7763504 复现两个真实缺陷。有限的 reset z=-1e308 与 post z=+1e308 相减产生 inf，或 drift=uncertainty=1e308 相加产生 inf；旧实现能在 min/clip 后生成有限 reward，却把无限 raw lift/slip 留在有效 packet。另一个入口是 SourceStamp 的可变子类：isinstance 接受它，子类恢复 __setattr__ 后可在结算后改写已发 packet 的来源。上一轮 243 项通过不能覆盖这两类问题。

本次先记录 RED，再修复：

1. 纯数学函数在裁剪、滑移上下界、成本求和与最终 reward 输出处拒绝派生非有限值，以 ValueError 暴露输入域失败；不把 inf 裁成合法进度/成本。高度的 median/min 两个 delta 分别校验后才能取 min，防止一个溢出被另一有限值掩盖。
2. source boundary 仅转换已核对的 reward 纯函数 ValueError/OverflowError 为 SourceFailure；reset、observe 与 settle 的来源失败均保留失败 endpoint、quarantine，并拒绝后续采样或可用 packet。派生 Phi、term raw/contribution、cost 和 numeric post_context 均有有限性检查。实现自身 RuntimeError 不被转换，既有回归继续通过。
3. SourceStamp、Measurement、SlipAnchor、RewardEndpoint、SubmittedRequest 和 RewardConfig 的入口均要求 exact dataclass 类型，拒绝可恢复普通赋值的子类。覆盖 reset/pre-request、begin、消费及附加字段 stamp、anchor 测量与 smoke 排除证据；不依赖子类的 frozen 声明或比较行为。
4. 正常有限饱和仍按原公式裁剪，滑移越界仍不新增终止。smoke 正向回归保留 drift=uncertainty=1e308、median/min=±1e308 等有限 optional full 原始证据，但不执行 full 派生数学；其 lift/slip 仍为 profile_excluded，reward 正确、未 quarantine。未增加 full 来源前置门槛。

本轮新增 42 项测试。首轮完整新测试 RED 为 31 failed / 10 passed；其中两个 clock metadata 测试最初同时含错 clock，会因既有 clock 检查通过，不能当目标域验证证据。修正为所有测量和 previous request 使用同一超大 tick/env clock 后，再实际得到 2 failed RED。原日志完整保留，没有把先前偶然通过冒充本次保护证明。

| 日志 | 实际结果 | SHA256 |
|---|---|---|
| c03-spec-final-fix-red-20261115.log | 31 failed / 10 passed / 168 deselected，exit 1 | cfa4422d1b186d5df7f5132939aa510057f2c96511f0b15f07b44eefbdd8f207 |
| c03-spec-final-fix-clock-red-20261115.log | 修正为完整同 clock 后 2 failed / 207 deselected，exit 1 | 27d3b4e7eba98347ac80767c90b22e1e44bd8689002fc5d09d617ed2c72983a1 |
| c03-spec-final-fix-green-20261115.log | 初轮 41 passed / 168 deselected，exit 0 | b73a3e3826575ccc2b5cfd58499b39d0282f223812bb6a1bb7a428229207dfaa |
| c03-spec-final-fix-regression-20261115.log | 284 passed，exit 0 | eb65153d1ce3a0c72a8b2189ef79d8ce62fbd641a64297ab394efdbf86a8700e |
| c03-spec-final-fix-final-regression-20261115.log | 加 smoke optional 大有限值正向回归后，285 passed / 0 failed，exit 0 | 8538169bab4755f756ca04ea71ddd23c0bbb61bcd5465a0673da993af4c9a8c8 |
| c03-spec-final-fix-protection-20261115.log | 528 protected sources + 2 historical Checkpoints 通过，exit 0 | f2fd6839bad780e97d55f54d04348ec94f7694baf175120ee37c6f5646892746 |

最终完整 CPU 回归仍严格使用前述统一命令与同五个测试路径，实际 **285 passed**：C03 **210** 项（此前 168 + 本轮 42），相关旧 R0/health/control **75** 项。独立 command/exit/log 清单 c03-spec-final-fix-command-evidence-20261115.json 的 SHA256 为 fafa4744a2c286731b12a35b72c31d3dba48bc02b048647f740d4dce92a0f74a。

再次按基线 3011160 逐个比较 **3044** tracked blob/symlink target，0 差异；新审计 c03-spec-final-fix-all-tracked-byte-audit-20261115.json 的 SHA256 为 c14c7973b671770e862e854d6b677df22531ee52b696be38835454b2e3f615b7。原恢复清单内 **80** 旧配置再次逐份匹配恢复 SHA，新审计 c03-spec-final-fix-restored-80-config-byte-audit-20261115.json 的 SHA256 为 c86cc97c95efda242ceda617641f6feb7335f4b6a561fb920d8695f53393d565。所有先前日志及事故证据保留；保护只调用 verify_protection()，未调用 verify_local_target()。

本轮仅改 source/tests/report 三个既定路径，累计相对基线仍仅五个新增路径。原数学、权重、成功阈值、schema 布局/digest 与两个新配置 bytes 未变：smoke SHA256 b4a74ce7390dbf825b3c132d11cddc40d2edeafa5af2d4ec27c15e20774bcac1；full SHA256 dcbab3aee773d1b36ade049aba7076a2ee72a2e0c3ee40bcbb082d3b117c5d9e。等待独立 SPEC/QUALITY 复核和 Windows 发布；未推送、未运行 native/GPU/训练。实际 source adapter、evaluator 共源、collector/PPO 绑定仍 pending，所有 native/admission 标志保持 false。


## 最终 QUALITY 复审：配置身份与真实文件来源

QUALITY 对候选 4a2e44ee15bf6aa0fc1039ac6349ee142ed03078 又复现配置身份别名：exact RewardConfig 经 dataclasses.replace 写入 list 型 config_sha256/profile_version 后，旧 TrainingReward 仍允许结算，外部修改会改写 packet 和 post_context 中的身份。此前 exact dataclass 类型检查不构成配置内容或 loader 来源证明，285 项通过也未覆盖此入口。

本轮先新增 56 项配置测试，实际 RED 为 48 failed / 8 passed，再完成以下修复：

1. 统一严格校验全部 retained config metadata：profile/schema 为原生已批准字符串；config SHA 为原生 64 位小写 hex；version 必须 exact int 且为 1；cost maximum/discounted bound 为有限原生数值并重算匹配。list/dict/比较兼容对象、bool 代整数、超大数值等在 engine 创建前以 ValueError 拒绝，不能进入 rollout。
2. weights/costs/events/ppo 只接受既定 flat schema，keys 为原生字符串，leaves 为已批准的原生有限数值或明确 None。计数为 exact int，拒绝 bool/float 代整数；拒绝未知 key、嵌套容器和任意 caller 修改的权重、成本、事件、PPO 参数。被实际接纳的四个映射均为 MappingProxyType 加不可变 scalar/NA leaves，持有深度不可变的 schema，不用浅 freeze 掩盖可变来源。
3. load_config 仍实际读取文件 bytes、验证设计合同并计算 SHA；新增 private init=False 的源文件路径绑定，dataclasses.replace/直接构造不会继承该来源。路径本身也不构成证明：TrainingReward 会重新读取该路径的实际文件、重算并校验完整配置，将 SHA 和全部 public payload 与 caller 对象核对，最后只持有这次 fresh loader 实例。没有以自报 SHA、bool 或 private 路径本身授予来源资格。
4. 源文件在初次 load 与 engine 创建间变更（包括仅追加换行）、丢失或参数变化都会拒绝旧身份；用户重新 load 合法文件 revision 后得到其新 SHA。已发 packet 仍保留原实际 bytes snapshot 的身份。三个额外保护测试证明：复制来源路径后伪造合法形状 SHA/自洽修改 gamma 仍拒绝；合法副本经真实文件重验后，即使外部使用 object.__setattr__ 修改该副本，也不会别名到 engine/packet。
5. 配置构造只是数据 API，不授予 actual file 来源；配置类的可变子类仍被拒绝。测试 helper 改为仅传递 dataclass init=True 字段，保持原 exact-type 子类拒绝测试仍实际执行 engine 边界，避免新增 private init=False 字段造成测试构造错误掩盖目标行为。

原两配置 bytes、数学、权重、成功阈值和端点 schema/digest 未改；RewardConfig 私有源路径只用于软件初始化的文件重验，不进入 packet schema。smoke 的 full 来源前置没有增加，既有 unknown/NA/optional measurement 及所有来源隔离、派生有限性回归继续通过。实际源文件需在 engine 创建时仍可读取且字节不变；需要改文件时应重新 load，而不能沿用旧 SHA。

| 日志 | 实际结果 | SHA256 |
|---|---|---|
| c03-config-origin-fix-red-20261115.log | 48 failed / 8 passed / 210 deselected，exit 1，无导入错误 | 61bd5d6345f9cee603c198d65e6c443bff3f56978da945fe3d9a0f1b44e6ae3e |
| c03-config-origin-fix-green-20261115.log | 首轮 56 passed / 210 deselected，exit 0 | 89456f1c5e2694eab56949a60799978d0ac2f4bfb31cb38d24511cf9a621edb5 |
| c03-config-origin-fix-regression-20261115.log | 加 3 项来源复制/所有权保护测试后，344 passed / 0 failed，exit 0 | 45b4a94c717a299345db9e7ed37848f967bfe2551739549bccf6105016909411 |
| c03-config-origin-fix-protection-20261115.log | 528 protected sources + 2 historical Checkpoints 通过，exit 0 | f2fd6839bad780e97d55f54d04348ec94f7694baf175120ee37c6f5646892746 |

最终完整 CPU 回归继续严格使用本报告统一前缀和同五个测试路径，实际 **344 passed**：C03 **269** 项（此前 210 + 本轮 59），相关旧 R0/health/control **75** 项。新 command/exit/log 清单 c03-config-origin-fix-command-evidence-20261115.json 的 SHA256 为 8b1f99dd5c81ea706fdf339ee2181cd788fa21c1634e89c997b4620bd1de09ce。

基线 3011160 的 **3044** tracked blob/symlink target 再次逐个相等、0 差异，新审计 c03-config-origin-fix-all-tracked-byte-audit-20261115.json SHA256 为 c14c7973b671770e862e854d6b677df22531ee52b696be38835454b2e3f615b7。恢复清单中的 **80** 旧配置再次逐份相等，新审计 c03-config-origin-fix-restored-80-config-byte-audit-20261115.json SHA256 为 c86cc97c95efda242ceda617641f6feb7335f4b6a561fb920d8695f53393d565。先前全部失败与恢复证据保留；仅调用 verify_protection()，没有 verify_local_target()。

本轮只改既定 source/tests/report 三路径，累计相对基线仍只新增原五路径；没有配置写入。两配置 SHA256 保持 smoke b4a74ce7390dbf825b3c132d11cddc40d2edeafa5af2d4ec27c15e20774bcac1、full dcbab3aee773d1b36ade049aba7076a2ee72a2e0c3ee40bcbb082d3b117c5d9e。仍待独立 SPEC/QUALITY delta 复审和 Windows 发布，未推送、未运行 native/GPU/训练。真实 source adapter、evaluator 共源、collector/PPO 绑定仍 pending，所有 native/admission 标志保持 false；配置软件来源重验不授予这些资格。
