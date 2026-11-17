# DH116 J0：最新 HEAD、源码保护与 readiness 审计

审计日期：2026-11-10。目标分支：`codex/dh116-stereo-privileged-dagger`。

**本次审计起点、复核中间点及写报告前 HEAD 均为 `7c808781ebd2cf969ca08776a83a65e3a2834582`。主 agent 已核对 GitHub 最新分支仍为该提交；本 agent 只读检查的 `refs/remotes/origin/codex/dh116-stereo-privileged-dagger` 也指向同一提交，没有发现审计期间 HEAD 变化。** 本报告以该 HEAD 已提交的 Phase 0–7 为对象，不把同轮其他 agent 正在新增的 J1 诊断源码算作基线已有能力。

结论：Phase 0–7 隔离软件路径存在；528 个受保护原文件 SHA256 全部匹配。现有 readiness 所列 Stereo 精度失败、Teacher/Student 未训练、正式 BC/DAgger blocked、最终分辨率未选定及物理资格未取得，均未被本次源码审计推翻。后续缺口同时包括**尚未实现的接口/测量工厂**和**尚未取得的原生数据/资格**，不能概括为只缺一次训练。

## 审计输入与边界

已完整读取 `D:/download/DH116_Current_Branch_Detailed_Codex_Tasks.md` 的 786 行，文件 SHA256 为 `14a1373d7c430286942ae309b8f70453083c434885323945c807a0b9359c8e5c`。该文件与早期 Phase 0 报告引用的 930 行任务书属于不同版本，不能混用其文件 SHA。

本次读取材料：

- `docs/contracts/stereo_phase0_7_readiness.json`，835 行；包括全部 141 项文件清单。
- `docs/superpowers/reports/2026-11-10-stereo*.md` 全部 13 份：bootstrap-snapshot02、cold-trials、native-resolution-final、phase0-7-integration、phase0-audit、phase1-integration、phase2～phase7、resolution-v2。
- 三份 Phase 0 人读审计：`docs/stereo_privileged_state_contract.md`（543 行）、`docs/stereo_action_contract.md`（156 行）、`docs/privileged_teacher_reward_audit.md`（202 行），以及对应机器合同和 source manifest。
- 私有 `artifacts/stereo-phase0-20261110/baseline.json`；现有原生结果公开摘要及其引用的本地汇总材料。
- Phase 0–7 共 43 个新增 src 模块的文件存在性、SHA256、AST 顶层符号清单；对深度/感知/状态估计、监督 mask、R0/R1/R2、原生采样、BC/DAgger、Student PPO、raw RGB inference 和 promotion 主路径核对具体源码。

开始时工作区已有 `?? artifacts/` 和本轮实施计划；审计中其他 agent 新增了 J1 文件。未 reset、stash、覆盖这些内容。本 agent 的唯一写入是本报告；没有改旧报告、合同、生产源码或配置。

## 源码与文件保护结果

下列范围互有重叠，数量不能相加为独立文件总数。

|核验范围|方法|结果|
|---|---|---|
|私有原 `src/`、`scripts/`、`configs/` 保护清单|对 528 条路径重算文件原始字节 SHA256|528/528 存在；missing `[]`；mismatches `[]`|
|公开 Phase 0 source manifest|核对 21 条 critical file 原始字节 SHA256|21/21 匹配|
|action 合同源码绑定|核对 30 条 source SHA256|30/30 匹配|
|reward 合同源码绑定|核对 32 条 source SHA256|32/32 匹配|
|state 合同源码绑定|核对 23 条 source SHA256|23/23 匹配|
|state 合同 AST 符号|按名称、起止行号、`ast.dump(include_attributes=False)` SHA256 核对|24/24 匹配|
|readiness 文件清单|逐项检查全部 141 条路径|141/141 存在|
|旧 224 配置|原始字节 SHA256|仍为 `f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`|
|原全局精度政策配置|原始字节 SHA256|仍为 `82a12907c342b58c5ac7e14e7bbd0315e35ffe9f219d4248fe4e0c4fc4e228d6`|
|公开 native summary|与 readiness 声明的 SHA256 比较|`e216463b4b311e06b823c132b3fe73c79eb7de98bddd08ec6aeea4893e8fa63b`，匹配|

私有 baseline 清单自身 SHA256：`83828ebd8c3e42e7461b5b6090416126f48a813c903d06ffbb1694839911e2fd`。canonical action-only 语义 SHA 仍为 `498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9`。这些结果证明受检文件内容身份，不单独授予物理参数运行值、历史 checkpoint 或原生任务资格。

### 历史报告中三处陈旧指纹

13 份报告中，按 Markdown 表格提取的 31 条显式本地 `src/tests/configs/scripts` path→SHA 引用，28 条匹配当前文件，以下 3 条与当前 HEAD 不同。当前工作区字节均与 `git show HEAD:<path>` 一致；换行标准化也不改变差异，不能归因为 CRLF。

|历史报告位置|文件|历史 SHA256|当前 HEAD/工作区 SHA256|
|---|---|---|---|
|`2026-11-10-stereo-bootstrap-snapshot02.md:33`|`tests/test_dh116_stereo_native_bootstrap.py`|`3e7b77b8b0cb37505d46be1f2b50ec31bb3f325fdd29762d9902eb8db0f9258b`|`efec4a3a843748fecde20e3989c4354b77b91926d3d59f763a21ca9d66c61d57`|
|`2026-11-10-stereo-cold-trials.md:92`|`src/go2_pvcnn/control/dh116/tabletop_stereo_resolution.py`|`cefac066fc212017716a4aea96369c150e686e3fb3fbdeed815bc645027ff15a`|`dc77aeb3f391d7a4bc95c41e8ad5b9ad58016ed4ec892653c42717e16832555d`|
|`2026-11-10-stereo-resolution-v2.md:98`|同上|同上|同上|

保留这些历史报告原样；其指纹只可按对应快照理解，不能用来声称当前 HEAD 已通过同一 SHA 的验证。`stereo_phase0_7_readiness.json:7` 的 `head_before_final_document_commit=a8d19ee...` 也是已注明语义的历史时点，当前 HEAD 应另行记录为上文 `7c80878...`。

## 当前 readiness 与资格差距

readiness 的 Windows `564 passed, 1 skipped` 和服务器 `565 passed, 1 skipped` 是已有报告记录，本次 J0 **未重跑该套测试**。两个平台测试重叠，禁止相加。相机 10 次 `measurements_verified` 也是已有冻结 run 的公共重读结果，本次 J0 不重新授予 native verification。

|状态轴|基线已有事实|仍缺少的证据|
|---|---|---|
|implemented|Phase 0–7 接口/路径存在；readiness 所列文件齐全|独立 geometry gate、真实 R1/R2 factory、通用 wire segmentation/tracking、depth decoder、不确定性/validity heads 等仍有实现缺口|
|CPU verified|已有两平台 CPU 软件验证报告；source/AST 身份本次重新核验|后续每项修改仍需自己的 RED/GREEN 与适当回归；本次静态审计不能重授全部测试通过|
|native capture verified|旧 rig 4、front v3 4、roll20 block5/9 共 10 次既有配置采集|没有本次新采集；旧 rig 4 次 Wire ROI 因 v1 reference 证据不完整仍 NA|
|accuracy qualified|10 次均 false；`selected_resolution=null`|全局 MAE≤5 mm、RMSE≤10 mm、coverage≥50%、有效像素/帧数等原门槛仍未满足|
|physical task qualified|Teacher/Student 均 not_evaluated；训练 not_started|完整来源、力学资格、两抓法各≥100独立 first-episode 初态和完整原生成功证明|
|deployment qualified|Actor-only 导出和 RGB runner 软件存在|真实 Student-only heldout 资格、Actor/完整 sensor-to-action 性能及硬件验收；当前不能标已部署合格|

371D 逐维合同统计已从 JSON 重算：`proprio=64`、`RGB/depth=63`、`history_inferred=118`、`history=26`、`known_config=14`、`unobservable=86`；监督 `conditional=199`、`NA=172`。合计 371。**199 个 conditional 不等于 199 个已可训练标签。**

`tabletop_stereo_runtime.py:46–102` 只开放经过当前 SDK q/qd、Task 测量、归一化和身份逐维交叉核对的 `[0:24]`；`snapshot:139–152` 将同一 mask 写入来源键，其他来源 mask 保持 false，`force_qualification=None`，硬件反馈资格 false。该路由的正向上限为 24/371 维仿真 encoder 标签，不能广播到其他 proprio、geometry、force 或真实硬件。`tabletop_stereo_losses.py:110–133` 仍按允许维度与每项 `required_validity_keys` fail closed。

## J0–J10 gap matrix

|任务 / 对应工作包|基线已实现或已有测量|实现差距|资格/数据差距与下一步|
|---|---|---|---|
|J0 / 全局审计|最新 HEAD、readiness、43 模块清单、528 原文件及公开 SHA 绑定可检查|历史报告存在上列三处陈旧 SHA；readiness 尚未独立列出所有新任务的验收轴|本报告保留历史身份并记录当前缺口；不据静态结果授训练/任务资格|
|J1 / WP1 误差诊断|已有 global、Wire ROI、部分共同有效集与离线 SGBM 汇总|基线没有完整八分区、逐拒绝原因、cost/纹理/方向/曲率诊断；同轮新增诊断代码需单独报告验证|必须取得原 native RGB/reference/instance/calibration/proof/completion；缺 raw 时列 path/SHA，逐像素结果保持 NA|
|J2 / WP1 matcher、分割、跟踪、depth head|SAD/blockmatch、可选 SGBM、浮点 subpixel/LR 检查、颜色 skeleton、Transformer heads 已存在|Census、ZNCC/gradient、coarse-to-fine、通用分割/逐点跟踪、真实 depth decoder 与统一诊断输出未齐|先以 CPU synthetic/analytic 测试实现候选，保留 baseline；没有新 native capture 不宣称算法改进|
|J3 / WP2 rig/分辨率|224×96/320×240/640×360/640×480、自定义、K rerender/resize、front/roll profile、benchmark 与资源读数路径|`disparity_budget` 仅给 near/far 和 search range；还缺每 profile 统一的 wire width、uncertainty、显存估算及全部阶段独立性能报告；未完成全部 rig 因素单变量研究|已有 10 次精度全部失败；需 J1/J2 后另作有界同 cohort 原生重测；Actor latency、完整策略/训练吞吐仍 NA|
|J4 / WP3 Teacher diagnostic|371D/24D Tanh Gaussian、saved-latent PPO、R0 closure、checkpoint/resume、single-instance lock 与 terminal capture 路径|需补任务书要求的动作组 noise/entropy、tanh saturation、clipping/安全投影统计；R0 trace 保存的是已合成贡献而非完整逐项物理 raw|实际 Teacher 2 env / 1 update bootstrap、rollout、保存恢复尚未运行；相机采集不能代替 Teacher PPO 诊断|
|J5 / WP3 Teacher 训练/验收|独立 Teacher 训练/确定性评估、冷初态小 batch runner、专家 BC 准入存在|curriculum 分布/阶段 SHA 与完整训练预算/噪声诊断需随实际实验落地|没有新 Teacher checkpoint、完整合格示范或每抓法≥100独立初态；旧 261D/26D/短 Hold 不可静默复用|
|J6 / WP4 force/support factory|typed R1/R2 算术、严格 force proof schema 和 expected support share 静力模型存在|真实逐 contact factory、ON_OBJECT/partner/逐指切向来源、external/authored-force/attachment 审计、独立 geometry gate 未接通|现 NativeBridge 全部 force/support unqualified、external/attachment 为 NA；R1/R2 仍关闭，需真实 SDK/PhysX raw 来源审计|
|J7 / WP5 estimator 训练|shared GN-ResNet18、stereo cross-attention、temporal Transformer、371D masked loss/SO3、按物理初态与 source episode 分组|没有逐 head uncertainty/validity output、校准评估；低覆盖/双目冲突时明确安全策略未实现；native 监督仅 q/qd 开放|缺合格 geometry/contact/force target、真实训练集与 heldout 各字段误差/抖动/ID-switch/恢复报告；NA 不算 loss|
|J8 / WP6 BC/DAgger|pre-action clone、Teacher 同刻 deterministic query、整 24D owner、真实 step、D0–D4、独立 Adam、成功池/visited-state 分离|depth loss 正权重仍因无真实 decoder/qualified target 被拒绝；完整 native 轮次统计需实际数据|正式入口先验 Teacher+Stereo；当前两门未过，没有成功 BC pool 或正式 DAgger 结果|
|J9 / WP6 Student PPO|`StereoAsymmetricPolicy`、独立 Critic、复用同一 PPO、clean auxiliary boundary、checkpoint、Actor-only export 已实现|需把任务书要求的验证链、训练预算和来源登记落实到实际训练；不能从软件包存在推断 BC/DAgger 效果合格|没有 Student native 短更新/恢复或后续训练；须等待 Stereo、Teacher 和 BC/DAgger 结果后推进|
|J10 / WP7 最终评估/promotion|cold trial runner、480Hz 回放、Wilson95、first-episode 去重、heldout lineage 与 source proof gate 存在|`promotion_ledger` 仅公开 Teacher/Stereo/Student 三门；需将 state heldout、BC lineage、DAgger lineage、Actor deployment 等八类 proof 显式分列；geometry 应独立于完整任务成功|两抓法各≥100真实 first-episode 初态未执行；Teacher/Student 成功率和 Wilson 分母均 NA，不能写 0/100|

## 后续修改的具体源码入口

下列位置均对应基线 HEAD；不是本轮已实施的修改。

|文件与行号|源码事实与可验证修改方向|
|---|---|
|`tabletop_stereo_depth.py:12–16,55–97,112–128,160–161`|方法只允许 blockmatch/sgbm，输出仅 5 个 Tensor。保留兼容返回合同并增加独立 diagnostics 侧信息；候选 matcher 共用明确 interface/config/source 身份。|
|`tabletop_stereo_depth.py:50–53,84–96,151–156`|纹理、uniqueness、subpixel curvature 和 LR 检查均存在；但 `occlusion` 合并右图支持/纹理/置信/LR 失败，并不等于真实遮挡。分开记录拒绝原因与其重叠，避免凭最终 mask 反推因果。|
|`tabletop_stereo_depth.py:147,155–161`|SGBM confidence 是有效范围的 0/1，不是校准置信度。内部 z 运算有 denominator clamp，但最终只暴露 valid 位置，其余 NaN；后续须测试无效/非正视差永不变成有效正深度，并使拒绝原因可审计。|
|`tabletop_stereo_perception.py:38–54,83–96,100–124`|颜色候选不等于 instance label；拓扑问题统一返回 None；跨帧关联只有端点位移和总弧长。新增独立分割/跟踪，显式区分 branch/loop/fragments，记录逐点 confidence/visibility/track_id/arc/frame；不能仅复用数组下标当身份。基线不存在 `tabletop_stereo_wire_segmentation.py` / `tabletop_stereo_wire_tracking.py`。|
|`tabletop_stereo_state_estimator.py:15–16,36,77–86`|所有 head 为 latent 的线性输出，没有 depth decoder、uncertainty 或 valid probability。新增实际 RGB stereo feature depth 预测，保留 matcher/learned depth 两路身份。|
|`tabletop_stereo_bc.py:73–80`|非零 depth loss 要求真实预测和 target/mask，否则明确拒绝；现 estimator 无对应 `depth_m` output。新增 head 时同步约定 `depth_pred_m`/兼容 loss key，防止命名不一致或以 GT 替代预测。|
|`tabletop_stereo_resolution.py:35–42,106–122`|budget 仅计算 disparity；比较器正确保留 Actor 缺测为 NA，capture+depth throughput 单列且不用于最终选择。扩充预算/各阶段计时，不把零 latency 或 capture throughput 填进完整 policy 指标。|
|`tabletop_privileged_cli.py:23–36,75–100` / `tabletop_privileged_ppo.py:146–156,174–190,257–289`|CLI 只接受 R0，PPO 已保存 latent/action/logprob 并复用现有 squashed Gaussian；noise 投影存在，但迭代汇总只记整体 entropy/KL 等，需独立加入 fingers12/translation6/rotation6 和实际安全投影统计。|
|`tabletop_privileged_reward.py:86–97`|R0 原样透传 reward 且检查 closure；raw/normalized 当前表示 native contribution，`physical_raw=None` 已明确标注。若要求完整物理 raw、coefficient、dt、mask，需增采同刻真实来源，不能把已合成 points 再冒充物理 raw 或重复加奖。|
|`tabletop_privileged_reward.py:20–65` / `tabletop_stereo_promotion.py:192–225`|geometry qualification 当前调用 `verify_native_episode`，后者强制完整 force proof 和成功回合；这是独立 geometry gate 的真实接口缺口。拆分 geometry/contact/force/support/external/full-task 资格时保持各自来源绑定，不能通过放宽完整成功门槛修补。|
|`tabletop_stereo_native.py:219–277` / `tabletop_stereo_promotion.py:114–183`|Bridge 当前输出 `source_scope='unqualified'`、force/support false、external/attachment None；与 verifier 要求的逐 digit sensor/partner/ON_HAND→ON_OBJECT/480Hz external audit 之间缺真实 producer。R1/R2 的纯算术不填补该缺口。|
|`tabletop_stereo_runtime.py:46–102,139–152` / `tabletop_stereo_losses.py:110–133,135–161`|维度级来源 fail closed 已实现；扩展监督必须逐维取得相应来源资格，不能由 finite、同一个全局 bool 或 q/qd 的局部成功给其他字段授信。|
|`tabletop_stereo_inference.py:54–61,74–84`|过期/异时图像会拒绝；但 depth/point 几乎全 invalid 时仍将 `image_valid=True` history 交 Actor。需定义覆盖/冲突失效条件及明确安全动作/停止接口，并测试不会静默切换 Teacher/MPC。|
|`tabletop_stereo_teacher_collector.py:134–187` / `tabletop_stereo_training_cli.py:24–38`|现有 online 路径确实查询同刻 pre-state、选整动作 owner 并调用一次 env.step；formal CLI 会重验 Teacher 与 Stereo calibration/matcher。后续复用，避免平行 DAgger。|
|`tabletop_stereo_student_ppo.py:90–130,146–166`|Actor/Critic 分离，PPO 数学复用，auxiliary 独立 optimizer/clean boundary 已有；后续训练不能把 Teacher label transition 插入 on-policy storage。|
|`tabletop_stereo_promotion.py:266–288,526–557,566–595`|Teacher/Student 每抓法≥100独立实测初态与成功门存在；ledger 仅接受三类顶层证据。任务书 WP7 的八类 proof 需扩展现 gate，而不是从 implemented/CPU 状态推导 qualification。|

上表文件位于 `src/go2_pvcnn/control/dh116/`。原生成功合同保持净 Lift≥0.03 m、qualified Hold≥5 s、formal fixed-anchor Slip≤0.0005 m 及其余原 gate；没有因这些缺口更改任何阈值。

## 已有 raw 数据的本机可用性

本机可读取的材料包括 `verified_results_summary.json`、roll20 的 `native_readback_and_rgb_comparison.json`、`offline_sgbm9_first_frame_metrics.json`、`artifact_manifest.json`、历史分析脚本和两张已有 PNG。这些主要是汇总/来源索引，不能替代完整逐像素 RGB、reference、instance-ID 与 process completion。

本次对工作区 `artifacts/` 的文件清单执行 `rg --files artifacts | rg '(stereo_probe|depth_reference|wire_instance_reference)'`，没有匹配；没有在 J0 中取得完整 native `.pt` 输入。报告明确引用、但不能仅凭本机路径获得的原始 capture 至少包括：

|对象|原报告绑定的远端路径|期望 SHA256|
|---|---|---|
|R5 capture|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/stereo_probe.pt`|`733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`|
|R9 capture|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block9/640x480/stereo_probe.pt`|`be1ba0bf5eadf82d4313e2ecc297bf016851df86a906489939591f57d1bc80d2`|
|旧 rig 640×480 首帧 depth reference|历史 `native_resolution_04/640x480/depth_reference_0000.pt`，完整 run 根见原报告|`55e2286209ef2c2b4331d4d6c6d1717cbca5746d1b8a246c29d7fdc6ecdfb62b`|
|同首帧 instance reference|历史 `native_resolution_04/640x480/wire_instance_reference_0000.pt`，完整 run 根见原报告|`ac7263910684709cb7dec3642c119e7635b611fafe159c4a7a6c7d43a9f40a07`|

完整 10 run × capture/reference/instance/calibration/proof/completion 的缺失路径、SHA 及远端可达性由同轮 J1 inventory 继续逐项报告。以上是本机未发现，不等于判定远端文件丢失。J0 未用 summary 伪造八分区误差、逐像素归因或新可视化。

已有报告能支持的有限观察是：R9 固定 Wire ROI 的 valid coverage 93.87% 与正确 3D 完整率 46.05% 是不同指标；共同有效集也比 R5 更差。离线 SGBM9 首帧正确完整率 81.96% 同时具有更差 P95 尾部。这提示需要检查匹配歧义、窗口背景混入和长尾，但**不能单凭这些聚合值完成八类原因的定量排序**，也不能宣布修复有效。

## 实际命令、环境与执行记录

实际使用 PowerShell、git、`rg`、`Get-FileHash` 和现有 Python 的只读标准库脚本。关键命令类型：

```powershell
git status --short --branch
git branch --show-current
git rev-parse HEAD
git log -8 --oneline
git rev-parse refs/remotes/origin/codex/dh116-stereo-privileged-dagger
Get-FileHash -LiteralPath <protected-path> -Algorithm SHA256
git show HEAD:<file-path>
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -B -
```

Python 脚本读取 JSON、重算 SHA256、统计 371 维、列出 AST 符号并比较 24 个绑定符号；没有导入 Isaac、训练入口或执行训练。解释器实际为 Windows Python **3.12.13**；metadata 为 Torch **2.14.0**、NumPy **2.5.3**、pytest **9.1.1**；该解释器 metadata 未安装 `opencv-python`/`scipy`，本次没有安装或升级依赖。`python` 命令默认解析为 WindowsApps 占位入口，因此采用上述既有 uv cache 解释器。

Git 状态/HEAD/remote ref、保护清单 SHA、公开合同 SHA、AST 检查与报告指纹核验均成功完成；本机 raw 文件名搜索无匹配，`rg` 按约定返回 1。审计脚本初次使用系统默认 GBK 读取中文 readiness 时出现 `UnicodeDecodeError`，随后显式 `encoding='utf-8-sig'` 完成核验；未改输入文件。

本轮 J0 没有运行 pytest，故没有新增测试通过数或 RED/GREEN 声明；没有新 native capture、Isaac 进程、Teacher/Student 训练、正式 BC/DAgger、依赖下载、checkpoint 写入、git commit/push/merge。报告只记录本次实际核验与现有证据边界。
