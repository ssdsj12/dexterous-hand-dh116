# DH116 原生 Teacher PPO 和奖励设计修订

日期：2026-11-15 UTC。

## 依据与批准范围

用户已完整提供并明确选择采用《DH116_PPO_详细修改与奖励设计.docx》的新版本方案。源文件SHA256为 `a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9`。正文841段和20张公式图均已读取；原件和公式转录仅保留在私有证据目录。

本修订覆盖 `2026-11-15-dh116-execution-chain-repair-design.md` 中将“完整371D有限网络输入”和“旧R0尚无数值”作为所有实际PPO前置的部分。原8项缺陷修复、旧模型保护、状态/动作来源、单实例和原生授权要求保持。

文档中的新增模块名、公式和权重是待实现并验证的设计；它们不是当前代码已有接口，也不是已通过原生训练的参数。

## 输入和模型

- 原371槽账本原义、validity、NaN和全部历史模型保留。
- 新独立Teacher只选择经过验证的66项公共字段，结合真实41节点、每节点13特征的rod分支及独立task_context。
- 当前measured模型只接受4个键；新增context需明确新适配器、编码器、尺寸和checkpoint schema，不能向旧字典静默加键。
- context包含阶段/接触状态、连续接触/Hold计时、固定材料锚及其validity、真实记录的前一提交命令、实际Reset基准/归一化尺度和有限任务所需剩余时间。
- 每个profile只使用其适用且来源有效的context。未知槽不以补零伪造测量；新模型的输入掩码和未启用字段语义写入合同。
- critic获得回报所需context；actor得到对应信息或经验证的时序表示，不能只给critic计时器就声称actor满足Markov性。
- 原371D网络和66诊断模型均不覆盖；新模型有独立namespace和明确迁移/恢复区别。

## 两个独立训练profile

| profile | 启用范围 | 允许证明 |
|---|---|---|
| approach_wiring_smoke_v1 | 实际同钟固定目标距离潜势、提交命令变化/时间成本、可信硬失败事件 | rollout、优化更新、保存重启继续更新 |
| flexible_train_r1_full | 六潜势、滑移/控制/时间成本、完整成功及失败事件 | 可开始完整抓取学习，成功仍需独立评价 |

新奖励版本名flexible_train_r1不表示启用旧R1/R2力学开关；旧R1/R2及未资格Force Proxy继续保持原门控。

旧十项null R0诊断配置不改。smoke不要求未启用Hold/slip来源；full的未知适用项不能靠零系数规避。初始未抓稳是正常负例，不是数值故障。

## 数学合同和建议候选

控制步奖励：
`R_k = gamma_c*Phi_post - Phi_pre - sum_j(dt_p*C_kj) + E_k`。

full候选：
`Phi = 2*p_distance + 2*p_contact + .5*p_persistence + p_geometry + 4*p_lift + 4*p_hold`。

- p_distance=1-clip(d/d_ref,0,1)，d_ref=max(本次实际Reset的d0,.005m)，运行中固定目标/尺度。
- p_contact=(左右required-pad有效覆盖率之和)/2。
- p_persistence=clip(连续双侧合格接触与对向时间/.25s,0,1)。
- p_geometry为原左右有效对向布尔结果均值。
- 净高度h=min(当前全节点中位高度−初始中位高度，当前最低节点−初始最低节点)。
- p_lift=I_lift*clip(h/.03m,0,1)，I_lift同时要求双侧承托、实体底部离桌及无禁止辅助。
- p_hold=clip(本次连续合格Hold时间/5s,0,1)，H不能包含“已经保持5s”。
- 滑移s为所有适用材料/切向漂移保守上界最大值，q_slip=clip(s/.0005m,0,2)。
- q_command=(1/24)*sum_i(((a_i−a_previous_i)/2)^2)，来源为归一化preprojection提交命令，不是SDK目标或实际能耗。
- full成本率候选C=1*q_slip+.05*q_command+.10，单位reward/s；速度成本首版0。
- 可信硬失败事件−40，禁止辅助−50，多个失败取最大罚一次；成功+100且每episode最多一次，失败优先。
- gamma从同一PPO配置读取，保留负进度；不裁负差或只用历史peak。
- full gamma=.999时Cmax=2.15/s，无限折扣成本上界约35.833；更改gamma/成本后重新核算失败罚，不能直接套40。
- smoke候选gamma=.995/lambda=.95、lr=1e-4、rollout32、epochs1/minibatches1，KL沿用.02/3回退；full gamma=.999/lambda=.97为另一个独立候选。

候选权重未经原生训练验证。它们是新配置的明确设计来源，原生物理测量来源仍须单独资格化。

## 适用性与时钟

已知没有接触是有效零；可信历史证明不适用时raw=NA、applicable=false、贡献可为定义好的零；应适用但缺测/损坏时隔离并停止本次采样。

正常窗口8个480Hz物理步。可信硬失败第3子步仅记录3步、积分3/480秒、生成一次可信terminal；NaN/错时不自动成为有限terminal，不伪造剩余5步。外部预算只在完整控制边界截断。

真实终止使用吸收Phi=0且不bootstrap；外部时间截断与buffer满保留Reset前真实Phi并用同一shaped V bootstrap；buffer满不Reset。bootstrap不再额外加Phi。奖励先产生post_context，再用真实末态+该context计算next_value。

5秒Hold需要2400完整物理间隔、2401有效端点。材料锚不因丢失/重接触重建来清零滑移。辅助污染锁存。奖励与evaluator共享底层测量函数，但独立重算成功。

## 运行和验收

静态验证 → 创建唯一owner → 本次真实Reset与源核验 → 创建新model/trainer/reward → actual collector。准入判断“本次数据与启用奖励是否可信”，不要求随机策略已经成功抓取。

smoke目标是1环境32条有效control transition后真实update1，clean boundary保存；独立第二次启动恢复Policy/Adam/RNG/计数/合同，用additional_updates=1完成update2。256测量SDK步之外，加实际setup/Reset/warmup；该预算必须新冻结，不复用旧234/240步、0更新授权。

保存不等于恢复完整物理求解器现场；第二次允许按明确合同重建环境Reset，再采样更新，不能称逐轨迹位级续跑。

先让可信来源的一个抓法推进；Tripod/Power各自有来源、初态和原生资格，不需等三者都已成功才开始第一笔PPO。完整奖励、长episode、独立成功评价、Student示范资格继续分开验收。
