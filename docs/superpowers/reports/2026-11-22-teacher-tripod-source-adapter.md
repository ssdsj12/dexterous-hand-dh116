# 三指 Teacher 独立状态来源接入

基线 30b934925116107c34d530b42a5a7921030d164c。Ubuntu 开发、Windows 导入同一 commit 发布。

## 实际改动

- tabletop_native_teacher_sources.NativeTripodSources 接入真实来源重建的 TripodResetManifest。
  使用 MeasuredTripodResetApproach 和 PolicyOwnedTripodApproach；左右各三个指腹、
  17 个碰撞槽与实际14个 link 绑定独立检查，不借用二指的距离证明。
- _NativeGeometrySources 仅复用参数、动作历史、数值健康、TaskContextOwner 和输入打包流程；
  NativePinchSources 保留原二指 Reset/策略距离。NativeEpisodeOwner 按抓法选择明确类，
  Power 尚无合格来源时拒绝，而非改一个 grasp 字符串伪装。
- 中指 SDF 覆盖缺失时距离保持未知并抛出 SourceFailure；
  不生成奖励端点，不填零，也不开放 contact/slip/Force Proxy。
- 固定41个材料节点、66项已验证公共字段、task_context 来源继续沿用现有流程。
  原371槽未知值、24D、相机、资产、物理参数和成功阈值未改。

## 历史回执和当前代码的边界

audit_dh116_tripod_fresh_reset 原先将整个仓库 frozen_files 的当前 SHA 等同于物理采集来源。
本包修改训练 consumer 后，历史 Reset 审计因此拒绝，但这些 consumer 未参与该次 Reset 采集。

仅 registry、release manifest 和三个明确的后续 consumer（sources/session/preflight）
改为核验 outer.head 对应的精确 Git blob；不是任意调用者豁免名单。
Reset、SDK、求解器、SDF、资产、配置等所有采集/物理文件仍严格核验当前 SHA，
BackendProfile 和实测数值/mesh/公共状态来源审计保持不变。
历史回执仅证明历史 Reset；不能证明本包新来源适配器已经运行过原生物理或 PPO。

## 验证与限制

- 新增两项 CPU 行为测试先因缺少独立 Adapter 失败，后通过。
- 初次集成夹具暴露 float32 radius 与 Python .005 的来源精度、缺失调用方 profile scope：
  修正测试夹具以保留真实原始半径和生产调用范围，未放宽生产来源校验。
- 最终选定回归55通过、0失败，319.51秒：独立三指来源、原生 Session、
  二指/三指策略距离及真实历史 Reset 审计。
- 测试使用真实保存的 Reset 数据，但 native getter 仅以显式 cpu_fixture 替代；
  所有资格标志仍为 false。这是 CPU 数据重放，不是新原生仿真。
- 本包新增原生0、控制0、求解器0、PPO更新0、合格抓取0。
- 初次失败与最终日志、SHA 保留在 Git 外；未上传 raw、模型或实验日志。

## 下一步

独立接通三指静态/实时准入、fresh17 Episode Reset 与实际 provider。
当前旧 static/live gate、Reset 路径仍有 Pinch-only 限制；本包不会移除这些必要门槛。
接通后先运行既有小预算更新/恢复诊断，再补全真实接触、固定手腕保持和离架 Lift 资格。
Power 保持独立全握与掌面来源要求，正式 BC/DAgger/Student PPO 尚未准入。
Antigravity 仍受此前观察到的额度限制，本包没有取得新的外部审查意见。

应用 Skill：测试审计、RL训练和发布前证据核验；新版示例命令不用于替换当前环境。

Registry：60通过、0失败、16.64秒。528保护文件、3份原始Checkpoint及两份固定相机配置SHA保持原值。
