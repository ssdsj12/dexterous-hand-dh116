# Tripod fresh 构造与 Reset 距离来源修复

## 范围与基线

基线 `17c34d665fac47c53fffd8869d3360f8c7f9699e`，Ubuntu 开发，Windows 导入同一提交发布到 `11-17`。
本包修复进入真实三指 PPO 前的构造和来源缺口，不启动 Isaac、求解器、Teacher 或 PPO。

## 实际修改

- `PowerSharedRodRuntime.__init__` 接受完整、双侧独立认证的 proximal grids/poses；在 fresh 六槽底座上直接构造原有17槽布局。拒绝不完整或左右来源混用的输入，发生在追加或占有底座之前。不通过隐藏 Reset、settling 或追加旧11槽 Reset 来伪造 fresh owner。
- `bind_palm_sources` / `bind_proximal_sources` 保留已由原生 USD 读取函数核验的 `frame=PhysX_link_origin`、`unit=m` 和对应 SDF 文件路径。严格读取器继续校验，不填造历史记录。
- 新 `MeasuredTripodResetApproach.capture_reset` 读取实际完成的零步 Reset、23个 body/17DOF、全部实际碰撞绑定和位姿副本、同钟参数与锁定 SDF。复用三指离散材料节点距离定义；缺一个必需指垫查询则 unknown，不使用二指均值或零值。校验实际 Reset 时钟、参数来源/版本、半径身份、原 USD/SDF SHA、q/qd/目标以及位姿副本。

## 证据和边界

CPU 测试使用已保存的 SDK2 实际 Reset 位姿与显式 CPU 参数夹具；不把重放写成新原生运行。原 Reset 原始 JSON SHA：
`1afdf3954fd98871b9735f4168d412e69de0b56cb770417ad17b1882c4825da4`。

| 手 | 拇指间隙 mm | 食指间隙 mm | 中指间隙 mm |
|---|---:|---:|---:|
| left | 2.395946 | 1.586157 | 1.858796 |
| right | 6.671616 | 9.878989 | 12.432757 |

这些是实际位姿下离散41材料节点的 SDF 表面间隙，不是完整连续线缆距离，也不是接触力或夹持证据。历史 Reset 只有11槽，明确记录 `observed_mesh_count=11`；新读取器不会给它授予17槽 Reset、R0、原生或 Teacher 资格。

## 失败复现与验证

1. fresh constructor 红测：缺少 `proximal_grids` 参数，2个测试失败；修复后直接17槽构造与拒绝不完整来源的测试通过。
2. metadata 红测：真实 producer 遗失 `frame`，1个测试失败；修复后 palm/proximal 的来源元数据和路径通过。
3. Reset distance 红测：缺少 Tripod Reset reader；补充篡改参数来源和未知 scope 的负例，确认旧实现未拒绝，再添加当前 profile 来源核验和明确 native_getter/CPU scope 核验。
4. 参数夹具曾保留旧 solver time，严格读取器正确拒绝；仅修正 CPU 夹具时间及派生参数重建，没有放宽生产校验。
5. 原生运行0、PPO更新0。最终 CPU/Registry 命令、数量和日志 SHA 见 readiness JSON。未执行全仓库回归。

既有封存历史 R0 replay 的源码 SHA 不一致仍保留原失败记录，不在本包降级校验或改写历史来源。

## 下一执行链

1. 完整17槽原始 SDF constructor plan、clear/add零步 Reset、实际 mesh_state_snapshot readback。
2. fresh supported episode owner、7路实际反作用力 exchange、独立 Tripod source/factory/preflight，保持原始安全保护。
3. 复用既有 PPO 两次短 update/resume 路径，随后分别做 Pinch/Tripod/Power 真实抓取验收。小诊断、距离读取和奖励连通均不等于成功。
4. 双固定 RGB、原24D/371D合同、物理参数、USD、Mimic、关节界限和原成功标准不变；R2继续关闭。

Antigravity 已完成该链路 R1 设计审查；新补丁外部复审受账户配额阻断，本包由 Codex 独立审查，不声称获得新的外部审查通过。


## 最终执行结果

- 11个相关测试文件：**102 passed / 0 failed，114.33s**；命令和逐文件列表见 readiness。
- Registry：**60 passed / 0 failed，17.31s**。
- 7个新增回归测试；覆盖独立来源拒绝、缺失中指、时钟/位姿/半径身份、17槽 fresh 构造和元数据保留。
- 日志仅存服务器 evidence 目录，Git 仅保存命令、结果和 SHA。原生运行0、PPO更新0。
