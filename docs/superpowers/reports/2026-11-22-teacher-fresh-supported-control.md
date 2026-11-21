# Fresh 17 槽 episode 与七路反作用力交换

基线：`6ace5cd7ee9e360c41e4fb54fad3e1c02ecb2c36`。本包接通现有 Reset、状态读取和控制交换，不改资产、物理参数、动作或成功标准。

## 修改内容

1. 新 `tabletop_fresh_supported_exchange.validate_fresh_supported_reset`：
   - 只接受原 `PowerSharedRodRuntime` 的17槽布局。
   - 校验完整构造计划、已完成 Reset/readback、同一 RopeBatch identity、同钟零 rod tick，并重新调用真实 mesh_state_snapshot 确认 current/target/solver 没有漂移。
   - 检查同一 Reset 的排他控制权；仅经过新的实际 Reset、递增 reset_id 和新 episode_uid 才可建立下一 lease。
   - 无动作、无 Reset、无物理或求解器推进；不产生接触测量资格。

2. `FlexiblePhysicalExchange.__init__` / `_step`：
   - 新显式 `measured_supported_reset`+`supported_mesh_plan` 参数，与已有 measured continuation 路径互斥。
   - 首步七路零载荷为 `commanded_reset_zero_not_measured_contact`，reaction_source_physics_step=None；其后逐实际 SDK 步递交上一 rod 步的七路 world/COM force 和 torque。
   - 初始零数组按真实路数创建，旧默认仍5路；原有6/7路实测 continuation 保留。
   - 每个物理行保留本次 Reset binding；原24D请求、8子步控制频率、关节/速度/Mimic/力与桌面保护不变。
   - 同一 Reset 禁止重复绑定；新 Reset 后旧 exchange 在任何新写入前被 clock/epoch 校验拒绝。

3. `DualHandPhysicalEpisode.reset`：
   - 新独立 opt-in 路径调用 `MeasuredSupportedMeshReset`。
   - 全部14个手部 link 进入原 `LiveGeometryReader`，保持所有原始碰撞形状的 clearance 检查。
   - 需要调用方提供独立 `record_reset_readback`；不把辅助 readback 伪装成 physical/history 行。真实落盘 flush/fsync 由原生调用方的 journal 负责，本包 CPU list recorder 不是磁盘耐久性证明。
   - 使用同一 post-Reset rod state 和原66项公共状态源；未观测字段保持 NaN/validity，随后接入 fresh 七路 exchange。

## 失败与测试证据

- 红测先复现 fresh exchange 参数缺失。
- 首版 CPU 夹具沿用旧3节点测试杆，与41节点/8364容量不一致，严格生产代码正确拒绝。改为从构造时创建41节点的 CPU double，没有放宽生产容量检查。
- 新多回合测试复现布尔占用标志导致第二次实际 Reset 永久无法绑定；改为经过源核验的 Reset epoch 控制权。
- episode 红测复现缺少 supported Reset 入口；新增真正的 Reset→pre-action→8物理行集成测试。
- 测试拒绝 foreign owner、pending readback、live mesh drift、重复 lease、fresh/continuation 混用，以及缺少独立记录器。
- source/SDK/solver 是明确 CPU doubles；每步原生资格和 Teacher 准入仍false。119项控制回归及142项 Teacher 回归为不同文件集合，最终命令、耗时、日志SHA和 Registry 结果见 readiness。

本包不调用 Isaac App、不运行原生 solver、不进行原生 PPO更新；没有生成成功轨迹、模型或视频。

## 审查与保护

使用 systematic-debugging、test-driven-development、isaaclab-auditing-tests 与 verification-before-completion。实际查看 Antigravity 桌面，角色仍为 codex-dh116-ssh-reviewer，但当前模型显示配额耗尽；未启用付费超额、未改安全设置，未取得新补丁的外部通过回执。此前R1的独立接线建议继续作为依据，Codex 独立核验实现和回归。

旧 `tabletop_mesh_reset_order.py` / `tabletop_mesh_reset_readback.py` 及旧 BackendProfile 封存保持不变；双固定 RGB、528保护文件和三份历史模型必须保持原 SHA。

## 剩余准入与下一包

- 从真实 SDK2 scene 建立 fresh17 constructor，首次唯一 Reset 由 NativeEpisodeOwner 持有；不能把诊断旧 owner 的 Reset/settling 当成 fresh 构造。
- 独立 NativeTripodSources、source/factory/preflight 和 worker 绑定，保留 current backend/source SHA、初态与 checkpoint身份。
- 再复用既有 PPO/GAE/KL/Adam/RNG 路径执行有界 update1/resume2，不重写算法。
- 当前正式三指 native collector 尚未接入，native_ready=False；本包不解除正式训练或成功专家池门控。
- 本17槽 Reset 仍保留 SDK≤240 的明确边界；长回合 Reset 后续需独立长时钟版本，不擅自放宽。
- Pinch、Tripod、Power 分别继续真实物理验收。R2/Force Proxy 未资格继续关闭；Lift≥30mm、Hold≥5s、Slip≤0.5mm、无外部支撑/Teacher接管标准不变。


## 最终结果

- 新旧控制链：**119 passed / 0 failed，24.75s**。
- Teacher preflight/session/collector 与两类策略距离：**142 passed / 0 failed，350.64s**。
- 上述为16个不同文件的261项相关 CPU 回归；未声称全仓库测试通过。
- Registry：**60 passed / 0 failed，14.88s**。
- 新增6项回归；命令、源码和日志SHA见 readiness。中间红测日志留在服务器。
- 旧 BackendProfile 重新 load+verify，通过且 digest 不变：
  `4b05944aaca8934b65ffa11636154204360c5ac0d5dd3921b7893ba9febacf14`。
- 原生运行0、原生 PPO更新0；正式三指 collector、三抓法成功资格仍未完成。
