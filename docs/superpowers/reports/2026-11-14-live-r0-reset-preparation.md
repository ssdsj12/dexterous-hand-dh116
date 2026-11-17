# 真实耦合 Reset 后 R0 初始化与原生观察器

基线 a8361087ba80b339c2646b292a03e700ded0d675。
本工作包仅增加独立入口；原双手有效指腹实验的物理、动作、初态、碰撞、保护和生命周期实现不变。

## 实际修改
- tabletop_r0_after_physical_reset.py：从已经完成的物理边界初始化原 R0 状态，不执行历史验证回调来伪装 Reset。初始化不调用 SDK、不推进时钟。
- tabletop_r0_live_observer.py：先验证本次真实 Reset 的左右手 q/qd、rod 和四个 owner 回执，再绑定原 371D/24D 同钟数据。使用原 R0，保留缺失值。
- prepare_dh116_live_r0_native.py：将实际事务、原日志、原生参数 getter 与 R0 观察器连接；临时 native 对象仅弱引用，成功和异常路径均清理。逐物理步即时写入；控制边界 flush/fsync。
- verify/run/validate_dh116_live_r0_*：独立入口、单次冻结预算和退出后审计。审计对齐原始事件 SHA、时钟、动作与结算，不单凭 summary 判定通过。
- 两份独立配置与四份新测试。没有修改既有任务、模型或原生运行结果。

## 验证及失败证据
最终 CPU 回归 **85 通过，0 失败，142.43 秒**（regression02.log）。CPU 使用真实 Python owner 与显式 SDK/solver doubles，不是原生资格。
初轮集成回归 84 通过、1 失败：新增测试后冻结指纹未同步，门控正确拒绝；保留 regression01.log。随后更新指纹，重新验证最终代码。
所有日志保留服务器 evidence/live-r0-reset-20261114，未上传原始日志。

## 原生计划与范围
新配方总计 234 SDK 步（2 初始化 + 232 测量），29 个控制步，1 环境，GPU3，0 PPO 更新，最多 180 秒、一次。
沿用已发布原始物理配置与动作轨迹；区别仅为真实进程中的 Reset/R0 数据连接。
使用最新用户“批准，后续自动批准”的既定有界范围；先发布冻结源码，再核对资源和单实例锁。
旧已消耗接触配方不重跑；新观察器失败即留证，不循环重试。

## 尚未解决
- 本报告不表示新原生执行完成；结果另包发布。
- 旧六个 hook 顺序资格仍为 false：本入口采用独立的真实耦合 Reset 后初始化合同，不改写旧回执。
- 稳定性、有效对向接触、物理滑移、手/桌外部支撑来源仍不完整。
- 原 R0 训练系数仍未指定；序数系数只用于诊断。training_reward 保持 null，Teacher 准入 false。
- 上一批 hold 必要条件扩展属于离线合同，本入口不启用，也不把离线资格带入原生。
- 没有开始 Teacher PPO、长训练或完整抓取评估；30 mm / 5 s / 0.5 mm 不变。

下一工作包：执行此冻结入口一次，核对真实 Reset、232 个同钟 R0 样本、29 个结算以及外层自然退出；随后继续稳定性与奖励来源资格。

日志SHA256：
- regression01.log `2c26da5991d529bdc7939fca967e744fcdd5fb2cc90808e5e6cacae1e757e325`
- regression02.log `9d44d53646505ff7207b01d1ba1edd3be46fe4695ec04777b469ea771bc5469c`
- live-red.log `2c5c3a6ca151e82bb429d3d1128cf192c0afc4b723892bc164805ab10a439920`
- session-red.log `8853ee81e32bf1aa2c56fca7c9203aefb5a00df226e79cb806de58cb693ee95d`

入口代码/配置SHA：`287586d40952b0cf4e0241bf16cda116193b7b66a4b64dfa14728fc24eeaa1f9`；命令配方SHA：`0d8f2d7258edf4cc7bf077499c9b3f54ef07c7984c5756c02db32b3f23fc2ea0`。
