# Contact Tripod fresh8：真实有界配方冻结

基线 `d5e3db6c7d53944abbc39d7d8bd5113294d1382e`，分支 `11-17`。本包没有新建 Solver/Collector/PPO，也不改变关节、资产、重力或双RGB；使用上一包已接通的 CONTACT worker，为真实 supported17 Tripod 创建独立 core 与 launch 配方，并以真实 Repository、原模型、严格 Reset/来源字节和CPU参考审计验证。

## 配方与证据

- Native 工作包：`teacher.tripod-contact-fresh8-v1`，初始 blocked，仅等待精确执行身份、资源和单实例锁。
- GPU3，单实例，1环境，8控制步，1次 PPO 更新；2 setup SDK +1独立零动作 prime +64策略SDK，总计67 SDK /65 solver calls /780内部子步。
- 内层180秒、外层240秒；四线程，最少4096MiB空闲显存和1GiB数据盘空间。不自动重跑，不自动强制终止。
- 奖励为独立 `approach_contact_prelift_v1`，R2/ForceProxy关闭，profile自己的checkpoint namespace与旧SMOKE隔离。原371D/24D及30mm/5s/0.5mm标准保留，本短诊断不可能授予完整Hold成功。
- 572文件完整来源；指纹 `d4ea1e10c76994c76532b3b6b7a6f6e5072be9d609a76030e0470e3b1d1d23e0`。核心SHA `ba95fff665d1e8120c6033dac1389d1d1c1b7679b9d6ca7486a49e1239bc62fd`；code/config SHA `182144382d36b5294a80528d4575be14124878b53346f77e6817c7833a088ee4`；launch SHA `5af05c745d3eefd67d0c82776e64f30dae05b3862896908d5dcf862d83f3285e`。
- 使用独立实际 Reset 证明 `c123ccbed1161471e2e34dc6941d7e06fcae93cb5ff87499bb8a5440eab0a6ad`，严格原 loader 验证。它只证明Reset，不是接触或抓取。
- 私有 training.json SHA `0f373f4502aaaa3b4852a72244d4c1888de7be842d976e390d19d57d8b8fd44a`。原始账本、模型、采集数据均不入Git。

## 修改与验证

新增 `tripod-contact-fresh8-core-v1.json`、`tripod-contact-fresh8-launch-v1.json` 和 `test_native_teacher_tripod_contact_frozen.py`，并注册软件冻结包及独立原生包。配置由已有的来源遍历、semantic/reset/target pins、recipe和actual-launch校验器生成，没有修改生产算法。

先写回归：新工作包缺失的KeyError构成1失败/1通过；原已消耗SMOKE来源仍拒绝。首次私有生成脚本遗漏明确jsonl上限与inner wall键，被原严格validator拒绝，未写入配方或启动Isaac；补齐生成脚本后成功，未放松生产校验。

联合CPU回归 **145 passed、0 failed**（45.21s），覆盖实际冻结配方/原CPU reference、旧配方不可重绑、worker合同、actual launcher和Registry。原参考模型仍为0更新、0环境步；没有SDK、CUDA、Isaac或PPO训练运行。新增软件冻结登记后Registry另60项通过（14.66s），最终145项已覆盖该登记；重复结果不累加。

Antigravity此前Contact Worker首轮实读建议作为本配方依据；新一轮受额度阻断，界面显示2026-11-22 06:55:13 UTC刷新。未提交重复请求、换模型或启用付费overage，不将缺少复审写成通过。Codex独立核对实际来源与运行上述测试。

## 后继和边界

发布后按用户持续全批准的授权，只有在同一指纹未消耗、GPU3确实空闲、单实例锁可用、数据盘空间满足和发布SHA一致时，才执行一次本配方。仍受67/65/780和180/240秒预算约束；失败保留账本、栈、自然退出和receipt，不能计入成功池或重跑同指纹。不会自行强制结束。

本冻结包原生次数0。实际接触、PPO更新、checkpoint、自然退出及后续恢复均需另看真实回执；Pinch17和Power五指/掌面来源保持独立，正式BC/DAgger/Student仍blocked。发布前核验528保护文件、3历史Checkpoint；Windows导入同一commit发布，PR#2保持Draft。

独立复核去掉了新测试对“永远未执行”状态的错误约束，改验同指纹最多一次的持久合同。对应测试源码变化触发严格来源拒绝；第一次未执行指纹和配方已私有留档，新指纹重新计算，不复用旧身份。
