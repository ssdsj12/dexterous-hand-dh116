# Fresh17 三指场景构造与原生 Reset 入口

基线：9621dffa3c5c043fa128273e20d11c61269dc342。Ubuntu 开发；Windows 发布同一 commit。

## 修复的执行断点
原诊断入口先执行11槽 Reset 再追加掌面/支架/近节，不能交给要求全新 episode 的 Teacher。
新增 tabletop_fresh_supported_builder.build_fresh_supported_owned：复用原双手USD、支架、材料和真实DeformX，
SDK0→2后以真实14个link姿态直接构造17槽；在全部槽位就绪后仅分配一次8364容量，返回状态new的原episode。
不做旧11槽 Reset、不静置、不控制、不调用求解器。task SHA、后端、左右独立USD/SDF来源均验证。
生产原生入口 record_dh116_tripod_supported_reset 使用已有R06生命周期，独占进行一次测量Reset，
保存17槽直接读回、数值状态、原34形状+10 cooked包络/桌面/支架几何、同钟66项common。
371D未观测字段继续NaN/validity；不补零，不提升资格。

## 实际验证
- TDD红灯：构造入口4例因缺功能失败；原生采集入口2例因缺功能失败，原日志保留。
- 构造/入口及旧Pinch builder回归：26通过、0失败，184.53s。
- Reset/七路线控制/原锁定路径回归：48通过、0失败，25.08s。
- Registry：60通过、0失败，15.47s。
- 合计74项不重叠选定CPU案例，不等于全仓测试或原生物理。
- CPU使用SDK/solver API替身；真实原始USD/SDF/来源文件用于构造审计，不伪称原生接触。
- 保留tests/test_dh116_dual_pinch_builder.py旧默认，只让新夹具可读取真实传入layout。
- 全部日志SHA、命令在readiness；原始日志保留在Git外evidence/ref902。
- Antigravity本轮界面仍显示额度耗尽，无新审查回复；未开启付费额度。建议由Codex独立核验。

## 原生范围与剩余问题
本包尚未启动Isaac。新入口只允许GPU2、单实例、1环境、2 SDK初始化步、1次零步Reset，
零控制/solver/Teacher/PPO，180s软上限，失败不自动重试或强制结束。
自然退出必须以外层GDB和实际资源释放回执验证；app.close返回或CPU测试均不能替代。
三指完整NativeSources/session/preflight仍待接通，17槽Reset仍仅在<=240 SDK范围；长时Hold的Reset版本尚未资格化。
二指已有历史短PPO更新/恢复证据；这不是三种抓法成功。Power仍需完整掌面/五指真实包围与力反馈。
固定两台RGB配置、24D/371D原始合同、原关节保护、Lift30mm/Hold5s/Slip0.5mm均不变，R2关闭。
