# Pinch 独立三维几何证明原生配方准备（2026-11-13）

基线 `1b7ea05b101f81a7217c9b1765d0176a23c2eb63`。已消耗旧234步配方不重跑。本包原生尚未执行，Teacher0/3。

## 实际修改

- `tabletop_segment_closure_geometry.SegmentClosureGeometry`继承现有实测捕获。仅在旧XZ下界为负时生成完整3D逐杆段证书；全部原顶点复核，半径5mm、数值裕量2um、零保护线不变。
- 新packet版本 `dh116-measured-segment-closure-geometry-v1`。保存实际raw身份、源SHA、全部回退证书；validate独立重算全packet，旧版本不能混用。
- `prepare_dh116_pinch_segment_contact.scenario_scope`只绑定新工厂和独立版本，异常后恢复。真实采集和post-exit磁盘审计使用同一新工厂；原始几何源码SHA仍 `3e90dca3dfc2448ad2a1e82f545c29e7229fbda7510b90fba45bf88a5a7132de`。
- 独立entry/wrapper/worker marker、调度recipe；不修改任何旧入口或冻结指纹。
- 复用完全相同 `flexible_pinch_margin_contact_v1.json`、Reset、B1 native参数后端、24D schedule和保护。只左手指闭合，右手/腕部零动作；1env，0PPO。
- source17+locked5范围保持，没有把源网格证明扩大为非指腹实时cooked或手-静态桌面接触资格。

## 预算与冻结指纹

234总物理步=2初始化+48静置+160左指闭合+24保持；29×8测量步；480Hz物理/60Hz控制；GPU3单实例；180s；一次；不自动重试。

- 配方SHA：`ab5e73125f1bae212b8a4865a362dd1dad208a3535f58deab2e84dcf04802c80`
- 调度code/config：`186855563b67e02e6998797fb57d416da79d35bfa70311b96815ff9f844f76df`
- worker preflight：`61dd815e39e6ea293b1f7dd51333db1843386b565aa119b4cb4f57162d071ed1`
- profile SHA仍 `47563ff24bf20b780e0b640ff801417982b5c45904e92c30d27bc16f4c2c3d69`
- 成功仅可标记 `left_pinch_segment_2_setup_232_guarded_contact_diagnostic_only`，不能算抓取、Hold或Teacher成功。

依赖使用真实 `flex.f7-native-parameter-task-native` receipt（SHA f0767fc9...），不把旧失败配方设成passed。该旧失败的实际原始数据用于单因素修复证据。

## CPU与已有数据重放

CPU新用例先3失败，随后
```text
DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q \
 tests/test_dh116_pinch_segment_geometry.py tests/test_dh116_polyline_support_certificate.py \
 tests/test_dh116_pinch_segment_contact_worker.py tests/test_dh116_pinch_segment_contact_owner.py \
 tests/test_dh116_pinch_margin_contact.py tests/test_dh116_measured_closure_geometry.py
```
**70 passed、0 failed，166.24s**，含真实Python owner全234次合成SDK调用、SDK11终止、参数错误、dedicated worker子进程、marker、防预算扩张、旧18步不可冒充234步、原文件哈希和证书反例。CPU日志：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-segment-full.log`。

原生旧账本SHA `05fb4242cc11fb23017b32f1d4d2820dc3f669a16f18f9f5a1c27f8136070cb9` 214条逐行重算：旧1条不通过，新0条；总12.095439s，最大单条evaluate+validate49.02ms。该计时是离线CPU证据，不是原生控制吞吐。
结果 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/segment-replay-r153qZ/result.json`，SHA `4e107b0c659730026e545c822db6f23eab9a1eb28766931c70f929e3417f8d9d`。
SDK216旧−3.674µm，新+0.497762mm，原记录/receipt不改。没有执行额外物理步。

## 后继和限制

发布后按持续授权执行新指纹单次。检查234步完整账本、真实左右pair/反作用wrench、原保护、闭合后保持与外层exit0；exit1或signal均失败，不以app.close返回代替。失败保留normal-finalize/信号栈和maps。

Teacher正式训练仍阻断：右Pinch/Tripod/Power尚无独立完整任务资格，371D实际来源仍66维、R0与材料滑移/稳定保持未齐，R2关闭。该0.4875s实验即使完整通过也不满足5秒Hold，不能进入成功BC池。
