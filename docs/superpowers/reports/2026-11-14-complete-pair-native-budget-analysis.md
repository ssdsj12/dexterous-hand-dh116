# 完整接触原生失败后的来源与开销审计

基线：`3bfad44e67087fa23303b078cbd66493bb44720e`。本工作包仅处理已结束原生的只读证据，不启动或重试Isaac。

## 实际原生结果

源 tested HEAD：`8c1c7f3525099207285a43bebe9e149e3fe0c3b4`。
receipt SHA：`208b6c6d16f7f7e23c8e72a721343bc74276d242fcc4ef0dd71b6d16182df508`。
配方 `complete_pair_r0_native_v1.json` 已消费；不可重跑同指纹。

- 234 SDK物理步，含2初始化、232测量；29控制窗口；1环境，0PPO。
- 外层与GDB inferior均自然exit0；signal=null；app.close返回。
- 实际 **196.211877563s > 180s**，scheduler已标记失败。没有SIGTERM/SIGKILL/os._exit，没有扩展物理步。
- 完整源与R0同钟不等于本轮整体原生资格通过；完整R0与Teacher admission仍false，training_reward=null。
- 生命周期审计先拒绝超时，因此失败receipt没有完整artifact_sha256 manifest。后续离线工具独立固定实际原始账本SHA，不补写或伪造失败receipt。

## 独立离线重算与真实接触

`validate_dh116_complete_pair_r0.audit` 对实际232条原始owned数据执行：原子capacity/attempted/overflow、原投影、独立solver聚合、实际Reset/物理时钟、原指腹条件、R0测量逐行核验。结果一致，仅为超时来源的离线数据检查，native_qualification仍false。

| 项目 | 左手 | 右手 |
|---|---:|---:|
| 原始hand/rod pair条数 | 0 | 2129 |
| 原指腹区域候选 | 0 | 0 |
| 已证实无有效接触的测量步 | 232 | 232 |

2129条右接触的唯一首拒绝和重叠拒绝均为 `outside_inner_ellipse`。没有放大ellipse、改法向、强制承托或把几何候选计为成功。

R0中required_contact、contact_persistence、grasp_geometry、lift_progress、joint_limit、external_support_violation各232步具有有效诊断数据。approach_progress、hold_progress、material_anchor_slip、instability仍各232步缺失。末态approach，Hold0秒，success_candidate=false。

## 存储与开销发现

新增 `profile_dh116_complete_pair_r0_ledger.py` 固定失败receipt和独立原始SHA，流式核验并统计来源、分项拒绝、累计source列表，不改原始文件：

- physical.jsonl：132,358,448 bytes，SHA `4d2b457e7e0c581c0aed139b25e2531f1554bada54ef7b836825c833b0536298`。
- live_r0.jsonl：92,994,363 bytes，SHA `6581685565bdea48ef66f2b35fd670a1c8790856ab889da737ac17d1630f389d`。
- physical行内R0 compact JSON合计42,425,591 bytes，其中term_stats为41,425,628 bytes。
- 累计source_identities出现162,168次，独立身份928个。原子样本SHA逐帧变化后被当作来源身份累计，形成显著重复。
- 仅required/contact_persistence/grasp_geometry三项的身份列表各6,351,812 bytes；lift_progress为14,973,744 bytes。

这是已经测量的存储增长原因，**不是隔离出的全部墙钟超时因果解释**。需要同证据CPU profile确定编码、累积统计和其余校验的独立耗时。

## 运行与测试

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/profile_dh116_complete_pair_r0_ledger.py --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/complete-pair-r0-20261114/ledger-profile.json
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_complete_pair_ledger_profile.py tests/test_complete_pair_native_audit.py tests/test_complete_pair_recipe_scope.py -q
```

**18 passed in 3.39s**。首次缺模块、失败receipt无artifact manifest的错误、对应hash缺失测试均保留于私有evidence。未覆盖模型或原始证据。
profile SHA：`3a14f06c83fb066a36fb8f4ba29413a8de06f7efba2048f6709e74f28ad71f5b`。
独立完整源审计 SHA：`c04a96308ef3a5762f4ed5c3262849a73e2be77af75a56f73e5fec791670b34b`。

## 下一工作包

建立独立的固定测量来源身份与逐帧样本证据分离实验；逐帧raw SHA、episode/tick和原始measurement source仍完整留存，不能删除困难样本或变更R0数值。先用同账本验证所有奖励项数值、validity、控制结算一致，再测CPU开销；只有有意义的新实现及冻结指纹满足条件才进入新的已批准有界原生。旧超时配方不重跑、不提高180秒。

并行依赖仍保留：真实approach来源、稳定性与固定材料滑移、正式R0系数、Tripod/Power独立物理资格。没有启用Teacher PPO、R2、BC或DAgger。