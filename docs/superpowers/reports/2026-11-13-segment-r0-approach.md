# 实测 Segment 几何接入 R0 approach 与完整短账本重放

基线69d39a96dda9bd3b3aa9c66c64dc82d00f42ec7b。本包新增物理步0、PPO更新0；只读既有234步原生结果，不能称作新原生采集、完整episode评估或Teacher训练。

## 修复的实际断层

旧MeasuredPinchApproach仅接受旧MeasuredClosureGeometry，最新SegmentClosureGeometry账本会被拒绝。新增tabletop_segment_r0_approach.py的SegmentMeasuredPinchApproach，显式选择原始segment validator，完整重算source、实际位姿、节点、半径、SDF及逐段证明后复用原approach计算。

新来源独立version/SHA，附继承metric实现SHA和geometry_contract_version。旧adapter、旧native配方、旧receipt和旧默认路径均不改。Reset值来自同一原生reset后getter，不能拿第3步代替第2步，也不把名义根姿态当实测。

scripts/replay_dh116_segment_contact_r0.py流式读取已锁定96MB JSONL，读取前后均核验SHA，拒绝软链接、空行、不完整行、错SHA和读取中变更。旧PhysicalR0Diagnostic检查真实24D请求/SDK目标、66个common有效字段、每步连续时钟、原guard、RodBoundaryState，并用现有opt-in applicability reducer结算，不重写PPO或奖励。

每个232实际物理样本只计算一次R0；每8样本结算一次，共29窗口。fixture系数仍仅CPU诊断用途，原任务训练系数保持未知。未创建模型，未填任何稳定性、滑移或有效对向接触标签。

## 原始证据与独立重放

- 原始scheduler failed receipt SHA97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746，保持原failed状态。
- 原234步physical.jsonl SHA170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf。
- 独立postexit复审SHA57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf，只证明限定短接触执行与自然退出。
- 本次输出/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/segment-r0-replay-20261113-dsubofqu/run01。
- result.json SHA56998139c8dc546a779367b4f555119ab682fbafbe71a41ba7ca7996a1608641。
- progress.jsonl SHA56051739a1905d8e5e6030f89490885b9f351d0db55058b4e5dc1187ea1a348b。
- CPU28.286388s，exit0，native_imports空；有界wall600s。仅重放原0.4875秒前缀，不是5秒Hold评估。

| 来源/结果 | 实际值 |
|---|---|
| setup/测量/总SDK步 | 原2/232/234，新步0 |
| 原测量范围 | SDK3～234，29×8控制窗口 |
| 原24D/371D合同 | 保留；common66有效/371，其余NaN+false |
| 同钟approach有效样本 | 232/232 |
| Reset左/右距离 | 1.149197/2.160668mm，实际SDK2 getter |
| 首测量左/右距离 | 1.149221/2.160681mm，SDK3 |
| 末测量左/右距离 | 0/2.161314mm，SDK234 |
| 累计approach_progress raw | 0.574275mm（双手均值差） |
| 有限完整diagnostic_reward窗口 | 0/29 |
| 原生资格/Teacher准入 | false/false |

左末态approach=0仅表示所需采样节点的正SDF间隙均已消除；不能替代有效接触、承托、滑移或成功。

## 准确定位的剩余缺项

- hold_progress、instability全部232步缺少合格稳定性来源，保持NA。
- SDK193～234共42步，左手同时有多个pad pair。旧测量adapter谨慎拒绝将多指力记录直接称为有效对向接触；因此required_contact、contact_persistence、grasp_geometry、lift_progress各42步缺失。
- 同42步接触历史变为未知，material_anchor_slip不能沿用“确定处于双侧接触前”的不适用贡献，仍NA。没有补零。
- 实际正向对向承托、固定材料锚/表面滑移、hand-table来源、正式训练系数及native R0 reset owner顺序仍待资格验证。未因短左接触解除双手抓取门槛。

## 测试与保护

先模块缺失RED，再通过新来源/同钟Reset/篡改拒绝/文件完整性/短前缀非完整episode等18新增测试；合并原approach、Reset、applicability及native-parameter重放回归共92 passed，0failed，34.31s。
日志/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/segment-r0-full-20261113.log，SHAa4813ababa7efae6a6b5cc84aa71fe0ee3fc79879f5bc21d59ae31daa2d1d2e7。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_segment_r0_approach.py tests/test_dh116_segment_r0_replay.py tests/test_dh116_r0_measured_approach.py tests/test_dh116_r0_reset_approach.py tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_native_parameter_r0_replay.py
```

CLI：python -m scripts.replay_dh116_segment_contact_r0 --output <不存在的独立目录>。已有本次结果不覆盖。原生0，GPU0，训练0。保护528文件及两个checkpoint由发布器复核；经Windows导入同commit发布，PR2Draft。

下一步：资格化同一solver子步的真实对向接触与稳定性来源，继续独立右Reset和Tripod/Power几何工作。不能以fixture系数或CPU重放启动正式PPO。
