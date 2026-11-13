# 2026-11-14 — 双手同杆实际接触与固定材料标记对应残差

基线 `9e00d69f3b5846be40ef181a070aaafbaf8a3693`。本包只读已完成的双手234步原生账本；没有新增Isaac实例、物理步、训练、外力、附着或参数变化。

## 问题和修改

原 `tabletop_cosserat_material_marker_reference` 已实现固定 rest-s/截面标记解析运输，但其测试不是实际碰撞点与材料点的对应证明。本包复用它，补上实际账本接线：

- `tabletop_contact_material_correspondence.contact_candidate`：验证末个求解子步接触点与原节点球面中心一致；在真实 segment XYZW director 下保留完整局部偏移、轴向残差、径向表面残差和候选材料点残差。不调整半径，不强行缩放截面偏移，不删大残差点。
- `bind_candidate/transport`：在第一次加载时固定node、rest-s、局部偏移及指腹参考法向；后续接触丢失或恢复不重新设锚。记录同一pad/node轨迹，不将不同环境或不同节点峰值拼成一条轨迹。
- `audit_dh116_bilateral_material_endpoints`：消费固定receipt/SHA和全部原artifact校验，核对同一SDK时刻的rod、director、pad、实际action identity。只在有同钟director的第11 solver substep计算候选，其他接触保留明确未比较原因。
- 真实 segment director 数为40；旧解析函数所需第41行只作永不读取的内部padding sentinel，不当成实测数据或真实segment。
- 使用新的独立模块/命令，不修改原R0、旧解析函数、求解器、物理资产、材料、动作或模型。

## 原始证据

receipt SHA：`8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8`。
ledger SHA：`d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d`。

输出：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-material-endpoints-20261114-run03/result.json`
SHA：`b96648b2a712aae6ef036d2acdf63d396393bda313f20853414e84373bcc35a5`。

同目录保存完整 `contact_mapping.jsonl` 和 `fixed_marker_tracks.jsonl`，原始数据不进GitHub。
只读耗时3.412688s；来源原生234步，不是本包新增。

运行命令：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_bilateral_material_endpoints --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-material-endpoints-20261114-run03
```

## 实际结果

232个SDK物理端点；2111条手/杆原子contact中，178条位于可与director对应的末子步，1933条因没有同子步director明确未比较。109870条桌面接触单列为本手部材料分析scope之外，不能据此视为没有外部支撑。

178/178候选保留，其中124条的**同节点截面候选残差**超过原0.5mm标准。此处是几何模型残差与标准的比较，不是直接认定真实滑移超标。

| 指腹 | 同钟端点数 | 候选残差P50 (mm) | 最大残差 (mm) | 残差>0.5mm数量 |
|---|---:|---:|---:|---:|
| 左拇指 finger14 |44|0.521474|0.533388|34|
| 左食指 finger23 |42|0.619519|0.655146|42|
| 右拇指 finger14 |45|0.353657|0.508206|1|
| 右食指 finger23 |47|1.331940|1.339164|47|

节点球面中心重构最大残差仅3.374668e-8m，因此该批误差不能仅解释为位置读回时钟错位。接触点含真实轴向偏移；将node-sphere point直接放进同节点Cosserat截面会丢掉该分量。

最大残差绑定：
env0，episode `dual-pinch-76ef2a55c15642c8a40a405096485601:one`，
右食指node3，SDK234，policy28/control substep7/solver substep11，timestamp0.4875s。
不将该结果推广为所有可能连续接触映射均不可行；它明确否定当前“同节点截面直接替代”的资格。

4条首次加载后固定的候选轨迹，无重新选锚；本批对应端点中未出现丢失，接触丢失不清零行为另有CPU测试：

| 固定轨迹 | 样本数 | 最大标记切向位移(mm) |
|---|---:|---:|
| left.finger14/node35 |44|0.076233|
| left.finger23/node34 |42|0.065586|
| right.finger14/node4 |45|0.640657|
| right.finger23/node3 |47|0.139952|

这些数值均为“真实端点驱动的截面重构假设”结果，**不是已资格化R0 slip**。右拇指0.640657mm不能用来声称真实滑移测量已通过，也不能将其隐去。480Hz端点不提供其间5760Hz路径上界，误差界仍NA。

## 测试和失败

- 先写失败测试：15项correspondence失败、4项driver失败，均为模块不存在。
- 首次只读run01失败：`compiled endpoint source differs`。实际原因仅为账本保存integration配置路径而当前审计运行在独立worktree，配置SHA相同。
- 修复仅允许固定原配置路径别名，并验证原文件仍为锁定SHA；不接受任意路径/内容替换。新增该负测试先失败，修复后通过。
- run01、run02完整保留；run03补齐原action的policy/substep身份后重算，仅CPU读盘。
- **54 passed，0 failed，2.89s**：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_contact_material_correspondence.py tests/test_bilateral_material_endpoint_audit.py tests/test_bilateral_material_actual_result.py tests/test_cosserat_material_marker_reference.py tests/test_contact_material_observability.py
```

日志保留在Ubuntu `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/` 下本包 `material-*` 与 `bilateral-material-*` 文件。原生通过次数为0新增，训练更新0。

## 对Teacher下一步的实际影响

- 不把固定node rest-s或当前球面点直接授予完整材料接触锚资格。
- 需要独立验证连续接触对应和表面重构误差，并补齐真实求解子步director/标记路径；仅增加endpoint getter不能解决已观测的1.339mm同节点模型残差。
- 必须保留0.5mm标准、原碰撞和求解器，不能静默改变物理来消除残差。若需新碰撞表示，作为独立物理方案和预算管理。
- R0材料锚、稳定性和Teacher准入仍blocked；本包把阻断从“缺来源”进一步定位为实际模型对应残差与子步可观测性缺口。
- Tripod/Power几何、状态/Reset和WireSeg仍可独立推进，不被此材料诊断阻断。
