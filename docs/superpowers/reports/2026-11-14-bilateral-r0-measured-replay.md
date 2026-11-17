# 2026-11-14 — 同一双手原生 episode 的 R0 实测接近与准入审计

基线：`2551c0260e1c4d6afc2168b313def39307252dae`。本工作包只读既有双手同杆原生账本，不启动 Isaac、不新增物理步、不进行 PPO。旧 R0/单手 adapter、控制、资产、限制和历史模型不变。

## 实际修改

- `tabletop_bilateral_r0_approach.BilateralMeasuredPinchApproach`：独立版本读取 `BilateralOwnedGeometry`；复用原测量、SDF、半径、实际 SDK Reset 和时钟验证。旧 adapter 继续拒绝本次新几何版本。
- `replay_dh116_bilateral_contact_r0.load_source/run`：绑定本次已消耗原生 receipt、recipe、source 和全部原始 artifact SHA，逐同一 episode 读取完整 JSONL，经原 `PhysicalR0Diagnostic` / `ApplicableR0ControlStep` 结算。
- `require_source_scope/replay_scope/summary_gate`：禁止其他 scope、失败/超预算源、缺行/扩预算和诊断器产生 training reward。
- 独立配置 `bilateral_r0_readonly_diagnostic_v1.json`：默认关闭；继续使用既有 CPU fixture ordinal weights，明确不是训练系数。原奖励配置不改。
- 三个新测试文件覆盖旧版本隔离、真实 Reset 基准、动作/节点/半径/身份篡改、缺失项、同一原生账本结算及禁止 Teacher 准入。

## 原生来源与独立运行

已完成来源 scope：`bilateral_owned_pinch_2_setup_232_guarded_contact_diagnostic_only`。
来源 receipt SHA：`8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8`。
来源 ledger SHA：`d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d`。
来源 234 SDK 步（2 setup + 232 measurement）、166.289762s、自然退出0；这是已完成的原生执行，不是本次复放新增。

只读结果：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-r0-evidence-20261114-run01/result.json`
SHA：`9720b12aecd6aeb98524d42e2ea6e04a795b06b4b1b08211b1fc253dbb0158e3`。
完整逐样本输出保存在同目录 `progress.jsonl`，不上传原始数据。

命令（Ubuntu，当前任务 worktree，独立新输出目录）：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.replay_dh116_bilateral_contact_r0 --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-r0-evidence-20261114-run01
```

实际耗时44.596716s；未导入 omni/isaacsim/isaaclab/ropesim/warp，无新物理步、PPO更新或成功样本。

## 结果

- 232/232 个实际物理样本接近距离有效；29/29 完整8子步控制区间正确结算。
- 首个测量样本（SDK3）左/右平均正间距：1.149221 / 1.067610mm；最后（SDK234）：0 / 0mm。
- 接近进度基准来自实际 SDK2 reset getter，未用 authored seed 或 SDK3 替代。
- 距离0仅是实际杆段/指腹几何结果，不代表稳定接触、材料锚、Hold或成功。
- 当前同步 common 状态66/371维有效；其余保留 NaN 和 false validity，不补零，不宣称完整371D Teacher状态。
- 全部29个控制步的 diagnostic reward 和 training reward 均为NA；没有训练资格。
- 原生双手四指同时有载荷连续85.9375ms，但全程桌面承托。不能用该事实补齐有效稳定对向接触。
- 无新增完整episode评估，原30mm/5s/0.5mm标准不变。

| 阻断奖励项 | 缺失物理样本 /232 | 阻断控制步 /29 |
|---|---:|---:|
| hold_progress | 232 | 29 |
| instability | 232 | 29 |
| required_contact | 45 | 6 |
| contact_persistence | 45 | 6 |
| grasp_geometry | 45 | 6 |
| lift_progress | 45 | 6 |
| material_anchor_slip | 45 | 6 |

前段已知未双侧接触域中的 slip contribution 不适用与原始 slip 测量缺失明确区分；接触历史不明后不再给予不适用零贡献。不能将 raw NA 解释为测到零滑移。

## 测试与失败证据

先写失败测试：
- approach：1 passed、10 errors，缺新 reader。
- replay：12 failed，缺新 replay。
日志保留于 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-r0-approach-red-20261114.log` 和 `bilateral-r0-replay-red-20261114.log`。

修复后相关新旧回归：**71 passed，0 failed，23.77s**：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_bilateral_r0_approach.py tests/test_dh116_bilateral_r0_replay.py tests/test_dh116_bilateral_r0_actual_result.py tests/test_dh116_segment_r0_approach.py tests/test_dh116_segment_r0_replay.py tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py
```

CPU回归包含对既有真实原生数据的只读验证；不计作新的原生实验。受保护528文件及两个历史checkpoint由发布门重新校验。此包没有接触任何模型或原始数据写路径。

## 下一步与未解除门槛

1. 同钟有效对向接触、固定材料锚/切向滑移、稳定性谓词必须由合格真实测量闭合；不能用多指载荷或SDF预测替代。
2. 原生R0 owner Reset顺序、完整状态来源、正式系数仍待独立资格，不因复放能结算而放行PPO。
3. Pinch已从单侧接触推进到同episode双侧短接触。Registry删除过时的“右手未原生执行”阻断理由，但保持Teacher blocked。
4. Tripod、Power继续独立几何/接触路线，不继承Pinch通过结果。无相机主线不等待WireSeg或C5。
5. 5s Hold至少2400物理步/环境，不扩用已消耗234步预算；短诊断不冒充完整抓取。
