# 双手 Pinch 原生账本逐求解器子步载荷审计（2026-11-14）

## 结果

基线 `224c04371af4d8cbb027aa1a757e7b88600ec0b5`。只读分析已消耗、自然退出0的原双手234步账本，没有重跑原生预算。

232个测量物理步、每步12个 DeformX 子步，共 **2784** 个独立时间窗口。按原 `surface_targets` 的严格载荷 **>0.05 N**、法向点积 **<=-0.25** 规则：

| 指标 | 左手 | 右手 |
|---|---:|---:|
| 原始法向配对的子步数 |493|527|
| 拇指与食指同时超过原载荷阈值 |493|527|
| 两条件同时成立 |493|527|
| 手对杆合力Z分量为正的子步数 |526|0|
| 拇指法向载荷峰值 N |0.263557|0.373068|
| 食指法向载荷峰值 N |0.287860|0.189797|

双手两个条件同时成立 **493个连续子步（85.590278 ms）**，首个在SDK193/solver2292，末个SDK234/solver2784。桌面在 **2784/2784** 子步都有载荷。因此不符合无外部承托要求，也没有5秒Hold或30mmLift成功证据。

这不同于此前“至少一条接触对”的85.9375ms计数：新统计严格执行已有载荷阈值，不把极小载荷视作有效承载。没有降低阈值。右手本段合力向下或为零只描述这次接近/观察轨迹，**不能外推成以后执行Lift一定失败**。

原始SDF网格接触尚无有效指腹区域资格；上述是“raw surface paired”诊断，不自动赋值原R0的有效对向接触。稳定性、材料滑移、完整R0及Teacher资格仍未通过。

## 修改

- `tabletop_pair_substep_load_audit.audit_capture`：按同一求解器子步分别聚合每手五个指腹的原生法向载荷、载荷加权法向、总力、接触对数及法向一致性；复用既有 `tabletop_surface_force_sharing.surface_targets`。同一指腹多个节点可以合法合力，但不增加指腹数；重复(node,mesh,substep)拒绝。
- `bind_clock`：验证run/env/episode/reset/rod、物理步、rod tick与实际SDK边界一致。未知/不完整捕获不能变成“实测零载荷”。
- `audit_dh116_bilateral_substep_loads.py`：只读取固定回执及SHA锁定账本，保存全部2784行及逐指峰值的真实env/episode/SDK/policy/solver身份。记录原物理端点时间；子步时间明确是基于tick和dt推算的 nominal 时间，不冒充每个子步独立时钟读回。
- `test_pair_substep_load_audit.py`：13项新增测试，包括时间错开的两指不能合成同时接触、严格0.05N边界、同指多节点、重复对、force/impulse一致性、缺失来源、越界子步、法向与切向分离、外来episode/时钟拒绝。

配置复用原 surface-sharing 合同及冻结原生回执；未新增奖励系数、未修改任何控制器/表面参数，也未接入训练奖励。

## 证据与运行

原生回执 SHA256：
`8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8`。

原生 physical.jsonl SHA256：
`d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d`。

原surface-sharing源码 SHA256：
`14cee806b3e6c339a7e1242b65f43391cee0435a1efa67ecc9f0ffe11b20f0d3`。

最终结果：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-substep-load-20261114-run02/result.json`，
SHA256 `f3a8076724fb0f706167192aecf0c0645fccb88b091fd7677bd5f290dbe4f102`。
相邻 `substep_loads.jsonl` 保留完整时序和峰值身份，不上传大型账本。不同指腹峰值可能来自不同子步，不组成一条虚构同步轨迹。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_pair_substep_load_audit.py tests/test_surface_force_sharing.py tests/test_dh116_bilateral_r0_replay.py tests/test_dh116_bilateral_native_evidence.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_bilateral_substep_loads.py --output <新的独立目录>
```

**CPU 48 passed / 0 failed**，3.14s；日志SHA256 `896d7b52efb80c4f4bdd2f2c9a097828a6c090acd95836b9d5244ab573d4c3a3`。先保存缺模块RED与时钟绑定缺失RED，再实现并运行回归。

新原生步数0，PPO更新0；这里是原生数据离线回放，不是新原生采集或训练。CPU测试通过不改变原native预算及失败回执。

## 后继

继续有效指腹区域、同子步材料身份/固定锚滑移、稳定性与R0系数来源审计。Pinch已有短时接触不等于悬空承托；Tripod/Power仍需各自几何与原生验证。原30mm、5秒、0.5mm及无外部支撑标准不变。PR #2保持Draft。
