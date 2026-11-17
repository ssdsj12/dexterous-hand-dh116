# Tripod 原 24D 动作路径与阶段预算审计（2026-11-14）

基线：`31a97153d5ceda20f4b9481d3ddc0a83c1f47a24`。
本工作包仅 CPU；新 SDK/rod 物理步 **0**、PPO 更新 **0**。未启动 Isaac，未重用旧批准或增加预算。

## 实际修改

- 新增 `scripts/audit_dh116_tripod_sdk_path.py`：`relative_intrinsic`、`compile_actions`、`target_at`、`budget_audit`、`runtime_reference`、`path_constraints`、`target_path_proof`。
- 增加4个测试文件：SDK路径、JSON数值序列化、required cooked contact guard、冻结实际CPU结果。
- 原控制器、24D合同、物理、关节限制、MPC、Teacher、checkpoint没有修改。
- 使用原 `finger_pre_guard`、`surface_final_guard`、`wrist_requested_targets`、`project_policy_wrist_targets`、`local_translation`、`compose_rotation`，没有创建替代控制器。

## 坐标和动作合同

保持12D手指、world XYZ6D、seed quaternion × intrinsic X/Y/Z6D。手指0.3 rad/s、平移向量0.01 m/s、旋转向量0.5 rad/s；480Hz积分，每8步保持同一60Hz动作。不把world rotvec直接塞入wrist Euler动作。不执行隐式速率裁剪、不增幅。

比较：
1. `intrinsic_linear`：相对初始掌位的intrinsic XYZ线性路径。
2. `geodesic_knots`：SO3参考路径在60Hz节点转换为intrinsic XYZ，节点间按真实保持动作积分。

每次CPU目标重建保留24D动作、policy/physics索引、关节目标、局部mount位移、intrinsic角、投影前后差和float32 target舍入误差。明确采用**理想目标跟踪假设**，这不是实际关节、接触或SDK读回。未参与的一侧仅是纯函数批次占位，不代表已验证另一只手。

## 冻结输入及真实CPU结果

来源为上一包实际 near-contact final：
`tripod-hard-constraints-20261114-run01/result.json` SHA
`a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df`。

本机目录前缀：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/`。

| 原始输出 | SHA256 | CPU秒 | 连续证明样本/区间 | 最小参考余量 |
|---|---|---:|---:|---:|
| tripod-sdk-path-20261114-intrinsic_linear-run04/result.json | f0726bb65f7ed3d2e645fb9d93226dd0e83f2100ee1cc928c95fa60fc74156ae | 3.266691 | 197/98 | 0.444078 µm |
| tripod-sdk-path-20261114-geodesic_knots-run04/result.json | 13afddee4057d00b0c51881aa6b9a251cd49d7b52eb048517536f2c6f151a42a | 3.088049 | 165/82 | 0.244524 µm |

两种路径都需要至少 **200物理步/25策略步**；192步及更短无法满足原速率。理想跟踪时安全投影改变目标的步数为0。
累计关节float32目标与理想解析目标最大差 `5.772407850601624e-6 rad`；
world目标重建最大差 `2.1630115527104927e-10 m`。

连续证明针对**分段插值后的float32 SDK目标参考轨迹**，同时计入root平移、intrinsic旋转总变化、主动/被动关节耦合。桌面保持2.1mm；非目标source/cooked碰撞形状保持原rod分离保护；三个目标指腹使用锁定cooked hull，保留原50µm几何接近边界和2µm数值guard。没有将端点通过当作整条路径通过。

参考余量很小，**不覆盖真实跟踪误差、自由杆动态形变、自碰撞或未实测的非指腹cooked变化**。最终三个SDF node gap仍为正，不能称为接触建立。native_ready、actual_tracking_validated、reset_verified、teacher_admitted全部false。

## 预算阻断

保持原阶段：2 setup +48 quiet +200 motion +24 hold = **274 SDK steps**，比本批最多240多 **34步**。
本包没有自动减少静置/保持，没有扩预算，没有生成可执行原生配方。
新阶段必须基于真实seed/静置条件重新核验，并保留逐步guard；若需要更大预算必须另行定界。200步接近不等于完整抓取，更不等于5秒Hold。

## 保留的失败与修正

1. run01第一次CPU计算在JSON输出遇到NumPy int64，留下完整异常日志与不完整result；不得作为有效证据。新增失败测试，显式转换summary计数，完整序列化后再写新目录。没有覆盖旧文件。
2. run02仅证明桌面与非目标杆路径；该证据不含required-pad全路径约束，不能升级为完整通过。
3. run03将CAD source指腹外包也套入cooked pad的50µm条件，得到source finger23/33下界-0.211631/-0.350736mm。这是保守下界、且碰撞表示不一致，**不是实际穿透深度证据**。记录保留，未据此更改物理。
4. run04按原endpoint采用的锁定cooked pad身份补齐required-pad连续检查，同时保留全source/cooked桌面检查与非目标分离；全部CPU参考通过。未删除困难形状或放宽原边界。

上述每次重算均有代码变化，仅CPU，不是原生重试。

## 测试

先RED：9项SDK路径测试；1项序列化；1项required-pad覆盖；1项cooked身份。保留失败日志。
最终回归 **117 passed /0 failed /0 skipped**，5.56s。
日志：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-sdk-path-20261114/regression.log`
SHA `b1671bc59fc3ab55ce6c2348b58d92a8174faf3a0a922f6a8bd924088ff68651`。

命令（Ubuntu）：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_tripod_sdk_path.py tests/test_tripod_sdk_serialization.py tests/test_tripod_sdk_required_contact_guard.py tests/test_tripod_sdk_actual.py tests/test_tripod_pose_path.py tests/test_tripod_fixed_root_path.py tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py tests/test_free_wrist_grasp.py tests/test_free_wrist_mount.py
# 两种mode分别在独立run04目录，CPU最多55秒；不能复用已存在目录
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_tripod_sdk_path --mode intrinsic_linear --output <new-private-output>
```

## 下一包

依据已保存的真实双手Pinch账本核对自由杆seed及静置响应，区分历史Pinch证据和新的Tripod资格。准备独立Tripod required三指捕获/validator与动态接近阶段预算；不能直接复用固定手腕Pinch validator。再完成真实原生接触、Reset和R0。当前Pinch/Tripod/Power均不因此取得Teacher PPO资格。
