# Canonical 24D action：Phase 0 源码合同审计

审计基线：`6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b`，分支 `codex/dh116-stereo-privileged-dagger`。本报告仅增加文档和 CPU 审计测试；没有修改控制器、Actor、模型、配置或 native 数据，没有启动 Isaac、GPU 或训练。规范文件 SHA256：`a09551b72c4540c3fb88f308461a2b0176cc5c625cbc974c8a138b765f90a2e6`。

结论：当前 surface 24D 路径的主动关节顺序、Actor 动作分组、解码、控制目标和按名称写入 SDK 的代码一致。有限幅、状态相关投影和机械跟踪延迟，因此 Actor 输出、issued action、目标及实际状态不能共用一个“executed action”字段。现有 Teacher 入口包括受资格约束的 24D 轨迹重放及 opt-in MPC；这不是规范 Phase 1 要求的 privileged PPO Teacher，也不是任意 Student 状态上的 DAgger 查询 oracle。

## 1. 唯一语义源与 SHA

唯一 24D 语义源是 `src/go2_pvcnn/control/dh116/tabletop_surface_contract.py::ACTION_CONTRACT`，版本 `dh116-surface-motion24-v1`，动作名 `24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1`。主动关节顺序来自 `linear_coupling.py::ACTIVE`，不依赖 SDK 数字索引排序。

Canonical action SHA256 为：

```text
498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9
```

计算严格复用 `tabletop_vision_command_shadow.py::action_contract_sha256()`：

```python
sha256(json.dumps(ACTION_CONTRACT, sort_keys=True,
                  separators=(',', ':'), allow_nan=False).encode()).hexdigest()
```

它覆盖完整 `ACTION_CONTRACT`，不是只对维度数或动作名做 hash。`surface_contract_sha256` 是完整 371D surface 合同的另一种摘要，含 observation、任务语义等，不能当作 action-only SHA 使用。controller source SHA 又是另一项，覆盖 command kernels、free wrist、tensor world、shadow 四个文件的名字和原始字节。

机器可读逐维合同、30 个源文件的原始字节 SHA256 和审计缺口见 `docs/contracts/stereo_action_contract.json`。源文件摘要用于固定本次基线；未来有授权源码变化时，应保留这份基线证据并显式更新后续阶段合同。

## 2. 逐维顺序、单位、缩放、方向与边界

表中 `a[i]` 全部是无量纲归一化输入，合法范围均为 `[-1,1]`；unit 表示解码后的物理请求单位。手指是主动关节目标的积分速度，平移是世界坐标目标偏移的积分速度，旋转是 intrinsic XYZ 角参数的积分速度。rate 是请求/目标变化速率，不保证 measured state 达到该速度。下面每一行的归一化 bounds、sign、rate 和控制规则均须同时应用。

|index|name|unit|scale|normalized bounds|sign|rate / control meaning|
|---|---|---|---|---|---|---|
|0|`left.finger11`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|1|`left.finger12`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|2|`left.finger21`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|3|`left.finger31`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|4|`left.finger41`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|5|`left.finger51`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|6|`right.finger11`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|7|`right.finger12`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|8|`right.finger21`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|9|`right.finger31`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|10|`right.finger41`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|11|`right.finger51`|rad/s|0.3|[-1,1]|named joint +rad|0.3 rad/s; J|
|12|`left.translation.X`|m/s|0.01|[-1,1]|world +X|per-hand norm 0.01 m/s; T|
|13|`left.translation.Y`|m/s|0.01|[-1,1]|world +Y|per-hand norm 0.01 m/s; T|
|14|`left.translation.Z`|m/s|0.01|[-1,1]|world +Z up|per-hand norm 0.01 m/s; T|
|15|`right.translation.X`|m/s|0.01|[-1,1]|world +X|per-hand norm 0.01 m/s; T|
|16|`right.translation.Y`|m/s|0.01|[-1,1]|world +Y|per-hand norm 0.01 m/s; T|
|17|`right.translation.Z`|m/s|0.01|[-1,1]|world +Z up|per-hand norm 0.01 m/s; T|
|18|`left.intrinsic_rotation.X`|rad/s|0.5|[-1,1]|intrinsic +X right-hand|per-hand chart norm 0.5 rad/s; R|
|19|`left.intrinsic_rotation.Y`|rad/s|0.5|[-1,1]|intrinsic +Y right-hand|per-hand chart norm 0.5 rad/s; R|
|20|`left.intrinsic_rotation.Z`|rad/s|0.5|[-1,1]|intrinsic +Z right-hand|per-hand chart norm 0.5 rad/s; R|
|21|`right.intrinsic_rotation.X`|rad/s|0.5|[-1,1]|intrinsic +X right-hand|per-hand chart norm 0.5 rad/s; R|
|22|`right.intrinsic_rotation.Y`|rad/s|0.5|[-1,1]|intrinsic +Y right-hand|per-hand chart norm 0.5 rad/s; R|
|23|`right.intrinsic_rotation.Z`|rad/s|0.5|[-1,1]|intrinsic +Z right-hand|per-hand chart norm 0.5 rad/s; R|

**J 规则：** `command += a[i]*0.3*dt`；随后 measured `q±0.1 rad` lead、affine mimic 可行主动关节位置区间、每 substep `|Δcommand|≤0.3*dt`；surface final guard 再限速和裁剪位置。所有手指正号只表示该手自身 authored joint 正轴的角度增加，不能泛称“闭合”或把右手做镜像取反。每手主动关节位置上下界（rad，按六关节顺序）为：

```text
lower = [0, 0.007912622218025698, 0, 0, 0, 0]
upper = [1.3962634015954636, 1.03, 1.2217304763960306,
         1.2217304763960306, 1.2217304763960306, 1.2217304763960306]
```

**T 规则：** 每手 3D 输入先除以 `max(||a_xyz||,1)`，再乘 `0.01*dt`。目标 offset 的 X/Y 区间是 `[-0.03,0.03] m`，Z 是 `[-0.004,0.05] m`，基准为 authored/reset root seed 加 task lift offset；世界 Z 正向向上。World transport 再做向量速率限制及 table clearance 投影。安全恢复可以覆盖普通 slew，例如已低于 clearance floor 时，目标会优先抬回 measured safe Z；它不是额外策略上升动作。

**R 规则：** 每手 3D 输入先除以 `max(||a_XYZ||,1)`，再乘 `0.5*dt`。每个累积角参数独立裁剪至 `[-π/2,π/2] rad`。XYZ 是串联 intrinsic chart，不是 world angular velocity，也不是 rotvec；多轴同时请求时不能把每轴各自看作 0.5 rad/s。

合同要求 physics 480 Hz / control 60 Hz，即每 tick 八个原生 substeps。无投影时每 substep 最大手指目标增量 `0.000625 rad`、每手平移向量增量 `0.0000208333333 m`、每手 XYZ 角参数向量增量 `0.00104166667 rad`；八步合计分别 `0.005 rad`、`0.000166666667 m`、`0.00833333333 rad`。后两项是向量范数限制，不能逐轴累计成更大向量。角参数范数速率也不等价于组合姿态的 world 角速度上界。

被动关节不是额外动作维度。`MIMIC` 是 `finger13←finger12`（ratio `0.88178789`, offset `-0.00697725445 rad`）及 `finger22/32/42/52←finger21/31/41/51`（ratio `0.9206`, offset `0.011072 rad`）。`active_limits()` 取 source、manual 和所有 affine follower 区间交集。线性展开是模型关系，不能伪造 measured follower 状态或 native 约束实际满足。

## 3. 坐标系、XYZ 组合与 SDK 链路

`compose_rotation(seed, angles)` 先归一化非零 wxyz seed，然后依次右乘 `qX(α), qY(β), qZ(γ)`：

```text
q_target = q_seed * qX(α) * qY(β) * qZ(γ)
```

因此正轴按右手规则，瞬时 world 轴分别是 `Rseed*ex`、`Rseed*Rx(α)*ey`、`Rseed*Rx(α)*Ry(β)*ez`。这些轴来源于 authored wrist seed，不能从发生驱动 lag 的 measured base 姿态逆推 seed。palm-local translation 只是 `Rpalm.T @ velocity_world` 的诊断表达，canonical 输入本身仍是 world XYZ。

实际调用链：

1. `tabletop_vision_training.py` 选择既有 `make_vision_algorithm` 或 `make_squashed_vision_algorithm`，`algorithm.act(obs)` 生成 normalized24；默认 mirror preprocessing 为 `m0_raw`。opt-in `m2_full_mirror` 会在提交前反变换 action，必须保存变换前后动作和模式。
2. `TensorEnv.step(issued)` 调 `TensorTask.begin_step()`；每个物理 substep 可调用明确安装的 `env.bimanual_action_adapter`，随后 `TensorTask.control()` → `tabletop_policy_control.control()` → `tabletop_free_wrist_grasp.control()`。
3. surface 路径从 `ACTION_CONTRACT` 取真实 `.3/.01/.5`，经 `finger_pre_guard`、`surface_final_guard`、`wrist_requested_targets` 形成 joint/palm/rotation targets。surface enable 默认 `surface_policy_owned_targets=True`、`redistribution=False`；有限 mimic error 不触发软件 master 纠偏。非 policy-owned profile 才在 `|mimic_error|≥0.012 rad` 等条件下叠加 bounded correction，再限速；不能误称 surface 默认一直纠偏。
4. `TensorWorld.advance()` 经 `palm_transport`、`per_hand_safe_z` 形成 root pose 和 commands。live 和 allowed 决定是否积分/保留；安全 table projection 可取消积压下降。`joint_requested_target` 与 `joint_governed_target` 另存，且前者已经历 pre guard，不是原始 Actor 输出。
5. `GpuWorld.__init__()` 用 `self.ids=[[h.joint_names.index(n) for n in ACTIVE] ...]`，SDK 写入 `hand.set_joint_position_target(commands[:,hi].float(), joint_ids=ids[hi])`。这里写的是位置目标；没有直接把 normalized24 当 SDK velocity 指令。
6. 外部 wrist `MountView.move()` 把 root world translation 通过 reset/current anchor 的逆旋转换为 prismatic local targets，并按名称 `wrist_x/y/z`、`wrist_roll/pitch/yaw` 写 float32 position targets。旋转角读写为 rad；USD authoring revolute limits 为 degree。外部腕部不是 DH116 手指 DOF。
7. `write_data_to_sim()` 后才发生 `sim.step()`、`refresh()`；实际 q/qd、wrist pose/velocity 和 backend target readback 与以上“提交目标”分别记录。目标和测量状态存在动态跟踪误差，CPU inert sink 测试不能证明 PhysX/hardware 跟踪。

`TrainingActionAdapter` 这个符号在审计基线的 `src/`、`scripts/`、`tests/` 中未找到。实际可选点是 `env.bimanual_action_adapter`，不能把一个未存在的类描述成已启用的生产接口。

`BimanualPhysicalActionAdapter` 为 opt-in 物理语义审计：左右手独立 USD 关节树，不做反射猜测；`to_sdk(decode(action))` 保持原 normalized24。其 `decode()` 给出的 translation/intrinsic rates 是 **raw per-axis nominal scale**，尚未做 controller 每手 vector-norm 限速，因此输入 `[1,1,1]` 的诊断线速度范数为 `sqrt(3)*0.01`，实际 controller 请求增量范数受 `.01*dt` 限制。这是诊断层级差别，不能把 decode 数值当最终 executed velocity。

`--bimanual-explicit-sdk-adapter` 是 `store_true`，默认关闭，仅 Teacher source collection 被允许显式安装；安装后在每次 decode/roundtrip 验证差异 ≤`1e-7`，保持原 action bits，独立输出 adapter 合同。checkpoint 推理只有明确匹配的 `physical_action_adapter` 声明及 runtime verification 后可用；旧 checkpoint 不会自动启用。普通 24D surface 路径不依赖该 adapter 才能按名称执行。

## 4. Actor、nominal、issued、applied、SDK、actual 分开保存

|字段/层级|精确定义|当前代码与审计注意点|
|---|---|---|
|`latent_mean` / `mu`|既有 squashed 策略的 unbounded Normal 均值|Actor 最后一层 `nn.Identity()`，不是 bounded action；无物理单位|
|`latent_std` / `log_std`|pre-tanh Normal 的 std / log std|`.001..0.3` 投影作用于 latent noise；不是 N、rad/s 或执行噪声|
|`latent_actions` / `z`|Normal 实际采样值|PPO 必须保存原始 z；不能从 rounded `±1` action 反算 `atanh`|
|Teacher sampled action|未来 PPO Teacher 自己 rollout 的 `tanh(z_teacher)`|现有 trajectory/MPC Teacher 没有该分布，字段应缺省/明确 unavailable，不能伪造 z/log_prob|
|Teacher deterministic mean action|未来 PPO Teacher 的 `tanh(mu_teacher)`|这是约定的 deterministic action，不是数学上 `E[tanh(z)]`；未来 BC/DAgger 默认用此标签|
|Student nominal action|Student 在任何 mirror、adapter、safety 或混合执行前提出的 normalized24|采样 rollout 是 `tanh(z_student)`；确定性推理是 `tanh(mu_student)`；须注明模式|
|`issued_action`|提交给 `env.step` 的真实 normalized24|可含 m2 反变换或未来 Teacher/Student 选择；history.previous_action 保存这个层级，done 后清零|
|task accepted / `last_raw_action24`|`begin_step` 处理后的动作副本|该名字虽含 raw，实际 continuous finite 输入已经 clamp 到 `[-1,1]`，invalid 行已归零，不能代替原始 Actor/issued 记录|
|physical action after optional adapter|每 substep 送入 `task.control` 的输入|仅无 adapter 或验证 identity 时才可等同 issued；不能默认所有实验回调都 identity|
|requested / governed joint, palm, angle targets|积分、lead、joint bounds、norm、clearance guards 各阶段的目标|物理单位 rad/m；不是 normalized24，也不是 measured physical action|
|SDK submitted targets / backend readback|按名称写入 articulation/mount 的 float32 targets 及实际 backend 读回|读回 unavailable 或时钟未对齐必须标无效；不能拿 Python cache冒充 backend 实证|
|applied/executed physical result|同一 interval 的 measured q/qd、wrist pose/velocity、接触及实际增量|以 pre/post measurement 和 native timestamps 表达；有有效性、recovery、terminal/reset 边界，不能由 action*scale 代算|

`begin_step` 的具体差异：continuous 模式把有限越界数值先 clamp，NaN/Inf 保持 nonfinite 再判 invalid，live invalid 行记失败并归零；shape 不对直接拒绝。独立 `surface_contract.validate_actions()` 更严格，要求 floating `[N,24]`、finite、区间内，越界不会静默 clamp。训练 tanh 路径在 env 前另做 finite/shape/range 检查。必须保存进入 `begin_step` 前的请求才能发现超大动作是否被裁剪。

已有 legacy v2 Actor 自带最终 tanh，但探索采用 Normal around bounded mean，训练提交前再 clamp；它不是当前 v3 `tanh(Normal(unbounded_mu,std))`。v3 重用 `tabletop_squashed_gaussian.py` 的 stable Jacobian、saved-latent log_prob、entropy、latent KL 及既有 PPO storage，不重新实现 boundedGaussian。Phase 0 测试没有构造 ActorCritic 或 optimizer，没有训练。

当前 `VisionActionDiagnosticRecorder` / physics telemetry / sequence sidecars 可提供各级目标及读回证据，但不是所有默认 rollout 都完整保存上述所有层级。未来 Teacher+Student/DAgger 数据合同仍须增加完整的双方 action provenance、deterministic label 和同 interval physical validity；本报告没有声称这些功能已存在。

## 5. Teacher / Student 同合同：现有强制检查与缺口

|边界|现有强制检查|资格和限制|
|---|---|---|
|旧 native 24D trajectory loader|`load_teacher_actions(require_action_contract=True)` 需要 metadata 完整等于 `ACTION_CONTRACT`；缺声明时必须有 source-file SHA 绑定的 audit 文件，拒绝 26D/未知语义|只接受 qualified native lift metadata 与 bounded24；schema/历史声明不等于本次实际成功|
|371D Teacher reference corpus|`load_teacher_reference_dataset()` → surface imitation validation → `surface.validate_contract()`，比较完整 canonical action dict；必要时核当前 critic 合同/reset family|原始24D references直接进入重放，不是对26D切片转换；known pinch/tripod、reset-family source SHA 等另有要求；函数返回 `state_query_oracle=False`|
|Student visual contract|`tabletop_vision_training._contract()` 写入同一 `ACTION_CONTRACT`；visual dataset validator要求完整 dict 相等，resume另核完整合同|共享 action dict 可按同一公式算 SHA，但并非所有普通文件都有单独 action-only SHA 字段|
|conditional command shadow|`controller_contract()` 写 canonical action dict、action SHA、controller source SHA；`_validate_contract()` 检查 action/free-wrist dict、profile、callback和sourceSHA|**现有 `_validate_contract()` 不独立比较其 `action_contract_sha256` 字符串**。CPU审计明确展示该冗余字段被改时这里不会拒绝；不能把它宣传为全面强制SHA验证|
|sequence sidecar boundary|`validate_sequence_context()` 默认计算 canonical action SHA并检查 context 字段；还绑定 dataset bytes SHA/可选 expected source SHA；join按run/env/episode/tick匹配|不得绕过该边界只调用 shadow 后声称文件已验SHA；显式传入 expected action SHA 时调用者也必须保证它来自同一canonical源|
|opt-in native sequence qualification|native capture、eight substeps、Teacher-shadow targets和SDK target readback parity另作证明|重放受资格限制；CPU fixture 输出不可当 native qualification|

现有 `tabletop_surface_teacher.py` / `tabletop_surface_teacher_control.py` 是 legacy 26D source（含 force2、旧 observation/success qualification），必须与 canonical24 入口区分。不能因为该文件仍为26D就否定当前受审计的24D trajectory，也不能认为切去最后两维即可满足语义和成功资格。`bind_contract()` 仅显式把已有 audited351D/24D metadata 扩展到371D，明确拒绝26D；它不改 action、不生成真值。

`run_teacher_collection()` 当前在同步 pre-action visual history 上记录 `reference_actions` 与真正送入 env 的 `actions`（可为 constrained MPC refine 后动作），native first episode、无 recovery、原 success、contact-style 和 provenance审计之后才生成 successful pool；失败/未完成另存。源码明确 `ppo_updates=0`、`student_policy_trained=False`、`dagger_state_query_oracle=False`。Constrained MPC可依当前 contact context refine轨迹，但不能据此当作规范里未来独立训练的 371D privileged PPO Teacher 或完整在线 DAgger。

未来 privileged PPO Teacher 和 Student 应各自显式保存同一个 canonical action-only SHA，load/query/BC/DAgger/rollout 边界都核完整 dict和该SHA，不接受只看24D shape。它们还需分别保存 Teacher deterministic/sample、Student nominal、issued、physical target/result与版本。此处属于后续阶段要求，Phase 0 没有新增强制逻辑。

## 6. 本次 CPU 验证与仍未证实项

新增测试 `tests/test_stereo_action_contract_audit.py`：24个基向量逐项经过真实 `TensorTask.control()` / free-wrist kernels / `TensorWorld.advance()`，与 conditional shadow和 SDK float32目标一致；SDK sink和 sim.step为既有 CPU测试替身。另覆盖向量norm、位置角度边界、mimic可行界、policy-owned默认、zero-action lead/clearance例外、非交换XYZ组合、physical adapter nominal roundtrip与按名称SDK顺序、`MountView.move()` 的anchor局部变换/真实joint_ids/float32提交、严格shape/dtype/nonfinite/range、continuous clamp/raw字段语义、既有saved-latent Gaussian端点、unsupported语义/SHA拒绝、shadow digest缺口和禁止SDK导入。

先运行 manifest 测试，按预期因文档 JSON 尚缺失而失败；新增 JSON 后完成全测试。指定CPU解释器本身无scipy，测试只读使用仓库已存在的 Windows CPU依赖目录；没有安装包。复现命令：

```powershell
$env:PYTHONPATH='src;artifacts/stage7_lifecycle_cpu_deps_v01'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_stereo_action_contract_audit.py
```

本次结果：`42 passed`，exit `0`。此前收集暴露 scipy/gymnasium环境缺包；最终测试直接构造 CPU controller state、使用 inert SDK sink，避免 Gym/environment构造，并读取已存在scipy依赖，无生产源码改动。

未由本次验证：当前真实 USD/PhysX 的 articulation/mimic运行值、原生backend targets跟踪、接触/摩擦/力、实际速度上界、真实 grasp成功、硬件轴标定、Teachercheckpoint的训练资格、任意状态Teacher查询及在线DAgger。独立的左右手资产解析/几何模型也不能替代这些物理项。只有未来明确授权的原生运行及相应新证据才可作此类结论。
