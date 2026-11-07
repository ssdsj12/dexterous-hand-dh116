# 视觉 Student：Teacher 参考动作与 MPC 监督

## 实现与边界

按用户要求增加独立链路：**历史 Teacher 参考 → 当前原生状态上的局部 MPC → 同步视觉动作监督 → 独立 Actor BC → 显式 v3 warm-start**。原 `tanh_gaussian` PPO、3844D Actor、371D Critic、24D 动作顺序/尺度/坐标系和默认 v2 保留。物理、执行器及净抬升30mm/持续5秒/滑移0.5mm原生成功标准未修改。PR #1 继续 Draft；没有长时间 PPO。

成功的历史 Teacher 是原生物理反馈控制/动作示范，并不是一个已训练成功的特权 PPO Actor。可用的 `dh116_teacher_multistyle_371_v1/surface_demonstrations.pt` SHA256 为 `f4fa715c58ce0d767588a6f2bf5aed248d62d22833d4794a97343baf31b8aaf9`，4891组371D/24D、四个回放episode，实际来自二指AM与三指AK两条原始轨迹。同一来源的重放全部绑定同一分组，不能随机拆相邻四帧历史来制造独立验证集。

### 同步采集

- 新 `--vision-teacher-source` 路径在普通 `TensorEnv.step()` 中执行 canonical24D参考动作；仅新监督模式可再加 `--vision-teacher-mpc`。旧 `--surface-mpc` 仍是独立 inference-only 选项，其旧限制不变。
- 每步先冻结物理采真实 RGB-D，再调用原 `SensorHistory/_observations` 构建3844D。输入仍只有图像、光学深度、标定、编码器/FK、上一步实际命令和时序/有效性；GT和PhysX仅用于训练专家与诊断。
- 记录同一时刻的 frame/env/episode/tick/physics-step/sim-time、来源SHA、原参考和实际issued动作。实际issued才是下一帧previous_action和BC标签。复制边界防止环境修改调用者动作。
- 原图通过默认关闭的只读tap按最多64帧分块落盘；无额外render/physics step。包含RGB、depth、K、外参和实测掌FK，失败实验原始数据保留。
- 只接收本次首episode、无recovery且实际原生success的轨迹；原生池还检查末5秒接触负载的实际抓法。失败/未完成样本单独保存，不能进入成功示范池。异常中断保留partial数据，明确末动作执行完整性未知。
- 历史tick动作只作当前回放的轨迹参考，不是Student任意离轨状态的可查询Teacher或DAgger oracle。

### MPC 范围

`TeacherReferenceMPC` 重用现有多步 `SurfaceGripMPC`：预测局部关节响应和pad运动，联合优化未来4步的12个手指命令，实际发出24D，腕部12维保持Teacher参考。保留原有关节/速率限制、净升至少3mm且双手有效对向接触才激活、坏预测回退。绑定实际q、command、pad/contact、body pose/COM和PhysX Jacobian快照原始字节SHA，记录每环境激活/修正/回退原因。

这是局部运动学模型，不预测物体/摩擦/力或完整碰撞动力学，不能声称24D联合物体动力学MPC。尤其不能承担尚未建立接触或尚未达到3mm时的抓取获取；此时跟随Teacher参考。优化cost下降和模型可行不等于原生抓取成功。模型标签诊断保持 `native_correction_qualified=False`、`student_state_dagger_label=False`，新成功示范资格只能来自随后实际执行的完整原生episode。

### Student 学习与 PPO 隔离

新 `fit_dh116_vision_supervision.py` 只构造现有视觉Actor，以单次 `tanh(mu)` 与实际issued24D计算有界MSE，支持±1标签，不反解atanh。PointNet和action MLP的12个参数tensor共同更新，使用独立BC Adam及采样RNG。

BC产物版本独立，不能直接当PPO resume。显式派生函数输出全新v3 warm-start；只替换actor权重，逐值保留源critic、log_std、PPO Adam、LR、PPO计数、保存的RNG和旧迁移来源，追加监督历史。源文件前后SHA和exclusive输出防止覆盖历史。后续PPO的保存/恢复继续保留该来源。

为保留exact v3合同，旧`teacher_distillation='disabled_without_synchronized_visual_labels'`字段未改；它描述PPO路径没有附加蒸馏loss，不能据此推断独立离线BC没有使用Teacher。实际监督初始化以`visual_supervision_derivation/history`为准。

PPO采样执行路径未添加Teacher/MPC postprocessor。Teacher实际动作、Student log-probability不会混配；BC不改rollout里的actions/latent/logp/advantages。3844D/371D/24D不变，也未将未知实机接触传感反馈追加到Student。

## CPU 验证

TDD先确认缺少实现/缺少raw tap/监督来源保存失败，再完成实现。专项覆盖：真实source合同与来源分组、输入尺寸/有限性/边界、同tick pre-action和previous action、首回合recovery隔离、raw chunk/时间/owned buffer、原MPC多步和实际body/Jacobian快照、坏环境隔离、actor梯度和单tanh、BC Adam/RNG连续性、critic/噪声/PPO状态逐值保留、失败示范/heldout泄漏拒绝、来源变更和并发输出保护。

已执行相关199项专项通过（11.58s）；追加BC与采集边界后独立66项通过（10.52s）。最终复审再补齐首回合晚到成功的完整terminal长度与preflight trace偏移保护，相关82项通过（10.85s）。真实Teacher源在CPU加载、与第1439轮Critic合同相容，四份回放与AM/AK实际源动作同长前缀逐值一致。这些CPU fixture均不计作原生实验或抓取成功。

最终完整CPU回归：**5356 passed、96 failed、28 errors、49 skipped、91 subtests passed，251.04s，exit1**。与上一阶段完整JUnit逐节点比较，原5416节点无缺失/状态变化；新增113项全部通过。原96失败/28依赖或资产错误不是本次通过项，不能称全仓库全绿。BC v2梯度/失活/初始化/RNG专项32项通过（9.37s），独立规格与质量复审通过。

完整命令（隔离目录，`USD`为现有`omni.usd.libs`，CPU无Kit实例）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$USD:src:.:vendor/DeformX" \
LD_LIBRARY_PATH="$USD/bin" PYTEST_DEBUG_TEMPROOT=/tmp \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider tests \
  --continue-on-collection-errors -q --tb=short \
  --junitxml=/tmp/dh116-visual-teacher-mpc-v2-publication.xml
```

JUnit SHA256 `370b8724731df453340e2ef9d77cd08fc94c56ea3d3c4b80e4350713112c9698`；原基线SHA256 `d57d9e2a3c8407e2638afbfad28a6e3f57bde219986953fdac079836e04de0c5`。机器可读节点比较随本地验证原件保存，不上传运行日志/模型。

## 本次有限原生实验

独立根目录：`/mnt/mydisk/xk/dh116/runs/teacher_mpc_supervision_20261106T1145Z`，冻结源码 `code_snapshot`，GPU3、单实例flock串行运行：先 `teacher_reference`，后 `teacher_reference_mpc`。每次8环境（二指/三指各4）、固定224×96真实RGB-D、种子42、60Hz/480Hz、相同物理/reset/相机/动作合同。最多1272控制步，不构造PPO或执行PPO更新。

第1439轮显式独立8环境v3基准为 `bc_base_v3_1439_balanced8.pt`，SHA256 `457eb9cd12f0f480aa5a7cfbe83eac4dc6bc706bb2c8a330c0fa4bb9a0dbc977`；与本次原生采集完整合同比较无差异。只有当本次生成成功监督池时才允许实际BC。该基准仍是第1439轮真实模型的派生，不能冒充历史269/482峰值策略。

### Teacher参考基线：本次实际结果

| 目标抓法 | 首回合原生成功 | 物体最高净抬升 | 实际结果 |
|---|---:|---:|---|
| 二指 | 4/4 | 31.619759 mm | 各自1174/1175/1175/1174控制步成功，满足原生完整标准 |
| 三指 | 0/4 | 0.139942 mm | 各自419/418/420/419步触发`excess_force`并recovery，排除成功池 |

该实验实际1175个RGB-D帧、74个chunk、6374条首回合执行样本，成功池4698条，SHA256 `bb2a8150ef5ba380ced9cedc90e773bc1f976e3122443e61bc640f76aec86beb`。这些是本次Teacher回放成功，Student尚未由此取得抓取成功。

独立审计校验全部raw chunk SHA、frame/physics time与动作同tick绑定，58条样本CPU重投影最大误差`2.83122e-7 m`。四个成功回合末300控制周期每手thumb/index持续承重，其他指腹负载最大为0，真实抓法为pinch；每回合样本均完整覆盖实际terminal，后加的严格完整性validator重新通过。实际冷reset的q/qd/command/roots/物体位置/四元数/参考底部/seed，与后续MPC实验逐值相同，最大差0。审计原件：本机`artifacts/dh116_lift_repair_20261106/teacher_reference_native_evidence_audit.json`。

另用未修改的`SurfaceLiftSuccess`独立CPU重放本次9400×8原生物理子步，四个pinch成功脉冲精确出现在各自第9392/9400/9400/9392子步，与原生terminal一致；三指无成功。最大合格保持5秒，合格窗口漂移分别为0.003543/0.003743/0.003576/0.002934mm。该重放是核验已发生的原生数据，不增加原生试验数量。

此成功池只有一个二指原始轨迹来源；三指目前没有本次成功视觉示范。不能把同来源四次回放当独立泛化验证，也不能据此声称Student已掌握三指。

### MPC 前缀诊断

独立审计固定tick0..949前缀：首次改变动作分别为335/336/337/335；各自首次改变前，baseline的480Hz指腹负载与MPC pre-action状态逐值相同（最大差0N），Teacher参考动作与腕部12维也逐值相同。随后二指四个环境分别于427、444/448、524/535、428步丢失至少一手的有效对向接触。

每次同状态求解的batch-local nominal/optimized累计cost为2.044883/1.721899，下降15.795%；198个active控制时刻中196次下降，0次增加。它不是跨时间cost单调下降，也没有每环境cost分配。真实接触同时恶化，表明较低代理cost不能作为抓握改善证据。关节预测最大误差0.001342rad、fallback0，未达到旧0.005rad门禁；现有joint-only预测校验没有发现接触保持失败。不能据此断定某个单独权重是唯一原因。

前缀原件为本机`artifacts/dh116_lift_repair_20261106/teacher_mpc_prefix949_diagnostic.json`，绑定完整行前缀SHA和原生480Hz对照。前缀SHA为`ab9ce06dc4d7ae42cd8e48b1b79d654d8242994e72153da94154ce92cf814630`。

### MPC 完整有限实验：未通过抓握验证

| 目标抓法 | 原生成功 | 480Hz物理最高净抬升 | 保持 | 实际结果 |
|---|---:|---:|---:|---|
| 二指 | 0/4 | 12.489469 mm | 0秒 | env0/1在822/763步`object_dropped`；env2/3到1272步预算结束仍未terminal |
| 三指 | 0/4 | 0.139942 mm | 0秒 | 与基线相同的`excess_force`终止；MPC未激活 |

实际1272个RGB-D帧、80个chunk、5611条首回合执行样本；MPC激活495个环境控制步、改变489次，均为二指。**没有成功MPC示范池，失败MPC标签未进入Student BC。该原型损害了本次二指抓握，不能宣称MPC改善抓取或已经解决Lift退化。**

全量帧时钟、参考/实际issued动作、chunk SHA与物理对齐通过独立审计；50条RGB-D独立CPU点云重建误差最大`2.68221e-7 m`。本次原生物理子步重新输入原始成功观察器，所有成功/无成功事件与原生记录一致。完整审计原件SHA256为`78363177ce92690d9d0c3e26f129c54494ec37667dd42d8fc4841f76f4db8478`，保存在本机`artifacts/dh116_lift_repair_20261106/teacher_mpc_native_evidence_audit.json`。

两个完整实验使用同一不可变源码。最终版本另补实际native类门禁、缺尾终止校验、preflight trace偏移处理；原生初始trace为空，基线成功池亦通过最新校验。冻结版失败/未审计episode默认`native_contact_style_audit='fixture_not_native'`是旧说明字符串错误，不代表这是真假夹具；两实验实际genuine构造、原始物理、RGB-D及成功重放均已独立验证，最终代码将该native缺测标记纠正为`not_audited_failed_or_incomplete_episode`。

最终代码另在独立8环境20步原生MPC冒烟中通过实际类门禁，`native_runtime_verified=True`、RGB-D/动作/原子轨迹正常、exit0、PPO0。这个短冒烟没有完整episode或成功资格。

## 实际离线学习：发现并保留视觉失活反例

首个独立BC100只使用已验证的二指Teacher成功池4698条环境帧样本；没有使用失败MPC。源1439/8环境v3保护状态完整保留，独立BC Adam执行100步，MSE从0.0424834534降至0.0018478636。

但独立全数据核验发现：源1439的PointNet最后Linear输出最大值为-0.01493656，最后ReLU对全部有效点全零。BC100的6个PointNet梯度、Adam moments和权重变化均为0，真正改变的仅6个action MLP tensor；清空全部点云坐标时动作差精确0。因此这个v1产物只证明本体状态/动作历史拟合，**不能声称视觉编码器学会了Teacher**。旧产物与失败证据保留，没有覆写。审计原件为`actual_teacher_bc100_state_audit.json`，SHA256 `ddfb4f10a1b4c043fd675ad0b6ca072dcf02a329e584155a63f71709fd034970`。

该v1模型关闭Teacher/MPC后进行了8环境180控制步原生推理观察，exit0、无运行错误、PPO0，0个完整episode、8个未完成。不是100次/抓法验证，也不是视觉学习或抓取成功证据。

### BC v2：显式恢复视觉分支并核验

新增独立版本`dh116-vision-supervised-bc-v2-visual-gradient-verified`。默认`source`模式检测到来源PointNet对全数据失活时，在训练和写文件前拒绝；不再把仅本体拟合当合格视觉BC。新`--actor-initialization reset_pointnet`只在新副本中用私有CPU Generator、seed42重新初始化6个PointNet张量；Action MLP初态与源模型逐值相同，3844D/371D/24D/Tanh Gaussian架构不变。v2派生拒绝旧v1或缺少视觉证据的产物，原始文件全保留。

本次独立reset-PointNet BC100实际使用同一4698条成功二指环境帧。MSE为**0.0432773239 → 0.0019744662**；6个PointNet tensor在全部100步都有非零梯度，最大合并梯度norm0.08460835，独立BC Adam moments均非零且step100。全数据清空点云坐标、保留mask/本体/previous-action/timing时，归一化动作最大差为**0.13921228**，4698行均有变化。此干预只证明数值视觉分支参与，不是视觉语义、泛化或抓握成功证据。

独立22项实际产物审计全部通过：相对1439/8环境v3来源，仅全部12个Actor tensor改变；Critic、log_std、PPO Adam、LR、计数、RNG、完整合同和旧来源逐值保持。仅6个编码器张量被显式初始化，MLP初态未改；最终私有Generator版seed42生成的全部12个初始化SHA与实际CPU产物逐一相等。实际CPU快照在RNG隔离修正前、`CUDA_VISIBLE_DEVICES=''`，修正仅避免在可见CUDA进程中重设全局CUDA RNG，数值结果相同。

原件`actual_teacher_pointnet_reset_bc100_v2_state_audit.json` SHA256为`9ab9a6bf0236d68bc8e81811bf499afffef85535b1fa49f76ac98784d0d43ced`。新独立warm-start为`visual_actor_pointnet_reset_bc100_v2_warm_start_v3.pt`，SHA256 `949a014921ffea83ea3fe57fb53550042a748f8b9decffe72eff69c11c6602d3`。这是显式权重派生，PPO iteration仍1439，BC100不是100次PPO更新，不能称精确续训或best模型。

新模型的实际8环境原生初始Actor输入还做了只读CPU坐标干预：8/8行敏感，最大动作差0.04392020；输入SHA256 `9215ed33c272b036e1411bd0de80310b5f2165a622ddb7f7bc48ed3fb3f92815`。没有额外增加原生试验次数，也未把Critic/接触数据注入Actor。

新v2模型关闭Teacher/MPC后完成同一8例manifest、180控制步的独立原生推理观察，exit0、0无效测量、无运行错误、PPO0，0完整/8未完成episode。它只覆盖3秒物理时间，不能替代5秒成功保持条件或100次/抓法验证。全部原始数据、两个BC版本和新warm-start均保留在独立目录；历史129/1439/1750模型SHA核对不变，模型与原始运行数据不上传Git。

## 未解决项与下一步

- 当前MPC只有局部pad/关节模型，未通过原生抓握验证；失败标签隔离。须先验证接触保持/载荷约束与Teacher轨迹约束，逐因素做短原生对照，不能直接开启长PPO或按代理cost选模。
- 当前成功视觉监督只有一个二指原始来源；三指成功视觉示范和独立每抓法验证集缺失。fit-only不等于泛化，不满足每抓法100次原生能力验证。
- 新Student视觉分支已参与数值学习；仍需完整原生episode及固定30mm/5秒/0.5mm评估。短推理观察不证明可靠抓取；没有本次Student成功结论。
- Grasp/Hold/Lift课程、R0～R5及保留/重置PPO Adam对照、泛化仍未启动。无长PPO，PR继续Draft、不转Ready、不合并。

当前不得宣称Student已学会抓取、MPC改善已证实或Lift退化已解决。无独立每抓法多来源验证集时，BC只能明确报告fit-only，不能作为选模/泛化证据。
