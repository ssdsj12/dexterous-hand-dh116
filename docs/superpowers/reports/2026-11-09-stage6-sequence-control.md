# Stage 6：连续视觉监督与闭环控制实施记录

基准`26f1bbcc5a0e596da55c5d89896befcb5622cda7`，分支`codex/dh116-lift-repair-20261106`。本阶段按用户已指定方案直接修改代码和训练流程。3844D Actor、371D Critic、24D具名动作、原M0/Teacher/物理与30mm/5s/0.5mm标准保持；条件命令展开不冒充物理世界模型。

## 提交1：连续数据构建器已经接入

新增`tabletop_vision_sequence_dataset.py`，并在原监督数据模块末尾增加`validate_sequence_context`和`join_sequence_context`。原v1数据及`validate_dataset`函数保持原样。新的sidecar按`run_uuid/env_id/episode_uid/tick`联结实际动作、真实previous_action、传感/控制时间和8个物理子步；缺测窗口排除，重复身份/格式/SHA错误拒绝。

16步连续窗口不跨episode/reset/terminal/来源/缺口。长episode的完整滚动窗口覆盖末尾，只有原本短episode允许有效mask padding；从长episode缺测切出的短片段不能伪装短完整示范。支持来源均衡及40%全轨迹/60%事件混合，缺类别重新归一化，每个合法窗口保持正概率。CONTACT/LIFT_ONSET/HOLD取原生事件元数据，RECOVERY需要独立资格，均不进入Actor输入。单来源明确fit_only。

实际CPU命令（服务器新可写副本`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/source`）：

```bash
CUDA_VISIBLE_DEVICES='' DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest tests/test_vision_sequence_dataset.py tests/test_vision_supervision_dataset.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-data-commit1-20261109
```

结果：76 passed，5.10 s，exit0。合成CPU数据仅验证schema/窗口/采样，不计原生成功；本阶段尚无BC模型或来源纯Student成功可报告。

## 独立原生采集与固定输入

新实验根为`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01`。输入manifest从已冻结Stage5 launch的真实参数解析模型路径，并验证源c58模型SHA，不猜权重文件名。原第1024轮、c58、4698样本池、全部Stage5冻结目录继续只读。

训练前已冻结新200例manifest，seed116602、SHA `298f379c599b34c5e58c3fdecb3957912503d42b81ce62a3f3c6b6f03bae9bed`。原200例保留回归用途，新集仍是同reset族内小XY/yaw扰动，尚为authored_unverified，原生次数0，不声称来源族/材料/柔性线缆泛化。

原Teacher/MPC正在新目录采集真实pre-control上下文。每次作业均独立冻结源码、记录SHA并持有原单实例锁；任何未完成、未通过完整目标一致性或失败的数据不放行B2。PointNet激活、可微shadow、B0/B1/B2、分支与双侧恢复的后续实现和实际结果将在对应提交补入，不以本数据提交提前授予资格。

## 条件展开实现与第一次实际采集

新增共享`tabletop_command_kernels.py`及`tabletop_vision_command_shadow.py`。原native free-wrist control与TensorWorld使用相同纯函数；指令积分、lead/限位/slew、掌心norm/clearance/safe_z、live/allowed和seed intrinsic XYZ顺序保持。shadow拒绝不支持的callback及不同源码SHA；Teacher与Student只使用同一已录制反馈，各自维护累计命令。

完整finger+palm+rotation反传测试发现原slice-inplace会破坏autograd版本计数，改为同算术的out-of-place拼接。独立审计直接加载`26f1bbc`完整旧control/advance，并将其相对import也指向旧模块，160组合×16子步、97,280次张量比较全部bitwise一致，包括SDK float32编码及诊断NaN bits。这是CPU前向等价，不能替代原生抓取。

实际Teacher采集01已自然结束（exit0），输出`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/teacher_sequence01`：

| 证据 | 实际结果 / SHA256 |
|---|---|
| 二指 / 三指首回合 | 4/4成功 / 0/4成功；三指2掉落、2超时 |
| 成功视觉样本 | 4698，四个完整二指首回合，同一参考来源族，fit_only |
| report | `fc1ac351461a093cfd79378f7866e2a62c1c175d77355b5a8794cce52515c639` |
| successful_visual_supervision.pt | `28a2da63642393a4a91c91041c7c5f0ca52d9f8d279a1a5ac395c8ce85d162f8` |
| successful_sequence_context.pt | `50ef46a896e96d147b9efedbf58295b3a952282f0f66fe132984a9a41bcaa011` |
| 物理目标parity | 37584个live子步，float64及SDK cast最大误差均0；真实backend/cache分别available、missing0、误差0 |
| 整回合累计展开 | 每episode从真实起始state展开1174/1175个policy步，finger/palm/wrist_angle最大误差均0；4638个合法L16窗口 |
| 来源冻结完整性 | 1413文件SHA复核全部一致，无丢失/改写 |

独立CPU脚本与报告保留在服务器`command_validation/forward_baseline/`，报告SHA`c0a679dd5bfc3aa81668a742414f5d13cd5931ad08dd43a493f86290478c5ae9`。整回合原件为`teacher_sequence01_whole_episode_shadow_validation.json`。这些Teacher/MPC成功不计Student成功，刚性圆柱代理也不计柔性线缆或硬件验证。

采集01冻结于autograd修复前，source controller SHA为`c8a96ed0cd0fee0a41c8d75f0c351aa0b6165e7dac6831cd82fb122a1e375780`。当前新版直接准备此池实际触发`controller source SHA256 mismatch`，失败日志`prepared_sequence01.log`与已解析输入manifest保留，未覆盖01或放宽验证。新版采集02已独立冻结启动；完整目标与当前源码SHA一致之前不放行B2。新增post-physics记录明确与pre-control输入分开，只有实际advance返回的q/qd及SDK getter同钟时才可标定跟踪阈值。

## 当前CPU回归与工程入口

独立冻结`cpu_integrated01/source`的实际聚焦回归为**533 passed，52.67s，exit0**，涉及25个测试文件：sequence数据/损失/训练器、shadow、workflow、branching、跟踪标定、PointNet合同，以及旧BC/hemisphere/Tanh/控制/限位与surface兼容。完整argv、逐文件SHA、pytest原始输出与exit单独保留在服务器`cpu_integrated01/`，这些合成/CPU测试不计原生成功。后续Teacher-replay mode互斥变更另跑`tests/test_stage6_workflow.py`，11 passed。

文件入口`python scripts/run_dh116_stage6.py`提供`prepare-data`、`train-bc`、`diagnose-branch`、`evaluate-source`、`evaluate-heldout`。prepare需要完整资产/相机/模型/预处理/数据SHA，或从固定发现manifest及实际新采集report解析；不猜权重文件。训练再次读取绑定manifest与原文件，不接收替换数据；主实验显式要求ReLU/M0。source/branch请求模式必须与配置一致，Teacher prefix预测误差不当作实际Student动作分叉。

提交2的独立控制范围另跑以下命令，**96 passed，9.45s，exit0**（不是把不同检查次数相加）：

```bash
CUDA_VISIBLE_DEVICES='' DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest tests/test_vision_command_shadow.py tests/test_vision_teacher_collection.py \
tests/test_free_wrist_grasp.py tests/test_dh116_tensor_world.py tests/test_surface_baseline_compatibility.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6_command_commit2_cpu01
```

Git对4个controller来源文件做字节保留：新增模块固定LF，两个提取后的legacy模块保留实际混合换行。实际Git clean/filter→blob往返逐字节相等，没有改动正在运行的冻结文件。历史checkpoint/原件均保持；实际B0/B1/B2、来源闭环与首次物理分叉结果仍待后续填入，不以CPU训练fixture替代。

## 提交3：当前源码原生池与实际三组BC

以上01是保留的历史实验；本节记录新02与真实模型，不回填或复用01的资格。`teacher_sequence02`实际自然退出0，2623.21秒。二指env0/3在tick1173、env1/2在1174达到原生首成功，4/4；三指env6/7在1091/1168掉落，env4/5在1800原超时，0/4。没有恢复标签或三指成功数据进入BC。

| 原件（均位于本实验根） | SHA256 / 实际检查 |
|---|---|
| `teacher_sequence02/teacher_collection_report.json` | `0388b03469350d76f3921279c544a5deb650a5f10b02b2d09e7f7308fc35ec82` |
| `teacher_sequence02/successful_visual_supervision.pt` | `76a7d05ca6c84cedc45f9ed653dd7006d09f24a09deb3bb1710aa185442811bc` |
| `teacher_sequence02/successful_sequence_context.pt` | `7cc6ea548d9c4d37c99360b9a7ce1cf5e4e16f3b2a6670f1cd74ea3e5277850f` |
| 来源冻结manifest | `590e21419617619c4bbfd706b082d9c44fdf7f9cb51e14f64a94237487312103`；2837文件复核无变化 |
| controller源码合同 | `dbb9a303993783b65f79dc169ed4cd592c1b16e3e629b9f2f90e573982ecdebb` |
| 原生目标 / 整回合累计shadow | 4698成功记录、37584 live子步，float64、SDK cast、实际PhysX backend、cache各自最大误差0，missing0；每回合累计展开也为0 |
| `prepared_sequence02/prepared.json` | 4638个完整16步窗口，593664次重叠窗口物理比较全部误差0；fit_only，只有一个参考来源族 |

`prepare-data`实际输入为固定发现manifest、新02 report与训练前冻结的final200 manifest；解析到完整`prepared_sequence02_inputs.json`后再次硬核模型、资产、相机、观测预处理、数据及上下文SHA。旧发现manifest中的Stage5 launch只保留模型发现出处，不拿它的200环境配置当8环境来源场景；原生来源明确绑定`teacher_sequence02_launch.json`。

实际批训练命令使用服务器环境Python，经`bash scripts/python.sh`运行有限CPU驱动`run_bc_batch01.py --source .../source --prepared .../prepared_sequence02/prepared.json --output .../bc_sequence02`，`CUDA_VISIBLE_DEVICES=''`、`DH116_NUM_THREADS=2`。驱动先冻结`bc_sequence02/frozen_source`，三组各2 CPU线程，固定seed116601、学习率1e-4、新BC Adam、梯度裁剪0.5、4×16批、3000更新；不是PPO或Isaac实例。共享条件shadow的实测CPU单次前反传0.351秒、GPU2为0.717秒，因此三组使用同一CPU后端。

| 预设候选 | 最终实际文件（`bc_sequence02/`下） | SHA256 | 有效曝光 / padding |
|---|---|---|---|
| B0，均匀动作监督 | `B0/B0_update_03000_completed.pt` | `02f3d5f983e0315d8a0858699ad8cca2e3d088375826cd31f126179305c8855d` | 192000 / 0 |
| B1，事件动作监督 | `B1/B1_update_03000_completed.pt` | `26bf92dc127bba9a809361397b5d6b6346d7aa245fec26077c05358597c894fd` | 192000 / 0 |
| B2，事件动作及条件命令监督 | `B2/B2_update_03000_completed.pt` | `510c31239e7a08e5c58d6408eed77af71a1070690e2f72de2b5df8e5044ba8b8` | 192000 / 0 |

三组均实际完成3000、每250保存Actor、独立BC Adam、采样RNG、最后训练输入和固定PointNet探针；B2命令权重在250轮前为0，500轮达到0.1。源c58 Actor继承，不重置，主候选仍为ReLU/hemisphere；仅派生Actor参数改变，Critic、探索log_std、原PPO Adam、LR、计数及原contract等逐项保持。原c58和历史模型保持只读。新激活合同仅支持显式`leaky_relu_001`独立派生，不给旧模型静默换激活。

训练返回的`loss_before/loss_after`明确仅指固定首16步窗口动作MSE，不能当全部4698行的总体MSE。所有候选按预设最终3000轮送评；未用fit_only训练loss或历史峰值选最佳，也没有把本次BC3000写成PPO3000。

最终START/CONTACT/LIFT_ONSET/HOLD四类固定合法视觉探针全部非dead；点云置零和0.1mm有效点扰动有响应，点排列干预保持不变。探针同时保存三层激活及各PointNet参数梯度。连续失活则保存模型/输入并只停该候选，真实三组没有触发该停止条件。视觉敏感性不授予抓取资格。

本次提交范围实际CPU回归命令与结果：

```bash
CUDA_VISIBLE_DEVICES='' DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest tests/test_vision_sequence_losses.py tests/test_vision_sequence_training.py \
tests/test_vision_point_activation.py tests/test_vision_supervised_bc.py \
tests/test_vision_squashed_policy.py tests/test_vision_quaternion.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-bc-commit3-20261109
```

**101 passed，16.86秒，exit0**。这不是与先前533项相加的独立总数。旧`fit_actor`与旧ReLU/Tanh/quaternion路径继续通过；合成masked损失/梯度测试和实际训练分别报告。

原生来源和Teacher前缀短分支正在另一个冻结目录、同一Isaac实例锁下逐项执行。当前模型是fit_only候选，尚未通过来源纯Student闭环或独立测试，B3和短PPO门禁未开放。新最终200集仍是训练前冻结且尚未使用的authored_unverified清单，不能把它写成200次已验证。

## 提交4：原生来源/分支入口与真实Teacher重放基线

已新增`tabletop_vision_branching_eval.py`、独立只读比较缓存和五命令入口。每个配置绑定实际模型/数据/context/report/真实初态路径与SHA。Student原生launch会删除Teacher、MPC、旧评估和M2参数，强制M0/Tanh；真实场景来自8环境seed42的02 launch，GPU3环境和Kit activeGpu3分别校验。`evaluate-source`从reset起只有Student控制；`diagnose-branch`每次真实冷重置，再重放Teacher prefix，Student接管1/4/16/60步。三种事件和四个来源分别报告，前缀先终止的case不能算完成Student分支。

原模型发现launch与真正原生来源launch曾存在200环境/8环境配置差异；在首次启动前已通过测试及显式native-template修复，没有拿模型发现参数冒充来源重现。相机先构建、再真实冷重置，并逐字段核对q/qd/command/roots/物体位置/四元数/参考底高，以及相机warmup前后状态和SDK clock；不把pose写回当PhysX solver state恢复。

60Hz记录实际issued及真实previous action，480Hz记录首terminal前实际物理payload和另存的后端/q/目标流。execution UUID、env和episode UID与Teacher来源身份分开。首terminal后退休环境，不覆盖其最终有效样本。Teacher前缀期间的未执行Student预测，只计同输入误差，不能作为实际动作第一次分叉。首次数值差异、绝对任务物理事件和对重复Teacher范围的分叉也分别保留，后者需要实际可用重复证据，不能由数值差异直接推断因果。

只读cache仅复用同一saved Teacher observation上的前向/条件shadow结果；它绑定模型、dataset、context、原stream/report及13个实际实现模块SHA，CRC与JSON行完整性校验。每个原生job仍独立冷重置、执行完整前缀、采集480Hz物理和首terminal，`native_trials_reused=0`。代码/输入不符则拒绝，不靠缓存减少或伪造物理次数。

冻结有限驱动`evaluation_driver01.py`的SHA为`fad4a09fc57ff62d6fee768dce57dc9c11907f597c57925c67be775455cec5ef`，原生代码为`evaluation_source01`，冻结2843文件。预设总表为1个实际Teacher tape重放、4个E0/B0/B1/B2纯来源任务，以及两臂E0/B2各3事件×4时长的24个短分支。仅实际完成的job计入结果；这一预设表不等于29项已通过。

首项已实际自然退出0：

| Teacher tape原生重放，`native_e0_replay01/jobs/teacher_replay_E0` | 实际结果 |
|---|---|
| report SHA256 | `133ccd86a7f8377f5a37172a28d9fdaad5d63eeff2c8494b1f6b3892ecd4b5c0` |
| 首回合 / policy总步数 | 4/4原生成功，1175步；`pure_student_native_success_count=0` |
| 实际UUID | `e4668000-6287-4883-8a3f-118d1f6f7708` |
| env0 / 1 / 2 / 3净lift峰值 | 31.591481 / 31.537555 / 31.531521 / 31.627703 mm |
| 逐环境合格Hold | 原5秒；浮点记录4.99999999999999s，使用原判据，未改容差 |
| 有效480Hz样本 | 9392 / 9400 / 9400 / 9392，合计37584 |
| raw physics SHA256 | `6b93602053ed81f994fddbbf3a798f406f6cdb39c68ad4b4448c64e32d9ea0e4` |

该重放证明这四条Teacher tape在真实reset/相机路径下可以再现，仍是Teacher控制，不是Student或新独立初态成功。新纯来源/分支正在同一单实例锁下继续；没有中断其他训练或并行第二个Isaac。

完整回合边界另外对冻结实际`TensorTask.elapsed`做原f64 masked累加审计：14400次原1/480累加后为30.00000000001039s，首次原`>30s`在physics14400/policy1800触发，现有来源循环可以观察该原超时。审计SHA为`dac894046fba52b14e7fbe01b026085ea21f9c652e988d76c964e6b121954609`；没有因收尾疑问修改时限、物理dt或重启原生job。

完整saved-pool只读前向在4698个原始观测上得到总体动作MSE：E0 `0.00019526530494252625`，B0 `0.0000683884250602903`，B1 `0.000070870184078405`，B2 `0.00007078445551577331`。报告`offline_actor_pool_analysis/all4_analysis_v01/actual_all4/saved_actor_pool_report.json` SHA `aa54492a5c380beee4912e03928b22aac3391821228146b52ee42320e8b832a1`，包含24维、具名动作组及事件分层，输入/文件/RNG与全部非Actor模型状态前后不变。B1/B2的这点差异不构成B3或PPO晋级依据，也不能拿B0最低训练MSE替代原生选择。

CPU命令分别实际执行：`pytest tests/test_vision_branching_eval.py -q -p pytest_mock -p no:cacheprovider`为39 passed；冻结batch/cache/entry组合91 passed。新增Teacher扰动参数隔离测试先得到3 failed/1 passed，补充Student launch剥离后，`pytest tests/test_stage6_workflow.py tests/test_stage6_workflow_integration.py -q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-tier1-cli-green01`为**25 passed，13.29s**；均经同服务器`CUDA_VISIBLE_DEVICES='' DH116_PYTHON=... DH116_NUM_THREADS=2 bash scripts/python.sh`执行，不计原生验证。

## 提交5：真实扰动专家入口与来源隔离（原生扩充仍待执行）

新增`tabletop_vision_teacher_perturbation.py`。显式命令`python -m go2_pvcnn.control.dh116.tabletop_vision_teacher_perturbation --checkpoint <已核验c58实际文件> --primary-initialization <原primary文件> --secondary-initialization <原mixed文件> --teacher-source <原canonical Teacher文件> --output <新manifest.json> --seed 116603 --count-per-style 4`读取实际输入及SHA，生成独立8 mixed初态；XY各±0.5mm、world-Z yaw±1°，左乘实际物体四元数。两档不混用：Tier2在Tier1可重复成功之前明确拒绝。parent source group由原参考来源确定，run/seed/env不制造独立来源族。

原生Teacher执行必须显式增加`--vision-teacher-perturbation-manifest <该manifest>`，保留原constrained MPC、224×96相机及sidecar。`TeacherPerturbationSession`在原最终reset之前安装对象pose缓冲、reset后完整读回，并在第一份真实相机包进入SensorHistory前再次核对全部物理状态、clock、有效率和原初始几何。未知/不一致输入拒绝，失败证据独立保存；不以合成fixture授予native资格。

一个实际开发RED发现：直接复用原evaluation writer会将Task原double grasp command seed替换为float32编码关节q，虽只有约1e-8差，也不符合“仅扰动物体”。因此共享core的Teacher分支现在只写`seed_wire`的XY与四元数，原q/qd/root/velocity、Z及Task reset seed逐位保留。原evaluation公开wrapper仍运行原100例/抓法验证和原默认完整buffer写入顺序，未放宽reset或安全判据。

每次仍由原Teacher/MPC真实闭环执行到原首terminal；原生成功、未recovered且实际接触抓法一致的二指才可进入本轮新增池。三指即使出现观察成功也单列，不能混入这份二指扩充池。失败/未完成的数据另写`failed_or_incomplete_supervision.pt`及manifest；异常中止也独立保存partial audit，不能当成功标签。实际source group、run UUID、env、episode UID、reset authored/readback SHA及原输入/producers SHA都有出处，不改观测、动作、previous history或物理标签。

目前只完成真实输入的CPU manifest和冻结/启动草案验证，**扰动原生次数仍为0、尚无新增专家或重新拟合结果**。`tier1_entry_validation02/entry_validation.json`明确`isaac_imported=false,native_jobs_executed=0`。其实际manifest SHA为`f15dc7fce9b10a57d59bd236c50a5dd5d6b94983a494483b077b7f348886772c`，完整critic/physics/camera/两初始化SHA与真实Teacher02合同一致。2845文件的冻结manifest SHA为`a0be0c865a43356fa593c7c31eb27ae9798fef060c2996c9032555ef3fb5620a`；启动草案SHA为`b13e49037e50c89e1c4187d850a96e24c2301b1d14f52de931dca8e8b68dc00e`。该草案早于最后lineage加固，后续正式运行重新冻结，原草案不改写。

显式来源lineage新增拒绝：row与episode都有`source_group_id`时不得冲突；同一物理episode不能前后绑定不同显式group，即使只后半段出现显式字段，也不能将前半段放train、后半段放heldout。纯legacy轨迹ID变化继续作为原窗口边界，原first-terminal后记录不制造新来源组。TDD先复现两个冲突被接受，再复现“后半段才声明group”的不对称漏洞，最终修复并通过旧窗口兼容。

独立双namespace加载当前与实际BC冻结版本，在真实Teacher02上比较4638个窗口、segments、全部event labels、sampling probabilities、完整report、原prepared index以及1000次同seed采样，全部exact；data/context/index SHA前后不变，仍1来源fit_only。只读报告`lineage_actual_teacher02_readonly_comparison01.json` SHA `6306cccc486508a587b8f5400182e6df232292cc082d7326bd80614eb3e224ef`。因此新校验没有使既有真实BC输入变化，也没有回写其冻结源码或模型。

有限native驱动现以`scripts/run_dh116_stage6_evaluations.py`进入仓库，与实际已执行`evaluation_driver01.py`字节SHA `fad4a09fc57ff62d6fee768dce57dc9c11907f597c57925c67be775455cec5ef`一致，配套`tests/test_stage6_evaluation_batch.py`。默认prepare/list不启动Isaac，显式execute才使用Linux pgrep/flock及当前GPU3 template白名单；output必须在source外，源码/driver变化会使旧plan拒绝。每个模型仍从实际completed报告选择唯一3000轮快照，不猜文件名或用loss选模。

本次实际CPU命令均使用同服务器环境前缀：

```bash
# 数据、训练、旧schema与可发布的有限驱动；125 passed，21.01s，exit0
pytest tests/test_vision_sequence_dataset.py tests/test_vision_sequence_training.py \
tests/test_vision_supervision_dataset.py tests/test_stage6_workflow_integration.py \
tests/test_stage6_evaluation_batch.py -q -p pytest_mock -p no:cacheprovider \
--basetemp=/tmp/stage6-lineage-batch-green04

# 扰动/原Teacher/reset、旧command与评估、回归；200 passed，22.74s，exit0
pytest tests/test_vision_teacher_perturbation.py tests/test_vision_teacher_collection.py \
tests/test_vision_evaluation_reset.py tests/test_vision_teacher_hold_tail.py \
tests/test_vision_command_shadow.py tests/test_vision_evaluation_trials.py \
tests/test_stage6_workflow.py tests/test_stage6_workflow_integration.py \
tests/test_stage5_teacher_regression.py tests/test_vision_supervision_dataset.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-tier1-commit5-cpu02
```

两组测试有重叠，均为CPU或纯USD fixture，不是325个原生验证。一次命令使用不存在的旧缩写测试名而未collect，随即用实际文件名纠正；没有把该退出算通过。原生四组来源任务、首次物理分叉、真实Tier1新专家及重训仍按实际结果分别收口，未启动新PPO。

## 提交6：独立共同暂停与完整回合证据入口

新增`tabletop_bimanual_tracking_pause_runtime.py`和`tabletop_full_episode_tracking_experiment.py`，扩展guard但保持旧默认。显式`--vision-tracking-experiment-config`与PPO/Teacher/其它评估互斥，有限运行original、unilateral和common；source固定c58、原8环境seed42和30秒物理时限，不能当旧M2 env799/932/661的精确复现。候选不默认装入Student，也不改变3844D/371D/24D、物理或成功判据。

共同暂停状态为NORMAL→PAUSE_REQUESTED→PAUSED→RESUME_PENDING→NORMAL/TERMINATED。实际跟踪误差取PhysX `root_physx_view.get_dof_position_targets()`减真实q，命令cache单独记录且不回退冒充backend。每关节触发阈值为`max(0.01rad,1.5×Teacher P99.9)`，另保留20ms外推预警；连续至少0.05秒误差及速度安全后才能恢复。原控制之后、原advance/transport之前在实际offset/seed intrinsic-XYZ图中取消排队双腕目标，再经fresh-action/rebase恢复，不把旧排队目标跳回。原强制卸载优先；普通保持无法表达时，advance前明确拒绝，保留pending证据，不伪造原task terminal。

Teacher/MPC强制卸载priority接口尚未完成共同暂停晋级。当前Actor hybrid只允许经源码审计的`surface_policy_owned=true,redistribution=false,pinch_force_hold=false`且无其它callback的原profile，不能把负policy动作猜成mandatory release。未知callback/profile拒绝；原Teacher02的4/4成功不是共同暂停Teacher/MPC回归。完整原生保护实验尚未运行，不能宣称消除了历史越限或新增抓取能力。

实测Actor F32输出按原Env begin_step规则转F64进入控制器/guard，未加gain/clamp/tanh。外层分别保存原始Actor、env请求、controller cast、投影/最终目标及实际位姿。独立execution UUID与真实diagnostic episode身份分开，首terminal立即冻结，后续reset/no-op不覆盖。旧日志`actual_palm_pose_m`实际测的是受驱动`base_link`/SDK腕掌根原点；新完整回合入口同时记录实际body-link姿态和具名palm_link，模型中没有该具名link时为NA，不用CAD中心或base冒充掌心。

实际02v02跟踪标定原件为`teacher_sequence02_tracking_calibration_v02.json`，SHA `bbbca0656b94bfac0eeeddd1f7152fc2e41a092cb0381a9a8231a915c7933adc`。4个成功Teacher首回合共37584个live子步/手，合计12个主动关节各有37584个有效样本。原advance返回的post-physics q/qd在TensorEnv reset前，与实际backend getter按独立SDK index/time同钟关联；不是pre-control q减新目标，也不用cache。

最初按metadata_time加名义1/480拟合同钟时，SDK报告时间存在每步约1.09e-10秒的累计表示差，原失败标定已保留为不合格。修复改为用全部9400个实际不同SDK step、重复step及policy anchors验证单一affine cadence/origin，仍使用8 ULP算术界而非放宽秒级误差：实际cadence `0.0020833334419876337`、origin0、最大残差0。SDK Python源码证明其报告时钟累加callback step_size；C++绑定的具体精度未独立证明，不声称改变了物理dt。原control dt始终是`1/480`，没有逐row单独调时钟来消除错位。该标定只是对真实原件的只读统计，没有安装控制器或启动Isaac。

实际12维阈值rad为`[0.01,0.0119347053,0.0190166838,0.01,0.01,0.01,0.01,0.0122126748,0.0140647740,0.01,0.01,0.01]`。输入仍是本报告02成功dataset/context的原SHA。原失败标定、installed SDK源码证据和qualified版本均保留，不覆盖原件。

完整回合实现额外一次policy tick只用于在不同浮点累计表示下观察原严格`>30s`超时，未延长task时限或给已结束首回合新动作权限。本阶段来源评估的实际F64审计已确认1800步足够，不因此修改或重启其冻结job。

本次实际CPU命令：

```bash
CUDA_VISIBLE_DEVICES='' DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest tests/test_stage6_tracking_calibration.py \
tests/test_stage6_common_tracking_pause.py tests/test_stage6_common_pause_runtime.py \
tests/test_full_episode_tracking_experiment.py tests/test_bimanual_tracking_guard.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-tracking-commit6-cpu01
```

**98 passed，13.16s，exit0**。此前双侧/跟踪12文件联合210 passed/53.30s的日志SHA `e8975694e3f35923e508c0371a0eaf645485ba984a39cf6ff3a4a5f6cf610c3b`另保留在`bilateral_validation/`；两者不相加，不计native成功。新三组完整物理结果、原Teacher/MPC共同暂停回归及移动腕资格仍分别待验证，默认M0和旧guard不变。

### 当前来源任务实际结果：E0完整基线

`native_sources01/jobs/source_E0`已自然退出0、status=completed，4个首terminal齐全，纯Student成功0/4，1349个policy步。report SHA `ce4f5369ef46fd0933dfde85219604ddcdf6fff36ca1230152f4b667cb83dc29`，raw physics SHA `c9d1e506111be96890a0f3046582e1559b0e7e028981f929874debfb69144f34`。执行UUID为`ee642205-8717-44d9-84ea-f53fc37e4252`，各episode UID为`<该UUID>:<env_id>:0`，与Teacher来源UID分开。

| env_id | 首terminal policy tick（从0起） | World physics step | 实际timestamp_s | 原生失败 |
|---|---:|---:|---:|---|
| 0 | 1269 | 10159 | 21.202084439108148 | object_dropped |
| 1 | 762 | 6099 | 12.743750664638355 | object_dropped |
| 2 | 1348 | 10792 | 22.52083450788632 | object_dropped |
| 3 | 983 | 7869 | 16.431250856956467 | excess_force |

四个合格Hold都为0。此处不把峰值高度拼接成同一轨迹，也不根据summary回算峰值时刻；逐环境峰值argmax和首分叉将由真实raw后处理独立保存。完整同环境issued动作和480Hz轨迹均保留。B0/B1/B2尚在同一预设串行计划继续，不把未完成候选写成失败或成功。
