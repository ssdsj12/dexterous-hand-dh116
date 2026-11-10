# 第五阶段视觉迁移：源模型审计与原生门禁

本报告把源模型诊断、离线BC、Teacher/MPC原生回归和视觉Student原生结果分开。原生成功仍要求净Lift至少30 mm、连续Hold至少5 s、固定锚滑移不超过0.5 mm及双侧有效对向承托。未完成的实验不计成功。

## 实际第1024轮源模型

源文件为服务器`/mnt/mydisk/xk/dh116/runs/dh116_gpu3_vision1024_ppo1024_20261107T131048Z/vision_student_ppo.pt`，SHA为`f37f25c29c8458f4c23e68ef1bd907c30f0607dcbd7a801e3f76c20eb82a1563`。实际合同为旧`gaussian_clip`，3844D四帧Actor、371D Critic、24D动作。新版Tanh实现保留，不能把这份旧模型改名冒充已经使用Tanh训练。

该run的原始`training_start.json`记录：`initial_iteration=0`、`initial_timesteps=0`、`initial_episodes=0`、`resume_source=null`、`optimizer_state_entries=0`、`privileged_actor_inputs=false`，`teacher_distillation="disabled_without_synchronized_visual_labels"`。因此这份模型没有从已成功的视觉Teacher标签蒸馏启动；不能把Teacher已有的成功直接视作它已经学会。

## PointNet激活与坐标干预

对同一只读模型执行CPU探针。只将点坐标清零，保持有效点mask、关节/腕部本体状态、previous action、四帧历史顺序和时间字段不变。使用实际网络权重和既有`audit_actor_visual_branch`，没有物理推进、PPO更新、BC更新或模型派生。

| 输入来源 | 3844D样本数 | 有效手坐标点数 | 非零有效坐标分量数 | 非零有效PointNet特征数 | 清零坐标后的动作最大变化 |
|---|---:|---:|---:|---:|---:|
| 本次Z0的200个原生冷初态冻结输入 | 200 | 27814 | 83442 | 0 | 0 |
| 历史已合格二指Teacher/MPC同步输入 | 4698 | 3176824 | 9530472 | 0 | 0 |

每条3844D样本包含四帧历史，4698不是独立初态数量。两组输入均有真实有效且非零的点坐标；在这些输入上，源PointNet输出为零、动作对坐标清零完全不敏感。该证据表明此模型在已测状态上没有利用点云，不等于相机没有采集数据，也不能推广为所有未测输入上的数学证明。

本次原生冻结输入文件SHA为`d947270e8151786e13c64336a101cffb7effca19222da202613b9b1ed2e19871`。历史成功输入文件SHA为`e9b5ed7057d8996a0c8ae6b0fbf30baa4c42cbaa40257ad8ab14371277f652dc`；这里只用它做固定输入诊断，不将历史数据算作本次新采集或Student成功。原始探针为阶段总目录下的`t7_encoder_frozen_inputs_preflight01.json`及`t7_encoder_readonly_preflight01.json`。

当前没有留存可用于精确定位这次视觉分支失活起点的原始第513轮模型，也没有用重新随机初始化的权重冒充实际初始模型。单个终点和这些固定输入不能证明右Z偏置形成的唯一因果链。

## 独立迁移合同与数据资格

后续迁移必须先通过T2真实映射与T5二指原路径/Adapter路径各4/4的门禁。原Checkpoint只读，所有派生模型使用新文件。

1. 显式从旧模型派生完整Tanh V3，保持确定性均值，按实际第1024轮和源SHA核验。
2. 如同步数据为8环境，采用显式平衡batch投影，并以实际Torch数据池中的完整visual contract精确断言；不因JSON的tuple/list序列化差异放宽物理合同。
3. BC仅使用本次重新合格的二指同步RGB-D和实际60Hz Teacher/MPC标签。PointNet若仍失活，必须显式选择独立的`reset_pointnet`初始化，不能静默跳过视觉梯度检查。网络尺寸、Critic、24D动作、执行器与物理标准保持。
4. 如采用既有quaternion半球规范化，必须写入独立新模型合同；它仅统一同一姿态的q/−q表示，不是M2左右镜像变换，也不修改旧模型推理。
5. V4绑定记录实际Adapter版本、代码/资产SHA及原生资格，推理前重新核对实际SDK关节顺序、源USD和合同。BC拟合、原生评估及后续PPO分别报告。

三指协调候选的480Hz执行与原60Hz源请求不同，其样本明确排除于BC池；即使候选出现物理成功，也不能直接将源请求当作实际专家标签。三指没有独立合格专家数据时，本阶段不进行三指BC/DAgger迁移。

当前Teacher是参考轨迹加MPC，报告明确`dagger_state_query_oracle=false`。它不是任意Student访问状态上的已验证查询专家。不能把时间索引重放改名为DAgger，也不能用离线MSE下降代替恢复能力或原生成功。

## 结果状态

为后续Student验证冻结新的200个独立 authored 初态（二指/三指各100），seed116507，manifest SHA为`28197144e00ca71e25a051934219744e27466cc21e4684a1f76e5c015b59b511`。与已有200例清单无身份重叠；仅在线缆XY与world Z yaw上施加有界扰动，保留Z、手根/关节/速度、相机和物理配置。此为同一reset族内的小扰动验证集，不是长度、质量、摩擦或柔性线缆泛化验证。清单生成时仅为`authored_unverified`，该历史manifest元数据保持不变；后续实际原生reset读回核验及200个完整首回合结果见下文，不以authored身份本身授予原生资格。

新增`bound_student_eval`独立入口，仅接受已合格完整V4，显式核验模型SHA、manifest、相机/物理身份及模型的raw/hemisphere预处理合同；等待单实例锁后再次核对输入SHA。它不添加paired arm、M2、Teacher或重放控制，原M0默认路径保持。

原路径与Adapter路径现已本次各复现二指4/4，T5原生证明通过，实际控制动作与主要物理状态一致。三指两臂仍各0/4。候选三指协调实验也已完整结束，二指4/4、三指0/4；没有三指合格专家标签。

本次Adapter路径的成功监督池为4698条、4个二指首回合，SHA为`1db1bfd46f69fdcf334d9740873b035767aff8e202887397e7b74875af15ce37`。它的路径、SHA和样本数与T5证明固定的采集报告完全相同；从同一批已检查字节反序列化。执行使用预先固定的独立1000次离线BC更新、batch64、LR1e−4、seed116507、显式reset_pointnet、quaternion半球规范化和0.25启动样本混合（tick<4）。本次BC仅使用二指专家标签（源1024轮PPO曾混合训练二指和三指），无独立来源holdout，不能声称源族泛化；首次独立原生验证不用于选择这些超参数。

## 已完成的离线拟合

在T5/T6完整结束后执行，独立目录为`/mnt/mydisk/xk/dh116/runs/stage5_physical_mapping_20261108T0840Z/t7_pinch_bc01`。训练集MSE从0.039165852降到0.000195265，手指/world XYZ/内禀旋转MAE分别0.005792710 / 0.007360959 / 0.005249295；最大单动作误差仍有0.300845，平均误差小不等于所有关键时刻准确。heldout样本为0。

拟合后，实际4698个样本的非零有效PointNet特征数为101538298，坐标清零使全部4698行的动作变化，最大/平均变化0.099634 / 0.036741。该证据只证明非零视觉梯度与数值坐标敏感性，不是物理抓取。未增加传感器维度，没有使用特权接触/滑移作为Actor输入。

新模型`pinch_bc_bound_v4.pt`的SHA为`c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97`。Critic、log_std、PPO Adam、学习率、全局计数及保存RNG保留；独立BC Adam另存。BC warm-start不是精确PPO续训，本次PPO更新数为0。执行前后原模型/数据/资格输入SHA均未变化，V4解包与新BC V3完整状态逐项一致。

## 本次原生冻结200初态只读探针

对本次原生reset已冻结的200条真实传感历史做CPU确定性前向，不新增原生试验、拟合或模型。旧raw模型`f37f25c…`的非零有效PointNet特征为0，坐标清零敏感行0/200；新hemisphere模型`c58f9cf…`为887123个非零有效特征、敏感行200/200，坐标清零动作差最大/均值为0.03021199 / 0.01322262。干预只清零点云坐标，保留mask、本体状态、previous action和时间。

下表是初始Actor归一化Z动作，左/右索引为14/17；标准差使用population std（correction=0），每种抓法100条，两手均为100/100负值。

| 模型 | 抓法 | 左Z mean | 左Z std | 右Z mean | 右Z std |
|---|---|---:|---:|---:|---:|
| 旧raw `f37f25c…` | 二指 | -0.12523611 | 0.00003402457 | -0.16618668 | 0.00008271281 |
| 旧raw `f37f25c…` | 三指 | -0.14337855 | 0.00003094570 | -0.28499357 | 0.00008153721 |
| 新hemisphere `c58f9cf…` | 二指 | -0.01750937 | 0.00003716591 | -0.02871013 | 0.00008497025 |
| 新hemisphere `c58f9cf…` | 三指 | -0.06630496 | 0.00003910229 | -0.11186252 | 0.00008880939 |

源1024轮PPO曾二指/三指混合训练，本次BC仅使用二指专家标签。以上只是运行时guard之前的初始Actor请求，不是SDK目标、实际掌位移、整回合因果解释或成功证据。验证集继续只作评估，不根据此探针重新拟合、调参或选择Checkpoint。

原件为`t7_frozen_initial_actor_audit01.json`，已核对SHA `3b7eff6e71ecf13b61ba43fb235e7c6942c79a10f0dd85f0ab2571cf9559fee2`；输入、源码及模型字节前后不变。

另核对本次标准conditional遥测的首200条Actor请求：env顺序0–199、trial身份及冻结reset顺序全部匹配。`c58f9cf…`的CPU hemisphere前向与原生提交请求的最大/平均绝对误差为`1.04308128e-7 / 1.25236632e-8`，无动作符号差；错误使用raw预处理时最大误差为`0.100190595`。微小数值差不能唯一归因于TF32；半球规范化是幂等操作，也不能仅靠输出一致推断其调用次数。原件`t7_native_initial_action_parity01.json`已核对SHA `cfc9e972f07c3ea5592fe4805463e9ce3e98f9729b098cb840f4df3d8c1f8418`。

本次这些conditional原始行没有显式policy step或时间戳；仅本段“policy 0、该次step的首个物理substep 1”的对齐来自冻结代码调用链，而非原始行直接记录。它不替代T1/T5/T6的跟踪链，也不能证明新策略的右掌物理退化已解决；旧T1/T5/T6目标来源还须区分提交前表达式、SDK缓存和backend。首200条请求只是初态核验，完整回合统计见下文，验证集不据此重新拟合或调参。

独立验证目录`t7_bc_validation01`使用原M0物理执行语义和显式hemisphere/V4合同，未启用M2、Teacher、MPC或本阶段特权协调器，沿用原SDK及安全控制；资产/Adapter身份在首次动作前核验。二指100例评估拟合后的Student，三指100例仅作诊断；本次BC未使用三指专家标签，源PPO曾二指/三指混合训练。

已保存执行元数据中的`diagnostic_only;untrained_grasp_style`，其`untrained`仅指本次BC阶段未使用三指专家标签，不表示历史PPO未进行三指训练；该执行原件不改写。

## 首轮200例原生结果：未通过抓取标准

本轮已自然完成1800控制步、200个完整首回合，0个无效测量，执行退出0。独立审计流式重放全部1353010条物理记录，200个trial各恰有1个terminal、之后0行覆盖；report/certificate、5份实际证据、模型、真实reset读回及相机身份核对一致。二指/三指各100个实际初态，不含CPU fixture。审计`t7_native_validation_audit01.json`的SHA为`508babfba1f6b47ae877f7a147dd89b6e5f7c66a16929c5f3dcd0707d4d5b18c`。

| 抓法 | 原生评估成功 | 净Lift P50 / P90 / P95 / max (mm) | 达30mm比例 | Hold max (s) | 双侧对向接触时间率 / 最长连续段 |
|---|---:|---|---:|---:|---|
| 二指 | 0/100 | 5.963853 / 14.627709 / 17.124066 / 25.235939 | 0% | 0 | 3.490306% / 3.2625 s |
| 三指 | 0/100 | 0.001133 / 0.276490 / 0.451067 / 0.651073 | 0% | 0 | 0.004984% / 0.020833 s |

二指有17个trial出现可用固定接触锚，最大相对滑移1.246515 mm；其18.904167 s适用窗口测量覆盖100%，其余83个trial为NA。三指均无有效锚，滑移为NA，不补0。两组均无合格5 s Hold，接触锚滑移不能改名为成功Hold窗口漂移。模拟隐式执行器估算力矩饱和时间率分别6.526028% / 15.434577%，判据为估算绝对值达到原effort limit；不是硬件电机反馈。

证书只证明本轮conditional原生执行及数据完整性，不证明策略合格：`conditional_telemetry_complete=true`，旧严格`telemetry_complete/validation_complete/native_validation_complete`仍为false；合法无锚NA与原生证书分层处理。没有改标志宣称成功，也没有降低30 mm /5 s /0.5 mm标准。刚性圆柱代理结果不是柔性线缆或真实DH116硬件验证。

本轮标准raw未保存失败码、显式policy/physics step及时间戳、完整SDK掌位链，不能按回合时长猜失败原因，或由物体高度回算左右掌位。每抓法100个非成功terminal的原因在本轮均未记录。后续已保持同一模型、同一200初态和超参数，在独立`t7_bc_diagnostics02`源码/目录完成被动诊断复采并退出0；不算400个独立初态，不回填旧文件或重新拟合验证集。

新复测200个首episode仍二指/三指各0/100，逐trial11项汇总与标准臂相同。1353010个首有效物理样本已关联60Hz Actor、480Hz掌位/载荷/真实backend与cache、实际时钟和首terminal；失败二指excess_force/joint_limit/object_dropped/timeout为16/18/51/15，三指为40/7/53/0。二指掌峰值可达50mm而同刻物体净Lift接近0，不代表抓取。逐手分位、峰值env/UID/时刻、同环境完整轨迹及来源边界见[11-09执行链审计](2026-11-09-stage5-student-execution-audit.md)。
