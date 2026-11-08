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

三指协调候选的480Hz执行与原60Hz源请求不同，其样本明确排除于BC池；即使候选出现物理成功，也不能直接将源请求当作实际专家标签。三指没有独立合格专家数据时，不进入三指Student训练。

当前Teacher是参考轨迹加MPC，报告明确`dagger_state_query_oracle=false`。它不是任意Student访问状态上的已验证查询专家。不能把时间索引重放改名为DAgger，也不能用离线MSE下降代替恢复能力或原生成功。

## 结果状态

为后续Student验证冻结新的200个独立 authored 初态（二指/三指各100），seed116507，manifest SHA为`28197144e00ca71e25a051934219744e27466cc21e4684a1f76e5c015b59b511`。与已有200例清单无身份重叠；仅在线缆XY与world Z yaw上施加有界扰动，保留Z、手根/关节/速度、相机和物理配置。此为同一reset族内的小扰动验证集，不是长度、质量、摩擦或柔性线缆泛化验证。初态尚须通过原生reset读回核验，不能将200个authored身份计为已经完成的200次原生评估。

新增`bound_student_eval`独立入口，仅接受已合格完整V4，显式核验模型SHA、manifest、相机/物理身份及模型的raw/hemisphere预处理合同；等待单实例锁后再次核对输入SHA。它不添加paired arm、M2、Teacher或重放控制，原M0默认路径保持。

原路径Teacher现已本次复现二指4/4，Adapter对照及后续门禁实验仍在串行采集。本节将在实际完成后记录模型SHA、同步数据资格、离线拟合与独立原生Student统计；此时尚未从上述CPU探针产生新模型或成功声明。
