# Stereo 运行链路修复：P0／P1 验证报告

基线为 `9bfee426363527190011dbca164c104a00cbc999`，本轮开始时再次以 `git ls-remote --heads` 核对远端，HEAD 未变化。沿用 `codex/dh116-stereo-privileged-dagger`，没有重做 J0/J1/J2-A/J2-B，没有 push、merge 或启动长时间训练。旧 Teacher、Student、MPC、Checkpoint 和原 528 份保护文件保留。

本轮增加了实际软件修复、CPU/GPU 存储测量、三模式短时原生执行及有界 Teacher 诊断。它们没有授予原生深度精度、成功 Teacher、Student 抓取或部署资格。当前原生任务明确为 `rigid_proxy/solid_cylinder`。

## P0-1：可见路径与时序关联

`tabletop_stereo_wire_tracking.py::TrackedWirePointExtractor.extract` 先从 RGB mask 建二维拓扑，随后查询深度。数字骨架仅在已有轴向中间像素提供同一路径时移除冗余对角边；真正的对角线保留。已是细骨架的阶梯不再被 Zhang–Suen 二次侵蚀。厚、薄连通分量分别处理。真实分叉、交叉、环保持保守拒绝，断片独立编号，不连通隐藏区。

最多 128 个点输出 `xyz/valid/component_id/depth_segment_id/arc_length/tangent/confidence/visibility/track_id`，同时保留旧 `points_world/point_valid` 名称。深度空洞保留二维路径槽位，三维与米制弧长设 NA；完整深度按可见三维弧长采样，部分深度明确使用二维弧长，不插值填补隐藏三维形状。端点、切向、曲率、弧长和运动预测用于跨帧关联；竞争不明确时 ID 为 -1。Reset、时间倒退及过大间隔清理历史。

`tabletop_stereo_perception.py` 原 `WirePointExtractor` 和 `_visible_path` 未改写，提供显式 legacy 对照。`tabletop_stereo_preprocessing.py` 将新 tracking 模式及源码 SHA 纳入处理合同；`tabletop_stereo_runtime.py` 和 `tabletop_stereo_inference.py` 实际调用新提取器并保留跟踪诊断。旧 raw-RGB 包不会被静默改用新处理语义。

TDD 包含旧实现实际拒绝普通阶梯、两像素方向翻转、混合厚薄片段、透视下弧长采样不均等 RED，及阶梯、弯曲、分叉、交叉、环、遮挡、断裂、深度空洞、关联歧义、Reset 回归。单项 65 项通过（含旧 perception/geometry，和完整套件重叠）。

首次短原生配对保留了实际失败：二指 4 步均 0 有效点，三指为 0、9、9、9 点。复放同标定、同实际初态的保存图像发现，二指 7 个独立连通分量里，一个 4 像素歧义分量触发了整帧拒绝，掩盖另外 6 个可见路径。新增三参数 RED 后，将 `_paths_from_mask` 改为逐分量拒绝；输出 `rejected_components/rejected_component_count/all_visible_components_simple`，不将坏分量连接到好分量。纯分叉/环仍拒绝，`full_cable_observed` 不作真值声明。

final05 对原始四帧的只读复放中，二指每帧恢复 4 个有效点，三指每帧保留 9 个；二指的坏分量仍带原因和 bbox 被拒绝。输入 capture SHA `2f0ee494d1339f7d68d661eb81af23d8578843b935ae21863802c9c1bc60bdec` 未变，复放不计作新原生采集。三模式另外使用新独立目录复验，结果见机器摘要。

final05 新原生复验中，三个模式各自的二指有效点均为 `[0,4,4,4]`，三指均为 `[0,9,9,9]`；二指拒绝分量数为 `[0,1,1,1]`，三指均为0。首帧两抓法 mask 分别有2639/2375像素，其中各45个候选像素具有有效深度；但是二维路径采样点仍无有效深度，不能将首帧补为4/9点。该剩余稀疏支持问题保留原始证据。

**限制：**分割仍是颜色 mask 基线；track ID 是可见片段假设。这些稀疏局部点没有通过两抓法正确三维点完整率门槛，不能把拓扑修复或非零点数当作完整线缆重建成功。

## P0-2：TaskStateAdapter 真正进入动作网络

`tabletop_stereo_task_state_adapter.py::TaskStateAdapter` 按现有 371D schema 选择字段；默认候选 143 维，每项都有 value、validity 和 uncertainty，形成 429D 任务特征。直接本体状态不从估计头重复预测，几何/速度/接触使用指定预测行；分类接触取概率，连续接触量仍按合同解释。Force Proxy 重新读取合格证据，无法凭布尔配置或自称 qualified 解锁。

已修复的边界：NA/不可观测字段禁止进入有效状态；缺少关节归一化上下界的 q 保持 masked；速度及完成区间的预测至少要求两个有效历史帧；默认配置不宣称 `feedback.valid` 或 tracker 窗口已合格。额外 deformation/velocity 等头必须声明单位和坐标系。没有 palm→wrist-root 同帧标定时，不将 palm-link FK 当作 371D wrist-root 字段。Uncertainty 的 1 是未知哨兵，不是已校准误差。

`tabletop_stereo_state_estimator.py::StereoStudentActor.latent_mean` 明确提供 `latent`、`estimated_task_state`、`hybrid`，后两者将适配后的状态送入 `action_net`。Action-only loss 能回传选定估计头；未选择和 NA 行梯度为零。Critic 不进入这个调用。

`tabletop_stereo_checkpoint.py` 绑定输入模式、掌/世界模式、字段配置、当前 371D schema 和源码；相同宽度而字段含义不同的包也拒绝导出。三份独立 `stereo_student_{latent,estimated_task_state,hybrid}_palms640_v3.json` 保留 VGA 和独立模型身份；旧 latent/world 工厂默认路径保留。旧包需要它原先绑定的源码，不自动迁移。

三模式进行了固定输入 CPU 导出/载入逐值对照，随后各自使用独立随机初始化包做 2 环境×4 步原生执行。两个抓法的实际 SDK 初态 SHA 在三个进程中分别完全相同；同相机、seed=116；发出动作与原生 raw 逐条一致。它们是**未训练模型的运行通路比较**，没有策略效果优劣或学习成功结论。

final05 三个进程均 exit0，合计24条环境动作与192条480Hz子步记录对齐，左右掌点逐点符合实测 FK。各模式4步完成，但零完整回合、零成功，最高净抬升约 `7.15e-7 mm`（接近0）；不能据此评估30mm/5s/.5mm任务成功率。未将这24条动作、原始模型或 raw 数据写入公开汇总。

## P0-3：左右掌局部点进入 Encoder

`tabletop_stereo_observation.py` 在 SensorBatch 和四帧 History 中保存左右掌点、独立 validity 和 confidence；由同一帧实测 FK 生成。`tabletop_stereo_encoder.py::PalmPointEncoder` 共享点投影权重，使用两个独立 hand identity embedding。`StereoStateEstimator._spatial` 的 palms 模式读取这些点，同时去掉本体位置里的共同世界平移。World 点保留为显式辅助输出，不能绕过 TaskStateAdapter。

整体平移相机、双手和线缆的 CPU 测试检查局部点及估计输出不漂移；原生三模式检查左右掌局部点逐点符合实际 FK。无效深度/图像/历史有独立 Mask。独立 RGB 推理遇时间异常会清历史并拒绝当次动作。

## P0-4：Teacher 原生身份和真正执行动作

`tabletop_stereo_native.py` 新增/使用 `bind_camera`、`bind_teacher_checkpoint`、`teacher_pre_action_identities`、`set_teacher_action_diagnostics`。标定取实际 `bridge.camera.calibration.sha256`，Checkpoint 取文件实际 SHA；首次物理动作前绑定 owner。全零 SHA、过期帧、错 episode/reset/tick 和动作不一致均拒绝。

`tabletop_privileged_cli.py::collect_or_evaluate` 分开记录 sampled、mean、actually issued 和 applied；eval/collection 实际执行 deterministic mean，随机动作只作分布诊断。`applied_action` 没有归一化逆读回来源时保持 null，实际控制器和 SDK 目标另外保存，不能把 issued 改名 applied。

原生训练 raw 64 个物理样本验证 sampled==issued；另一次 2 环境×4 步 collection 的 64 样本验证 mean==issued、sampled!=issued（最大差约 .281487），Checkpoint/标定/frame/owner/时钟一致。数据仍为不合格候选，未进入成功专家池或 DAgger。

## P1-1：高分辨率 PPO 帧池

`tabletop_stereo_rollout_storage.py::IndexedStereoRolloutStorage.capture/resolve` 保存唯一 RGB 图像池、独立深度/Mask/小状态帧和紧凑历史索引。只有 float32↔uint8 `/255` 逐字节相等的 RGB 才压为 uint8；任意 float32 RGB 保留原字节，深度等不量化。

`tabletop_privileged_ppo.py` 增加可选存储钩子，原 PPO 公式、GAE、随机采样与安全合同保留。收集、旧 logprob 复核、minibatch 前向都按 `max_visual_batch` 恢复；梯度按原 minibatch 权重累加，优化器步数不变。`tabletop_stereo_student_cli.py::train_student` 直接将 History 交帧池，移除实际入口多余 clone。更新使用可训练 Encoder，没有用旧 latent 替代图像。

实测输入为两步、四帧、640×480 合成传感序列；每个 CPU/GPU 用例独立进程、真实视觉前向和 PPO backward。下面是 GPU2（4090）结果，显存为 PyTorch peak allocated，RSS 单列；不包括 Isaac 渲染器。

|环境|CPU 峰值 RSS MiB|GPU 进程峰值 RSS MiB|GPU 峰值显存 MiB|GPU 采集 env-step/s|GPU PPO update s|CPU PPO update s|
|---:|---:|---:|---:|---:|---:|---:|
|2|1574.79|2042.86|411.77|1.45|1.28|10.44|
|8|1869.71|2394.77|455.97|8.00|2.43|42.97|
|16|2152.91|3257.54|455.98|8.51|4.77|81.61|
|32|3427.47|4347.49|456.01|8.51|9.21|166.69|

池载荷 44/176/352/704 MiB，相同输入完整历史重复张量载荷 154.78/619.10/1238.20/2476.41 MiB，减少 71.57%。这是张量载荷对比，不是 RSS 同比例下降。恢复前后抽样 logprob 最大误差 CPU `1.90735e-6`、GPU `0`，显式容差 `2e-5`；逐字节输入还原及 Encoder 梯度检查通过。采集速度包含合成输入插入和 Actor，不是原生相机采集吞吐。

## P1-2：audit 与 accelerated

`tabletop_stereo_pyramid.py::match_pyramid` 默认 `audit` 保留全范围检查；与冻结父版本三个数值输出完全一致。`accelerated` 使用可信 coarse prior，在 `tabletop_stereo_matching_costs.py` 中实际跳过不需要的视差平面。局部置信、边界或支持不够时 global fallback。双向、逐层记录候选数、实际代价计算数和 fallback，`StereoDepth` 绑定显式 mode。

同一 192×80 解析图像、8976 个固定参考点、每模式 1 次预热+7 次实测：

|场景/模式|深度 MAE mm|RMSE mm|P95 mm|正确三维点≤5mm完整率|实际代价计算数|平均/P95 ms|全尺寸 fallback 左/右|
|---|---:|---:|---:|---:|---:|---:|---:|
|单平面 audit|0|0|0|100%|8,609,280|1857/1931|全范围审计|
|单平面 accelerated|0|0|0|100%|6,336,000|1742/1810|23.89%/23.89%|
|双表面 audit|2.241|32.111|约0.000030|94.285%|8,609,280|2017/2120|全范围审计|
|双表面 accelerated|2.242|32.116|约0.000030|94.251%|7,918,080|2007/2223|33.27%/33.16%|

正确三维点完整率分母固定，不删失败点；深度误差统计 accepted reference，并另外保存共同支持集。双表面存在大误差尾部，P95 不能代表所有点。搜索工作减少没有保证端到端更快，双表面 P95 反而更慢。本次不将 accelerated 设为默认，不宣称 VGA/GPU 或原生精度升级。

## P1-3：任务身份

`stereo_native_task_v1.json` 仅新增 `rigid_proxy/solid_cylinder` 身份字段，原 arguments 逐项不变。`NativeTensorBridge.task_identity` 进入原始样本和 Teacher Checkpoint source identity。

`tabletop_stereo_rod_state.py::DeformableRodStateProvider` 独立验证后端文件 SHA、同步物理边界、单位/坐标、节点位置速度、中心线、切向、曲率、轴向形变和分布接触来源；`stereo_native_deformable_rod_v1.json` 不借用圆柱 root pose。现有旧 TensorTask 启动器不支持该柔性入口时在创建 Isaac 前拒绝。

**阻断：**真实 DeformX/Cosserat backend→DH116 闭环仍未接通。已有旧后端按网格汇总的回执、含内力/重力的数组不能冒充逐节点接触。本轮没有柔性线缆物理成功或 CPU/GPU solver 验收。

## P1-4：复用 Teacher PPO 的有界诊断

使用现有 PrivilegedTeacher/PPOTrainer，371D 输入、24D tanh Gaussian，R0 原生奖励只加入一次。新增按手指/平移/旋转的噪声、饱和、collection MC entropy 和最终 old→new KL 诊断；关节越限无法逐手归因时写 NA，原始关节/SDK 目标保留。Checkpoint 保存后用独立实例回读 policy、Adam、计数、版本，不改变采样 RNG。

第一次原生诊断为 2 环境、4 rollout steps、1 update，8 条 transition，40 minibatch updates，原生 64 条 480Hz 样本及两个 episode header。每手式约 .0667 秒，零完整 episode，零成功。R0 closure 为 0；更新约 6.59 秒、优化器过程平均 KL 29.79。如此小 batch 的高 KL 是明确限制；不自动启动正式训练。补充分组统计后使用独立目录重复相同有界验收，详细 SHA/最终指标见机器摘要。

原生进程使用现有 `/mnt/mydisk/xk/envs/dh116/bin/python` 和项目 `scripts/python.sh`，未升级任何环境。运行期间仅一个 Isaac，GPU0/1 的其他进程未改动。日志存在 inotify change-watch `ENOSPC` 警告；当时 `/tmp` 文件系统尚余约 57 GB，未修改系统 watch/驱动配置。

旧 launcher 的 Isaac fast shutdown 会使 outer CLI 返回后的 `completion.json` 不落盘；本次以 callback/scene 已持久化结果、Checkpoint、raw 样本及外部进程退出码联合确认，未伪造 completion 文件。该退出包装限制单独保留，不能仅凭 callback 结果断言完整进程成功。

**多更新边界：**本轮原生验证限定单次更新，或固定 Checkpoint 的 collection。若未来同一物理 episode 跨两次以上 PPO 更新，新的 rollout 已绑定新 Checkpoint，但初始 `episode_start` 仍保留首个 SHA；需要增加明确的 policy-epoch 切换证据和校验后才能进入该类正式训练。不能把本轮单更新身份一致性扩大为多更新全回合证据已通过。

## 测试、文件与失败证据

最终代码对应的两平台完整回归、实际文件 SHA、分组统计和原生证据摘要保存在 [机器验证摘要](../../contracts/stereo_runtime_repair_validation_summary.json)；[readiness](../../contracts/stereo_phase0_7_readiness.json) 与 [文件清单](../../stereo_phase0_7_file_inventory.md) 同步更新。

最终 final05 代码：Windows **1030 passed / 1 skipped，153.36s**；服务器现有 Python 3.10.12、PyTorch 2.7.0+cu128 的 CPU 运行 **1030 passed / 2 skipped，180.92s**，均 exit0、无 Isaac 模块导入、测试前后源码 SHA 不变。服务器多跳过一项需要本地私有父版本快照的数值比较，该项已在 Windows 实际执行；既有可选路径跳过保留。完整选择为全部 `tests/test_dh116_stereo_*.py` 和现有 11 份 privileged/Phase0/surface/squashed-Gaussian 回归；子组、两平台计数不能相加。

所有新问题先保存 RED 再修复。首次完整 1024/1skip 运行发生源码并行变化，source guard 正确失败，不能作为最终源码通过；随后冻结代码重跑。Schema/字段导出、palm-root 混淆、区间首帧、单位缺失、逐分量拒绝等集成 RED 与 native/benchmark 原件保存在私有 `artifacts/stereo-runtime-repair-20261111/`，不提交 raw、模型或日志。

```text
PYTHONPATH=src;artifacts/stage7_lifecycle_cpu_deps_v01
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_runtime_wire_tracking.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_runtime_palms.py tests/test_dh116_stereo_runtime_teacher_identity.py tests/test_dh116_stereo_runtime_integration.py tests/test_dh116_stereo_runtime_rollout_storage.py tests/test_dh116_stereo_runtime_pyramid_modes.py tests/test_dh116_stereo_runtime_rod_identity.py
```

原生命令须使用独立冻结源码/新目录并遵守单实例：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 \
bash scripts/python.sh scripts/train_dh116_privileged_teacher.py \
  --config configs/dh116_training/privileged_teacher_ppo_v1.json \
  --native-task-config configs/dh116_training/stereo_native_task_v1.json \
  --camera-config configs/dh116_training/stereo_camera_front_640x480_v3.json \
  --num-envs 2 --rollout-steps 4 --updates 1 --enable-experiment --execute-native --output NEW_DIRECTORY
```

## 晋级边界与后续

- P0 软件运行链路及有界执行已验；独立歧义分量导致二指整帧0点的问题已修复，但两抓法深度精度、正确三维点完整率仍阻断视觉精度晋级。
- 三种 Actor 尚未获得合格监督数据或训练完成模型；当前配对不是策略成功率实验。额外形变、力代理、未知标定状态继续 masked/disabled。
- Teacher 只有极短 PPO 诊断和 mean collection，未获得完整原生成功轨迹，不能拿来正式 BC 或 DAgger。
- P2 仍执行既有门禁：BC 只收完整成功原生 Teacher；DAgger Student 执行、Teacher 同一 pre-action 查询且不推进；Student PPO 使用自身 on-policy rollout。没有将候选收集或诊断当成功专家数据。
- 下一项先完成两抓法深度精度与稀疏可见点完整率复验、policy-epoch 身份边界，再进行独立状态监督和 Teacher 完整回合验证。30mm、5s、0.5mm 门槛及双手有效对向接触标准不变；PR 保持 Draft。
