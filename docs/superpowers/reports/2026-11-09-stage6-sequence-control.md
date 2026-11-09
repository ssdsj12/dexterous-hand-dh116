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
