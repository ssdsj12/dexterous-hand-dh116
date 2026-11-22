# 历史抓取证据复核与接触阶段 Teacher Reward

基线：`2767819c76be23c7cd38f8408d9b3b60ca83b0e8`，Ubuntu 分支 `11-17`。
本包实际修改奖励运行逻辑、配置及测试；没有启动 Isaac、Teacher 原生 PPO 或新采集。

## 历史成功中可以借鉴什么

独立只读检查 `/mnt/mydisk/xk/dh116/runs/wp3_mpc_t5_first_episodes_v01`，
结合现存源码，与 Antigravity 进行了 Historical-Success R1–R4 四轮讨论。
Antigravity 只读通路允许读取源码/测试，不能读取本报告、配置和原始 runs；
后者由 Codex 独立读取，未将“对方已审阅”扩张到不可访问的内容。

### 原始 T5 刚性任务

原报告 [T5 原生证据](2026-11-12-mpc-t5-first-episode-native-evidence.md)：
`solid_cylinder` 二指 4/4，三指 0/4。控制来源是旧 Teacher 参考轨迹加局部手指 MPC，
不是当前柔性 PPO；不能写成新的柔性成功。四个二指环境共享源轨迹，不是 100 个独立初态验收。

本轮按 raw post-SDK 与样本 pre/post-SDK 配对，避免控制 tick 的 0/1 起始差异：

| env | 首次必要二指载荷及对向条件 SDK | 达到 30mm 的 post SDK | 原始成功 post SDK | 净 Lift mm | Hold s | 固定锚点漂移 mm |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 993 | 7010 | 9410 | 31.589504 | 5 | 0.002354 |
| 1 | 992 | 7018 | 9418 | 31.535705 | 5 | 0.002205 |
| 2 | 993 | 7026 | 9418 | 31.530101 | 5 | 0.006494 |
| 3 | 992 | 7010 | 9410 | 31.627685 | 5 | 0.079632 |

首次必要接触是对 480Hz 原始 accepted positive normal load、contact mask 和受力方向做的离线核对：
每手 thumb/index 各 >0.05N，方向余弦 <=-0.25；出现在约 2.067s。
只解析压缩 states 文件至 SDK1200 的前 9456 行，不能据此声称全回合逐 contact 资格通过。
整文件 SHA 另作完整字节核验。其余表中终局来自 1800 行 60Hz post-control 摘要。
漂移是旧固定物体锚点定义，不能改称新柔性材料节点切向 Slip 测量。

首次任一正腕 Z 请求是 control tick55/pre SDK458（left +0.0780/right -0.0331），属接近微调；
首次两手 Z 请求同时达到约 +0.3 在 tick242/pre SDK1954。
不能把前者当成 Lift 相位，也不能把后者复制为柔性任务固定抬腕时刻。
旧 `hold_paired` 是嵌套的两手 bool，且被 30mm 等物理窗口条件门控，不能用 `bool(list)`
或此字段推断首次抓取接触。

4698 条“successful ... unqualified”样本确有 (371D,24D)，但每行只有 **24/371** 维有效；
`qualified_for_bc=False`。不能补零后作为新 common66 + rod41x13 + context50 的正式 BC 标签。

### 三份旧 GPU Teacher 模型

实际读取保留的 `gpu_teacher_ppo/*/best.pt` 和 metrics：

| 抓法 | 选中 update | legacy stage2 | hold30_s 摘要 |
|---|---:|---:|---:|
| Pinch | 8 | 16/16 | 2.64 |
| Tripod | 3 | 16/16 | 2.78 |
| Power | 10 | 16/16 | 2.23 |

它们是实际存在的历史模型/训练摘要，不能一概称为资源或几何检查。
但当前旧入口 `scripts/train_dh116_gpu_teacher_ppo.py` 的判据为：
双手总 wrench 模长各 >0.3N + min Lift >=30mm；stage2 仅需 hold30>=1s；
失效时扣 2dt，而非连续时长清零。缺逐指对向、离架、0.5mm 滑移和 Power 掌面资格。
入口注册 11 mesh（左右各 5 指端及桌面），不含掌面/两端支架/近端；且关闭桌面 PhysX 碰撞。
其评估分支使用 `os._exit(0)`。本包不采用这些做法。

checkpoint 未保存完整 source commit/config/force 资格绑定；
当前源码定义不能反推每个历史运行使用逐字相同的代码。
上述摘要自身不能证明连续 5s 正式验收，不能作为当前成功专家池准入。

### 采用与拒绝的内容

- 采用“实际目标与增量动作分开”的经验：零速度命令保留累计目标，不能假定它恢复绝对 q。
  旧 absolute-reference 的不重锚与限速思路可作独立候选；不覆盖旧控制器或直接回放旧 q。
- 采用真实接触反馈指导“调腕就位/手指闭合 → 连续稳定承载 → 离架 Lift/Hold”课程。
  旧成功不是始终固定腕；闭合初期允许安全范围内就位，稳定保持阶段再核验腕部。
- 旧三指报告暴露“手指参考归零、缺失指不再恢复，但手腕继续上行”的空抬风险。
  几何代价降低、手腕升高都不能代替三指承载。
- Power 必须五指包覆加掌面，与二/三指独立；本包新接触 profile 拒收 Power。
- 不迁移刚体 CAD gap/根姿态至柔性材料节点，不复制固定 tick 抬腕、旧力目标或不合格监督。
  固定双 RGB、左右手身份、原 24D/371D 和 30mm/5s/0.5mm 均保留。

## 本包实际代码

`tabletop_training_reward.py` 复用原 `TrainingReward`，
新增独立 `approach_contact_prelift_v1`：

1. `_bind_contact_owner/_contact_source` 仅接受精确的 `PrimedContactContextOwner`，
   从真实 separately-primed receipt 绑定单一 Reset、history、backend 和 grasp。
   weakref 不延长 native view/CUDA 持有链；owner 关闭、回收或绑定变化后禁止新动作窗口。
2. `_validate_contact` 将每个奖励端点与 owner 已记录的同一物理事件核对：
   clock、物理观测计数 +1、固定材料41、真实 required pad IDs、6项接触/桌面值及来源均必须一致。
   接触时长只读取 owner，首次为0、断触清零，没有第二套衰减计时器。
3. Reset 保留 SDK2 的 S0/dref，即使奖励首个可用端点在 primed SDK3。
   priming 请求必须为真实 event SHA、零24D及 `source_priming_no_policy`，不算 PPO transition。
4. 势能仅为 `2*p_distance + 2*p_contact + p_opposition + 0.5*p_persistence`；
   连续接触尺度0.25s仍是候选课程尺度，不是正式成功门槛。
   Lift/Hold/Slip/velocity/prohibited-support/+100 在本 profile 排除，Hold 原值为 NA/None。
5. 保留既有动作变化/时间代价与原护栏失败 -40。初始两端支架允许且原证据保留；
   真实桌面接触终止 -40，不把支架承托判为抓取成功。
6. Antigravity R3 发现并由 Codex RED 复现：首次 Reset 失败后可用合法 token 复活。
   CONTACT 入口增加一次性 Reset 门控；失败/已用对象必须重建，旧 profile 的 Reset 语义保留。

配置：`configs/dh116_training/approach_contact_prelift_v1.json`，PPO gamma=.995，
沿用 smoke 的有界诊断参数与 KL Guard。没有改旧 SMOKE/FULL/CURRICULUM 配置、奖励行为、
网络输入合同或任何 native 物理参数。

## 测试与失败证据

所有执行在 Ubuntu 项目 Python；CPU 环境显式 `CUDA_VISIBLE_DEVICES=""`。
最终运行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
  tests/test_native_teacher_contact_reward.py \
  tests/test_native_teacher_reward.py \
  tests/test_native_teacher_contact_context.py \
  tests/test_native_teacher_collector.py \
  tests/test_native_teacher_session.py \
  tests/test_surface_force_sharing.py \
  tests/test_vision_teacher_absolute_mpc.py
```

最终版本 **402 passed, 0 failed, 92.43s**，其中新奖励12项；不累加此前重复运行。
另有调度器/Registry **60 passed, 0 failed, 15.22s**，与上述主回归不重叠。
初始新包11项 RED（旧 loader 无新 profile/旧构造器无 owner lease）；
桌面 guard 删行 mutation 确认原测试会失败；Reset 复活也先独立复现再修正。
日志与原始数据留在服务器私有 evidence 路径，不上传模型、压缩 states 或 JSONL。

## 仍未完成的真实链路

这是 **CPU 接触奖励组件**，不是已完成原生训练：

- 实际 `_NativeGeometrySources`、session 和原 Collector 仍只放行旧 smoke；
  下一包需复用该 Collector 接入 separately-primed source + Context + 本奖励。
- 桌面失败必须在原 physical recorder/history 之前提供可信 physical guard，
  才能在小于8步时一致结束窗口；不能只有奖励终止而原 history 继续积分。
- 相同 pre-action 身份、sampled request/logprob、实际 SDK projection、raw contact/source、reset 和
  checkpoint 身份必须端到端贯通后，再冻结新有界原生配方。不可重跑已消耗指纹。
- Power 继续独立五指掌面 owner；正式三抓法柔性 Teacher、成功 BC/DAgger/Student 均未准入。
  R2/Force Proxy 关闭。本轮 native0、原生PPO0、新成功轨迹0。

相关源码、原始证据与回归 SHA 见
[readiness](../../contracts/dh116_supported_contact_reward_v1.json)。
