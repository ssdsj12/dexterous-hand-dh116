# DH116 无相机柔性特权 Teacher CPU 模型

基线为柔性任务/R0 CPU 提交 `8511a78b6ee716a0380f5808b25293807e96ad27`。新增独立模型 `DeformablePrivilegedTeacher` 和三份配置。配置均要求真实双手同杆接触、Reset 与 371D 来源、柔性 R0 资格、独立预算批准后再接原生。当前三者的 `native_ready` 都为 `false`。没有相机、WireSeg 或 MPC 数据依赖，专家初始化可选。

| 抓法 | 独立 Checkpoint namespace | 配置 SHA256 |
| --- | --- | --- |
| Pinch | `deformable_teacher_pinch_v1` | `4324785c66ff8557aca20b4c5ae7e40f802455432ae6db45b4eb9d827cd174d8` |
| Tripod | `deformable_teacher_tripod_v1` | `349918998360794df3645fe2ab6ffad3540496ab1bac02a9d55f52063b92a853` |
| Power | `deformable_teacher_power_v1` | `80bca302c100acb18d69fa7615f23ae76be368cc61f111ce6be7463894e8eca1` |

模型身份绑定独立版本、抓法、完整配置和任务 SHA、原 24D 动作与 371D 状态源、DeformX 扩展 SHA、节点 schema 及实际模型和 Tanh 源码。CPU 回归验证交叉抓法身份和旧 Teacher 身份均被拒绝，旧权重无法严格载入新结构。没有写出模型或 Checkpoint。

## 输入与推理

Actor 与 Critic 各有独立材料节点编码器。原 371D common 分支保留 256/256 层；材料分支通过有 mask 的均值和最大值聚合，随后分别融合到 24D Actor 和标量 Critic。沿用原 `PrivilegedTeacher` 的 Tanh Gaussian 采样、确定性 mean、`log_std` 和 likelihood 代码。原 Teacher、PPOTrainer、CLI、动作合同和检查点格式均未编辑。

`pack()` 接受有限的原 371D common 张量，以及带环境、episode、reset、rod UID、PhysX tick、rod tick 和物理时间的 DeformX world-frame 节点读回。两者身份不一致、schema/后端不匹配或已标记有效的数据包含 NA 时拒绝。每节点输入为世界位置 3、世界速度 3、世界单位切向 3、曲率 1、归一化 rest-material station 1、相邻有效 segment strain 1 及其有效位 1。所有力学和触觉字段均未输入。已明确无效的节点在编码与聚合前屏蔽，相关 strain 不能进入有效相邻节点；未知 common 字段拒绝输入。CPU 样本验证了 material station 跟随节点特征置换时的推理不变性。

## 结果与剩余接入

先由缺少独立入口的测试失败，再验证错配时钟/rod UID、非法切向、节点 shape 和无效掩码、独立配置和权重身份。CPU 测试覆盖采样动作、mean 和旧 Tanh log probability 等式，分别验证 Actor/Critic 的材料分支梯度与无效节点零梯度。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_privileged_teacher.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_privileged_teacher.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py
```

149 项 CPU 测试通过，其中 7 项为新模型测试。528 个受保护文件和 2 个历史 Checkpoint SHA 校验通过。未启动 Isaac 或训练。当前 PPOTrainer 的 Critic 输入仍为单一 371D Tensor，CLI 也仍构造旧模型。将新模型接入训练须先通过原生材料读回、Reset、R0 和同杆接触资格，再实现独立的 Critic structured input、Checkpoint 与原生入口。三种抓法均无原生 Teacher 成功数据。
