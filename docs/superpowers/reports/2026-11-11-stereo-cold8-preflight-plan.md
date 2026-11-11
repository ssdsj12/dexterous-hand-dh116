# 双侧同步候选数据：cold8 预检和待批准执行计划

本文件是执行计划及源码/CPU 预检结果，不是新的 Isaac 采集报告。用户本轮未批准新增同步采集；新增原生采集次数为 **0**。不得据此文件启动 Isaac。后续只有用户明确批准，且单实例和全部身份预检通过，才可以执行独立有界实验。

## 固定范围与身份

基于 `codex/dh116-stereo-privileged-dagger`，本轮起始远端 HEAD 为 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。最终执行必须重新冻结当前源码并记录快照文件 SHA，不能把该历史 HEAD 当作未提交新代码的精确身份。

|项目|已固定输入或边界|
|---|---|
|Teacher|`/tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000001.pt`|
|Teacher 文件 SHA|`c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`|
|原生任务合同文件 SHA|`71d1d95d8e7630d941af119fad08ff2724a0ef7d40393d513f746fbf900e6179`|
|物理语义 SHA|`2bd771fb22f951dd0bdd8f52a0af55c01d24382ad5efe09ab19e22603a936566`|
|cold8 manifest|`/tmp/dh116-wp2-cold8-plan-20261111-v01/candidate_cold8.json`|
|manifest 文件 SHA|`e2e1280e02cfe193678b2474a179e4e7207c4239d75641b7e2efc2ad5c969e83`|
|manifest 语义 SHA|`afbb1984ab20c296921e251a78846eb9cab665582b35a6328eec73eeea5fed78`|
|相机配置|`configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json`，两侧均 640×480|
|相机配置文件 SHA|`5840b57a98b42c21458302c8405fb49aa75b17270abf14955ec7e69c383a597c`|
|标定语义 SHA|`02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`|
|采集配置|`configs/dh116_training/stereo_candidate_cold8_v1.json`|
|并行和预算|2 env，每批二指/三指各 1；4 批；每初态最多 4 个 60 Hz 控制步；最多 32 个逐环境候选样本|
|物理精度|原 480 Hz；每控制步原 8 个物理子步；不修改物理、材料、执行器或成功合同|
|动作|同一 pre-action 371D 输入的 Teacher deterministic mean；不训练、不采样探索、不调用 Student/MPC 接管|
|任务身份|`rigid_proxy`，不是 Cosserat/DeformX 柔性线缆|
|GPU 计划|`CUDA_VISIBLE_DEVICES=3`，计算设备 `cuda:0`，renderer `activeGpu=3`；真实 GPU UUID 必须执行前读回，当前未填写|

8 初态来自真实 Teacher 合同及原始左右手初始化，通过既有 generator 生成完整 200 个 authored case 后固定取每种抓法前 4 个；不是按结果筛选。完整来源 sidecar 与 200-case manifest 保留。原生成器只在合法 reset 状态内应用微小变化：主动关节半径 0.0005 rad、物体 XY 半径 0.00005 m，沿用真实被动关节耦合。它不改变材质、执行器限制或训练物理参数。

`authored_unverified` 不等于原生通过。既有实际初态指纹按 10 μm/10 μrad 量化并排除环境 clone origin；8 个实际 SHA 必须全部不同，否则立即拒绝。数值不同也不意味着覆盖足够的任务分布，不能用于宣称泛化。

## 逐字段同步审计

|字段|当前实现与 CPU/静态证据|原生待证|
|---|---|---|
|左/右 RGB|`_NativeRGBPairReader.read()` 从同一 tiled RGB 产品按实际 numeric slot 顺序切出 2N 图像；读前后核对 ReferenceTime/IsaacReadTimes；左右 frame/time 一致；`StereoCamera.sample()` 核对当前 sim time、标定读回、尺寸和无物理推进|cold reset 后四个实际 Camera prim、两侧分辨率/slot 顺序、新鲜 render frame、无混帧|
|左/右 instance ID|新增右侧切片与左侧同属同一 tiled reference 产品；每侧独立解析实际 Wire prim/render inventory；缺 ID 或解析冲突拒绝，而不是把颜色阈值当标签|新右侧 ID 通道尚未做本轮原生采集；实际右侧 inventory 完整、ID/metadata/visible mask 一致|
|左/右 optical-Z reference|与左右 ID 共用 reference render 产品；与 RGB 的 frame/time/calibration 三项逐项一致；reader 本身不 render/step，重试只重新读取完整同步 RGB/reference 对|双产品 ReferenceTime 与 renderer 更新时间真实一致；参考帧不能滞后一帧|
|Actor 的 depth/点云|仍从左右 RGB 计算视差和深度，保留中心线/observed surface 来源编码；reference 光轴深度和实例 ID 不进入 `SensorBatch`、Actor 或任务输入|RGB-only 输出在实际遮挡下有效性/误差；不预报改善|
|371D Teacher 状态|snapshot 同一 pre-action 边界复制 `bridge.critic`；privileged identity 与 RGB identity 一致；24 维关节编码器标签独立读回并比较归一化值|实际 q/qd/掌 FK/状态同钟；有限 371D 值不等于 371 维监督都合格|
|61–63 中心 XYZ 标签|离线对同一 episode/reset/physics tick/time 的 raw object/wrist pose 重算 `(object_center-mean(wrist_origins))/0.05 m`，再与原 371D 比较；必须有通过独立 ID/optical-Z 参考验证的可见点|新初态实际物理读回及可见点；每初态首帧没有同钟前一 raw sample，继续为 NA，不补零/借后一帧|
|24D issued action|保存 latent mean、std、`teacher_mean` 和 `issued_actions`；`teacher_sample=None`；每步要求 issued 与实际被选中的 mean 逐元素一致；raw 每个 live 子步保存 `actor_issued_action` 并与预动作身份绑定|实际 raw 480 Hz 全链、terminal 边界和 SDK 读回；不得把 issued action 重命名为实际关节运动|
|applied targets|独立保存控制器 command、SDK joint targets、实测 palm pose；terminal 后的 post-reset 读回标为 NA，使用原 terminal raw 证据|实际 SDK 目标来源与终止子步保持；未提供完整 normalized applied-action，不伪造 24D applied action|
|身份|每条候选有 run/env/episode_uid/reset/policy_tick/physics tick/time/frame/calibration；Teacher SHA 在加载前、加载后、cold 采集结束检查；raw/candidate/ref/ID 文件按 SHA 读取|运行中新快照 SHA、实际所有文件 SHA、独立 output 未存在、单实例/GPU UUID|

参考包含 GT，用途仅为监督/误差评价。`native_instance_targets()` 的 unknown 是“渲染实例 ID 未解析”，不是遮挡真值。没有独立隐藏表面监督的 `occluded_wire` Head 保持关闭；不能以右侧无标签时的共享权重输出宣称右侧监督已通过。

## 本次预检实际发现并修复的问题

1. **冷 reset 的完整数据读回被旧终止条件拒绝。** 每 4 步人工冷 reset 时原 episode 通常没有 terminal，旧 `validate_rows()` 会拒绝 UID 变化。没有放宽该正式验证器。新增 `_validate_cold_candidate_rows()` 仅用于明确的 Teacher candidate-only payload：核对保存的 manifest、cohort report、逐批 reset-readback 文件 SHA、实际/作者初态匹配、唯一 UID/初态、每批预算、全局时钟以及清空的历史，再分别调用原 `validate_rows()` 验证每个连续 episode。之后仍执行完整 raw ledger 核对。正式 BC/DAgger 的完整成功轨迹与 terminal 条件不变。
2. **异常时 RGB/Teacher 中间记录会丢失。** 后续 batch 异常现在 exclusive 保存 `partial_candidates.pt`、SHA、失败原因及已收集原张量，格式为 `dh116-stereo-teacher-failed-partial-v1`。它不是正常候选格式，不能被 `read_candidate()` 或成功池接收；不写完整 cohort report。
3. **Teacher 文件在模型加载间变化未被双检。** `_load_frozen_teacher()` 在调用既有严格 loader 的前后核对实际文件 SHA，cold collector 结束再核对。同一个内存 policy 的 tensor digest 仍逐次检查。文件变化产生失败证据，不绑定过期 SHA 为成功结果。
4. **相同 policy tick 的不同 reset 参考文件重名。** 参考文件名现包含 episode UID 集合摘要。同 tick 新 epoch 独立保存，不改写时钟或旧文件。
5. **两条初态证据链没有互相连接。** 复核发现 manifest/readback 与 raw header/样本 SHA 原先各自一致仍不够：错槽的 header 可以自洽地指向另一物理初态。现 `_check_reset_readback_header()` 将同 env 的 raw 初始 object pose/velocity 对到 reset readback 的 wire pose/velocity，显式处理 World 到 env-local 原点平移及四元数 ± 等价，容差仍为 2e-5；同时检查 env_id 对应真实 batch slot。root pose 不等于 palm pose，full joints 不等于 active12，没有用这些不等价字段强行比较。

测试均先复现失败再修改。partial-save 回归为 50 passed / 16.72 s。最终冷 reset 完整 ledger、窄准入反篡改、checkpoint 双检/结束变更、状态监督和训练模式专项为 **43 passed / 10.44 s**，日志 `cold_preflight_final_v05.log`。自洽错物体 pose/velocity 和自洽错环境 slot 均拒绝，正确 World 原点转换和相反四元数符号通过。fixture 使用 CPU 构造数据，仅证明软件检查逻辑，不计为原生验证。最新全量 CPU 汇总由主任务报告。

此轮源码冻结时 `tabletop_stereo_state_supervision.py` SHA 为 `e9082164de2d82450582524fda3df801862052c0b1eabd60b1e59574bb030fb7`，`tabletop_stereo_training_cli.py` SHA 为 `03026017937086a21f99550a3bd4de43f7bce8120f50efafa002986135000423`。执行前仍必须核对最终快照，文档中的 SHA 不能替代实际验证。

## 现有状态拟合与后续数据使用

已有真实数据离线拟合使用 8 行、2 个实际物理初态，每种抓法 4 帧。只有 6 行的 61/62/63 三维有独立同钟监督。`diagnostic_fit` 的全部误差均来自这些训练样本，validation/test 为 `None`；没有按相邻帧制造 holdout。`formal_training` 因实际独立组不足而停止。unknown uncertainty=1 是 sentinel，不是已校准的误差。

已有两个实际初态 SHA 为：

- `098176add663f869e9e10d19dbd85ec4bf89160c96032faf4d5476a321a0eb26`
- `5521434f566fb235e0322e1fcd8ab6b1661aea997195456f51eccc910e59e479`

最终冷 reset/身份修复后的源码已重新运行三个 CPU 模式，而不是只引用修复前的结果。v04 输出 `/tmp/dh116-stereo-wp2-modes-20261111-v04` 使用最终快照 manifest `fab9451a7be5decf9e5f484aa62ecfadb8bcd623f8db05d7ce3fae71a6eade6e`。schema_test/diagnostic_fit/formal_training 分别退出 0/0/2，耗时 6.327/51.829/5.679 秒；正式模式仍因 2 个组不足而阻断。模型 SHA 仍为 `4bc8c94a1fe8095e599e122fd99ad77277aab2bbdda29df184ecaa5126befa49`，只训练 61–63 输出行，关闭 Head 未变。读回比较文件 SHA `fb6798d4455f70410291bbb7a7dd7f21f2e9d378af8fcde1cce591a970af5ad4` 与修复前一致，所有泛化/部署/成功池资格仍为 false。

拟议 cold8 最多提供 32 个候选，其中中心标签最多预计 24 行，仍须逐点/逐时钟实际准入，不能预填 24 为完成数。它的 manifest `split=test` 是已有冷试验生成器的冻结 cohort 标签，**不是**该小数据集已经拥有独立 train/validation/test 的证明。本轮不授权用这 8 个初态正式训练或校准不确定度；先审查真实数据，再另建按物理初态及 source family 连通分组的明确拆分。多回合/相邻帧不能跨 split。Geometry 其他字段、Velocity、Deformation、Contact、Slip、Force Proxy 在各自真实来源/样本门槛通过前继续关闭。

## 批准后的执行及停止条件

执行前只读核对输入文件 SHA、真实 Teacher contract、manifest/camera/物理语义和最终源码快照；运行 CLI `--check-config`。读取进程列表确认仅允许本次一个 Isaac 实例，不停止其他训练。GPU UUID、显存、环境解释器、渲染设备映射必须写入新执行计划。任一检查不符，停止并报告。

输出预定为全新 `/tmp/dh116-stereo-cold8-native-20261111-v01`；若已存在则拒绝，另行选择新独立目录，不能清空或覆盖。源码使用已冻结、SHA 绑定的独立 snapshot，并通过项目现有 `scripts/python.sh` 启动。下面只表示批准后的命令形态，**当前未执行**：

```text
scripts/python.sh scripts/collect_dh116_stereo_teacher.py --config configs/dh116_training/stereo_candidate_cold8_v1.json --teacher-config configs/dh116_training/privileged_teacher_ppo_guarded_v2.json --teacher-checkpoint /tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000001.pt --camera-config configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json --native-task-config configs/dh116_training/stereo_native_task_v1.json --candidate-only --state-reference --cold-trial-manifest /tmp/dh116-wp2-cold8-plan-20261111-v01/candidate_cold8.json --cold-minimum-per-style 4 --num-envs 2 --steps 4 --seed 116 --output /tmp/dh116-stereo-cold8-native-20261111-v01 --enable-experiment --execute-native
```

逐批复用既有 cold reset writer/readback；验证初始化匹配后清除传感历史；执行最多 4 步，任一 env 首次 terminal 立即结束该批。重复初态、时钟不齐、Teacher/标定 SHA 变化、NaN、文件冲突、原始证据不完整均判为失败，保留日志、reference、raw 和 partial candidates，不重试到“通过”、不更改原阈值。没有长训练，也没有新的奖励或执行器参数。

采集后先写文件清单与 SHA，再运行原生 raw/candidate/reference 对齐验证、cold cohort 窄准入、每视角像素来源与拒绝原因统计、按抓法逐帧 3D 点误差/覆盖率审计。只有这些结果出来才能讨论新的分割/三维匹配或状态估计训练。该时长根本不足以验证 5 秒稳定保持；Teacher、BC、DAgger、Student 原生成功资格均不因本次小采集而改变。
