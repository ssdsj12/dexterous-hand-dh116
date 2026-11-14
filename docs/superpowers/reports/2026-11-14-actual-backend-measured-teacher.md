# 实际 backend 的独立 measured66 Teacher 输入适配（2026-11-14）

本包关闭的是一个已复现的 **CPU 输入身份缺口**：原 MeasuredCommonTeacher
绑定 c47942…，真实 Pinch/Tripod 杆读回来自 2fea8036…，原 pack 正确拒绝。
新适配器明确绑定实际 backend、完整 source lock、原始参数 snapshot 与分抓法证据，
复用原网络、杆特征、24D、PPO、checkpoint 验证/序列化算法。

本包 **0 新 SDK/solver 步、0 optimizer updates、0 native、0 下载**。
旧三份 Teacher 配置、旧模型/Checkpoint 和默认入口未改。
不授予 mesh Reset 同步、奖励、接触成功或 Teacher 训练资格。

基线 06b11c3c95ff1c2937eb41e861b6fbffed944e32；
工作树 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-actual-backend-20261114；
分支 codex/teacher-actual-backend-20261114。

## 最小实现与接口

- tabletop_actual_backend_teacher.py：
  load_adapter_config 在创建模型前核 source/receipt/artifact/config；
  read_first_boundary 只读绑定账本的首个 pre-action；
  ActualBackendMeasuredTeacher 继承 MeasuredCommonTeacher；
  pack_actual(pre_action, raw_rod, parameters) 重建来源后调用原 pack_measured。
  直接调用新对象的 pack_measured 会拒绝，要求完整来源包。
- 三份 actual_measured_teacher_{pinch,tripod,power}_v1.json：
  各自绑定原 base config SHA，使用独立 namespace；没有覆盖原配置。
- tabletop_actual_backend_checkpoint.py：
  新 version、model/source/parameter identity，复用旧 measured checkpoint 函数代码；
  原 writer/loader、全局字典及旧格式不变，新旧 namespace 双向拒绝。
- tests/test_dh116_actual_backend_teacher.py：30 项新测试。
- 本报告。

源码位于 src/go2_pvcnn/control/dh116/，配置位于 configs/dh116_training/。

~~~python
from pathlib import Path
from go2_pvcnn.control.dh116.tabletop_actual_backend_teacher import (
    load_adapter_config, read_first_boundary, ActualBackendMeasuredTeacher,
)
root = Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-actual-backend-20261114")
binding = load_adapter_config(
    root / "configs/dh116_training/actual_measured_teacher_pinch_v1.json", root)
model = ActualBackendMeasuredTeacher(binding)
observations = model.pack_actual(*read_first_boundary(binding))
# 未构造训练器、优化器或原生 owner；不执行 update。
~~~

## 实际 CPU 结果及分抓法边界

| 抓法 | 绑定的真实来源 | 本包结果 | 未取得的资格 |
| --- | --- | --- | --- |
| Pinch | approach 234 SDK / 29 窗的已归档诊断；outer/inferior exit0，169.025s | 首态 pack 通过 | mesh Reset、正式奖励、训练、正向抓取 |
| Tripod | 原 199 SDK 失败执行的独立首态；exit1，left_tripod_failure_prefix | 首态 pack 通过；失败状态原样保留 | 不能把左 Tripod/右参考手称为双侧 Tripod 准入 |
| Power | 无实际源账本；evidence及parameter_static_sha256均null | 创建 model/optimizer 前拒绝 | 不能借 Pinch 或 Tripod 的输入结果通过 |

两份通过的 CPU 输入均保持原索引
0–23、78–87、153–184，共 **66/371**；其它 **305 项仍 NaN、valid=false**。
原 owned 浮点账本未改；网络副本显式转换为其既有 float32 dtype。
杆使用同一真实 41 节点、40 rest segments，原 encoder 输出 [1,41,13]，
actor 仍为 24D；没有圆柱姿态、固定底座占位或补造 contact/slip。

实际首态核验中两个模型权重均前后完全相同，原输入摘要前后完全相同。
两侧采用同一个显式 CPU seed，所以初始权重 SHA 相同不代表 checkpoint 身份相同；
Pinch 与 Tripod 的 config、receipt、grasp、namespace、model identity 明确不同。

| 项目 | Pinch | Tripod |
| --- | ---: | ---: |
| 加载配置/完整原文件身份核验 | 0.423554s | 0.333598s |
| 一次 pack_actual | 0.054557s | 0.052983s |
| 有效 / 未知字段 | 66 / 305 | 66 / 305 |
| 新 optimizer steps | 0 | 0 |

pack 每次重新核验完整 compiled source lock、依赖文件和配置。
当前约53–55ms的单次测量适合 CPU 验证，**不能宣传为可直接满足60Hz的在线 PPO入口**。
后续需要独立设计有生命周期边界的 session source binding，再验证实时预算；
本包不缓存、跳过源码核验或改物理控制频率。

## Source、参数和时钟校验

实际 backend：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu/build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so
SHA 2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec。

复用 NativeParameterEndpoints.source_basis，验证72个编译源码文件、完整动态库、
原 getter delta 和 zero-step probe 的源/结果。source_tree SHA：
3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472。
本包不会 import 该动态库、启动 CUDA 或调用 solver。

只允许原配置文件路径在显式 source_project 下定位；实际源 SHA 不变，
没有任意后缀匹配或忽略缺失路径。小型 metadata 校验与解析使用同一份 bytes。
实际 native source 项、参数和旧保护依赖全部进入新 model/checkpoint identity。

参数使用原 check_radius_packet / validate_snapshot：
检查 raw/derived fields、正 rest、EA/EI 一致性、node identity、radius provenance、
SDK/rod/solver clock和同一边界。除动态 time_s 外的原始参数完整摘要固定为
36f26367d66fa4ddff7e4a07ce3525cabf6141c2130ccaf5cecec0621ca6f65e；
参数改变、固定节点注入、gravity/radius/rest 等变化均不能静默接受。

common 来源由 validate_mounted_observation 从原 source_packets 重建。
raw rod 通过 RodBoundaryState 在纯 owned clock 下重新生成；
保存的 rod boundary 必须与重建包完全一致，不能只检查 shape。
随后仍由原 measured model 和 rod encoder 验证 mask、有限值、tangent、strain、
相同 env/episode/reset/rod/SDK/tick/time。这里是已归档事实的一致性校验，
不是再次证明 live SDK/mesh Reset 时序。

## Checkpoint 兼容性

新格式 dh116-actual-backend-measured-checkpoint-cpu-v1。
旧 measured writer 的签名仅为 (path, trainer)，没有 identity/version 注入接口；
本包不编辑旧 checkpoint 模块，因此用 FunctionType 复用其函数代码，
在独立 globals 副本中仅替换 VERSION 与 checkpoint_identity。

该方式未重写 Adam、RNG、拓扑、有限性、clean boundary、独占写入或恢复算法。
保留函数 closure、positional defaults、keyword defaults 和 annotations；
测试检查原 module globals 不变、函数 code 对象直接复用。
这仍是对已冻结实现的窄适配；父实现变更会改变依赖 SHA，需要重新测试，
不能承诺对任意未来函数内部改动自动兼容。

CPU checkpoint 测试只保存/加载 **update0** 状态：
新→旧 reader、旧→新 reader、跨抓法、参数身份篡改、runtime backend 属性漂移
均在恢复前拒绝。实际旧模型文件没有加载或修改，临时测试 checkpoint 不进入 Git。
本包没有正式训练 checkpoint 或 resume 迁移。

## 真实来源与证明

Pinch receipt：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-approach-r0-native/96fc30001e0d26e7421b683fc0e18ae9240054e3b6fcffb3e010aa6d6d0d70b3/attempt-wy27jh8g/receipt.json
SHA 9b0a8eb4f37ce20e27d5054a4d95043e21fb99febb00128a878fffa739608ee6。
其 physical.jsonl SHA 4e9b42b1fc96119888cce34d4977d2fd8720b4385409dda45a10f59fcbb8b83b；
director_reset.json SHA c524f8c8a142da5e3909464146d991230ead8474aad428bb67f12a94552631ab。

Tripod receipt：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-tripod-owned-contact-native/71f0fe738be3559c02c2a32ca5f1c2702f98cdb46ed403f3b2339fe732fa536a/attempt-r_cnsg84/receipt.json
SHA 9bb31008160da84e31c1e089eed925ba658d40c82a830173a958e21491ecf0eb。
失败 receipt 没有成功 artifact map，物理/参数文件分别显式锁定：
physical.jsonl SHA fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33；
director_reset.json SHA 7ef7b21981c584b09e6e2a3860a64b229ffe44eec63938bccafc08cfce23c217。
失败账本未补成功标记，未丢弃其 terminal。

独立 CPU proof：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-actual-backend-20261114/actual-input-pack.json
SHA fe1cb66bf043c6a42e8dbaaf18c43e888b9f1463bc41ee052975683eeb8ca213。
该命令只 pack 首态：0 forward、0 optimizer、0 SDK/solver；测试中的独立 forward 另算 CPU 功能检查。

## TDD 与回归

原 MeasuredCommonTeacher 的真实首态 backend 拒绝先作为明确测试保留。
新模块未实现时5 failed/19 setup error/1 passed；源码身份重封与backend属性漂移
再先复现2 failed；delegate keyword defaults 先复现1 failed，再修复。

最终 **30新 +62既有 =92 passed，1 deselected，10.37秒**。
被排除的唯一旧测试会调用一次 CPU fixture optimizer update；
本包包括所有本次新测试均不执行 optimizer.step。

~~~bash
cd /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-actual-backend-20261114
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
tests/test_dh116_actual_backend_teacher.py \
tests/test_dh116_deformable_privileged_teacher.py \
tests/test_dh116_flexible_measured_common_teacher.py \
tests/test_dh116_teacher_mounted_wrist_source.py \
tests/test_dh116_flexible_rod_boundary.py \
tests/test_dh116_native_parameter_endpoints.py -k 'not original_ppo_update'
~~~

同 evidence 目录的 regression-handoff.log SHA：
b260a75c1e742b20efc7507d48692ecda53b4f3b7ce22a85e53a5b82ada9995d。
red-binding.log SHA 23b903bfe679ba04a0c4a64c68412699170d5849d314ab83f7be8b4768c7e7b8；
red-delegation.log SHA aab34b411692c971dc638751b586b5cd0bd51859870c730df75a5918dc9aa2cc。
root只读审查提出完整每步核验成本与delegate兼容要求，已补测试/实际测量；
其他未完成审查不计为通过。

源码 SHA：
teacher 1bc486e17b111c1cb3d03eab4b2253af298920a77914ea6f356cea325fb679a4；
checkpoint b4a282e415afd8ceccd75ad9ace5f19c091ffbd202c5ddf49f0ea4b47a200f7c。

## 仍阻止1-update无相机PPO的事项

1. **新发现的 mesh Reset 同步风险仍为硬阻断。** 原 runtime reset调用
   set_mesh_poses(..., teleport=True)，首step再次传False。
   root独立审计指出该排队teleport可能被首step覆盖，首solver从构造add_mesh姿态插值；
   左thumb before_hands→hands约65mm。此数值由root独立审计提供，本包未做新的原生重演。
   历史outer PASS不能证明真实mesh Reset同步。所有新config/model显式
   reset_mesh_pose_qualified=false；本包不修Reset、不改原账本、不借Pinch通过授予资格。
2. **正式R0未定义。** 全configs检索显示原 flexible_dual_hand_grasp_v1 的10项
   coefficients及coefficient_source仍null；Applicable/NecessaryHold/HealthSeparated
   配方均diagnostic-only，training_reward仍None。ordinal诊断权重和disabled shaping
   candidate不是批准的训练配置。既有FlexibleR0ControlStep只接受frozen_eval/updates0；
   不能将diagnostic_reward直接改名作为训练reward。
3. **来源有条件适用，不应先要求成功抓取。** 已证实无双侧必要接触的Approach，
   可以按已有Applicable/NecessaryHold规则把不适用的贡献记0，raw仍NA；
   未知接触、进入双侧接触后所需材料滑移/Hold来源仍不能补0。
   新HealthSeparated已分开数值健康和Hold运动，但实际last_error历史未持久化，
   不能把低速度、无guard或未Hold当成失稳/稳定事实。
4. **还没有实际无相机on-policy collector。** 可复用DualHandPhysicalEpisode的
   reset/pre_action_state/control、MeasuredApproachSession、已验证sealed生命周期，
   再连接原DeformablePPOTrainer.collect_step/record_step/update及新checkpoint。
   现有privileged CLI仍走旧刚体bridge并无条件bind_camera，不能直接当柔性无相机runner。
   需保持同钟action24、实际执行回读、每8子步exactly-once reward、terminal先落盘、
   truncated同pre-reset endpoint bootstrap、GAE不跨reset；不重复已有算法。
5. **精确预算/来源仍需独立准入。** 当前native实例是一环境；原smoke示例是2环境×32控制步，
   setup/solver来源仍null。例子不等于已批准的一次训练运行；不得重复已消费234步配方。

Pinch现有测量最完整，但不因正向接触/30mm/5s/0.5mm尚未成功而永久禁止普通探索；
这些是正向成功验收，不能替代上述来源/Reset/数值健康/标量reward硬门。
Tripod的finger22保护终止应作为失败事实保留，不能变成功或绕过安全门；
Power只有静态来源，仍无可复用实际source/Reset。相机和WireSeg不阻断无相机准备。
