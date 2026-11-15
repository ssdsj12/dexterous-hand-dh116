# C01 独立 BackendProfile 与两阶段预检

日期：2026-11-15。状态：CPU 源码实现完成，等待独立 SPEC／QUALITY 审查与 Windows 发布。工作分支 `codex/teacher-c01-profile-20261115`；实施 base `3c63f68f72a3ee9213b86bfccac8b4240a611928`。

本包新增 6 个文件；不修改旧 core、旧模型、Registry、readiness、README 或 CHANGELOG。后四类由集成方维护。没有启动 Isaac、导入 native 扩展、创建 native owner、构造策略、执行训练／采集、编译／安装或 Ubuntu push。

## 实现与来源

- `BackendProfile.load(project)` 重新调用原 `tabletop_mesh_observation_source.source_basis()` 并重开实际文件。独立 binary 为 `8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340`；source tree 为 `34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64`；patch 为 `1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41`。
- source_basis 原样提供 156 个实际文件、2 个本地来源、72+72 source 文件及 5 个差分。本包合并候选来源、2 个实际 USD、锁定 manifest、10 个 SDF、SDF summary 和 reader schema 后，共 191 个独立文件条目；inventory SHA 为 `b3e009979992354f8d37c43afc94e7001bc783ca9309710532868ad880bae757`；profile digest 为 `18f7de208e771d834190874f8c4d37cc463a5bac3e7794b4c1b6f5dfdffe0c0d`。
- Profile 使用 frozen/slots dataclass；嵌套 inventory 为 tuple[FileIdentity]，各条目也 frozen，没有可变 dict 成员。禁止直接构造；`from_sources(binary_sha256=..., source_inventory=...)` 是严格的实际 loader wrapper，0 SHA、旧 2fea、缺失／篡改 inventory 不能重贴新 identity。`verify()` 每次重读全部实际字节。
- 新常量：`MODEL_SCHEMA_VERSION=dh116-native-context-teacher-tanh-v1`、`INPUT_SCHEMA_VERSION=dh116-native-context-teacher-input-v1`、`CONTEXT_SCHEMA_VERSION=dh116-native-teacher-context50-v1`。输入精确 6 键：common371/common_valid、rod_nodes/rod_mask、task_context/task_context_valid。原账本 371 槽保持 66 selected + 305 NaN/false；后 305 槽不作为新网络的阻断前置。
- `required_capabilities(profile)` 返回 frozenset。smoke 要求 backend/reset/common66/rod41x13/current parameters、同钟固定41材料集合 distance、实际 normalized request24／SDK targets、原 guards／health、context、8 SDK/control 和有限预算；不要求 Hold、slip、完整 contact success。full 明确增加真实接触／对向／承托／离桌／无外援／材料与切向滑移／Hold 和连续历史。
- `static_preflight(config, *, project, backend_profile=None)` 只核源码、schema、数值 reward loader、同源 gamma、固定 manifest、预算授权和真实 kernel singleton lock，不要求先创建 owner／live proof。C03 module 存在时调用其 `tabletop_training_reward.load_config(path, ppo_gamma=...)`，没有复制奖励数学；model/context 缺失分别列 implementation_pending。
- `load_reset_manifest()` 只接收完整 SHA 的实际文件绑定，核原始 receipt artifact inventory，重读小型 mesh Reset／numeric 原件，用既有 `validate_numeric_reset_audit(audit, *, reset_receipt)` 核 typed state，逐值比较 manifest seed/rod/mesh。Pinch 原件不能借给 Tripod／Power，pose_preview_only 拒收。manifest 为禁止直接构造的冻结 bytes identity，live 前再次重读。
- `retained_reset_source()` 复用公开钉住 SHA `fcf8f92c0b2838e72a6fe6d026cea61c54759b30350b607c0da6b26db6db6a15` 的 R08 sealed report，核 216 项 replay source SHA；未重跑 R08 CLI、未解析 621MB physical stream。已有 producer_profile 仅用于重放小型历史 numeric artifact 的原 producer 绝对路径来源。原 receipt `6bc124...adc4ef` 仍 failed，旧 native 234=2+232／0 PPO；2 个 Reset snapshot／23 invariants（10 typed arrays）不授予新在线 Reset 资格。
- `LiveResetContract`／`live_preflight(owner, contract, *, backend_profile, manifest)` 要求 owner 共享同一 Profile 实例及 sim/rod/bindings handles。本次 numeric audit 与实际 rope id／Reset UID 同钟绑定；重新读取 SDK q/qd/targets/body poses、CAI rod p/v/director、mesh current/target/solver 和 mounted source，对照实际 Reset 与 immutable manifest。当前 parameter raw packet 与 fresh getter 精确匹配；使用既有 bilateral Reset distance 与 actual health getter。返回本次 identity、source digest 和 live_reset_verified，不授予 native／Teacher admission。调用顺序保留 static→创建 owner→实际 Reset/live→策略／trainer／reward。
- CLI 使用 checkout 外 exclusive 私有输出。损坏 schema／覆盖输出 exit 2；合法但缺源 exit 3；全部静态条件满足 exit 0。输出 components 含 source_complete/applicable/valid/reader/source_sha256/blocked_reason/release_condition；永远记录 new_native_steps=0。

## 实际 CPU 验证

私有证据目录：

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c01-native-teacher-20261115-p0eZyW`

命令统一通过：

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_native_teacher_profile.py tests/test_native_teacher_preflight.py tests/test_mesh_observation_profile.py tests/test_mesh_reset_numeric_state.py tests/test_dh116_teacher_mounted_wrist_source.py tests/test_r0_health_separation.py
```

最终实际结果：**159 passed，0 failed**；其中本包 54 个测试。CPU fixture 明确限于协议／reader／kernel tempfile lock，不授予 native qualification。

RED 保存：缺实现断言 3 failed；新增 mesh/mounted reader 接口 10 failed；直接伪造 manifest 构造器 1 failed。GREEN 覆盖深层 mutation、未知／旧 binary、actual source tamper、no-native meta_path、smoke/full 差异、数值预算／独立授权／真实临时 kernel lock、原始小型 Reset artifact manifest 正例与篡改、SDK／mesh／时钟／跨 Reset 源、gamma mismatch、fake bool 和 CLI exit/exclusive 行为。

| 私有证据 | SHA256 |
|---|---|
| red.log | 2f0dc74b86e4480984d13ab0a81f52a0577b69ab0aecce87fdaa043be7504f9f |
| live-red.log | 4bf8e4d34aa98c15cfc7fa9128977f5857273262739ebe44b996619332cd1624 |
| manifest-constructor-red.log | 4ac692798a1b106ba60da79b216b7fd9e1ec60f4b900c51d71f741990f33c48d |
| green-final.log | 6626c69c39035095bbe7ccd3301f431a89124a5af411e5c43e79540d9b31b5cd |
| protection.log | 007c685118b53bfd4b6f7bb8bed0354e379437b44487a3bd9663a33ba9372bf5 |
| pinch-preflight-final.json | e3b5b200806e7ae9788383e054940578cb8b36569e8f5f9f2965e07c6efc3b48 |
| tripod-preflight-final.json | 2748361acc7e382c8c5427382ad1eb173147f72326223956a5c20f16e4707029 |
| power-preflight-final.json | 1f357faa633f52c1e214c8bed7de82a65c7ea7b0597a4de7b7cfcc1078a6ecd0 |
| pinch-full-preflight-final.json | 713007870f931515c7b8aa5a69af395688f3afd81ca307677cd5563cef272863 |

三种 smoke 和 Pinch full CLI 均实际 exit 3，static_ready=false、live_reset_verified=false、new_native_steps=0。Pinch backend／当前 reader 源／R08 retained source 有效；当前 default 没有 resolved_reset_manifest、独立预算／授权／resource reservation，C02 model/context 未实现。C03 不在本 base，具体记 reward_implementation_pending；其已批准数值设计不称缺失。Tripod／Power 另缺独立 distance reader，retained Pinch evidence 不适用。full 另列其真实 contact/slip/hold 来源待实现。

`Repository(Path.cwd()).verify_protection()` 已实际核对 **528 源文件 + 2 历史 checkpoint**。`git diff --check` 通过。

## 限制与下一步

本包证明 CPU source identity、preflight 契约与真实 Python reader 验证逻辑。whole live owner 组合仍待 C04 实际 owner 接入；没有在线 Reset、真实 PPO transition／optimizer update／checkpoint 或抓取成功。C02/C03/C04/C05 与 C06 独立新预算／授权仍按原计划分别推进；本代码包没有提供 native execution 授权。集成方审查后进行同 SHA Windows 发布，PR #2 保持 Draft。


## QUALITY P1 桥接修复（发布前，2026-11-15）

独立 QUALITY 指出实际 bridge 会在 BilateralMeasuredPinchApproach.capture_reset → MeasuredPinchApproach.capture_reset → scripts.replay_dh116_native_parameter_r0.check_radius_packet 失败。既有 observation source_scope() 已绑定 parameter、health、builder，但该 radius replay 模块另有独立旧 BACKEND_SHA=2fea8036...；合法新 88170101... raw packet 因而报 ValueError("raw radius/provenance differs")。已通过规范 CPU wrapper 和真实 geometry constructor 复现，原始 failed receipt 不改写。

修复仅更新既有 C01 新增文件。新增 profile_reader_scope(backend_profile) 要求 exact、实际 source-loaded BackendProfile，每次进入先重读验证所有实际文件，组合既有 observation scope，再临时绑定上述 radius 模块的独立 SHA。未知调用方 SHA、旧 packet、外来部分绑定均不能授权；正常退出、异常、同线程嵌套及既有 source scope 组合均恢复进入前绑定。live_preflight 在 owner/Profile/runtime/manifest 核验后使用此 scope，static → 创建 owner → live Reset → policy 的顺序保持。旧 radius reader、旧物理／recipe／模型和全局默认源码均未修改，scope 外仍为旧 2fea API。

新增 radius reader 本身的固定来源 SHA 70c3b7b8b999564edd266a093fe4ef2df393d00a8b8551235b2fb37acff05e0c，来自原 R08 sealed inventory。修复后的当前 Profile 为 **192** 个文件，inventory SHA c1266bbf6b78cf8566f0537f630d65d3e44f3371b8fb6dc7159da4cd8f1055a1，profile digest 76e39e27d43c70a7e65454814d214ab403e7e971de0988606ce53c38c098a786；上文 191 项与旧 digest 是首版提交身份。binary／source tree／patch 与 schema 常量保持原身份。

实际正例直接使用 BilateralMeasuredPinchApproach(ROOT)、真实 capture_reset、真实 check_radius_packet 和既有 locked-SDF 距离数学。只由原 receipt SHA 6bc124...adc4ef inventory 重读 director_reset.json（51,539 bytes，SHA 6769069d488d1e18e96bb1396c520b8256580dbda0eaa2d5e36ec211073709df）和 mesh_reset_state.json（52,707,334 bytes，SHA 59332d1a2ded777806c3078e9b16cf2a6b8c3f112855164d23a3c3c084e14a6b）取得新 packet／Reset；没有 mock 这些方法、geometry 或 SDF。规范 wrapper 提供 CPU USD 库路径；没有启动 Isaac 或加载 RopeSim native owner。没有解析本 receipt 的 621,194,597-byte physical stream。

本次独立私有证据目录：

/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c01-bridge-fix-20261115-0gvlwwww

RED：真实未扩展 observation scope 的完整桥接 exit 1，精确复现上述 ValueError；原 C01 源码运行新增契约测试为 12 failed／54 deselected（exit 1）；将修复中的唯一 radius assignment 暂时移除后，真实完整桥接正例精确因相同 ValueError 失败 1 项（exit 1），随即恢复。GREEN：本包两份测试 66 passed（原 54 + 新 12），exit 0。最终 CPU 回归命令：

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_native_teacher_profile.py tests/test_native_teacher_preflight.py tests/test_mesh_observation_profile.py tests/test_mesh_reset_numeric_state.py tests/test_dh116_teacher_mounted_wrist_source.py tests/test_r0_health_separation.py tests/test_dh116_native_parameter_r0_replay.py tests/test_dh116_r0_measured_approach.py tests/test_dh116_r0_reset_approach.py tests/test_dh116_bilateral_r0_approach.py tests/test_dh116_parameter_packet_validation.py

实际结果 **252 passed、0 failed，78.61s，exit 0**。覆盖新参数→实际双手 Reset distance、未知／旧 packet 拒收、正常／异常／嵌套恢复、foreign alias 保留拒收、进入 scope 前实际 binary／reader 字节重新验证，以及原 radius／measured／Reset／bilateral／parameter packet／source-scope／health 回归。

| 本次私有证据 | SHA256 |
|---|---|
| quality-real-bridge-repro.log | f09771921f768bc19e94eddd67a44aebb7e02dc3bf601fec5d9b734b37eacaf9 |
| real-bridge-red.log | d0f31d2d7144a1940f8c7f1d7c3280d966463ba9c742129c002b58d036575528 |
| bridge-behavior-red.log | 5950ec9c63e5cad1f5e1dab8612d006e79c3ca7b45ecf5d335ea9f5d0c290687 |
| c01-green.log | 0187778da3f8634b6988686c2fcf8b3defcf664c40c19f0ca609eccf4e300671 |
| green-regression.log | e24c9aed6d7f8862b3479b441f60d63d6e45d9d0ff0836129d72233b40696589 |
| protection.log | 380408fa481edf8326f494c3c26d75ecfca7deaa77bc24691a8b438599b70ed1 |
| prior-evidence-preserved.log | 11d9e8b1d967c50f3bcfa8c45b236793c2961ccbd68590c4d63dec4c81c878a0 |

每条测试／复现的命令与真实 exit 另存同目录 *-command.json。原 9 份证据 SHA 已逐项重读核对，全部保留。Repository(Path.cwd()).verify_protection() fresh exit 0，核对 528 源文件 + 2 历史 checkpoint；相对实施 base 的累计改动仍严格为原 6 个新增文件，旧 reader／recipe／模型／codeSource 未修改，git diff --check 通过。

本修复只证明 CPU source bridge 和 reader 数学通过。C04 whole live owner 组合仍待接入，没有在线 Reset、native qualification、PPO transition／update／checkpoint 或物理抓取成功。原 failed receipt、历史 native 234=2+232／0 PPO 的结论保持；等待同一 QUALITY 审查者独立复核，再由 Windows 发布。
