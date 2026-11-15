# Fixed S0 Context51 v2：W3 Task 3 CPU 实施报告

日期：2026-11-15。基线：de02ae7a0bef6c87185fc698f6cd3a66db90b2e3。
独立工作树：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/v2-context51-20261115；
任务分支：codex/v2-context51-20261115。本报告尚待 root 对精确提交进行 SPEC、QUALITY 和正式发布。

## 范围与设计依据

执行已批准的独立 v2 phase/context 计划 Task 3，不重复 Task 1–2。
原方案 DOCX SHA256 为 1ea4da5342de9a64e5adb3ef5c0a50482d2158bd9a9d3e7f5fab7f6b04e318dd。

只新增三个文件：

- src/go2_pvcnn/control/dh116/tabletop_native_teacher_context_v2.py；
- tests/test_native_teacher_context_v2.py；
- 本报告。

保留旧 context50、原 Teacher/Student、Tanh Gaussian/PPO、371D 原合同、24D 动作、
物理/材料/执行器/保护与全部历史 Checkpoint。未新增模型、controller_state、相机、
训练入口、W4 source adapter、长 Reset 或 native worker。本包没有 App/Isaac/GPU/native 运行，
没有安装、下载、原生训练或 push。共享 Registry/readiness/README/CHANGELOG 由 root 集成。

## 实际 API 与数学行为

新模式为 fixed_s0_v2，独立于旧 approach_wiring_smoke_v1/flexible_train_r1_full。
它不是生产资格，也不是配置 profile=fixed_s0_pinch_v2 的替代。

- CONTEXT_FIELDS_V2 严格 51 项。0:4 为 Close/PreliftHold/Lift/LiftHold；
  4:50 字段名称、单位、归一化与旧表一致；50 为 prelift_stable_duration。
- CONTEXT_SCHEMA_SHA256_V2 覆盖独立版本、完整字段表和 phase 源代码 SHA。
  phase_contract_sha256 来源为实际已导入的独立 PhaseOwnerV2 模块字节。
- ContextMetadataV2 保留原 sidecar 字段，并增加 phase_contract_sha256、
  fixed_s0_manifest_sha256、prelift_scale_s。scale 必须显式正有限。
  metadata 不新增 episode_cap_s/external_cap_s；它们仍由构造参数参与实际预算计算。
- TaskContextPacketV2.owned() 分别 clone values/mask 并 deepcopy metadata。
- validate_context_v2 严格拒绝旧 packet/metadata、旧宽度、旧 mode、异时钟/身份、
  原始 NaN 被填零、有效项 NaN、缺失物理字段、错误 applicability/NA reason 或源 schema。
- encode_context_v2 严格接收 float32[N,51] 与同设备 bool mask，输出 N×102。
  未知原始值保持 NaN/false；只在网络编码分支内转零，梯度仅通过有效值。
- TaskContextOwnerV2 迁移了原已验证的 history、固定 d0/初高、材料锚、预算、
  event latch 和 pre/post copy-on-write 逻辑。freeze_pre_action、
  advance_physical_endpoint、commit_post_context 分开。
- reset 从实际 Action24HistoryOwner 的 Reset receipt 读取同钟身份与原请求零种子；
  phase 必须是同一个 exact PhaseOwnerV2，初始 Close 且两个时钟为零。
- phase owner 同实例、同 episode/tick、已捕获旧 receipt 突变检查贯穿推进。
  不读取 caller phase_receipt；不会替换原 FlexibleGraspEpisode.step 或 reset。
- slot50 = phase prelift_stable_s / 显式 prelift_scale_s；
  slot7 = phase final_hold_s / 5。二者来源与时钟独立。旧 slot6 接触持续历史仍独立。
  0.25s 只是候选，最终 5s、30mm、0.5mm 标准没有改变。
- post 返回值的修改不改变 pending；commit 核原 hash 且只能一次。
  正例执行实际 CPU history settlement → 原 trainer.record_step → context commit。
  该 owner 不另外冒充 trainer/storage 的原生证据。
- close 清除借用 phase/anchor 源和 pending 引用，不调用任何物理停止或 native 释放，
  不修改源 task.step；close 后不能 commit。

## 真实来源门与 CPU fixture 边界

W4 的真实 SourceAdapterV2/PhaseSourceWitnessV2 尚未实现或资格化，本包不创建同名空接口。
任何 owner_source_unqualified 生产构造或手工换 scope 的 packet 都以
full_v2_phase_source_adapter_missing 拒绝。任意 caller object 也不能冒充 adapter。
只有显式 cpu_fixture 可用；native_qualified 永远为 false。

CPU fixture 沿用真实 Python 控制链/历史 owner 和原 PPO 方法，但 SDK/rod 为 CPU doubles，
其接触布尔、phase 样本、物理字段、固定 S0 字符串均不授予物理来源资格。
缺 measurement_snapshot/phase_source_witness 的生产路径不会通过。
当前缺的是实际 adapter，因此还不能声称已实现 adapter 存在时的 witness 校验；
后续必须由 W4 的实际 source 实装后接入，不能让本报告替代该工作。

CPU fixture 在既有 state 键外显式使用 fixed_s0_manifest_sha256、measurement_snapshot、
phase_source_witness；为独立检查旧材料锚，允许 state["anchor_owner"] 借用实际
FlexibleGraspEpisode 原方法和同 epoch/clock/rest 的实例。state 自报的 station/UID 不能建锚。
首次原 task 合格接触后从其 material_anchor/slip_references 读取固定参考；
后续失接触保留原 station/UID，禁止 caller 重锚；已存在的左右 material/tangential reference UID 全部逐名锁存核验，不只核材料 station。此借用只用于 CPU 回归，不是 W4 native 接线。

旧 371D 中 305 个未知槽及 rod41×13 模型处理未在本包重写或重新资格化；
原模型不变，新六键 model/profile 属于下一 Task 4。

## S0 身份的准确状态

默认 candidate_config 的 resolved_reset_manifest 仍为 null；本包没有替换它。
主代理另包从保留的实际 R08 三来源解析并通过原 loader 的私有 manifest，
本包只读核对其 16537 字节与 SHA：
a39de62de8ea97b3b154af2f910ba9dc097ccf64f9d650dad627ee2d8516cdce。
这说明存在可解析的保留 Reset 身份，不等于新 v2 source、稳定接触或 Teacher 成功资格。
CPU fixture 的 "1"×64 仅为显式测试身份，不冒充上述实际文件。

## TDD、失败证据与回归

私有日志目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/v2-context51-20261115

| 日志 | 实际结果/根因 |
| --- | --- |
| task3-red.log | 3 failed，新 v2 模块不存在 |
| task3-first-green.log | 1 passed/2 failed，迁移时误命名 MODES_V2_V2；已修 |
| task3-green-initial.log | 3 passed |
| task3-boundaries-red.log | 47 passed/2 failed，伪造 scope 可绕 constructor；close 后可 commit |
| task3-boundaries-green.log | 49 passed，两个边界已在新模块修复 |
| task3-encoder-red.log | 2 passed/1 failed，原 generic encoder 可接受非51宽；新版本改严格宽度 |
| task3-borrow-release-red.log | 1 failed，close 后仍保留借用 anchor owner；已清引用 |
| task3-record-before-commit-green.log | 178 passed，36.88s：52新 context51 + 59旧 context + 67 phase |

最后组合命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_context_v2.py tests/test_native_teacher_context.py tests/test_native_phase_contract_v2.py
~~~

首次178项组合日志 SHA256：
a5112489b86bc3ad93d3c76275382ab5b0ff0a62ba33aea0abba797f21a8c6a7

提交前再次执行原 Repository.verify_protection：528 个保护源码与 2 个历史 Checkpoint 均通过。
仅暂存三条明确文件路径，执行 diff --check、基线范围和 clean 状态核验。
原生步数 0、原生 PPO 更新 0、抓取成功次数未验，不将 CPU doubles 计为真实运行。

## 下一步与未解决项

本包仅完成 Task 3 软件合同，仍需精确 SHA 的独立 SPEC/QUALITY 和 root 回归发布。
Task 4 的新 profile/model 六键身份、W4 同 owner 实际 snapshot/witness、
W5 长 clock/guarded Reset、新 checkpoint/admission/collector 与 C06 actual provider 分别推进。
真实阶段稳定性、三种抓法、原生保存恢复和严格 Lift/Hold/Slip 成功均未被本包证明。


## SPEC 复查发现与最小修复

未发布的候选 0a4d27d44522fe7fd93deb9347d059f72bb9e222 在独立 SPEC 复查失败：
原 task 同一实例通过 drive 逐物理端点推进，第二个控制窗结束后仅注入右切向 UID 损坏，
材料 station/UID 均不变，context 原先会接受并提交 foreign-tangent。

根因为 advance 只比材料 station/UID，_anchor_state 重构了完整 refs 却未核此前切向 UID。
修复仅在独立 v2 接受 proposed_refs 之前增加逐名比对，锁存全部已存在 reference UID。
未改旧 task、v1、材料参数、关节保护、成功门槛或预算；首次未建锚的 NA 规则保持。

原 SPEC probe 已复制到本包私有目录并先重新验证 RED：
spec-tangent-lock-confirmed-red.log 为 exit1，打印 accepted/committed foreign-tangent。
最小修复后 spec-tangent-lock-green.log exit0，明确拒绝 changed reference UID；
probe 内的 528+2 保护核验亦通过。

新增5例使用原 task 每端点 drive、原 history/控制与 trainer.record_step：
左/右切向 UID、左/右材料 UID和双侧材料 UID。
task3-all-reference-lock-red.log 为 2 failed/3 passed，正是左右切向此前漏检；
task3-all-reference-lock-green.log 为 5 passed/52 deselected，10.96s。

每种拒收均核 pending/pending hash/collection tokens、anchors/UIDs/refs、current/pre、
phase source/receipt/seen reference/value 完整不变。
只纠正源 UID 后，同一 settlement/token 仍可成功推进、record/commit，未重新执行物理动作。
该修复仍仅 CPU 契约，native 步数与原生更新数均为0。

修复后的最终相关组合：183 passed，42.12s（57新 context51 + 59旧 context + 67 phase）。
日志 task3-reference-lock-final-green.log SHA256：
4ba2c55f0294b834395b5c550635c10a9416788d4cd6ebfa5a05c668b38787da
