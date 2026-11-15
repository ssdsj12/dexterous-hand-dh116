# DH116 Fixed S0 Phase and Context v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 建立固定S0四阶段执行合同、context51与独立六键教师身份，冻结旧v1。

**Architecture:** 新phase owner只消费同钟typed测量资格，旧FlexibleGraspEpisode独立评价。复制旧context/model已验证非阶段逻辑到独立v2模块，修改新schema/packet/phase receipt输入，绝不改旧metadata绕过exact-type。无W4真实来源时只验CPU合同。

**Tech Stack:** Python3.10、dataclasses、NumPy、Torch CPU、pytest、现有scripts/python.sh；不安装依赖。

---

基线731a97298c2a1498f9fddd7a2ce7f35ce3dfa0b9；设计见docs/superpowers/specs/2026-11-15-dh116-fixed-s0-v2-design.md。实施另建Ubuntu worktree，当前docs工作区不写生产代码。以下“CPU前缀”均是完整命令开头，后接指定测试路径：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
~~~

## Task 1：固定S0与新phase类型

**Files:** Create src/go2_pvcnn/control/dh116/tabletop_native_phase_contract_v2.py；Test/create tests/test_native_phase_contract_v2.py。只读tabletop_native_teacher_preflight.py:125–195、tabletop_native_teacher_context.py:343–354、tabletop_deformable_grasp_task.py:384–414,512–519；后文模块名均位于src/go2_pvcnn/control/dh116/。

- [ ] 只读核对resolved S0：candidate_config、load_reset_manifest、ResolvedResetManifest.data/verify。config若有binding，读取其path/SHA并用loader验证三原始来源；若没有则记录fixed_s0_manifest_unresolved，继续CPU但拒绝实际factory，不能造manifest或根据照片拼S0。

~~~python
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config, load_reset_manifest
c = candidate_config(style="pinch")
assert c["resolved_reset_manifest"] is None
try:
    load_reset_manifest(c)
except ValueError as error:
    assert "resolved_reset_manifest_source_missing" in str(error)
else:
    raise AssertionError("unresolved S0 admitted")
~~~

- [ ] 创建实际失败测试：

~~~python
def test_reset_is_close_old_phase_schema_unchanged():
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import CONTEXT_FIELDS
    p = PhaseOwnerV2(prelift_stable_s=0.25)
    r = p.reset(episode_uid="fixture:e1", physics_tick=2)
    assert (r.phase, r.prelift_stable_s, r.final_hold_s) == ("Close",0.0,0.0)
    assert [x["name"] for x in CONTEXT_FIELDS[:4]] == [
        "phase_one_hot.Approach","phase_one_hot.Close",
        "phase_one_hot.Lift","phase_one_hot.Hold"]
~~~

- [ ] 运行CPU前缀加tests/test_native_phase_contract_v2.py::test_reset_is_close_old_phase_schema_unchanged，预期ModuleNotFoundError，留RED。
- [ ] 在新模块定义以下全部类型及owner。__init__(*,prelift_stable_s:float)拒bool/非有限/≤0；reset拒空UID/重用UID/bool tick；observe要求同episode、连续tick+1、exact bool。范围明确为CPU数学/fixture，不给布尔值、dataclass类型、state.source_scope或caller True以native资格。生产入口需W4实际同owner adapter与PhaseSourceWitnessV2；没有时分别以full_v2_phase_source_adapter_missing或full_v2_phase_source_witness_missing拒绝。

~~~python
from dataclasses import dataclass
PHASES_V2=("Close","PreliftHold","Lift","LiftHold")
@dataclass(frozen=True)
class PhaseSampleV2:
    episode_uid:str
    physics_tick:int
    bilateral:bool
    prelift_stable:bool
    final_hold_eligible:bool
@dataclass(frozen=True)
class PhaseReceiptV2:
    episode_uid:str
    physics_tick:int
    phase:str
    prelift_stable_s:float
    final_hold_s:float
    success_candidate:bool
~~~

PhaseOwnerV2.reset(*,episode_uid,physics_tick)->PhaseReceiptV2清计时并返回Close；PhaseOwnerV2.observe(sample:PhaseSampleV2)->PhaseReceiptV2实现下一Task算法。不得保留未定义helper调用。
- [ ] 重跑单测预期PASS；git diff --check；仅add新module/test，commit feat: add fixed S0 phase v2 contract。

## Task 2：逐步转移与两个独立时钟

**Files:** 修改上述两个新文件；旧task.py不改。

- [ ] 添加边界RED：

~~~python
def test_prelift_does_not_count_toward_final_hold():
    from go2_pvcnn.control.dh116.tabletop_native_phase_contract_v2 import PhaseOwnerV2,PhaseSampleV2
    p=PhaseOwnerV2(prelift_stable_s=.25)
    p.reset(episode_uid="fixture:e1",physics_tick=2)
    def step(t,b=True,s=True,h=False):
        return p.observe(PhaseSampleV2("fixture:e1",t,b,s,h))
    assert step(3).phase=="PreliftHold"
    for t in range(4,123): assert step(t).phase=="PreliftHold"
    assert step(123).phase=="Lift"
    assert step(124,h=True).final_hold_s==0
    for t in range(125,2524): assert not step(t,h=True).success_candidate
    assert step(2524,h=True).success_candidate
    assert step(2525).phase=="Lift"
    r=step(2526,b=False,s=False)
    assert (r.phase,r.prelift_stable_s,r.final_hold_s)==("Close",0,0)
~~~

- [ ] 运行CPU前缀加该单测，确认语义RED。
- [ ] 最小算法：Close首次bilateral→PreliftHold；稳定start只由连续bilateral+prelift_stable建立，首端点0，(tick-start)/480达到配置才Lift。Lift后final资格首次→LiftHold，独立start；完整2400间隔才candidate，最终资格中断回Lift清最终时钟。失bilateral回Close清两时钟。Prelift阶段即使final=True也不累计最终时间。锚不属于phase owner，不接收/创建锚。0.25只在新配置中是候选，不改原success常数。
- [ ] 新增test_wrong_episode_or_skipped_tick_rejected、test_unknown_bool_rejected、test_prelift_interruption_restarts_window、test_final_interruption_clears_hold。None不能bool(None)吞成负例；bool tick不得当int。
- [ ] 运行phase测试预期全PASS；commit fix: separate prelift and final hold clocks。

## Task 3：独立context51与copy-on-write

**Files:** Create src/go2_pvcnn/control/dh116/tabletop_native_teacher_context_v2.py；Test/create tests/test_native_teacher_context_v2.py；只读旧context全文件。

- [ ] 创建RED：

~~~python
def test_context51_has_independent_phase_and_appended_duration():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import CONTEXT_FIELDS
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context_v2 import CONTEXT_FIELDS_V2
    assert len(CONTEXT_FIELDS_V2)==51
    assert [x["name"] for x in CONTEXT_FIELDS_V2[:4]]==[
        "execution_phase_one_hot.Close","execution_phase_one_hot.PreliftHold",
        "execution_phase_one_hot.Lift","execution_phase_one_hot.LiftHold"]
    assert [dict(x) for x in CONTEXT_FIELDS_V2[4:50]]==[dict(x) for x in CONTEXT_FIELDS[4:50]]
    assert CONTEXT_FIELDS_V2[50]["name"]=="prelift_stable_duration"
~~~

- [ ] 运行CPU前缀加tests/test_native_teacher_context_v2.py，预期新模块缺失。
- [ ] 从旧context独立复制已经验证的ContextMetadata/Packet/Owner及validator/encode实现，逐一重命名ContextMetadataV2、TaskContextPacketV2、TaskContextOwnerV2、validate_context_v2、validate_context_tensors_v2、encode_context_v2、CONTEXT_FIELDS_V2。在新模块导入旧ContextIdentity作为未变时钟值类型；新owner内部删除旧_PhaseStepObserver（原context.py:257）对task.step的替换/lease，改用Task1 PhaseOwnerV2返回receipt，核同episode/tick。新metadata保留原所有字段，追加phase_contract_sha256、fixed_s0_manifest_sha256、prelift_scale_s；新schema digest覆盖全部。
- [ ] 实现新owner接口，方法体从旧对应方法逐条迁移/修改，不留空骨架：reset_from_verified_state(state,*,history_owner,phase_owner)、freeze_pre_action(identity,*,history_owner)、advance_physical_endpoint(state,*,history_owner,collection_token,phase_owner)、commit_post_context(packet)。state仍用旧固定节点/距离/rest/target/external_budget/event_history/physical_fields键，另有fixed_s0_manifest_sha256、measurement_snapshot、phase_source_witness。TaskContextOwnerV2 constructor在原backend_profile/mode/scope/episode_cap_s/external_cap_s/grasp参数之外显式接source_adapter=None：只允许CPU fixture缺adapter，生产必须绑定W4实际SourceAdapterV2并按该adapter复核state内snapshot/witness，不能由state.source_scope赋予资格；phase receipt由owner读取而不是调用方任填。新packet.owned()克隆values/mask/metadata；validate严格N×51、同钟及新schema；index50按正有限prelift_scale_s归一化，encode输出102列。
- [ ] 新CPU fixture采用tests/test_native_teacher_context.py已有真实history构造方式，在新文件独立建立phase/context，不改旧fixture默认值。逐个先RED后GREEN覆盖：旧Packet拒绝、wrong epoch/tick拒绝、phase owner不同实例/receipt突变拒绝；raw未知NaN/false保留；pre snapshot在advance后不被改；record后commit；index50与slot7独立；Reset两个计时清零；失接触锚UID保留且不可换；固定d0/初态高度/预算/事件不变。生产physical_fields待W4 SnapshotV2接入，当前CPU不能获得native资格。
- [ ] 运行CPU前缀加tests/test_native_teacher_context_v2.py tests/test_native_teacher_context.py tests/test_native_phase_contract_v2.py，预期全PASS；commit feat: add independent fixed S0 context51。

## Task 4：六键新模型/profile身份

**Files:** Create src/go2_pvcnn/control/dh116/tabletop_native_teacher_model_v2.py、src/go2_pvcnn/control/dh116/tabletop_native_teacher_profile_v2.py、tests/test_native_teacher_model_v2.py、configs/dh116_training/native_fixed_s0_pinch_v2.json；旧model/profile/collector/checkpoint不改。

- [ ] 创建RED：

~~~python
def test_v2_keeps_exact_six_input_keys():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import INPUT_KEYS
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile_v2 import INPUT_KEYS_V2,CONTEXT_WIDTH_V2
    assert tuple(INPUT_KEYS_V2)==tuple(INPUT_KEYS)
    assert CONTEXT_WIDTH_V2==51
~~~

- [ ] 运行CPU前缀加tests/test_native_teacher_model_v2.py，预期ModuleNotFoundError。
- [ ] 新profile定义MODEL_SCHEMA_VERSION_V2、INPUT_SCHEMA_VERSION_V2、CONTEXT_SCHEMA_VERSION_V2、PHASE_SCHEMA_VERSION_V2及独立SHA；仅复用真实BackendProfile不重标签。新config显式profile=fixed_s0_pinch_v2、prelift_stable_s=.25、prelift_scale_s=.25、resolved_reset_manifest=null、native_ready=false、checkpoint_adapter_ready=false。null不能通过生产准入。
- [ ] 独立新model以旧NativeContextTeacher源码为起点，定义NativeContextTeacherV2/model_contract_v2/pack_native_inputs_v2，复用原PrivilegedTeacher Tanh及_MaterialEncoder；不super旧__init__后改identity。common66/rod41×13/actor24保持；context encoder首层nn.Linear(102,64)。_parts exact six keys+context51新validator；get/set_extra_state与load_state_dict核新model/input/context/phase/fixed S0身份。新packet不得传旧validator。生产pack_native_inputs_v2与TaskContextOwnerV2的reset/advance必须调用W4 validate_phase_source_witness_v2，确认witness是同一个实际adapter从同一owned SnapshotV2和PhaseOwnerV2 receipt签发，并复核owner/epoch/clock/source SHA；fixture packet走显式pack_cpu_fixture_v2且不能转为生产packet。仅typed PhaseReceiptV2、state.source_scope或调用方填True不通过。pack_native_inputs_v2的完整签名固定为(pre_action,raw_rod,parameters,task_context,*,backend_profile,source_adapter,measurement_snapshot,phase_receipt)，task_context.metadata持有已签发phase_source_witness；pack_cpu_fixture_v2使用独立fixture metadata并不生成生产见证。phase_source固定为execution_phase_owner_v2，不借用old.task.step；W4尚未实现该真实来源时，生产pack拒绝full_v2_phase_source_adapter_missing，缺/异钟witness拒绝full_v2_phase_source_witness_missing。
- [ ] 从tests/test_native_teacher_model.py现有CPU方法建立独立新fixture，先RED后GREEN覆盖actor/critic感知index50、旧50失败、添加controller_state键失败、305未知槽仍NaN/false、owned snapshot、old_extra_state双向拒绝。W8 controller50D/74D八键在另一版本。C05正式集成前不重写序列化器：其exact-type拒绝v2模型是当前预期；新模型额外状态已提供将来checkpoint身份，未有正式adapter时不能称保存恢复闭环。
- [ ] 运行CPU前缀加tests/test_native_phase_contract_v2.py tests/test_native_teacher_context_v2.py tests/test_native_teacher_model_v2.py tests/test_native_teacher_context.py tests/test_native_teacher_model.py tests/test_native_teacher_session.py，预期全PASS；git diff --check后commit feat: add versioned six key fixed S0 teacher。

## Task 5：审查与交接

- [ ] 比对spec每条与Task映射；检索未定义helper、旧identity误用、prelift混入5s、旧6keys被改、S0被预览代替。
- [ ] 只读确认scripts/run_dh116_work_packages.py:630 Repository真实constructor/verify_protection调用，执行既有保护核验输出实际528源码+2checkpoint。只检查相关CPU回归，不重复全库审计。
- [ ] 独立SPEC后QUALITY对精确commit复审。W4实际来源、W5长Reset、C05正式adapter、C06 worker/预算未齐时full仍拒绝。记录CPU通过、native0、grasp未验，交接W4，不原生、不push。

## 后继注册位置：V2-I01

W3完成后交接规划项native.fixed_s0.v2.integration（V2-I01），不更改旧C04/C05 exact-type拒绝。该包文件/API/静态→Reset/live→实际adapter witness→模型/trainer→collector顺序及新checkpoint/admission/actualworker完成门在spec“后继集成包 V2-I01”定义。W3不单独标native-ready；W4真实same-owner witness缺失时生产context/pack具体拒绝，不能以PhaseSampleV2或scope升格。V2-I01在W3/W4/W5及正式C05/C06实际provider之后、W6/W7之前实施，CPU跨进程update1/save/restore/update2与实际原生闭环分别验收。
