# DH116 Slip Source and Independent Evaluator v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 将已建锚失接触表示为有来源NA并保留固定锚，使用同钟owned测量独立重算奖励与严格评价。

**Architecture:** 新lifecycle/source snapshot模块独立于冻结C03 v1。v2奖励显式消费contacting/detached/never_anchored，独立evaluator仅消费测量快照；生产源adapter按字段来源准入，不以fixture/布尔标记充当真实来源。

**Tech Stack:** Python3.10、immutable dataclasses、NumPy/Torch CPU、pytest；复用C03数学与原guard，不改PPO/GAE/物理。

---

基线731a97298c2a1498f9fddd7a2ce7f35ce3dfa0b9。依赖W3 PhaseReceiptV2/TaskContextPacketV2；W3可先以CPU接口完成，本包不假设C05或C06已发布。模块均在src/go2_pvcnn/control/dh116/。CPU命令前缀：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
~~~

## Task 1：以真实旧代码复现生命周期冲突，定义新类型

**Files:** Create tabletop_native_reward_sources_v2.py；Test/create tests/test_native_reward_sources_v2.py。只读tabletop_deformable_grasp_task._slip:384–414与step:433,512,537、tabletop_training_reward._measurement:464–488与_read_anchors:550–589；不改二者。

- [ ] 在新测试文件记录旧冲突的精确行为，不需要native/fixture模型。以下第一测试是旧代码证据，会PASS；第二新接口测试必须RED：

~~~python
from types import SimpleNamespace
def test_old_evaluator_retains_anchor_but_returns_no_contact_slip():
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    anchor=((.01,.02),(.00001,.00001))
    task=SimpleNamespace(material_anchor=anchor)
    assert FlexibleGraspEpisode._slip(task,(None,None),False)==(anchor,None,None,None)

def test_new_detached_state_cannot_erase_anchor():
    from go2_pvcnn.control.dh116.tabletop_native_reward_sources_v2 import SlipLifecycleOwnerV2,FixedAnchorV2
    owner=SlipLifecycleOwnerV2()
    refs=(FixedAnchorV2("left","material","m",2,(.01,)),
          FixedAnchorV2("right","material","m",2,(.02,)),
          FixedAnchorV2("left","tangential","tl",2,(0.,)),
          FixedAnchorV2("right","tangential","tr",2,(0.,)))
    first=owner.observe(episode_uid="fixture:e",tick=2,contacting=True,references=refs,pairs=((0.,.00001),)*4)
    lost=owner.observe(episode_uid="fixture:e",tick=3,contacting=False,references=refs,pairs=None)
    assert first.state=="contacting"
    assert lost.state=="detached" and lost.references==refs
    assert lost.contact_pairs is None
    assert lost.na_reason=="fixed_anchor_detached"
~~~

- [ ] 运行CPU前缀加tests/test_native_reward_sources_v2.py；预期第一PASS、第二ModuleNotFoundError；留实际RED。
- [ ] 在新模块定义下列全部类型及SlipLifecycleOwnerV2.observe。references固定左右×material/tangential四组，UID/reference/tick首次冻结，之后exact equality；不得用当前最近点换reference。observe空contacting=None拒绝，接触恢复仍原refs。初态无refs且无接触为never_anchored/NA no_bilateral_contact_since_reset。已锚无接触为detached，当前接触必须4组有限非负drift/uncertainty，否则SourceFailure。geometry_drift是单独字段，存在时验证有效同钟，不替换contact_pairs。

~~~python
from dataclasses import dataclass
from typing import Optional,Tuple
@dataclass(frozen=True)
class FixedAnchorV2:
    side:str
    kind:str
    uid:str
    reference_tick:int
    reference:Tuple[float,...]
@dataclass(frozen=True)
class SlipLifecycleV2:
    state:str
    references:Tuple[FixedAnchorV2,...]
    contact_pairs:Optional[Tuple[Tuple[float,float],...]]
    na_reason:Optional[str]
    geometry_drift:Tuple[object,...] = ()
~~~

geometry_drift的object在Task2收紧为旧Measurement，不向生产开放任意对象。SlipLifecycleOwnerV2.__init__新episode reset/uid历史；observe(*,episode_uid,tick,contacting,references,pairs,geometry_drift=())->SlipLifecycleV2；其clear/reset只允许新Reset UID。当前CPU helper不认证native来源。

- [ ] 新增失败测试：contacting少一组/NaN/负uncertainty拒绝；detached refs消失或变更拒绝；never_anchored不伪造refs；重接触drift=.0007不被清零；unknown contacting=None拒绝；geometry有值不能因detached被丢弃。
- [ ] 运行该文件预期全PASS；commit feat: add versioned fixed anchor slip lifecycle。

## Task 2：owned同钟快照与逐字段source gate

**Files:** 修改tabletop_native_reward_sources_v2.py；Create tests/test_native_measurement_snapshot_v2.py；Create configs/dh116_training/native_fixed_s0_sources_v2.json。

- [ ] 首先只读确认真实可用源，不启动App：tabletop_deformable_r0_replay.PhysicalR0Diagnostic._measurements:236–287会保留未绑定AUXILIARY为None，不能把它当full完整reader；tabletop_native_teacher_sources.NativePinchSources当前仅smoke。列每字段producer函数/输入/单位/clock/unknown/NA规则及available/missing理由，存私有source-map.json。对已有contact/material/tangent/支撑guard，沿实际调用链读取到原始getter，不根据上游ABI猜测。source-map必须逐字段给出现有具体producer或真正缺失的证据；已存在可读字段必须在本Task实际绑定并消费，不能将所有字段统一missing包装后称桥接完成。
- [ ] 建立失败测试：相同钟snapshot被reward/evaluator各自消费后原nodes不变；wrong clock/owner拒绝。新module API在下面定义，测试先import应RED：

~~~python
def test_snapshot_schema_requires_owned_source_identity():
    from go2_pvcnn.control.dh116.tabletop_native_reward_sources_v2 import SnapshotV2,validate_snapshot_v2
    import pytest
    with pytest.raises((TypeError,ValueError),match="source|clock|snapshot"):
        validate_snapshot_v2({"scope":"actual","qualified":True})
~~~

- [ ] 定义SnapshotV2为frozen dataclass：stamp使用既有SourceStamp；target_ids为固定41字符串tuple；fields为Mapping[str,Measurement]；required_pad_ids与C03 RewardEndpoint同语义；slip为SlipLifecycleV2；node_positions为owned tuple[41][3]；owner_uid/source_context_sha256/fixed_s0_manifest_sha256均非空身份。构造时将输入arrays/tensors复制成owned finite tuples，fields/refs深冻结。validate_snapshot_v2(snapshot)->SnapshotV2逐字段核stamp.clock_identity同钟、单位、scope、有效性、来源SHA及required-pad固定分母。geometry_drift收紧为tuple[Measurement,...]，每项记录独立source/unit/NA理由，未知不得零填。
- [ ] 把已存在可读字段真正接通：同一C04 NativePinchSources.observe(event):142–164已串联PolicyOwnedPinchApproach.capture、validate_post_step_clearance和同owner health，并返回_endpoint:48–73的distance/numerical_health/constraint_failure/original_guard；该smoke范围只证明这些字段，不能据此授予full。节点从该已验证physical row的exchange.rod.node_positions复制并对同owner material_director packet核验；时钟从原owner.rod.clock与event identity交叉核验。contact/table/joint分类可复用PhysicalR0Diagnostic._measurements:236–287的具体已有计算，但只保留它实际支持的负接触/桌面SDF/关节guard结果，multiple-pad unknown不能推成正opposed。这些是现存producer路径；v2实际binding调用同底层distance/health/guard reader组合或同owner已验证返回字段，不把full配置塞给明确拒收full的旧NativePinchSources构造器，也不修改其smoke准入。逐字段source-map必须记录实际调用的具体producer和返回数据，不能仅列名字。
- [ ] 定义ReaderBindingV2为新frozen dataclass，字段field_name:str、producer_module:str、producer_qualname:str、producer_source_sha256:str、unit:str、clock_contract_sha256:str、available:bool、missing_reason:Optional[str]。SourceContextV2.reader_bindings为这些typed binding tuple；available=True时必须由源锁resolver找到具体callable且校验SHA，capture真实调用一次并验证结果；available=False必须给沿实际源码确认的缺失原因，不可绑定统一抛missing的callable。新增测试为每个available字段安装显式CPU producer spy并断言确实被调用且数值流入snapshot；只对确实缺失字段断言逐名SourceFailure。实际readable字段仍缺接线即本Task未完成，不能以native_ready=false作为豁免。
- [ ] 定义capture_snapshot_v2(*,event,history_owner,source_context)->SnapshotV2。event只接受C04已durable记录并history验证的完整physical event；source_context为本模块新frozen SourceContextV2(owner_uid, backend_sha256, source_inventory_sha256, reader_bindings, scope)，reader_bindings为明确字段→已核producer绑定。未有真实binding时抛SourceFailure("full_v2_measurement_source_missing:<field>")；不得给NativePinchSources贴full标签。逐物理步capture一次，生成same snapshot供两个consumer，不能另建scene/多sim.step。
- [ ] 本模块同时定义PhaseSourceWitnessV2和生产见证API，供W3消费。PhaseSourceWitnessV2只允许模块内已验证adapter签发，公有constructor拒绝调用方构造；字段至少adapter_uid、owner_uid、epoch、clock_identity、snapshot_sha256、phase_receipt_sha256、source_context_sha256、scope。定义SourceAdapterV2(owner,source_context)绑定唯一实际owner及已解析producer；capture(event,*,history_owner)->SnapshotV2消费完整durable event；capture_phase_witness_v2(*,adapter:SourceAdapterV2,snapshot:SnapshotV2,phase_owner:PhaseOwnerV2)->PhaseSourceWitnessV2从snapshot有效测量导出phase sample并调用phase owner，不接受调用方替换资格bool；validate_phase_source_witness_v2(witness,*,adapter,snapshot,phase_receipt)->None复核签发登记、同adapter/owner/epoch/clock与三个SHA。类定义在本Task完整实现，签发登记只由实际capture成功路径建立，typed dataclass或scope标签不授予登记。CPU fixture adapter使用独立显式scope，只能生成fixture见证；生产验证器拒绝fixture见证。缺实际phase adapter拒绝full_v2_phase_source_adapter_missing，缺/伪造/异钟见证拒绝full_v2_phase_source_witness_missing。
- [ ] source config仅声明需要哪些reader，不授予能力；initial native_ready=false。真正native source要源函数/当前backend字节/同owner/同钟/单位与合格历史全满足。不确定geometry漂移定义时，在source-map中明确其unavailable来源并拒绝依赖该量的full profile；先完成CPU lifecycle/共源接口，不编造ABI或几何算法。
- [ ] 逐项先RED后GREEN：caller boolean绕过拒绝、异钟Measurement拒绝、原输入后续mutation不改变snapshot、NA reason不匹配状态拒绝、应有source missing拒绝、真实contacting=False保留负例而非SourceFailure。运行CPU前缀加tests/test_native_measurement_snapshot_v2.py tests/test_native_reward_sources_v2.py，预期全PASS；commit feat: add source bound owned measurement snapshot v2。

## Task 3：保持C03数值语义的独立v2奖励

**Files:** Create tabletop_training_reward_v2.py、tests/test_native_teacher_reward_v2.py、configs/dh116_training/native_fixed_s0_reward_v2.json；旧tabletop_training_reward.py和旧配置不改。

- [ ] 建立纯数学RED；新API先精确写入测试：

~~~python
def test_detached_cost_is_na_and_time_cost_continues():
    from go2_pvcnn.control.dh116.tabletop_training_reward_v2 import slip_cost_v2
    from go2_pvcnn.control.dh116.tabletop_native_reward_sources_v2 import SlipLifecycleV2
    slip=SlipLifecycleV2("detached",(),None,"fixed_anchor_detached")
    result=slip_cost_v2(slip)
    assert result.raw_m is None and not result.applicable
    assert result.rate==0.0
    assert (.05*0+.10+result.rate)/480 == .10/480
~~~

注意此math-only值不经过source validator；完整detached必须有已冻结refs。运行CPU前缀加该测试预期ModuleNotFoundError。

- [ ] 定义SlipCostV2(raw_m:Optional[float], applicable:bool, rate:float, reason:Optional[str])与slip_cost_v2(lifecycle)->SlipCostV2：contacting采用C03 conservative_slip(pairs)/.0005 clip0..2；never_anchored/detached原始量None、明确NA、rate0，绝不输出“测得零滑移”。生命周期/实际来源校验先于数学。
- [ ] 独立RewardEngineV2复用原smoke_potential/control_total/command_rate/slip_status等纯数学函数，移植TrainingReward的reset/begin/observe/settle_once顺序、source failure隔离、terminal Phi0、完整8或可信partial terminal及事件一次。v2 config/profile/schema均新身份，但原权重2/2/.5/1/4/4、.05命令/.10时间/滑移、-40/-50/+100、gamma/成本界检查不变。不得修改原FULL全局常量后继承伪装v2。
- [ ] 给RewardEngineV2完整公开接口：reset_from_verified_state(snapshot,*,previous_request)、begin(submitted_request,*,remaining_substeps)、observe(snapshot)、settle_once(token,*,truncated=False,buffer_full=False)。snapshot只接受SnapshotV2；_read_slip使用lifecycle，不调用旧_read_anchors。detached时contact persistence/lift/hold资格归零，coverage/opposed保留真实部分值；正常下降保留负势差；不因所有slip超限新增硬终止。
- [ ] 使用tests/test_native_teacher_reward.py现有endpoint/clock构造模式，在新测试建立SnapshotV2 fixture并依Task1/2验证，逐项RED/GREEN：建锚→detached→同锚重接触；partial3可信guard只3步/-40一次/Phi0；source坏第3步隔离无transition；下降、hold中断、失败优先、重复settle拒绝、buffer full非done、外部截断保Phi和pre-reset value。新schema拒绝旧v1 packet/config混入。
- [ ] 运行CPU前缀加tests/test_native_teacher_reward_v2.py tests/test_native_reward_sources_v2.py tests/test_native_measurement_snapshot_v2.py tests/test_native_teacher_reward.py，预期全PASS；commit feat: preserve C03 mathematics in slip aware reward v2。

## Task 4：独立evaluator与W3消费同钟receipt

**Files:** Create tabletop_native_success_evaluator_v2.py、tests/test_native_success_evaluator_v2.py；新context_v2仅在实际必要处接SnapshotV2，不改旧context/task。

- [ ] 建立独立性RED：evaluator拒绝reward packet或含success布尔值的dict，仅接受SnapshotV2；测试导入新EvaluatorV2尚不存在应RED。
- [ ] 定义EvaluatorV2.reset(snapshot)、observe(snapshot)->EvaluationV2；EvaluationV2为frozen dataclass，字段stamp、eligible、hold_intervals、hold_s、success、reasons。evaluator从snapshot节点+S0高度、qualified support/opposed/coverage/slip/guard/health独立计算，不能导入RewardEngineV2或读reward packet。首次合格端点0，2400完整间隔且≥5s再success；中断清计时，prohibited_support锁存；保留原30mm与0.5mm不确定性分类。
- [ ] 新tests使用Task2完整fixture，检查2399/2400间隔、最低节点不足高度、支撑缺/假、0.5mm边界、detached清hold但保anchor、重接触同锚；篡改reward事件不会影响evaluator（测试中任意改变独立reward对象后重复同snapshot得到同评价）。
- [ ] W3 TaskContextOwnerV2从phase receipt读执行四阶段，从同SnapshotV2读source facts；slot7与最终hold同钟，index50与prelift同钟；不将旧物理phase改名冒充新执行owner。首次生产full仍要W5+正式S0+C06预算worker交集。
- [ ] 运行本包全部新tests及旧reward/context、W3新phase/context测试，记录精确SHA/exit/output；git diff --check、528+2保护、独立SPEC/QUALITY，commit feat: evaluate strict success independently from shared measurement。
- [ ] 交接报告分别列CPU全通过、actual native0、source native qualification未验、grasp未验；真实producer缺失项逐名保留阻断，不用已写config消除缺口，不push。

## 后继注册位置：V2-I01

Task2完成条件包含所有现存可读字段的具体producer接线、逐字段source-map及调用证据，统一missing空bridge不满足完成。SourceAdapterV2另定义capture_reset(reset_receipt,*,history_owner)->SnapshotV2，绑定本次已验证S0/live Reset且不推进SDK。PhaseSourceWitnessV2每个physics snapshot仅签发一次，生产context/pack只验证已签发见证，不重复推进phase。RewardPacketV2在tabletop_training_reward_v2.py定义为frozen dataclass，保留旧RewardPacket全部结算字段，追加slip_lifecycle/source_context_sha256/phase_witness_sha256，不能将v1重标签。

W3/W4/W5完成后进入规划项native.fixed_s0.v2.integration（V2-I01），其新session/collector/checkpoint/admission/CLI与C06 actualworker依赖见spec同名节。旧exact-type此时仍不兼容，不能称现工厂已能用v2。完成门包含新入口实际消费本adapter/独立evaluator/phase witness、逐字段缺失拒绝及CPU跨进程update1→save→restore→update2；native/grasp分别等独立预算与完整来源证据。
