# DH116 Native Teacher Source Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Repair R03 and activate a new 66-common + 41x13-rod + explicit-task-context Teacher; preserve the 371-slot provenance ledger and historical models.
**Architecture:** New immutable BackendProfile, profile-specific capabilities, actual reset verification before policy construction, two independent reward profiles, original PPO/GAE and clean checkpoint restart.
**Tech Stack:** Python3.10, PyTorch2.7, NumPy, pytest, existing DeformX/Isaac runtime; Ubuntu development and Windows relay publication.

**Status:** Planning Markdown only; no production code, tests, model, native run or commit performed.
**Source:** User-approved DH116 PPO DOCX SHA256 a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9; extracted.txt and formula-transcription.md under evidence/ppo-reward-doc-20261115.
**Supersedes:** Previous C01-C05 full371-network prerequisite and missing-numeric-design conclusion. R03 source binding repair remains.
**Task count:** 7 (R03, C01-C06).
**Weight status:** Design-approved engineering candidates; not implemented or native-validated.
**Commands:** Run in a new isolated Ubuntu worktree. PY=/mnt/mydisk/xk/envs/dh116/bin/python; PYTHONPATH=src:.:tests; PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES=; OMP_NUM_THREADS=2; MKL_NUM_THREADS=2.
**Paths:** Control modules below are under src/go2_pvcnn/control/dh116/ unless a full repository-relative path is shown.

## Task R03：统一Teacher专家来源重放（当前可实施）

**Files:**
- Modify: src/go2_pvcnn/control/dh116/tabletop_privileged_warm_start.py:27–72
- Modify: src/go2_pvcnn/control/dh116/tabletop_privileged_dataset.py:38–120
- Modify: src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py:90–109,335–408
- Test: tests/test_dh116_teacher_expert_assembly.py
- Create: tests/test_teacher_expert_source_binding.py

### 已有API与最小复用

~~~text
verify_supervised_rows(payload)
read_expert(expert_dataset, expert_sha256, identity)
validate_expert_dataset(payload, *, expected_identity)
_episode_source_rows(source, episode_uid, physical)
~~~

read_expert的现有返回值是(payload, bindings)，必须保留。不能改成单payload后留下既有调用方解包不一致。

现有verify_supervised_rows已经：
- 重开source字节并校SHA；
- weights_only=True读取；
- 按episode/tick选择原始行；
- 严格比较observation/action的dtype和数值；
- 验证371D来源validity。

现有assembly._episode_source_rows已经：
- 验证canonical24动作合同；
- 验证完整371D来源；
- 校验每个8-substep或terminal部分窗口；
- 对齐actual physical issued action和原policy/physics clock。

仅补“所有入场都调用”和“source与对应native process/raw physical关联”，避免重写整套数据层。

- [ ] **Step 1：写失败回归**

复用tests/test_dh116_teacher_expert_assembly.py的artifacts、CPU_only_gate。后者明确模拟已通过的下层native gate，只用于隔离上层绑定，不授予native资格。

~~~python
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from test_dh116_teacher_expert_assembly import artifacts, CPU_only_gate
from go2_pvcnn.control.dh116 import tabletop_privileged_expert_assembly as build
from go2_pvcnn.control.dh116.tabletop_privileged_dataset import validate_expert_dataset


@pytest.mark.parametrize("damage", [
    "observation", "action", "mask", "order", "dtype",
])
def test_same_proofs_cannot_bless_changed_training_rows(tmp_path, monkeypatch, damage):
    identity, manifest = artifacts(tmp_path)
    CPU_only_gate(monkeypatch)
    build.assemble_experts(manifest, identity, tmp_path / "assembled")
    payload = torch.load(
        tmp_path / "assembled/privileged_native_expert.pt",
        weights_only=True,
    )
    changed = deepcopy(payload)
    if damage == "observation":
        changed["observations"][0, 0] = 123.
    elif damage == "action":
        changed["actions"][0, 0] = .9
    elif damage == "mask":
        changed["valid_mask"][0, 0] = False
    elif damage == "order":
        changed["actions"] = changed["actions"].flip(0)
    else:
        changed["observations"] = changed["observations"].double()
    with pytest.raises(ValueError, match="source|issued|mask|dtype"):
        validate_expert_dataset(changed, expected_identity=identity)
~~~

再增加：
- 同UID但来自另一文件/进程的source被拒绝；
- 原3844D来源不能裁切、投影或补值为371D；
- intact assembly产物可重建；
- 直接ActorBCTrainer.step与CLI expert路径触发相同校验；
- 当前无专家数据的PPO路径不要求成功专家样本；
- read_expert返回tuple，bindings含episode/source SHA/行数；
- 单次统一入场过程中source replay恰好调用一次。

- [ ] **Step 2：红灯命令**

~~~bash
"$PY" -m pytest -q tests/test_teacher_expert_source_binding.py -x
~~~

预期：observation/action/mask篡改因“未拒收”导致失败；不能将导入错误当作绑定漏洞的有效红灯。

- [ ] **Step 3：保留接口并消除重复source join**

新增共享函数，集中原validate_expert_dataset的metadata/native验证及source replay：

~~~python
def validate_bound_expert_dataset(payload, *, expected_identity):
    _validate_expert_metadata(payload, expected_identity=expected_identity)
    from .tabletop_privileged_warm_start import verify_supervised_rows
    bindings = verify_supervised_rows(payload)
    return payload, bindings


def validate_expert_dataset(payload, *, expected_identity):
    checked, _bindings = validate_bound_expert_dataset(
        payload, expected_identity=expected_identity
    )
    return checked
~~~

_validate_expert_metadata是将现有validate_expert_dataset第38–119行原样移入私有函数，不改变proof/episode/schema规则；避免新共享函数调用公开wrapper形成递归。

read_expert保留原tuple API，但不再先validate再额外verify_supervised_rows：

~~~python
def read_expert(expert_dataset, expert_sha256, identity):
    raw = _checked_bytes(expert_dataset, expert_sha256, "expert dataset")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return expert.validate_bound_expert_dataset(
        payload, expected_identity=identity
    )
~~~

每个正式入场调用只执行一次source replay。独立的后续BC调用仍不能因为payload先前通过就跳过自身入场；跨step缓存若要引入，必须另有不可变payload/session设计，本修补不加入该优化。

- [ ] **Step 4：在已有verify_supervised_rows内部补强raw/process绑定**

延迟导入已有assembly helper，不复制它的join：

~~~python
from . import tabletop_stereo_promotion as promotion
from .tabletop_privileged_expert_assembly import _episode_source_rows

qualification, base = promotion._read_object(episode["qualification_evidence"])
binding = {
    "path": str(Path(episode["source_path"]).resolve()),
    "sha256": episode["source_sha256"],
}
raw_source, _ = promotion._process_output(qualification, base, binding)
source = torch.load(io.BytesIO(raw_source), map_location="cpu", weights_only=True)
raw_physical, _ = promotion._artifact(qualification["raw_artifact"], base)
physical = [
    row for row in promotion._jsonl(raw_physical)
    if row.get("kind", "sample") == "sample" and row.get("live", True)
]
measured, issued = _episode_source_rows(source, episode["episode_id"], physical)
start, stop = episode["start"], episode["end"]
for name, expected in (("observations", measured), ("actions", issued)):
    observed = payload[name][start:stop]
    if observed.dtype != expected.dtype or not torch.equal(observed, expected):
        raise ValueError("expert rows differ from replayed source " + name)
if not bool(payload["valid_mask"][start:stop].all()):
    raise ValueError("canonical issued-action source does not authorize altered mask")
~~~

保留已有source cache、bindings、contiguous tick检查。metadata proof验证先完成，再进行source join；helper不能自己接受caller给的“verified=true”。

canonical assembly当前把所有实际issued24通道设为true；这里的mask规则只适用于该来源合同。旧partial-mask产物不能靠重新计算自己的摘要入场，须从原source重建或另行提交有来源支持的mask选择合同。不能静默制造false标签或忽略篡改。

- [ ] **Step 5：统一CLI加载路径**

- train_teacher新增expert_dataset_sha256 keyword。
- --expert-dataset与--expert-dataset-sha256必须成对。
- 参数检查发生在模型、optimizer或native启动之前。
- 删除CLI中的直接torch.load专家入口，调用read_expert并保留tuple解包。
- ActorBCTrainer继续调用统一validate_expert_dataset，不信任单独传入的张量。
- 不要求无BC的PPO探索提交专家proof。

- [ ] **Step 6：绿灯与提交**

~~~bash
"$PY" -m pytest -q \
 tests/test_teacher_expert_source_binding.py \
 tests/test_dh116_teacher_expert_assembly.py \
 tests/test_dh116_privileged_cli.py
~~~

验收：篡改拒收，intact source通过，loader调用次数正确，所有新测试的mock scope明确。记录实际passed/failed与命令；仅暂存本任务源码/测试/报告/readiness，独立提交。


## C部分共同规则

- [ ] 原371账本仍为float32[N,371]和bool[N,371]，仅0:24、78:88、153:185共66项进入common网络，其余305项保留NaN/false。删去旧稿“完整371有限网络”和conditional_371d_policy_input_undefined前置。
- [ ] 新Teacher有独立model/input/checkpoint schema；不向旧模型四键字典塞第五键，不改旧identity，不把新backend重贴到旧模型。
- [ ] 两种profile分别声明所需capabilities。smoke不等待Hold、slip、完整接触成功或三抓法全部成功；full只在适用来源真实完整时启用。
- [ ] 权重来自已获用户批准的新设计，但仍标记design_approved_native_unverified；旧R0 null JSON/诊断输出保留原义。
- [ ] 真实原生预算尚须单独冻结与核验；本计划不启动native，不扩大旧234/240步、0-update接触诊断授权。
- [ ] 除R03之外，C01–C05也可实现CPU源码闭环；当前只因真实源/初态/预算缺失的具体项目拒绝生产启动，不再无条件拒绝所有成功路径。

## Task C01：BackendProfile、两阶段preflight和当前审计修复依赖

**Create:** tabletop_native_teacher_profile.py、tabletop_native_teacher_preflight.py、scripts/preflight_dh116_native_teacher.py、tests/test_native_teacher_profile.py、tests/test_native_teacher_preflight.py。
**Dependency:** 总计划R08的新独立验证器应覆盖entry.numeric与entry.prior.sha两条动态属性错误，保持已消耗numeric wrapper与配方只读；只覆盖sha不完整。Teacher包不重复改写冻结文件。

- [ ] **Step 1：先固定已有API与来源**

复用mesh source_basis()对全部实际源文件、已登记差分及编译/import/binary SHA验证；先只读重审留存真实Reset产物。历史failed receipt不改写，离线复审不计新native。

已有签名使用文本声明，避免将keyword-only签名写成非法Python调用：

~~~text
tabletop_mesh_reset_numeric_state.validate_numeric_reset_audit(audit, *, reset_receipt)
tabletop_bilateral_r0_approach.BilateralMeasuredPinchApproach(project)
MeasuredPinchApproach.capture(exchange, *, grasp="pinch")
MeasuredPinchApproach.capture_reset(reset, parameter_packet, *, grasp="pinch")
tabletop_r0_health_separation.capture_solver_health(runtime, *, backend, sim, expected_clock)
~~~

已有distance reader是Pinch-only；不得把它的证据用于Tripod/Power。目标定义固定为原41个材料节点集合与该抓法required pads，版本为fixed_material_nodes41_set_distance_v1；禁止运行中改目标集合或材料身份。

- [ ] **Step 2：失败测试**

~~~python
def test_unknown_backend_cannot_be_relabelled():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    import pytest
    with pytest.raises(ValueError, match="backend"):
        BackendProfile.from_sources(binary_sha256="0" * 64, source_inventory={})


def test_smoke_preflight_does_not_require_full_task_sources():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import required_capabilities
    needed = required_capabilities("approach_wiring_smoke_v1")
    assert "same_clock_distance" in needed
    assert "native_health" in needed
    assert "fixed_material_slip" not in needed
    assert "hold_motion" not in needed


def test_full_profile_requires_real_slip_and_hold_motion():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import required_capabilities
    needed = required_capabilities("flexible_train_r1_full")
    assert {"fixed_material_slip", "tangential_slip", "hold_motion"} <= set(needed)
~~~

运行：
~~~bash
"$PY" -m pytest -q tests/test_native_teacher_profile.py tests/test_native_teacher_preflight.py -x
~~~

- [ ] **Step 3：实现不可变profile与static/live两段核验**

BackendProfile使用frozen dataclass，保存实际binary绝对路径及完整SHA、source inventory SHA、资产/SDF/rod schema、profile version及digest。由受限source loader生成，不接受native_ready=True。新模型、动作owner、health/distance reader、reward和checkpoint引用同一实例identity。

static检查：配置schema、source字节、固定初态manifest引用、动作/频率、数值reward、profile/modelcontext匹配、独立预算。不可变manifest内容必须来自用户已确认的resolved配置，不能从截图估角、把pose_preview_only当最终初态，或凭空填root/q/rod节点。

live检查发生在owner真实reset之后：同一实例与reset_id、实际SDK q/qd/targets、碰撞mesh current/target与numeric状态、rod p/v/director、参数/距离/health同钟；比较本次resolved_reset_manifest。不得以历史receipt替代本次在线Reset。

smoke required capabilities明确为：
- backend/current-source身份、actual resolved reset、common66；
- 41节点13特征、当前参数和同钟读回；
- 固定目标定义的pre/reset/post双侧距离；
- normalized request24实际归属、SDK目标读回；
- 原guards与native numerical health；
- task_context使用项、8SDK/控制频率与显式资源预算。

full在此基础增加真实required-pad有效接触/对向、无外援/桌面支持、固定材料和切向锚滑移、hold_motion及连续时间。

- [ ] **Step 4：readiness可执行输出**

required_capabilities(profile)返回静态定义的集合；preflight按真实文件/reader检查返回component条目，不接受caller资格Boolean。

输出字段：profile、style、model_schema、static_ready、live_reset_verified（未运行为false）、components[{source_complete,applicable,valid,reader,source_sha256,blocked_reason,release_condition}]、blocked_reasons、new_native_steps=0。始终保留371ledger仅66测量的事实，但它本身不再阻断新schema。

缺初态/距离/health/预算时精确列缺项；配置值源已存在，不再输出formal_r0_numeric_source_missing。输出reward_implementation_pending或具体源缺项，直到对应软件/真实验证完成。

CLI使用exclusive output，schema损坏exit2、有效但缺源exit3、全部静态条件通过exit0；static PASS不意味着live Reset/PPO/native成功。

- [ ] **Step 5：验收与独立提交**

测试no-native meta-path guard、三抓法范围、old/new backend交叉、budget/manifest缺项、相同gamma来源、profile-specific gates。R08离线复审结果作为独立输入，原failed不改。记录实际测试数量，不预写通过。

## Task C02：精确task_context50与独立新Teacher模型

**Create:** tabletop_native_teacher_context.py、tabletop_native_teacher_model.py、tests/test_native_teacher_context.py、tests/test_native_teacher_model.py。
**Reuse:** MeasuredCommonTeacher的66投影、_features的41x13材料特征、_MaterialEncoder、原Tanh sampling/logprob和DeformablePPOTrainer。

- [ ] **Step 1：冻结schema**

新输入键精确为common371、common_valid、rod_nodes、rod_mask、task_context、task_context_valid；全部张量owned clone。context元数据另存同钟identity/source/applicable/reference UID，不把字符串UID编码成任意浮点。

| context索引 | 字段 | 单位/归一化/来源 |
|---|---|---|
| 0:4 | phase_one_hot Approach/Close/Lift/Hold | bool来源明确状态机；smoke仅Approach阶段 |
| 4:6 | required_contact_fraction L/R | [0,1]，完整有效接触源；smoke excluded |
| 6 | bilateral_contact_duration | s/0.25，连续时间；smoke excluded |
| 7 | continuous_hold_duration | s/5，首端点0；smoke excluded |
| 8:10 | fixed_material_anchor_station L/R | m/rest_length；未锚定NaN/false |
| 10:12 | fixed_anchor_valid L/R | 布尔，实际reset锚状态及可信历史 |
| 12:36 | previous_normalized_request24 | 原[-1,1]值，来自Action24HistoryOwner真实reset/issued记录 |
| 36 | reset_median_node_z | m，实际Reset41节点 |
| 37 | reset_min_node_z | m，实际Reset41节点 |
| 38 | distance_ref | m=max(actual_d0,0.005)，回合内固定 |
| 39 | actual_reset_distance_mean | m，真实双侧距离均值 |
| 40 | lift_target | m=0.03，原成功合同 |
| 41 | hold_target | s=5，原成功合同 |
| 42 | slip_limit | m=0.0005，原成功合同 |
| 43 | episode_elapsed | s/显式episode时限，真实物理clock |
| 44 | external_time_remaining | s/显式外部时限；用于透明上下文，不伪装task deadline |
| 45 | finite_task_deadline_enabled | bool；首版false |
| 46 | finite_task_time_remaining | 首版不适用NaN/false；启用需新任务合同 |
| 47 | prohibited_support_latched | bool，合格来源；smoke excluded |
| 48 | success_event_emitted | bool，真实内部一次性历史 |
| 49 | trusted_failure_event_emitted | bool，真实内部一次性历史 |

总宽50。固定尺度进入schema digest；float32[N,50]与bool[N,50]配对。未知raw值始终NaN/false，显式非适用保留applicable=false与原因。

网络context encoder采用显式mask编码：在完成profile required-source检查后，cat(where(valid, value, 0), valid.float())得到100维编码输入。这是新schema声明的缺测表示，不改变raw值/validity，不声称未知被测为零；required且适用的未知在编码前拒绝。不得对371ledger做nan_to_num。

actor/critic分别有context MLP100→64→64；common66→256→256、rod per-node13→64→64做mean+max128；融合448→24 actor/448→1 critic。两个分支独立；log_std24与原Tanh分布不变。

- [ ] **Step 2：失败测试**

~~~python
def test_context_schema_has_exact_fifty_slots():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import CONTEXT_FIELDS
    assert len(CONTEXT_FIELDS) == 50
    assert [field["index"] for field in CONTEXT_FIELDS] == list(range(50))
    assert [field["name"] for field in CONTEXT_FIELDS[12:36]] == [
        "previous_normalized_request24." + str(i) for i in range(24)
    ]


def test_mask_encoder_keeps_unknown_out_of_numeric_branch():
    import torch
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import encode_context
    values = torch.tensor([[1., float("nan")]])
    valid = torch.tensor([[True, False]])
    encoded = encode_context(values, valid)
    assert torch.equal(encoded, torch.tensor([[1., 0., 1., 0.]]))
    assert torch.isnan(values[0, 1])
~~~

encode_context接受任意最后维的值/mask，模型入口另强制宽50；因此上述2维单元测试不伪造model schema。

其余测试用新测试文件内明确定义的CPU数据fixture，复用test_dh116_flexible_measured_common_teacher.setup()返回的config/identity/ledger/valid/rod/clock/mask；context fixture按上述50字段明确构造并标记scope=cpu_fixture。不得把fixture标为actual/native。

必须覆盖：
- 新6键字典通过、旧模型仍拒第五/第六键；
- common mask固定66，305NaN不进网络；
- 41节点不允许mask藏掉坏节点；
- context有效项NaN、跨episode/time、重锚、previous_request错token拒收；
- actor动作损失梯度进入context encoder与rod encoder；
- collect_step->_check_frozen_policy->update可用，不能只测forward；
- weights/model/schema与旧checkpoint交叉拒绝。

- [ ] **Step 3：实现接口与兼容性**

NativeContextTeacher继承DeformablePrivilegedTeacher体系，保持_parts(observations)返回(common,nodes,mask)三项供原trainer使用；其内部先检查6键和context。latent_mean/value另读取已验证context，拼接新的context编码。不能改变原DeformablePPOTrainer对_parts的调用协议。

pack_native_inputs(pre_action, raw_rod, parameters, task_context, *, backend_profile)重建原始rod数据、validate_mounted_observation、同钟和profile，不调用固定2fea历史ActualBackendMeasuredTeacher._actual_inputs作为新scope入口。

TaskContextOwner接口：reset_from_verified_state、freeze_pre_action、advance_physical_endpoint、commit_post_context。前后上下文copy-on-write；奖励先返回post_context，用它计算next value，再commit，避免bootstrap用了旧上下文。

- [ ] **Step 4：红绿与提交**

~~~bash
"$PY" -m pytest -q tests/test_native_teacher_context.py tests/test_native_teacher_model.py \
 tests/test_dh116_deformable_privileged_ppo.py
~~~

先单次失败后实现；新schema/模型namespace独立，旧模型确定性输出不变。未创建任何正式模型/训练结果的规划阶段不预写验证数。

## Task C03：两种正式奖励profile与可重算结算器

**Create:** tabletop_training_reward.py、configs/dh116_training/approach_wiring_smoke_v1.json、configs/dh116_training/flexible_train_r1_full.json、tests/test_native_teacher_reward.py。
**Reuse:** 已有实际distance/health/guard读取、原成功阈值和滑移底层测量。旧R0 null合同与diagnostic_coefficients不改。

- [ ] **Step 1：把已批准建议落为独立版本配置**

| 项 | smoke | full |
|---|---|---|
| profile | approach_wiring_smoke_v1 | flexible_train_r1_full |
| 潜势 | 2*p_d | 2*p_d+2*p_c+0.5*p_p+p_g+4*p_l+4*p_h |
| command_rate | 0.05 reward/s | 0.05 reward/s |
| time | 0.10 reward/s | 0.10 reward/s |
| slip | profile_excluded | 1.0 reward/s，q_slip截[0,2] |
| velocity | excluded，首版关闭 | 0，首版关闭 |
| trusted_constraint_failure | -40，最多一次 | -40，最多一次 |
| prohibited_support | 未启用额外奖励项，原guard仍保持 | -50，失败取最大罚一次 |
| success | 不授予完整任务成功 | +100，episode一次，失败优先 |
| PPO gamma/lambda | 0.995/0.95 | 独立候选0.999/0.97 |

gamma_source固定ppo.gamma，reward不得再有独立默认gamma。smoke PPO：rollout32、lr1e-4、epochs1、minibatches1、clip0.2、entropy0.005、value0.5、max_grad_norm0.5、target_kl0.02、kl_backtrack_steps3。full rollout256/epochs3/minibatches4仅作为后续独立候选，不自动占用预算。

数值配置注明设计文档SHA、weight_status=design_approved_native_unverified，不把该字段当来源通过Boolean。配置SHA进入session/checkpoint；加载时检查fixed成功阈值与原合同完全一致。

- [ ] **Step 2：实现可单测的纯数学内核**

这些Python块是未来实现内容，语法必须有效。来源/身份/finite/范围校验在结算adapter进入此核之前完成；此核本身不授予资格。

~~~python
import math
import numpy as np


def command_rate(action, previous):
    action = np.asarray(action, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if action.shape != (24,) or previous.shape != (24,):
        raise ValueError("normalized request24 required")
    if not np.isfinite([action, previous]).all() or np.any(np.abs([action, previous]) > 1):
        raise ValueError("finite bounded normalized request required")
    return float(np.mean(((action - previous) / 2.0) ** 2))


def smoke_potential(distance_m, distance_ref_m):
    if not math.isfinite(distance_m) or distance_m < 0:
        raise ValueError("finite actual distance required")
    if not math.isfinite(distance_ref_m) or distance_ref_m < 0.005:
        raise ValueError("fixed reset-derived distance scale required")
    return 2.0 * (1.0 - float(np.clip(distance_m / distance_ref_m, 0.0, 1.0)))


def control_total(phi_pre, phi_post, cost_per_second, gamma, event, terminated):
    if len(cost_per_second) < 1 or len(cost_per_second) > 8:
        raise ValueError("one to eight actual SDK samples required")
    values = [phi_pre, phi_post, gamma, event, *cost_per_second]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("finite reward components required")
    if not 0 < gamma < 1 or any(value < 0 for value in cost_per_second):
        raise ValueError("bounded gamma and nonnegative costs required")
    endpoint_phi = 0.0 if terminated else phi_post
    return gamma * endpoint_phi - phi_pre - math.fsum(cost_per_second) / 480.0 + event
~~~

smoke成本为每个实际子步0.05*q_action+0.10；q_action来源必须是owner实际核对过的normalized_submitted_command24_preprojection，不能用SDK投影后的targets或sample副本冒充。

- [ ] **Step 3：先写数值红灯**

~~~python
def test_smoke_distance_progress_uses_same_ppo_gamma():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_training_reward import smoke_potential, control_total
    phi0 = smoke_potential(0.010, 0.010)
    phi1 = smoke_potential(0.005, 0.010)
    actual = control_total(phi0, phi1, [0.10] * 8, 0.995, 0.0, False)
    assert actual == pytest.approx(0.995 - 0.10 * 8 / 480)


def test_true_failure_integrates_only_three_actual_substeps():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_training_reward import control_total
    actual = control_total(1.0, 2.0, [0.15] * 3, 0.995, -40.0, True)
    assert actual == pytest.approx(-41.0 - 0.15 * 3 / 480)


def test_potential_contact_loop_has_no_discounted_profit():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_training_reward import control_total
    gamma = 0.999
    gain = control_total(0.0, 2.0, [0.0] * 8, gamma, 0.0, False)
    loss = control_total(2.0, 0.0, [0.0] * 8, gamma, 0.0, False)
    assert gain + gamma * loss == pytest.approx(0.0)
~~~

运行后实施纯核：
~~~bash
"$PY" -m pytest -q tests/test_native_teacher_reward.py -x
~~~

- [ ] **Step 4：full潜势与适用域**

- p_d=1-clip(d/d_ref,0,1)，d_ref=max(actual_reset_d0,0.005)，目标集合和尺度回合内固定。
- p_c=(cL+cR)/2，分母为本抓法required_pads；真实无接触且完整来源时为0。
- p_p=clip(T_contact/0.25,0,1)，双侧required接触+原对向判据连续满足；首合格端点计时0。
- p_g=(gL+gR)/2，复用原可信对向判据，不新增未验证力阈值。
- h=min(current_median_z-reset_median_z,current_min_z-reset_min_z)；p_l=I_lift*clip(h/0.03,0,1)。I_lift要求真正双侧承托、离桌、无辅助及适用来源完整；允许回落，禁止历史max刷分。
- p_h=clip(T_hold/5,0,1)，瞬时H不得含“已保持5s”。2400个完整间隔/2401端点才满5s，H中断即清0并回退潜势。
- s=max适用材料/切向drift+uncertainty，q_slip=clip(s/0.0005,0,2)。原合规/违规/unknown三值阈值保持；reward与evaluator共享测量函数。
- 固定材料/切向reference UID只按合同建一次；正常失去接触保持锚，重接触继续相对原锚，不能重锚清漂移。
- numerical_health、constraint_status、hold_motion独立。正常没抓稳/无接触/掉落不自动成为数值故障。

每项记录raw、unit、valid、applicable、source、contribution：
1. known negative：真实0或负进度可入PPO；
2. proven not applicable：raw=None，applicable=false且有可信history依据，贡献按profile阶段为0；
3. applicable but unknown/corrupt：SourceFailure，保留原始行，停止采样，整份未完成rollout隔离，参数不更新；不能用-40伪装可用样本。
smoke未启用项明确profile_excluded，不要求其来源；full不能以系数0绕过所需源。

事件一次性：trusted failures全部日志保留，罚取max一次，failure覆盖success；损坏源优先SourceFailure，无success。不得擅自把slip>0.5mm改为新物理中止，除非原任务定义要求。

成本上界loader测试：full Cmax=2.15/s，gamma0.999无限折扣成本=2.15/(60*(1-.999))≈35.833，-40覆盖；gamma0.9995对应≈71.667，原-40不再满足，配置必须拒绝并要求重核。smoke Cmax=.15/s/gamma.995上界=.5，仍保持设计-40。

- [ ] **Step 5：统一终止与bootstrap**

- trusted success/failure或正式task deadline：reward终端Phi=0，日志仍存真实Phi/post；V bootstrap=0，done=true。
- 外部20s episode cap（候选需显式配置）：保留实际pre-reset Phi和V_shaped，truncated=true、done=true，先算V再reset。
- rollout32满：保留Phi和V，done=false，不reset；GAE buffer末端自然止carry，不虚构done。
- 外部预算只能在完整8子步窗口开始前截断，余额不足不启动；可信硬失败可3子步终止只积分3/480。
- bootstrap只加gamma*V，不再加Phi。

结算API：TrainingReward.reset_from_verified_state、begin、observe、settle_once。RewardPacket含token/policy/profile/config SHA、真实duration、pre/post各潜势、成本、事件、endpoint、post_context。重复token、日志重试或重复settle拒绝，不能重发动作。

测试覆盖full Hold2400/2401端点、下降/断Hold负进度、锚未知/重接触、三值slip、failure/success冲突、sourcefailure无update、gamma同源、无第二次Phi。新profile数学CPU通过不等于原生成功。

## Task C04：真实factory与单物理owner collector成功路径

**Create:** tabletop_native_teacher_session.py、tabletop_native_teacher_collector.py、tests/test_native_teacher_session.py、tests/test_native_teacher_collector.py。
**Modify:** tabletop_no_camera_collector.py的actual工厂导向正式新session，保留CPUFixtureCollector隔离；不删除其actual backend拒绝条件。

- [ ] **Step 1：静态→本次Reset→模型奖励的明确顺序**

新factories是有明确属性的dataclass：episode(contract)、policy(model_contract)、trainer(model, ppo_config)、reward(reward_contract)。新增NativeEpisodeOwner包装原DualHandPhysicalEpisode与本session已有场景/资源的所有权，不再造第二个物理owner。它的reset签名与原episode一致，close执行R06定义的阶段清理，不使用强退伪装自然退出。

~~~python
def create_native_collector(config, factories):
    contract = verify_static_contract(config)
    owner = factories.episode(contract)
    try:
        reset = owner.reset(
            contract.reset_seed,
            episode_uid=owner.episode_uid,
            owner_resets=owner.owner_resets,
        )
        live = verify_reset_and_sources(owner, reset, contract)
        model = factories.policy(live.model_contract)
        trainer = factories.trainer(model, contract.ppo_config)
        reward = factories.reward(live.reward_contract)
        return NativeTeacherCollector(owner, trainer, reward, live)
    except BaseException:
        try:
            owner.close()
        except BaseException as cleanup:
            owner.record_cleanup_failure(cleanup)
        raise
~~~

Python3.10使用owner.record_cleanup_failure持久化收尾异常并保留原始异常，不能使用仅Python3.11提供的add_note。

python
    except BaseException:
        try:
            owner.close()
        except BaseException as cleanup:
            owner.record_cleanup_failure(cleanup)
        raise
~~~

owner.record_cleanup_failure仅持久化cleanup异常，不抛新异常覆盖原始失败。outer launcher持有native App，不由collector重复close。初始化后所有stage均可cleanup；原生自然退出需独立预算验证。

现有create_actual_collector(project, grasp, *, policy_factory, trainer_factory, episode_factory)保留关键字参数；内部组装NativeFactories，reward使用已验证profile的TrainingReward构造。actual_admission改为调用同一静态verifier，不再hardcode admitted=False/固定失败原因；live条件在reset之后判断。API成功返回实际collector，不因full371网络缺失永远raise。

- [ ] **Step 2：factory顺序测试（实际mock对象，非未定义double）**

~~~python
def test_reset_is_verified_before_policy_construction(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session as session
    events = []
    contract = SimpleNamespace(reset_seed={}, ppo_config=object())
    live = SimpleNamespace(model_contract=object(), reward_contract=object())
    owner = Mock()
    owner.episode_uid = "cpu-isolated:one"
    owner.owner_resets = {}
    owner.reset.side_effect = lambda *args, **kwargs: events.append("reset") or {}
    monkeypatch.setattr(session, "verify_static_contract", lambda config: contract)
    def verify(*args):
        events.append("verify_live")
        return live
    monkeypatch.setattr(session, "verify_reset_and_sources", verify)
    monkeypatch.setattr(session, "NativeTeacherCollector", lambda *args: "constructed")
    factories = SimpleNamespace(
        episode=lambda static: owner,
        policy=lambda model_contract: events.append("policy") or object(),
        trainer=lambda model, ppo: events.append("trainer") or object(),
        reward=lambda reward_contract: events.append("reward") or object(),
    )
    assert session.create_native_collector({}, factories) == "constructed"
    assert events == ["reset", "verify_live", "policy", "trainer", "reward"]
~~~

这条仅证明factory顺序，显式mock下层verifier和collector constructor，不宣称实际sources通过。另用真实CPU SDK/rod fixture验证NativeTeacherCollector本体，不mock它的collect。

补静态拒绝不创建owner、live拒绝不创建model、policy/trainer/reward各失败close一次、cleanup失败保留两异常、无成功抓取也可构造smoke、full缺slip拒绝但smoke不因此拒绝。

- [ ] **Step 3：原API与实际控制**

保持DeformablePPOTrainer exact type，供Action24HistoryOwner._policy使用；新模型作为其已支持的继承类型。
原签名：
~~~text
trainer.collect_step(actor_obs, critic_obs)
trainer.record_step(reward, done, next_value, *, truncated=None, bootstrap_mask=None, executed_actions=None, action_owner="policy", collection_token=None)
history.capture_pre_action(pre)
history.begin(pre, trainer)
history.end(control_result)
history.issued_for_pending(trainer, settlement)
gae_boundary(terminated, truncated, *, endpoint_identity, next_value_identity=None)
~~~

collector顺序固定：
- owner.pre_action_state，context冻结；
- history.capture_pre_action，pack新6键obs且前后owner clock/内容不变；
- trainer.collect_step，history.begin，reward.begin；
- owner仅调用原episode.control(sample["actions"])一次，collector不sim.step；
- recorder先durable写，再history/reward.consume；health/guard每个实际子步检查；
- reward.settle_once返回post_context，history.end返回settlement；
- actual request-derived issued=history.issued_for_pending(trainer, settlement)；
- 使用同一Reset前endpoint和post_context计算next value；
- 原gae_boundary检查；record_step传done=terminated|truncated、truncated、bootstrap_mask=not terminated、executed_actions=issued、原token；
- 记录transition_recorded，然后commit上下文；真正terminated/外部truncation后才新Reset；
- buffer满不关闭episode，不reset，不取消原锚/时间；原PPO update后继续同一episode新策略。

额外hard guard或sourceerror必须即时从exchange observer传播，复用_ObserveTerminal思想但不沿用CPUFixtureCollector的类型/owner限制；不能仅让episode recorder返回reason（原函数忽略其返回值）。

- [ ] **Step 4：collector失败/GAE回归**

用已有test_deformable_r0_replay.setup(monkeypatch,tmp_path,mode="empty")的原物理owner fixture和新C02明确构造的模型/context fixture；真实CPU model/trainer和正式smoke数学，来源fixture标签明确。

逐条失败测试：
- 第3子步trusted guard：只有3个物理样本、有限terminal Phi0、-40一次、无第4次调用；
- 第3子步NaN/错identity：保存3行，raise SourceFailure，未完成rollout隔离，不trainer.update、不checkpoint；
- actual request != sample、SDKtarget不匹配、token/weights/policy变更拒绝；
- reward/replayed token重复不推进、不结算；
- truncated next_value刻意与下一episode不同，确认来自pre-reset；
- terminated b=0/c=0；truncated-reset b=1/c=0；normal/buffer-end b=1且原done=false；
- post_context更新不会改写前一行owned tensor。

SourceFailure后停止本次session，不把private pending清空伪装clean继续；last clean checkpoint是唯一恢复点。实际已消耗SDK/solver预算保存sessionledger，不能因rollout作废而减账或自动扩预算。

## Task C05：独立无相机CLI、clean checkpoint与additional_updates

**Create:** scripts/train_dh116_native_teacher_ppo.py、tabletop_native_teacher_checkpoint.py、tests/test_native_teacher_training.py、tests/test_native_teacher_checkpoint.py。
**Reuse:** 原PPOConfig、DeformablePPOTrainer、OnPolicyRollout GAE、KL rollback、exclusive checkpoint数值/Adam/RNG校验。不修改frozen tabletop_privileged_ppo.py算法或其源SHA。

- [ ] **Step 1：失败测试与训练配置**

新CLI参数：--config、--output、--additional-updates（默认1）、--checkpoint、--preflight、--execute-native。--preflight不构造policy/optimizer/native；--execute-native仅在新session/worker/显式预算通过后。默认不绑定StereoCamera。

配置包含独立model/context/reward/backend/target/reset manifest SHA；run budget必须有max_sdk_steps、max_resets、max_solver_calls、max_episode_count、wall_timeout_s等明确值。没有已核实值时null并在preflight具体拒绝，不猜258之外的Reset成本。

额外更新目标纯函数，测试不用未定义double：
~~~python
def test_resume_one_more_update_means_update_two():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import stop_update_target
    assert stop_update_target(0, additional_updates=1) == 1
    assert stop_update_target(1, additional_updates=1) == 2


def test_resume_target_rejects_nonpositive_or_boolean_additions():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_native_teacher_checkpoint import stop_update_target
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            stop_update_target(1, additional_updates=value)
~~~

实现：
~~~python
def stop_update_target(loaded_updates, *, additional_updates):
    if type(loaded_updates) is not int or loaded_updates < 0:
        raise ValueError("nonnegative loaded updates required")
    if type(additional_updates) is not int or additional_updates < 1:
        raise ValueError("positive additional updates required")
    return loaded_updates + additional_updates
~~~

- [ ] **Step 2：runner保持原trainer字段与掩码**

停止条件为trainer.counters["updates"]达到loaded_updates+additional_updates。采样原collector到trainer.rollout.full；遇SourceFailure停止并诊断，不能补行。调用原trainer.update后要求metrics["minibatch_updates"]>0、policy_version/counters真实增加、loss/grad有限且参数变化有限非零；KL全部拒绝不手动增加updates。原update即使0笔接受也会清空rollout；runner遇minibatch_updates==0必须有限停止并记录optimizer_update_rejected，不能自动再采32controls或扩预算。

普通rollout末端模型计算V_shaped，不额外加Phi，原GAE自然结束该buffer计算但物理episode继续。full和smoke各用自己的同源gamma；不在runner重定义reward折扣。

- [ ] **Step 3：checkpoint独立schema**

保存clean条件：trainer._pending is None、len(trainer.rollout)==0、更新完成。保存：
- actor/critic/common/rod/context weights与log_std；
- Adam所有状态、policy_version、updates/environment_steps、accepted minibatch计数；
- Python/NumPy/Torch CPU RNG；未来CUDA迁移需独立完整RNG/device验收；
- 固定观测缩放schema；running normalization若未启用明确None，不留未保存隐式统计；
- backend profile、model/input/context/reward/target/reset/action/source/config SHA；
- 当前scope及训练起点来源（fresh/resume/explicit initialization migration）。

沿用既有序列化/完整校验算法，在新identity层和新版本保存，不靠改旧ActualBackend checkpoint的globals或namespace混入。trainer仍CPU，真实GPU→CPU numeric复制、sync和耗时计入报告，不能称PPO全程GPU。

普通buffer满且episode继续时，可保存train-state clean checkpoint；checkpoint不声称保存了完整物理状态。restart从verified Reset开始新episode/context，训练state恢复，不能声称逐轨迹位级续跑。

- [ ] **Step 4：CPU完整保存恢复测试**

测试文件内使用C02新模型fixture和C04原物理ownerfixture，标签scope=cpu_fixture；真实optimizer至少接受一笔minibatch：
1. frozen likelihood检查通过；
2. update1，权重有限非零变化；
3. clean保存exclusive文件；
4. 新模型/trainer加载，比较model/Adam/RNG/counters/contracts完全一致；
5. 同一owned CPU数据/RNG回放下一sample或下一update一致；
6. additional_updates1，从loaded1达到2；
7. 换grasp/backend/model维度/context/reward/target/reset contract明确拒绝；
8. 旧4键模型/旧backend/旧measured66 checkpoint拒绝；
9. 非clean保存拒绝，不覆盖历史文件。

smoke切full是新reward目标；默认新critic/Adam与normalization训练state，actor权重继承只能显式初始化迁移，不当作同任务resume。

运行：
~~~bash
"$PY" -m pytest -q tests/test_native_teacher_training.py \
 tests/test_native_teacher_checkpoint.py tests/test_dh116_deformable_privileged_ppo.py
~~~

CPU两次update不计真实PPO通过。新模型仅在实施测试时创建测试内实例，不写入成功池。

## Task C06：独立预算与两次原生启动验收（执行门）

### 先完成且可发布的软件包

- [ ] R03来源绑定/tuple loader；
- [ ] C01 backend/profile/static/live核验与R08双属性审计修复；
- [ ] C02独立66+rod+context模型/schema；
- [ ] C03已批准候选smoke/full奖励及来源适用性；
- [ ] C04真实成功factory与collector CPU组合；
- [ ] C05clean恢复及additional_updates；
- [ ] 相关CPU回归、保护文件/历史checkpoint核对、报告/readiness；
- [ ] 新原生配方冻结SHA后再核验资源/授权，规划本身不启动。

### 每次新运行预算

1env×32controls=256 SDK measured。初始setup是读回核实值，若仍2步无中间Reset则总258；后续Reset/warmup/其他推进按实记录并另有max budget。solver实际子步按原配置核算，不把8SDK与rod内部子步混为一项。

预算余额不够完整下一窗口则停止，不为填rollout自动扩额；窗口中可信硬失败可部分终止，成本只积分实际行。wallclock单独限制，不承诺180s必完成；超时、受控取消和自然退出分别记录。

必须是新PPO update1配方，不能扩用旧234/240step、0update接触诊断范围。当前资源/初态/来源未满足项保持明确blocked，不虚构预算值。

### 两次启动目标

- [ ] 启动A：Pinch verified当前fixed初态，smoke32有效transitions，真实原policy采样/issued请求，接受update1，权重有限非零变化，clean checkpoint；自然退出/审计实际完成。
- [ ] 启动B：读取同checkpoint核对model/Adam/RNG/counters/contracts，重新建立真实owner并verified Reset，新采样32controls，additional_updates=1达到update2；保留所有实际setup/reset预算与退出证据。
- [ ] 当前只证明接线可更新和恢复，不授予完整抓取成功；不要求三抓法先成功。
- [ ] full profile在真实contact/slip/hold_motion等适用来源完整后独立启用，20s外部episode cap作为明确新候选；5s Hold必须2400间隔/2401端点。
- [ ] Tripod/Power各自manifest、distance/source/backend/receipt验证；Pinch-only reader不能借用。

### 结果报告

每包分开记录software CPU通过、native source/Reset、有效transition/SDK/solver预算、source failures、accepted optimizer steps、参数变化、checkpoint恢复、自然退出与完整任务物理成功。原日志/数据/模型私有保留；528保护文件与2历史checkpoint校验，明确暂存范围，Ubuntu commit→Windows同SHA正常发布，PR保持Draft。

### 最终CPU组合命令（实施后）

~~~bash
"$PY" -m pytest -q \
 tests/test_teacher_expert_source_binding.py \
 tests/test_dh116_teacher_expert_assembly.py \
 tests/test_dh116_privileged_cli.py \
 tests/test_native_teacher_profile.py \
 tests/test_native_teacher_preflight.py \
 tests/test_native_teacher_context.py \
 tests/test_native_teacher_model.py \
 tests/test_native_teacher_reward.py \
 tests/test_native_teacher_session.py \
 tests/test_native_teacher_collector.py \
 tests/test_native_teacher_training.py \
 tests/test_native_teacher_checkpoint.py \
 tests/test_dh116_deformable_privileged_ppo.py \
 tests/test_dh116_no_camera_collector.py \
 tests/test_action24_history_owner.py
~~~

这是未来实施命令，不表示新文件当前已存在或已通过；每任务先保存真实红灯、最小实现、绿灯，最终记实际测试数量。没有新增native或权重文件的规划稿不能填通过。

## v2自审与source target修订

- [ ] 旧稿R03保留，read_expert仍(payload,bindings)，共享validator/sourcejoin一次；API签名块用text，无非法keyword-only调用。
- [ ] 新66common+41x13rod+50context网络独立；371ledger仍305NaN/false。context mask neutral0仅在encoder内部，raw及metadata不改；required适用字段NaN/false在sample之前拒绝。
- [ ] phase切换只由本次真实物理/已记录task条件决定，不能按rollout数/episode轮数切换。smoke阶段恒Approach是该接线profile明确任务定义，不冒称已接触/抬升。
- [ ] target_material_ids/材料区段在verified Reset或用户manifest中确定，写入reset/profile/context metadata/checkpoint，episode内不重选。跨episode若变化须作为新模型可辨识输入或新manifest identity，不能隐藏变化。
- [ ] 若复用原all41-set SDF距离，明确是独立核验的fixed_material_nodes41_set_distance_v1集合距离语义，不称固定材料点；同一41个材料节点UID集合不变，节点重排/换UID/候选集变化拒绝。增加nearest-node换位轨迹测试：最小距离项切换仍须忠于同一集合函数、无隐藏目标切换或d_ref重算；若该语义不能满足训练目标，则改用manifest固定材料子集并显式升级target schema后再运行。
- [ ] smoke reward有已批准数值设计但尚未实现/原生验证，不等于缺数值方案；只对真正required源gate，不等待full成功/全量数据。
- [ ] full潜势和成本权重全部候选，保留负进度、吸收Phi0/截断Phi实际、pre-reset V_shaped与原GAE。
- [ ] static→owner真实Reset核验→model/trainer/reward→actualcollector成功路径可达，不再永远raise。
- [ ] update1→saveclean→restart追加1→update2，32controls256measured+实际setup/reset新预算独立，不扩用旧授权。

- [ ] 追加CPU三episode Reset回归：正常结束后与trusted-failure后各含一次新Reset，context/anchor/contact/hold/previous request/一次性事件不跨episode泄漏；live源故障不走无证明恢复。
