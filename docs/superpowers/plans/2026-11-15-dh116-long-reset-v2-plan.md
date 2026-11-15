# DH116 Long Clock and Reset v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在保持真实SDK时间与预算消费的前提下，独立支持绝对SDK240、258、>2400之后的完整Reset并继续下一episode。

**Architecture:** 显式ClockPolicyV2封装已核SDK时间累积模式/有界步数，新Reset/order/readback/numeric/preflight链全部消费同一policy，不放宽冻结v1。保留零步Reset、同scene/rod、11mesh和所有controller/source历史归零证据；真实native只由独立冻结worker与批准预算执行。

**Tech Stack:** Python3.10、NumPy/Torch CPU、pytest、现有SDK/rod接口及scripts/python.sh，不安装/下载。

---

基线731a97298c2a1498f9fddd7a2ce7f35ce3dfa0b9。W3/W4可并行CPU设计；full/native实际运行须三者共同通过。所有新模块位于src/go2_pvcnn/control/dh116/。完整CPU前缀：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
~~~

## Task 1：确认时间来源并建立独立clock policy

**Files:** Create tabletop_native_training_clock.py、tests/test_native_training_clock.py。只读tabletop_mesh_reset_order._clock:27–39；tabletop_mesh_reset_readback._validate_clock:29–55；tabletop_mesh_reset_numeric_state._capture:94–121；tabletop_native_teacher_session:67–81,261–289；tabletop_dual_hand_reset._clock/run。

- [ ] 只读检查实际SDK current_time/current_time_step_index/get_physics_dt实现及已留setup/clock证据，记录其时间累积模式与源码SHA到私有clock-source-map.json。旧v1包络假设float32(dt)→float64累积，但该假设不得直接当新长程原生证明。已只读安装源码：SimulationContext._physics_timer_callback_fn:1360–1362在Python累加step_size并增整数步；current_time:253–266返回该Python值，_physx.pyi:1860仅声明Callable[[float],None]，未提供C++底层精度。SimulationManager.get_physics_dt:197–218返回1.0/physics_hz。记录callback精度仍需既有真实receipt或获批native核对，不把pyi float当其每个数值精确来源。
- [ ] 创建真实旧边界测试与新clock RED，不修改旧模拟时间：

~~~python
def test_old_reset_source_remains_bounded():
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import reset_clock_preflight
    import pytest
    owner=SimpleNamespace(sim=SimpleNamespace(current_time_step_index=258))
    with pytest.raises(ValueError,match="bounded_mesh_reset_clock"):
        reset_clock_preflight(owner)
    assert owner.sim.current_time_step_index==258

def test_new_policy_accepts_accumulated_time_without_rewriting_it():
    from go2_pvcnn.control.dh116.tabletop_native_training_clock import ClockPolicyV2,validate_sdk_clock
    import numpy as np
    p=ClockPolicyV2(max_sdk_step=9600,timestamp_mode="fp32_dt_float64_sum")
    stamp=0.0
    for _ in range(2402): stamp+=float(np.float32(1/480))
    result=validate_sdk_clock(step=2402,timestamp_s=stamp,dt_s=1/480,policy=p)
    assert result.timestamp_s==stamp and result.sdk_step_abs==2402
~~~

- [ ] 运行CPU前缀加tests/test_native_training_clock.py；预期旧测试PASS，新模块缺失RED。
- [ ] 定义frozen ClockPolicyV2(max_sdk_step:int,timestamp_mode:str)、SdkClockV2(sdk_step_abs:int,timestamp_s:float,dt_s:float,tolerance_s:float,policy_sha256:str)、validate_sdk_clock(*,step,timestamp_s,dt_s,policy)->SdkClockV2、read_clock_v2(sim,rod,*,policy)->dict。max_sdk_step为正exact int，不是运行授权；mode仅接受已定义fp32_dt_float64_sum和float64_dt_float64_sum；CPU成功不自动声明某native模式成立。
- [ ] 实现包络：令u=np.finfo(np.float64).eps/2、n=step、term=float(np.float32(dt))或dt，gamma_n=n*u/(1-n*u)，误差界n*abs(term-dt)+gamma_n*n*abs(term)+8*np.finfo(float).eps*max(1,n*dt)。必须n*u<1、界<dt/2、实际stamp有限且abs(stamp-n*dt)≤界；不沿用旧2e-7绝对上限，不以任意宽atol掩盖偏移。输出保留原timestamp，任何函数不得给sim.current_time赋值。连续reader另核步数/时间单调，rod.clock保留原solver_time/tick来源。
- [ ] 加RED/GREEN：240/241/258/2402/9600边界；超max拒绝；bool/NaN/负步拒绝；dt≠1/480拒绝；误差包络外的stamp拒绝；正确两种累积模式；episode elapsed从实际Reset起点；sdk absolute不归零；constructor rod未注册字段保留None/clock_valid=false，不造SDK0。
- [ ] 运行clock测试预期全PASS，git diff --check，commit feat: add bounded actual clock policy v2。

## Task 2：完整独立Reset源链，显式依赖注入

**Files:** Create tabletop_dual_hand_reset_v2.py、tabletop_mesh_reset_order_v2.py、tabletop_mesh_reset_readback_v2.py、tabletop_mesh_reset_numeric_state_v2.py、tests/test_mesh_reset_v2.py。冻结对应四个v1文件。

- [ ] 先在tests/test_mesh_reset_v2.py写import新RebuiltMeshResetV2应RED；另外保留旧readback validator对258拒绝的测试，证明不只order._clock有240限制。运行CPU前缀加该文件留实际RED。
- [ ] 从冻结源码建立独立对应实现并记录每个来源SHA，不通过临时改旧_clock全局让旧receipt冒充v2。公开接口统一：

~~~python
# 下列class分别定义在对应新模块；ClockPolicyV2来自Task1。
# constructor将clock_policy存入self；所有run/read/validate都使用它。
# 输入/返回保持现有owned dict结构，并增加schema/policy/reset_epoch身份。
DualHandResetTransactionV2(seed, *, episode_uid, clock_policy)
RebuiltMeshResetV2(seed, *, episode_uid, mesh_plan, clock_policy)
MeasuredMeshResetV2(seed, *, episode_uid, mesh_plan, clock_policy)
~~~

三类都有run(sim,rod,bindings,*,owner_resets)；Measured另有record_readback=None，与原API一致。order_v2复用原_plan/mesh数学与原phys动作，不能调用含v1_clock限制的原RebuiltMeshReset.run。dual_hand_v2保留原_hands/_targets/_verify_after/_seed等已验证纯helper；其run独立使用read_clock_v2，不全局替换旧_clock。所有原gain/effort/reset API与顺序不变。order_v2的_state改为实例state_reader参数（默认新numeric reader），避免旧module monkeypatch；typed capture通过同一个实例policy。

- [ ] 在readback_v2定义capture_mesh_state_v2(sim,rod,*,clock_policy)、validate_capture_v2(packet,*,clock_policy)、validate_reset_readback_v2(packet,*,reset_receipt,mesh_plan,clock_policy)。复制现有readback的11mesh current/target/quaternion/teleport、pair buffer、epoch/clock验证，替换每一处原_validate_clock到新policy。不允许只扩大order再让readback仍≤240。
- [ ] 在numeric_v2定义capture_numeric_state_v2(rope,sim,rod,*,clock_policy)、validate_numeric_reset_audit_v2(audit,*,reset_receipt,clock_policy)。复用原CAI-aware _copy_endpoint/_copy_velocity及state_differences，保存2份owned before/after typed state、真实time、原native last_error/参数、byte equality；不给新geometry view外借。audit schema/policy SHA新身份；不会调用旧order._clock或旧v1 validator。
- [ ] 测试在新文件独立复用test_mesh_reset_order.native_semantics(monkeypatch)的CPU mesh current/target/teleport模型与plan数据构造；对新order_v2 source/runtime verifier的CPU替身必须显式scope=cpu_fixture，不能重标旧native包。fixture函数完整定义在新tests/test_mesh_reset_v2.py，名字long_mesh_fixture(monkeypatch)，返回sim/rod/bindings/seed/owners/mesh_plan。复用原helper函数可以import，但不得用旧next_step（它把timestamp赋成step/480）；新增fixture advance必须由sim.step累加真实fixture时间再调用rod.step。
- [ ] 分别RED/GREEN验证：11mesh同槽重建；S0关节/腕target+rod位置/速度/director一致；反力/composer清零；clear_meshes后旧view epoch失效；mutation时clock poisoned且退出后有效；部分add_mesh/capture失败不多推进且下一步拒绝；所有owner callback失败保留主异常和单独cleanup证据。新receipt不能被v1 replay validator接受。
- [ ] 运行CPU前缀加tests/test_mesh_reset_v2.py tests/test_native_training_clock.py tests/test_mesh_reset_order.py tests/test_mesh_reset_numeric_state.py。预期全PASS；commit feat: add independent long clock reset evidence chain。

## Task 3：真正推进至边界后Reset再继续，预算不返还

**Files:** Create tests/test_native_long_reset_episode_v2.py、src/go2_pvcnn/control/dh116/tabletop_native_reset_preflight_v2.py、src/go2_pvcnn/control/dh116/tabletop_native_episode_v2.py。只读test_dh116_dual_hand_episode.setup与test_native_teacher_session预算测试。

- [ ] 定义新CPU fixture long_episode_fixture(monkeypatch,tmp_path)：以test_dh116_dual_hand_episode.setup实际CPU hand/rod/SDK对象为起点，构造新FixedS0PhysicalEpisodeV2并显式physics_budget_end=3000，record=owned rows；原sim.step每次累加DT，不把计数硬赋240/258/2402。在tabletop_native_episode_v2.py定义FixedS0PhysicalEpisodeV2(DualHandPhysicalEpisode)，constructor保留原参数并新增reset_factory、exchange_factory、clock_policy。仅独立实现原reset:44–76与_read_state_clock:78–87，以显式reset_factory创建MeasuredMeshResetV2并用read_clock_v2校验；原control/pre_action_state算法继承，旧类/default factory保持不变。reset代码必须显式使用C04现行ClearanceGuardedExchange，不因原episode模块默认FlexiblePhysicalExchange而丢失post-SDK clearance guard；原live geometry、2.1mm floor、控制/gain/作用延迟全部保留，来源未长程适配则拒绝。fixture控制仍继承原DualHandPhysicalEpisode.control，实际exchange必须是ClearanceGuardedExchange；不直接实例化裸FlexiblePhysicalExchange，不用for循环直接改rod_tick。
- [ ] 在tabletop_native_episode_v2.py定义ExchangeFactoryV2 callable协议和guarded_exchange_factory_v2，确切调用签名为(sim,rod,bindings,*,root_seed_w,read_geometry,physics_budget_end,read_geometry_after_sdk)->ClearanceGuardedExchange。guarded_exchange_factory_v2显式return ClearanceGuardedExchange(sim,rod,bindings,root_seed_w=root_seed_w,read_geometry=read_geometry,physics_budget_end=physics_budget_end,read_geometry_after_sdk=read_geometry_after_sdk)，参数与现有FlexiblePhysicalExchange.__init__:24–25一致；生产只接受该独立factory源锁及exact ClearanceGuardedExchange返回类型，不能让调用方传裸FlexiblePhysicalExchange或同名wrapper。factory来源锁记录本新模块SHA、原tabletop_deformable_clearance_guard.py SHA及其_pins():24–29原五文件inventory；创建前复核源字节与2.1mm常数，禁止通过字符串标签升格。source inventory进入v2 session/checkpoint身份。测试在真实fixture reset后断言exchange exact type，造成post-SDK低净空时原guard第三步终止且无第四步；错误factory/type/SHA在SDK推进前拒绝。
- [ ] 添加参数化语义RED（完整fixture由上一checkbox定义）：

~~~python
import pytest
@pytest.mark.parametrize("target", [240,258,2402])
def test_reset_after_real_fixture_progress_then_next_episode(monkeypatch,tmp_path,target):
    scene=long_episode_fixture(monkeypatch,tmp_path)
    scene.advance_to(target)
    before=(scene.sim.current_time_step_index,scene.sim.current_time,scene.rod.rope.time)
    receipt=scene.reset("fixture:next")
    assert (scene.sim.current_time_step_index,scene.sim.current_time,scene.rod.rope.time)==before
    assert receipt["mesh_reset"]["new_sdk_steps"]==0
    assert receipt["mesh_reset"]["borrowed_mesh_views_invalidated"] is True
    scene.control_zero()
    assert scene.sim.current_time_step_index==target+8
~~~

long_episode_fixture返回的新LongEpisodeFixture测试类在同文件定义，字段sim/rod/owners/seed/rows/clock_policy；方法advance_to(target)、reset(uid)、control_zero()。advance_to对258用32完整controls、2402用300完整controls（从setup2起），对240用29完整controls到234、然后在第6个实际substep注入原可信joint guard，得到terminal6行再Reset。guard之后恢复CPU被测joint至S0由真实Reset完成，不能跳过post-reset下一episode。

- [ ] 运行CPU前缀加tests/test_native_long_reset_episode_v2.py，确认失败落在新Reset接线/clock，不是undefined fixture。
- [ ] 实现LongEpisodeFixture方法及必要生产接口：新的ResetPreflightV2.require_reset_admitted(owner,*,clock_policy,manifest)只接受W3真正ResolvedResetManifest与Task2 v2 reader证据，输出新policy/epoch范围。它不是原C04 reset_clock_preflight修改；v1仍拒绝241/258。实际worker将来使用显式v2 owner/factory合同，未定义前不声称原C04能接受v2。
- [ ] 在新生产preflight模块定义validate_live_reset_v2(*,owner,receipt,numeric_audit,mesh_plan,manifest,clock_policy)。复用C01 live中的真实mounted/rod/backend校验，调用新v2 numeric/readback validators，不调用旧load_reset_manifest中v1 numeric分支给新receipt背书；manifest的原S0身份仍由旧loader认证原始来源，再单独比较本次v2 reset exact数值/seed/mesh。新receipt不是新的S0定义。
- [ ] 增加预算测试：复用C04 SessionBudget构造显式limits，record_setup真实setup事件，所有实际SDK推进调用charge一次；Reset不退款，已隔离window不退款，partial6只消费6solver calls/72internal substeps；下个episode预算接续。W3 phase/context/anchors/history新reset_epoch清空；view旧epoch拒绝。测试末态只用CPU计数，native_steps=0。
- [ ] 运行CPU前缀加tests/test_native_long_reset_episode_v2.py tests/test_mesh_reset_v2.py tests/test_native_training_clock.py tests/test_native_teacher_session.py tests/test_dh116_dual_hand_episode.py，预期全PASS；commit test: prove long reset continuation and budget accounting v2。

## Task 4：长程clock消费者验证门

**Files:** Test/create tests/test_native_long_clock_consumers_v2.py；只读tabletop_deformable_bimanual_step._clock:56–78、tabletop_deformable_live_geometry.LiveGeometryReader:58–68、tabletop_deformable_dual_evidence._sdk_boundary:28–34、tabletop_deformable_dual_hand_episode._read_state_clock:78–87。

- [ ] 用Task1两个确定累积模式产生240/258/2402/9600端点，不改变endpoint timestamp，调用上述真实reader记录通过或拒绝。已有abs1e-9/2e-7约束必须保留在v1。该测试把结果作为明确兼容表而非强求所有旧reader接受新时钟。静态数学已算：double dt重复累加≤9600最大误差2.433608869978343e-12，全部落旧1e-9内，若实际SDK确认该模式就保留原连续control reader。FP32dt转Python float模式则240/258误差约2.608e-8/2.803e-8、2402约2.610e-7、9600约1.043e-6；这只是候选模式数学，不能当实际native失败结论。先修literal240的Reset/readback/order/numeric/preflight，避免因看到小abs便改整个timestamp生态。
- [ ] 对新v2 factory设置clock_consumers_ready字段，由兼容表和当前source SHA计算，不接受调用方True。只有实际SDK已核模式在所有经过的reader都通过，才允许发出v2 static-ready；否则blocked reason为long_clock_consumer_incompatible:<module.function>。新reader本身通过不足以解除该门。此项不复制整个物理/动作栈或放宽保护代码；若真实源触发该门，root按具体失败函数另审独立clock-consumer adapter变更，不让实现agent擅自扩大W5到全套物理核心。
- [ ] 加入测试：即使新Reset成功，consumer表存在拒绝仍native_ready=false；篡改表SHA拒绝；无source-map拒绝；float64 fixture连续controls仍使用原路径。完整CPU前缀加tests/test_native_long_clock_consumers_v2.py，预期PASS；报告逐consumer实际结果。

## Task 5：静态/CPU闭环与原生验收交接

- [ ] 独立SPEC/QUALITY审精确commit，检查没有保留v1≤240嵌套调用、旧全局binding、人工timestamp归一、提前释放view或预算返还。记录source-map中native时钟模式验证结果；未证实仍拒绝实际worker。
- [ ] 按scripts/run_dh116_work_packages.Repository(root).verify_protection()核528+2；git diff --check。保留旧clock/Reset全部回归与v1拒绝行为。
- [ ] 编写私有native验收请求数据（不执行）：冻结commit/新source/profile/S0 SHA、240/258/>2400三边界、每边界下一episode8steps、setup/Reset/solver/substeps/iterations、最大总SDK/solver/wall/log bytes/资源。预算必须计Close/Lift/完整5s，不能只给2400总步。root取得单独批准并有C06真实worker后方可原生执行；无GPU/App/下载/安装/push。
- [ ] CPU报告与native/严格grasp报告分开。通过长Reset不等于真实抓起，不改变480/60/12/8、24D和30mm/5s/.5mm。交接W6/W7分项profile前置，性能倍数留待设备计时。

## 后继注册位置：V2-I01

独立Reset链/下一episode通过后，规划项native.fixed_s0.v2.integration（V2-I01）在W3/W4/W5之后、W6/W7之前把新session/collector/checkpoint/admission/实际worker接入；文件、入口签名/完成门见spec“后继集成包 V2-I01”。不改旧C04 exact-type令新episode/epoch假装旧owner。V2-I01显式传已锁guarded_exchange_factory_v2，贯通同owner/history/phase/source/reset_epoch，验证原C04 post-SDK clearance第三步终止无第四步。C06真实provider/全来源/预算未齐仍拒实际执行，三组件局部CPU通过不代表工厂完成。
