# DH116 Actual Native Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 接通真实、有限、可恢复的 Pinch approach_wiring_smoke_v1 Teacher A→B 更新链，包含父调度器授权交接、真实双手/柔性杆来源、自然退出和独立进程恢复；不把工程更新成功等同抓取成功。

**Architecture:** 三个独立交付包 L01→L02→L03。保留 C06 CPU-only v2 recipe/validator/lease，新增严格 actual-execution companion v1 和 scheduler 小分支；执行复用 C04、C05、原 PPO、原 native_instance_lock 和 R06。模型、动作、物理和旧配方不重写。

**Tech Stack:** Ubuntu、Python 3.10、Linux AF_UNIX/SO_PEERCRED、/proc、fcntl FLOCK、现有 scripts/python.sh/GDB、Isaac/PhysX 现装版本、真实 DeformX RopeBatch、原 CPU Teacher/PPO/Adam/RNG。

---

## 0. 基线、资格与交付边界

- 计划基线：正式 HEAD 1b9472ac1064a68afbc97dadbde550d4643ba2be。
- C06依赖现已正式完成：code ce5af9bb650627a252e40c7c397c8efca1d25555，经Windows发布；manifest8193f9b9a00465f6e01df1de17d8b1fc1b425b76。QUALITY exact5d PASS；root CPU回归269 passed / 0 failed / 0 deselected，日志SHA e04c322275801259919fb3a1d94dac672f798436b95da4ce3219f4cc713c5681。L01从ce5正式代码实施；此前50746ec仅作为历史候选记录，下面历史接口审计SHA保留。C06此项是CPU软件资格，不表示ActualProvider/L02/L03或任何原生运行已经完成。
- 实际接口审计冻结：正式1b6daefceee73e6e5e7246afd125fac4597d4fb9、C06 0fce55e4479db9b5c581f385b23de1fa003e933e。32文件SHA/CPU证据在私有evidence/native-provider-feasibility-20261115/sources.json、feasibility-report.md。正式接口变化只更新相应步骤，不重复已通过工作。
- 已读正式 fixed-S0 设计 docs/superpowers/specs/2026-11-15-dh116-fixed-s0-v2-design.md。context51/新阶段/slip/长Reset是独立V2-I01，不静默送进本计划C04/C05 exact-type路径。先接现smoke，未来V2-I01复用合格actual provider/launch。
- 本计划只开发CPU软件和原生执行准备，没有新原生预算。A、B各1环境、additional1、setup2+32×8=258 SDK的新配方需实现后完整冻结并取得实际预算授权。完整Teacher/三抓法/5秒Hold另计，258步不等于5秒。
- 当前C06 artifacts的physical N>=3/model41 CPU隔离和nativeFalse保留。实际来源门必须证明真N41及同snapshot→model，不改bool绕过。
- 528保护文件、2历史Checkpoint、旧C01–C06/PPO/24D/371D账本不覆盖。模型只取已验证66公共字段+真实rod41+当前context，未知305槽保持NaN/false。
- 已验证真实S0小manifest SHA a39de62de8ea97b3b154af2f910ba9dc097ccf64f9d650dad627ee2d8516cdce，41材料节点。不重复原621MB全量审计/不重做R08；沿sealed R08和现load_reset_manifest执行每次准入必需的原始来源及52MB材料校验。不得以cache、曾验证或只读小manifest跳过现loader要求的真实字节验证。

### 0.1 不可误修的事实

1. scheduler不代持生产锁，旧entry在Python inferior本人进入native_instance_lock；真实CPU/GDB self与parent负例已通过，严格PID条件保留。
2. _native_child最后覆盖CUDA_VISIBLE_DEVICES=str(recipe budget gpu_index)，CPU父空mask不会直接传给native子。补设备计划/验证，不宣称修复“空mask继承”。
3. build_reset_owned已sim.reset且episode.reset，不能直接供给C04的new episode。
4. R06 output/ownership.jsonl为LifetimeTrace，C06同名为raw journal派生，必须分目录。
5. 当前合法v2不允许debugger字段，新actual合同必须显式绑定原GDB策略。

### 0.2 最少版本化方案：execution companion，不复制v3 trainer

新增 dh116-actual-native-execution-v1 companion；未来Registry package新增actual_launch={path,sha256}，绑定同一现有C06 v2 core recipe。

- 原v1/v2 JSON、validator、lease不改；无actual_launch保持旧分支、指纹算法及fixture预期。
- actual approval identity独立组合：core command_spec_sha256、core code_config_sha256、companion fileSHA、实际launch source closureSHA。旧core批准不能批准actual。
- companion精确字段：schema、core_recipe_identity、source_closure_sha256、role、device_plan、debugger、provider、checkpoint_origin。
- role仅a_fresh/b_resume；debugger必须gdb_on_segv，调用原native_debugger_wrapper；不靠隐含flag。
- device_plan精确为physical_gpu_index、cuda_visible_devices、logical_device、experience_path、experience_sha256。首版批准物理GPU3→实际mask字符串3→单可见logical cuda:0；逻辑0须实际枚举核验，不能把任意GPU硬写0。
- provider精确为受审相对模块path、entrypoint、SHA；L02仅唯一tabletop_native_teacher_actual_provider.run_a，L03才允许run_b；不接受任意import字符串/caller callable。
- A checkpoint_origin=null；B为真实A package+receiptSHA，取得真实A后才冻结，不能预造A路径。
- companion不包含自身SHA或组合identity，避免自引用；自身文件SHA由Registry/intent绑定。所有actual launch/provider/artifact serializer/scene imports及实际执行依赖必须列入core.source_files，companion.source_closure_sha256必须等于原worker_training_source_contract核出的core teacher_contract.source_sha256；C05 source_sha256和C06 semantic的source_identity使用同一个完整闭包，不允许checkpoint仍锁minimum CPU-only文件。companion JSON执行元数据不加入core，避免core_identity↔companion循环。
- role只选择fresh或恢复的训练起源，origin另由C05 checkpoint记录；GDB仅监督进程；物理GPU mask/logical映射仅选择已批准且实核一致的执行设备，均不改变模型输入/24D/PPO/奖励/DT/solver等训练语义，因此这些执行元数据只进actual identity。若experience、物理/算法配置或producer代码改变，必须改变core相应source/config/semantic；不得借metadata名义排除。companion experience SHA还须等于core source/config已锁的实际experience，不能在core外换App行为。
- 真正A/B原生冻结前，L03实现也必须已经进入同一core完整闭包。A与B共用完全相同代码/source/semantic，差异只有companion role/origin和独立进程资源。L02早期CPU产物不跨源码变化冒充B可恢复的正式A；闭包不同就拒绝，不迁移checkpoint。
- 原C06 WorkerLease仍核core identity/预算/本人kernel PID。ActualWorkerLease包住它并附加parent-issued session；不放宽旧lease PID。ActualWorkerLease只由request_actual_launch构造，公开只读process_config、role、resume_origin、device_plan、launch_identity、holder_pid，分别来自已核SHA的parent reply与core lease；revalidate核原core lease、同SHA session文件及首次SO_PEERCRED快照对应的当前/proc PID/starttime/boot，不重连；close只关闭本地handle，禁止pickle。
- 本计划不注册新配方、不发auth/resource、不改Registry/readiness；root在实现/回归/SPEC/QUALITY后集成。

## 1. 文件职责及精确API

新增：

|文件|接口与职责|
|---|---|
|src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_launch.py|validate_actual_launch(raw,*,core_recipe,project)->dict；actual_launch_identity(core_recipe,launch_binding,source_inventory)->dict；read_launch_binding(path,expected_sha256)->ParentLaunchEnvelope；peer_process_identity(connection)->ProcessIdentity；verify_peer_process(peer,*,spawn_root,expected_boot_id)->ProcessIdentity；issue_process_overlay(*,repository,package,envelope,peer,spawn_root,lock_identity,output)->WorkerLaunchReply；serve_actual_launch_once(listener,*,repository,package,envelope,spawn_root,deadline)->WorkerLaunchReply；request_actual_launch(*,binding_path,binding_sha256,config,repository,package)->ActualWorkerLease；ActualWorkerLease.revalidate()/close()。|
|scripts/train_dh116_native_teacher_worker.py|main(argv=None)->int；原C06限定argv解析、本人lock内握手；依真实role调用唯一run_a/run_b；不self授权。L01只交付解析/握手，缺provider明确非零退出。|
|src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_provider.py|build_actual_episode(contract,*,sim,owners,journal,lease,output)->NativeEpisodeOwner；build_actual_factories(*,sim,owners,journal,lease,output,resume)->NativeFactories；run_a(*,config,lease,output)->dict；run_b(*,config,lease,output,resume)->dict；_run_actual_update(*,config,lease,output,resume)->dict；validate_actual_device(*,lease,sim,hands,rod)->dict；new_actual_source_capability(*,lease,backend_profile,sim,hands,rod,episode,journal,static_contract,setup_receipt)->ActualSourceCapability；observe_source_pack(*,capability,owner,model,pre_action,raw_rod,parameters,context,packed)->SourcePackWitness。|
|src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_artifacts.py|derive_actual_artifacts(*,journal_path,output,training_result,lifecycle_dir)->dict；freeze_actual_artifact_bytes(spec,output)->ActualArtifactBytes；audit_actual_training_result(result,*,role,origin,artifact_sha256)->dict；audit_actual_update_journal(rows,result,*,role,origin)->dict；audit_actual_physical_rows(rows,*,expected_policy,source_pack_rows)->dict；audit_actual_artifacts(*,frozen,spec,role,origin,repository,package)->dict；validate_actual_execution(*,repository,package,launch_binding,output,outer_receipt)->dict；read_a_resume_origin(*,repository,package_name,receipt_sha256,semantic_sha256)->dict。|
|tests/test_native_teacher_actual_launch.py|companion、真实intent、peer/kernel租约、无self授权、设备、旧分支。|
|tests/test_native_teacher_actual_launch_process.py|临时AF_UNIX/FLOCK、真实普通Python与原GDB，0App/GPU。|
|tests/test_native_teacher_actual_provider.py|new场景/单journal/2setup+0Reset+256、同source、异常owned矩阵；明确CPU double。|
|tests/test_native_teacher_actual_artifacts.py|完整raw、artifact隔离、真41关联、outer/GDB/source拒绝。|
|tests/test_native_teacher_actual_resume.py|真实A receipt、独立B进程freshReset/restore/update；CPU不授native资格。|

只小范围修改scripts/run_dh116_work_packages.py：
- Repository.actual_launch(package)->dict或None：严格读companion及SHA，原core validator不动。
- Repository.native_execution_identity(package,*,recipe=None)->dict：统一返回core_identity、actual_identity或None、attempt_fingerprint；无companion的attempt_fingerprint保持原code_config_sha256，actual为组合identity的canonical SHA。
- Repository.resolve_actual_checkpoint_origin(launch,*,packages)->dict或None：A返回None，B仅L03后调用read_a_resume_origin；L01/L02收到B明确actual_resume_not_implemented。
- Repository._verified_actual_native_receipt(name,scope,recipe,expected,*,launch)->Path：actual来源/预算/outer/GDB/role审计，不能先走旧A-only通路。
- Repository._read_native_receipt_bytes(name,*,attempt_fingerprint,expected)->tuple[Path,bytes]：从原receipt读取逻辑提取“唯一实际路径＋同bytes SHA”纯低层；原旧分支行为不变。
- native_plan、register_native_attempt以及_run_registered_native_worker统一调native_execution_identity重算同一actual identity；所有native_attempts目录/已消费判据使用同一attempt_fingerprint，不能planning按compound而registration/恢复仍按core。
- _verified_native_receipt入口先读实际package；有companion立即dispatch _verified_actual_native_receipt。无companion保留旧v1/v2完整路径及fingerprint。通用resolve_native_dependency_receipt经同一入口；B additionally由resolve_actual_checkpoint_origin校验真实A，不能重新走旧A-only validator。
- _native_child增加可选actual_launch分支，复用原Popen/monitor/GDB/outer receipt；actual branch直接调用validate_actual_execution，不先调用原validate_native_artifacts(v2)。

## 1.1 交接值类型及默认拒绝门（均在上述新模块定义）

类型是表示合同，不是来源证明。所有生产factory均须重新核真实source、对象、kernel lease；只接受dict/constructor token或caller bool不能授资格。

**actual_launch.py：**
- ProcessIdentity为frozen dataclass，字段pid/starttime_ticks/uid/gid/boot_id/exe_device/exe_inode/cmdline_sha256。peer_process_identity从SO_PEERCRED和/proc取值；不从消息payload复制PID。
- ParentLaunchEnvelope为frozen dataclass，字段schema、package_name、registered_intent={path,sha256}、core_identity、actual_identity、core_source_sha256、semantic_sha256、frozen_config={path,sha256}、parent:ProcessIdentity、socket_path、nonce、issued_monotonic_s、deadline_monotonic_s、role、device_plan、checkpoint_origin。schema固定dh116-actual-parent-launch-envelope-v1。socket和nonce只本轮有效。
- make_parent_launch_envelope(*,repository,package,intent_binding,launch,socket_path,nonce,deadline)->ParentLaunchEnvelope：仅scheduler实际注册intent/批准通过后调用，重读真实字节，绑定当前server进程身份，exclusive/fsync写入本轮目录；不接受worker提供的授权dict。Popen root PID在启动后由server实际观察，不预造在envelope。
- validate_parent_launch_envelope(envelope,*,repository,package,connection,expected_binding_sha256)->ParentLaunchEnvelope：核文件SHA、真实Registry/intent/source/组合批准、parent SO_PEERCRED及starttime/boot、deadline、socket本轮路径，缺任项拒绝actual_parent_binding_invalid。此函数在entry已持原生产FLOCK、request唯一连接建立后调用；锁外只有entry对binding文件/真实intent/source及parent当前/proc静态身份的核验，不提前连接或申请lease。
- WorkerLaunchReply为frozen dataclass，字段schema、envelope_sha256、intent_sha256、actual_identity、nonce、server:ProcessIdentity、worker:ProcessIdentity、spawn_root:ProcessIdentity、lock_device、lock_inode、process_config={path,sha256}、authorization={path,sha256}、resource={path,sha256}、role、checkpoint_origin、issued_monotonic_s、deadline_monotonic_s。schema固定dh116-actual-worker-launch-reply-v1。parent由真实Popen及peer/kernel读回签发，JSON仅传绑定，不代替锁证据。
- validate_worker_launch_reply(reply,*,envelope,repository,package,connection)->dict：先复核parent peer/nonce/intent/当前worker PID/starttime/boot/inode/源和overlay digest，再调用原verify_worker_lease，返回受核core lease和owned process config。无reply或不一致拒绝actual_worker_reply_invalid，不App。
- ActualWorkerLease仅request_actual_launch用上述两个validator的结果与私有token构造；持原WorkerLease+envelope/reply SHA+owned config及握手时由SO_PEERCRED取得的kernel peer快照；grant后双方可关IPC，revalidate只复核原core lease、同SHA session文件和快照对应的当前/proc PID/starttime/boot，不重连已关listener、不把JSON peer当新来源；__reduce__拒绝pickle。源码冻结而非token本身才是资格根。issue_process_overlay只接parent内部validated envelope/peer/spawn_root，再独立读取当前intent/批准及/proc锁复核；它不以这些类型或caller bool为信任根，不开放worker self-issuer。

**actual_provider.py：**
- ActualSetupReceipt为frozen dataclass，字段owner_uid、episode_uid、sim_instance_uid、sdk_before、sdk_after、begin_raw_sequence、end_raw_sequence、begin_clock_sha256、end_clock_sha256、core_source_sha256；只由build_actual_episode实际sim.reset前后读回并写同journal后创建，要求0→2、同sim。
- ActualSourceCapability为不可序列化process-local handle，只有new_actual_source_capability能创建。它持严格ActualWorkerLease、原真实BackendProfile、sim/左右actual Articulation/实际RopeBatch和原episode的受控弱引用、同一journal、ActualSetupReceipt、resolved S0 SHA、source/method byteSHA及模型target IDs0..40；不由公开dataclass构造。
- new_actual_source_capability按实际profile loaded ABI核exact真实对象与Nenv1/Nnode41，核same sim/rod/episode、setup2、已完成C04 Reset/live、current epoch、原packing reader来源、预算和cap；缺/坏项拒actual_source_unverified，不能返回native flag。只能在build_actual_episode返回且C04完成真实live验证之后启用观察；create_native_collector的factory顺序保持不变。
- SourcePackWitness为frozen dataclass，字段schema、actual_identity_sha256、core_source_sha256、semantic_sha256、session_uid、owner_uid、episode_uid、reset_id、rod_uid、sdk_step、rod_tick、timestamp_s、solver_time_s、reset_manifest_sha256、material_ids、source_pack_raw_sequence、pre_action_sha256、raw_rod_sha256、parameters_sha256、context_sha256、model_input_sha256、packing_source_sha256、setup_sequences、live_episode_cap_s、external_deadline_monotonic_s。schema固定dh116-actual-source-pack-witness-v1。
- validate_actual_source_capability(capability,*,owner,model)->None：复核当前lease、exact真实资源/owner/epoch、target41/sourceSHA；expired/closed/foreign/missing拒actual_source_unverified。
- observe_source_pack调用该validator，记录实际同一次pre/raw/parameters/context/packed的owned source_pack原journal行；返回对应SourcePackWitness。validate_source_pack_witness(witness,*,journal_rows,decision,source_contract,setup_receipt)->dict在post-exit审计中对同bytes/raw_sequence重算digest、41材料ID、clock/cap和decision model input，缺项拒actual_snapshot_model_binding_missing。不回读GT替换输入，不二次pack改变动作。
- capability只能在完成live后签发：build_actual_episode返回原NativeEpisodeOwner；build_actual_factories将policy factory包为“C04已live→new_actual_source_capability→原NativeContextTeacher”，并按真实model建立observer。capability没有进入新的PPO观测键，不改模型算法。

**actual_artifacts.py：**
- ActualArtifactBytes为frozen dataclass，字段output_path、files:tuple[tuple[str,bytes],...]、artifact_sha256:tuple[tuple[str,str],...]。freeze_actual_artifact_bytes每文件只读一次，regular/path/row上限检查后在同immutable bytes上做全部role审计；不得解析后再重新读取另一版本checkpoint。
- ActualResumeOrigin为frozen dataclass，字段package_name、receipt_path、receipt_sha256、checkpoint_path、checkpoint_sha256、semantic_sha256、core_identity、actual_identity、loaded_updates=1、environment_steps=32、accepted_minibatch_updates、policy_weights_sha256。仅read_a_resume_origin在actual-aware receipt验证后产生；B source/semantic必须与A相同。
- role-aware审计API在L03.3列明；无companion、无source witness、无真实outer/GDB或B无ActualResumeOrigin均拒绝，不能把CPU fixture判native。

现有调用准确签名：
- prepare_scene(project,*,capture_dir,sdf_dir)、load_sdf_arrays(prepared)、validate_simulation_settings(sim,prepared)、assemble_prepared_scene(sim,project,prepared)、bind_actual_geometry(project,prepared,stage,hands,*,read_clock)、v2.external_wire_layout。
- NativeFactories(episode,policy,trainer,reward)；NativeSessionConfig(preflight,model_config,journal_path,episode_cap_s,rollout_steps,project)。
- NativeEpisodeOwner(episode,contract,*,episode_uid,owner_resets,mesh_plan,ledger,setup_events,initial_geometry,previous_reset=None)。
- create_native_collector(config,factories)；run_updates(collector,*,additional_updates=1,factories=None)。
- build_training_contract(trainer,*,scope,reward_sha256,target_sha256,reset_sha256,source_sha256,config_sha256)；save_checkpoint(path,trainer,contract)；audit_checkpoint_bytes(raw,trainer,contract,*,expected_origin=None)；checkpoint_trainer_factory(factory,path,contract_factory)。
- R06 run_lifecycle(app,*,make_sim,acquire,synchronize,clear_sim,stop,manager_is_none,freeze,output,cleanup)；它内部已经run_owned_call，不套第二层。

## 2. 工作树、CPU命令、提交门

每包Ubuntu独立工作树，先branch/HEAD/status。L01等待C06正式审查集成；L02依赖L01；L03的软件依赖L02，实际运行再依赖真实A qualified receipt。

~~~bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export CUDA_VISIBLE_DEVICES=
export DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
bash scripts/python.sh -m pytest -q -p no:cacheprovider TEST_PATH
git diff --check
~~~

日志私有保存；RED缺功能而失败→最小实现→GREEN→相关回归→原Repository.verify_protection()验证528+2→SPEC/QUALITY→explicit git add→commit。Ubuntu不push，root通过Windows导入同commit发布并核GitHub SHA/PR Draft。失败记录真实数量，没运行不写PASS；全局Registry/readiness/README由root集成避免冲突。

## L01：parent-issued实际启动握手（首先实现）

### L01.1 companion组合身份/旧路径

Files：actual_launch.py、test_native_teacher_actual_launch.py；scheduler有限actual分支。

- [ ] 写RED：合法core无companion保持CPU-only；旧v1/v2 command/code identity等于历史fixture；bool/未知字段/provider错/device错/旧批准拒绝。
- [ ] 单文件pytest确认缺函数失败。
- [ ] 实现validate_actual_launch、actual_launch_identity与Repository.native_execution_identity；调用原validate_teacher_recipe_v2。旧core JSON和算法不改；新的actual core recipe必须列完整producer闭包，已有旧recipe不回写。
- [ ] native_plan/register_native_attempt/_run_registered_native_worker/_verified_native_receipt四入口都用native_execution_identity；统一actual attempt_fingerprint消费/恢复，同名core_identity/actual_identity写真实intent。执行前复读Registry/companion/source/current HEAD/批准SHA；旧approval无actual_identity拒绝。增加actual companion改变后旧receipt失效以及原core无companion旧回执仍可验的正/反测试。
- [ ] GREEN本文件及test_dh116_work_packages.py、test_native_teacher_worker_contract.py、test_native_teacher_worker_semantic.py。
- [ ] commit：feat(dh116): bind actual native launch to approved core recipe。

测试的实际fixture必须创建临时git Repository、core/companion和明确CPU审批，不能生产auth或App。关键断言：

~~~python
assert repository.native_recipe(package)["command_spec_sha256"] == historical_core_sha
with pytest.raises(ValueError, match="actual launch approval"):
    repository.start_native(package_name, old_core_only_approval)
~~~

### L01.2 AF_UNIX peer/本人kernel锁/parent overlay

- [ ] 写真实CPU子进程RED：自报PID、错SO_PEERCRED、starttime/boot变化、非本轮Popen后代、parent持锁、错inode、nonce重放、过期/超wall、intent变化拒绝。
- [ ] RED运行test_native_teacher_actual_launch_process.py。
- [ ] listener为本轮私有0700目录/socket0600、AF_UNIX SOCK_SEQPACKET、一次accept、有限消息大小/墙钟；不用daemon。SO_PEERCRED读取kernel PID/UID/GID：
~~~python
pid, uid, gid = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
~~~
- [ ] verify_peer_process沿/proc/<pid>/stat真实PPID/starttime链到本轮Popen.pid，核同UID/boot且检查前后starttime不变；实际exe/cmdline/worker源SHA匹配intent。GDB允许真实inferior后代，不把GDB本人当worker。
- [ ] entry先read_launch_binding并静态核binding SHA、真实intent/source和parent当前/proc PID/starttime/boot；随后entry本人进入原native_instance_lock，锁上下文贯穿request、App/完整run及自然清理。request只在此锁内打开唯一连接、验SO_PEERCRED并申请lease，发nonce+实际dev/inode；parent independently读/proc/locks，拒parent/GDB代持。request绝不自行with生产锁后return而释放锁。
- [ ] issue_process_overlay只能server端读取真实Repository/current package/已批准intent签发，不接受worker自报approval。原C01 auth/resource schema不变，owner PID为peer，bind starttime/boot/dev/inode/session/config digest、derived五键预算/core/source/semantic。
- [ ] auth附actual_identity/intentSHA/nonce/server peer/role/deadline由新wrapper检查；原C06必需字段原样验证。原子exclusive写flush/fsync，regular/O_NOFOLLOW且本轮路径，拒symlink/越目录。
- [ ] request_actual_launch拿reply路径/SHA，先原verify_worker_lease，再ActualWorkerLease私有token；本次kernel peer快照存入lease。grant完成后request/server各关闭自己的accepted connection，listener仅accept一次；后续revalidate查原core lease、同SHA session文件及当前/proc PID/starttime/boot，不再连接。lease.close不释放生产锁；只有entry退出贯穿全run的原锁上下文才解锁。
- [ ] 握手耗时计原outer/inner预算，0物理步；失败明确非零，不App、不自动retry、不制造shutdown_complete/native成功。
- [ ] GREEN direct-self、原GDB-self、parent负例真实CPU三进程。只用私有临时锁；CPU fixture显式设置对应读取路径，不触生产lock，不把fixture发行物授native资格。
- [ ] commit：feat(dh116): issue worker leases through scheduler peer handshake。

server与原child monitor在同scheduler worker：Popen后select listener/child.poll，一次accept对应唯一连接，listener随后关闭；accepted连接只供一次hello/grant，grant后关闭并继续旧monitor，无新daemon或重连。parent消失/握手失败不App；超时保留真实状态，不kill/os._exit伪自然关闭。预算只消耗不返还。

### L01.3 GDB/device/entry实际接线

- [ ] RED：父空mask仍映射3；不匹配预算GPU/多mask/logical3拒绝；合法companion实际选原GDB wrapper；孤立argv不能self授权。
- [ ] entry仅原C06 argv。环境只新增DH116_ACTUAL_LAUNCH_PATH与SHA指向真实parent binding，provider不从任意环境import。
- [ ] _native_child actual分支env/debugger取validated companion；保留原budget覆盖GPU的事实和原wrapper/start_new_session/outer&GDB验证。
- [ ] App前检查actual mask与唯一logical plan。物理identity核当前可用枚举/已装资源API；若NVML UUID可读可升级批准physical UUID→logical0，不能猜torch属性。不确定明确拒绝，不自动换GPU。
- [ ] L01 provider模块尚无时main返回actual_provider_not_implemented，不构造空provider、不App。
- [ ] CPU进程/GDB回归及528+2保护后commit：feat(dh116): launch bounded worker with explicit device and debugger contract。

L01完成只表明可信启动/租约接线软件可用，L02真实资源仍未运行；失败仅阻其后继。

## L02：真实scene与A更新保存

### L02.1 new episode与单writer

Files：actual_provider.py、test_native_teacher_actual_provider.py，entry接run_a。原scene/builder/control/PPO不改。

- [ ] RED：active build_reset_owned不能返回为new；setup真实0→2；Reset零额外SDK；journal在构造前创建，owner/collector共享同writer。
- [ ] run_a严格lock/ActualWorkerLease成功后延迟import；沿旧真实entry：扩展路径/binary验证→AppLauncher→sim/scene；prepare_scene函数在App后，不能把模块可CPU导入当函数可App前调用。
- [ ] build_actual_episode借build_reset_owned首次episode.reset前最少构造段：prepare/assemble/settings、SDF、左右hands、FreeRodRuntime单RopeBatch、DualHandRodRuntime、真实geometry、原DualHandPhysicalEpisode；exact_state=new。
- [ ] sim.reset一次真实0→2，setup写同journal；无warmup/render/补步。DT480Hz/control60Hz/12rod子步/8迭代和原重力/材料/执行器/限位不变。
- [ ] 原BackendProfile.load/verify及profile_reader_scope覆盖所有reader；binary881、S0、USD/SDF/link原锁完整验证。实际device在sim/hands/rod全部核一致，SDK和rod不能各在另一张卡。
- [ ] factories.episode用真实mesh_plan/setup_events/initial_geometry/previous_reset构造NativeEpisodeOwner，不能预造Reset回执。
- [ ] policy=原NativeContextTeacher，trainer=原DeformablePPOTrainer，reward=原TrainingReward，参数按正式签名；C04顺序static→Reset→live→model→trainer→reward，未live不得先模型。
- [ ] sampled action由真实policy进入原owner.control；不使用旧冻结动作schedule/mean/MPC冒充on-policy。
- [ ] GREEN构造/异常矩阵及C04 session/collector/model/reward回归；commit：feat(dh116): construct owned fresh native teacher scene。

### L02.2 actual cap/raw41/model同snapshot

- [ ] RED：raw3+model41、materialID变化、foreign owner/epoch/tick、伪setup、episode cap/wallcap错、packed hash错拒绝；CPU fixture不授nativeTrue。
- [ ] observe_source_pack旁路包实际NativePinchSources.pack，读原pre/raw/parameters/context与原pack输出，tensor逐bit一致；scoped恢复原bound method、源SHA锁定，不改变动作/观测。
- [ ] 每witness绑定真实owner/episode/reset/rod、SDK/rod tick/time、raw N41/materialIDs0..40、positions/velocities/directors/parameters原source digest、common66/305mask、rod41完整模型输入digest、原context/clock、actual cap/external wallcap、packed hash。
- [ ] 按§1.1实际实现ActualSetupReceipt/new_actual_source_capability/ActualSourceCapability/SourcePackWitness及两个validator；C04 live之后、模型采样之前启用。冻结provider核真实BackendProfile、exact原RopeBatch N1×41、actual articulation/sim/epoch及严格lease；source_pack行存实际原参数与model输入而非只存hash，派生按同bytes重算，不靠CPU dict/token/bool晋级。
- [ ] actual cap来自实际episode/task构造属性并比sources.context_state；wall来自实际SessionBudget/lease deadline。无现成属性时本包旁路持有真实构造实参并绑定实际reader来源，不伪造属性/只抄config。
- [ ] 32×8=256 measured solver calls，共3072内子步；加setup2总258。意外推进立即停止保留账本，不退预算。
- [ ] GREEN同snapshot/source/第三步故障矩阵；commit：feat(dh116): bind native source snapshots to teacher model input。

### L02.3 accepted update与独占checkpoint

- [ ] RED：仅optimizer调用、KL rollback无accepted minibatch、非finite/零权重差、未满32transition、错origin不算accepted保存。
- [ ] 原run_updates(collector,additional_updates=1,factories=actual_factories)：loaded0→target1，accepted>=1、finite非零delta，真32 on-policy。
- [ ] 原build_training_contract scope=owner_source_unqualified，原source/semanticSHA及save_checkpoint独占teacher_clean.pt。保持C05原scope，smoke不提升full Teacher/grasp准入。
- [ ] audit_checkpoint_bytes检查同一bytes模型/Adam/RNG/updates1/globalsteps32，不二次加载冒充训练。
- [ ] save后pop返回dict中的collector，close一次，清factory/native closure；返回仅pure数值/hash。
- [ ] GREEN training/checkpoint及新增update失败测试；commit：feat(dh116): save accepted native teacher A update independently。

### L02.4 R06 owned表/派生artifact/自然退出

Files：actual_artifacts.py、test_native_teacher_actual_artifacts.py。

- [ ] RED：R06/C06同名文件混写、raw丢行、最后exception保留collector、缺右view、多app.close、STOP前view存活均失败。
- [ ] 原run_lifecycle调用一次，output=output/lifecycle；acquire经其原run_owned_call给真实NativeOwners；不重复套owned wrapper。
- [ ] scoped白名单按真实新构造：left/right hands/views/composers/runtime/rod/RopeBatch/episode/sim。不能抄旧table_contact_source不存在项；容器dict/tuple/SDF arrays由NativeOwners管理，不伪weakref aliveFalse。
- [ ] 关闭：collector.close→owner恢复包装；drop factory/observer/source；原run_owned_call resetcomposer→CUDA sync→清异常frames→drop nativealiases；R06 clear_known_simulation_instance→STOP→manager/view复查→gated update→app.close。实际GPU动作仅批准原生运行。
- [ ] App.close后仅写纯数字marker，不自称outer exit0；scheduler观测真实outer/GDB退出。不得os._exit/SIGTERM/SIGKILL。
- [ ] derive_actual_artifacts保留原training_events全部序号，physics256/transitions32/ownership派生raw_sequence引用；root七artifact沿C06 schema。R06完整文件在lifecycle子目录独立hash，由additional实际witness绑定。
- [ ] 原validate_teacher_artifacts_v2及checkpoint_from_a_receipt旧A路径不改且仍CPU nativeFalse。新actual分支必须从freeze_actual_artifact_bytes开始，用§L03.3的role-aware低层与C05同bytes审计；B不先经过旧A-only generic validator。actual A/B同走新actual审计入口，额外验证source/41/同snapshot/cap/owned/预算/真实outer&GDB，构造独立actual结果，不改旧dict bool。
- [ ] native_execution_qualified与teacher_admission/grasp_qualified分开；smoke可证明更新链，teacher/grasp仍false。
- [ ] GREEN C06、R06/cleanup、provider异常矩阵；528+2/sourceclosure；commit：feat(dh116): seal actual teacher evidence with natural owned teardown。

原事件矩阵256×4+32×3=1120，training_events现cap2048余928含setup/reset/source/训练/保存/清理/失败。实际完整事件超cap必须先提交有限schema变更审查，不删/compact事件、不扩物理预算。原生配方只准备不运行。

## L03：真实A回执→独立新进程B恢复

### L03.1 A来源解析/B实际审批

- [ ] RED：A未passed、receiptSHA错、nativeFalse、outer!=0/GDB signal、checkpointbytes变化、semantic/source错、预造路径拒绝。
- [ ] read_a_resume_origin不调用旧checkpoint_from_a_receipt。它读真实A Registry且要求actual companion role=a_fresh，经Repository._verified_native_receipt→_verified_actual_native_receipt取唯一真实路径/已核receiptSHA，再核actual A/source及同bytes checkpoint；返回§1.1 ActualResumeOrigin。旧helper保留供旧A-only路径，不进入新actual依赖解析。
- [ ] 真实A回执出现后才冻结B companion role=b_resume与checkpoint_origin；B actual审批绑定origin，A批准不能代B。
- [ ] parent B握手重读A receipt和checkpointSHA，origin只读传parent-issued reply，不加任意--checkpoint破坏core argv。
- [ ] GREEN CPU receipt fixture（nativeFalse）后commit：feat(dh116): bind resume launch to qualified A receipt bytes。

### L03.2 freshReset后真实load/additional1

- [ ] RED：同进程假B、A物理对象/episodeUID复用、load早于Reset/live、helper计数假更新、Adam/RNG丢失、覆盖A失败。
- [ ] run_a调用_run_actual_update(config=config,lease=lease,output=output,resume=None)，run_b调用同一函数并传已核resume；_run_actual_update独占整个App/R06/工厂/更新/保存路径，仅resume origin与trainer factory不同；新进程/App/scene/episode/Reset/live，不恢复物理状态。
- [ ] 原checkpoint_trainer_factory(factory,verified_A_path,contract_factory)包trainer；C04保障Reset/live/model后恢复policy/Adam/RNG/LR/global counters，合同不兼容拒绝。
- [ ] run_updates additional1从1→2/globalsteps32→64；本进程仍32transitions/256measured/setup2。
- [ ] 独占B目录teacher_clean.pt，expected_origin=真Asha；A bytes前后SHA不变。
- [ ] 两真实CPU进程fixture：A实际优化/保存/自然退出→独立B fresh fixtureReset/restore/更新；检查非零权重/Adamstep/RNG，nativeFalse，不人工改counter。
- [ ] GREEN新增resume及C05/C06 save/load回归；commit：feat(dh116): execute fresh-process additional teacher resume update。

### L03.3 B审计及申请准备

- [ ] RED：A-only validator误收B、缺origin、A/B共享output、foreign source witness、未自然退出拒绝。
- [ ] Repository._verified_native_receipt在任何旧v2 artifact调用之前按actual_launch dispatch；_verified_actual_native_receipt先_read_native_receipt_bytes定位actual attempt_fingerprint目录并核唯一同bytes SHA，再复核intent/approval/actual identity/current core source、真实outer lock/GDB/预算，直接调用validate_actual_execution。泛用resolve_native_dependency_receipt和native_plan重验同走该dispatch。
- [ ] 新validate_actual_execution调用freeze_actual_artifact_bytes、audit_actual_training_result、audit_actual_update_journal、audit_actual_physical_rows、C05 audit_checkpoint_bytes；不能调用旧audit_teacher_artifacts_v2/validate_teacher_artifacts_v2/checkpoint_from_a_receipt，也不能通过临时改counter/origin使B适配A。
- [ ] audit_actual_training_result(result,role,origin,artifact_sha256)：A严格loaded0/target1/updates1/envsteps32，expected_origin={kind:fresh}；B严格loaded1/target2/updates2/envsteps64，expected_origin={kind:resume,checkpoint_sha256:ActualResumeOrigin.checkpoint_sha256}。两组分别有positive test，A-only拒B也必须保留。保留原run_updates语义：training和accepted原行的accepted_minibatch_updates是累计；metrics.minibatch_updates是本次新增。B累计必须等于ActualResumeOrigin.accepted_minibatch_updates＋本次metrics新增，checkpoint累计值也相等，不能把原字段改成单次计数。
- [ ] audit_actual_update_journal(rows,result,role,origin)：保留全部原序号/操作/32×8/唯一本次accepted事件；按A policy version0→1或B1→2核origin/epoch/32 decision token和model/weights链。B collector_origin的weights须等于A同bytes审计返回的policy_weights_sha256，不只等于caller字段。
- [ ] audit_actual_physical_rows(rows,expected_policy,source_pack_rows)复用现_Sequence、_clock_row、decode_owned、_digest、_action、实际action/readback/rod低层合同。原C06 _physical_journal也硬policy_version0，故B不能调用它；新actual helper只实现参数化policy/source见证链与上述低层调用，不复制PPO/GAE/优化器或原整个artifact auditor，不改旧A函数默认语义。原_json/_jsonl/_view可直接复用；同一immutable bytes中匹配derived view与source_pack witness。
- [ ] 复用原_current_recipe_context及_cpu_reference构造只读CPU topology/Adam合同（这两者不硬编码fresh origin），用C05 audit_checkpoint_bytes(frozen checkpoint bytes,trainer,contract,expected_origin=实际role origin)。核A counters32/1或B64/2、policy version、权重/model SHA、累计Adam/accepted及raw journal；不先调用硬fresh/32的旧validate_teacher_artifacts_v2。
- [ ] read_a_resume_origin只解析actual A，不递归解析B作为A；resolve_actual_checkpoint_origin仅B查询A，形成有限A→B依赖，循环/role错立即拒绝。
- [ ] A/B两配方独立source/companion/budget；B批准只能真实A后签发，预算不自动叠加。若各获批准，两进程总516 SDK不等于抓取/5秒验收。
- [ ] 现有单实例锁A→qualified真实receipt→B，A失败只block B；同版本不自动重跑，其他CPU任务继续。
- [ ] 全CPU回归/保护/SPEC+QUALITY后commit：test(dh116): qualify bounded A to B native resume execution chain。
- [ ] 输出真实源码/semantic/companion/配方SHA及待核预算资源，不启动原生。

## 3. Source closure（必须实际全闭包）

L01 core.source_files必须纳新actual_launch、entrypoint、scheduler、scripts/python.sh、C06 contract/lease/artifacts、C01 profile/preflight及实际tests。actual companion执行元数据另SHA绑定且不进入core，避免循环；producer代码绝不可仅靠compound approval绑定而从checkpoint源闭包漏掉。
L02加provider/artifacts、C04 session/sources/collector、C05 training/checkpoint、NativeContextTeacher/TaskContext、TrainingReward及原配置、DeformablePPO/BASE_PPO、scene/v2、借builder段、FreeRod/DualHandRuntime/PhysicalEpisode、measured/numeric Reset/guard/mesh_reset_order、mesh_observation_source/parameter_endpoints、原24Dcontrol、SDF/geometry、R06lifecycle/ownership/acquisition/cleanup及installed runtime contract。
L03加A receipt reader、C05 origin/Adam/RNG decoder和B validator。
每个actual launch/provider/serializer/scene的受审transitive import实际清单都进入core.source_files，未列loaded源码在冻结前报错。原native_recipe_sources产生同一真实inventory，core.teacher_contract.source_sha256=SHA256(该inventory)，worker_training_source_contract返回这一闭包；C06 training_semantic_digest的source_identity与C05 build_training_contract的source_sha256使用它，审计逐项等值，不接受只在companion出现的producer源码。
A/B native冻结共用完整L01–L03源码，即使A未执行run_b也纳入同一文件/依赖。源码后来改变会诚实改变core/source/semantic并拒绝旧A续训；不能为“恢复”移除闭包或复用旧minimum SHA。
动态import唯一受审provider，不允许sys.path同名替代。USD/SDF/binary/experience/S0沿原source/config身份；若experience或物理算法身份变动，core也必须变，不只改companion。原load_reset_manifest仍执行必需的52MB来源验证；只免重复原621MB全审，不用缓存绕过loader。

## 4. RED→GREEN完整CPU命令

L01：
~~~bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_native_teacher_actual_launch.py tests/test_native_teacher_actual_launch_process.py tests/test_dh116_work_packages.py tests/test_native_teacher_worker_contract.py tests/test_native_teacher_worker_lease.py tests/test_native_teacher_worker_semantic.py
~~~
期望全PASS；真实CPU/GDB self/parent负例与peer失败，禁止生产锁/App/GPU。

L02（另跑L01）：
~~~bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_native_teacher_actual_provider.py tests/test_native_teacher_actual_artifacts.py tests/test_native_teacher_session.py tests/test_native_teacher_collector.py tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py tests/test_native_teacher_worker_artifacts.py tests/test_native_teacher_worker_checkpoint.py tests/test_native_teacher_policy_pinch_distance.py tests/test_deformable_grasp_runtime.py tests/test_simulation_owner_cleanup.py tests/test_dh116_physx_lifetime_frame_v2.py
~~~
期望全PASS；明确CPU doubles，import负测试无pxr/Isaac/App。真正物理仍unperformed。

L03（另跑L01/L02）：
~~~bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_native_teacher_actual_resume.py tests/test_native_teacher_actual_artifacts.py tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py tests/test_native_teacher_worker_checkpoint.py tests/test_dh116_work_packages.py
~~~
期望全PASS，独立CPU进程实际优化/恢复不授native资格。具体数量由实际输出记录，不能预填。

## 5. 实施交接

先实际实现L01，不增加算法接口或重审SDF；L02才构造native，L03才真实恢复executor。共享scheduler仅L01 owner修改，其他独立CPU开发继续。
本次仅文档；C06 QUALITY及正式CPU集成已按开头记录完成。ActualProvider/L02/L03实现、完整配方授权、实际A/B源资格与自然退出仍未完成，未执行原生。Strict30mm/5s/0.5mm、三抓法及Teacher/Student成功分开，不因计划或CPU数量宣称完成。
