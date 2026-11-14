# Projected Pinch health getter：仅 CPU 原生入口准备，Reset 阻断保留

## 交付与禁止执行状态

基线 ffd41d490f4146652224da5019bd7ec99615abc5；
独立 worktree /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/flexible-health-native-prep-20261114；
branch codex/flexible-health-native-prep-20261114。

只增加独立薄 observer、operation/worker/退出后审计、测试和准备 JSON。
没有修改原 projected Pinch 的动作、材料、mesh setter、原安全保护、App 生命周期、
旧配方或旧 R0。没有 Isaac、GPU solver、训练或原生启动。

**本包不能执行原生。** 新发现的 deferred mesh reset 问题尚未解决：
set_mesh_poses(teleport=True) 仅暂存目标/tele 标志，首个 False setter 会覆盖它；
首步可能从构造旧姿态插值，而不是从 reset 后姿态开始。
此外12个 solver substep 的 mesh pose 是插值，不是全部固定在 post-SDK pose。
本包不修改这些物理路径，也不把旧234步自然退出当作 Reset 同步或接触资格。

verify 的 main 在 --execute-native 时、direct execute 在任何资源调用前，固定抛出
unresolved_deferred_mesh_reset_teleport_overwrite；发生在 mkdir/App 前。
preflight 返回 native_execution_allowed=false、mesh_reset_qualified=false 和明确 blocker。
JSON 标记 execution_preparation_only=true/native_execution_allowed=false；
没有 registry 注册、没有新的执行批准。此 blocker 无可翻转运行开关；
需要独立后继源码修复和新冻结，不能用旧234步预算重跑。

## 在线薄观察器

tabletop_health_getter_evidence.HealthEvidenceMixin 只在 super._step 完成既有
SDK、rod、geometry、table、parameter 读回之后执行：

1. 从 sys.modules 取已加载 ropesim._ropesim，并核对 package 的同一模块 alias；
   不重新 import 或选择另一 .so。
2. expected_clock 只含 table_contacts.IDENTITY 的10个共享字段。
3. 调既有 capture_solver_health，读取实际 RopeBatch.last_error；捕获前后保存
   sdk_step、实际 SDK time、rope.time、runtime.clock 和实际 instance ID。
4. 保存小型 owned packet；不执行 observe_health、R0、额外物理步、solver step、
   CUDA gather、Reset 或额外 App update。
5. 原 guard 优先。getter 非零只补 numerical_health_failure；缺失、异常、
   owner/backend/clock变化只补 health_source_invalid。getter 异常保持整数未知，
   不填0。即便新 observer 自身抛异常，也先把 error/NA 附回已完成物理行再返回。

沿用原 owner 的 journal.record→SDK clock写盘→control终止→journal.end→抛错顺序，
不能丢失完成后的第9个真实行。新 mixin 无持久 native owner字段；
finally 清除 self/action/row/backend/package 等借用局部。
正常原run_owned_call、同步、owner drop、STOP、clear、App.close路径均未替换。

scenario_scope 先进入旧 projected scope，再幂等包装当前 DirectorExchange。
嵌套 execute/run/audit 不会重复 getter，也不剥除原 ProjectionEvidenceMixin。
作用域退出恢复全部绑定。原17DOF/手、2.1mm保护和48 quiet+160 closure+24短端点阶段不变；
最后24步不宣称原成功标准的5秒Hold。

## 退出后复算和资格范围

prepare.inspect_interval 完整路径先保留原 projection、参数、全部原 guard、
geometry和journal检查，再附健康观察。
新 health terminal 前缀只按真实 journal 的 identity、步骤、end和terminal复核，
不会将最终guard清零以通过旧 partial validator。
失败前缀不授予完整物理/生命周期资格。

scheduler wrapper 保留原 outer/GDB、所有权、PHYSICS_READY clear和clock门；
只有父审计通过且232条实际同 owner/error0 packet都通过，才可给
same_clock_health_getter_qualified。当前入口固定 blocked，所以本轮没有此资格。
结果始终 live_r0_qualified=false、mesh_reset_qualified=false、
substep_contact_frame_qualified=false、hold_motion_qualified=false、
contact_success_qualified=false、teacher_admission=false。

validate_exchange 重算 saved parent terminal、actual before/after boundary、
getter/parameter owner与backend锚，再在CPU调用 observe_health。
新scope缺原始health packet不能降级成unknown后通过。
postexit R0 重放则是另一个明确offline诊断路径：
使用原 PhysicalR0Diagnostic 加独立 HealthSeparatedR0ControlStep，
以现有 check_radius_packet 对实际 director_reset 和 journal reset rod核对半径，
不从配置猜半径，不改旧模型或奖励项。

offline replay 允许**历史账本**没有getter，此时只给solver_status=None；
它不会使新原生scope通过。其他原 approach、接触、slip、Hold来源继续可blocked。
使用既有离线示例的显式1..10 CPU诊断系数，标签明确不是训练权重；
原training coefficients与training_reward仍null。
offline reward域阻断不伪装成getter来源失败，getter来源通过也不宣称live R0已通过。
本包不凭旧post-pose patch统计授予正/负接触、Reset或Teacher资格。

## 独立来源与同一预算边界

新preflight沿用原合格parameter-task parent，还额外精确锁定已完成projected receipt及
physical/result/sdk_clock字节。来源锁用于确认原上下文，不会重复授权已消耗尝试：

- projected receipt SHA：04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3
- physical SHA：9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4
- 设想后继上限仍是234总SDK=2 setup+232 measured、180秒、GPU3、env1、0 PPO。
  当前只是 blocked 准备配置，不能据此执行。
- backend仍为2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec；
  没有重新编译或改SDK/solver。

worker ENTRY/WRAPPER/VERSION/环境marker各自独立；base.__file__重绑保证
真实调度子进程重入新wrapper。App前import blocker和原worker身份检查继续保留。
配方参数和源文件SHA一致性有CPU回归，旧receipt不被当作新recipe成功。

## CPU证据

新增37项与相关旧96项，共133项通过，分批命令和完整日志保存在私有目录。
所有测试中SDK/solver都是CPU-shaped doubles或只读历史文件，不是Isaac执行。

- 14项小observer：整数0/非零、getter异常、时钟/后端改变、before/after篡改、
  原terminal优先、嵌套scope、未知observer异常留行。
- worker与postexit等最终轻量合计33项：21.80秒（前一轮24.33秒也保留）。
  真实CPU subprocess执行父scheduler→新wrapper child→stub worker→audit→receipt，
  分别覆盖接受/拒绝；无Isaac import、无GPU调用。
- 真实历史234步离线R0：1项，80.96秒。逐文件校验原SHA，29个control窗口走完，
  所有缺getter的instability raw_sum仍为None。不是填0后的伪原生数据。
- 原Python owner完整及两种失败前缀：3项，475.85秒。
  完整2+232、getter失效在SDK11留下9行，以及原source-clearance guard同时失败优先；
  原SDK clock/end与app.close证据均保留，capture调用次数正好对应实际测量行。
- 旧回归：96项，18.08秒，包含原projected scope、deferred audit、clock projection、
  all-stop GDB小CPU fixture以及原R0 health/source回归。
- 530个保护指纹通过；tracked source diff为空，只有本包新增文件。

TDD保留的红例包括缺模块/审计、CLI薄层未把execute重绑到当前entry、
观察器异常原会丢行、旧真实账本缺health_getter、Reset blocker缺失。
CLI失败由CPU ImportBlocker在isaaclab导入前拒绝，未启动App。
修复后新增测试全部转绿。

长owner套件启动后增加了最终CLI固定Reset blocker、未知observer异常兜底、
offline radius来源交叉核对和false资格标记；原owner已覆盖的正常与getter-error路径未改。
新增分支分别由最终轻量测试和真实旧账本重放验证，没有反复执行昂贵owner套件，
也不把这批CPU输出当任何原生资格证据。

主线程还独立只读重放全部232旧行：数值health232Unknown、constraint232True、
13路finite全True，总29.94秒，health+独立复算13.24秒。
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-health-v2-full-historical-audit-20261114.json
SHA47e1a4206aab4fc804013b41579b3a14664babe3c81dbb766562dc0998809634。
这是将重型观察和R0放退出后CPU的依据，绝不是把旧last_error补成0。

## 命令与恢复点

统一环境：
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1

新轻量：
scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short
tests/test_flexible_health_native.py tests/test_flexible_health_worker.py
tests/test_flexible_health_postexit.py -k 'not actual_completed'

历史真实重放：
scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short
tests/test_flexible_health_postexit.py -k actual_completed

CPU原owner：
scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short
--basetemp /home/xk/.local/share/dh116-autonomous-3c80166/evidence/health-native-prep-cpu-apjo32nn/owner-red
tests/test_flexible_health_owner.py

旧回归：
scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short
tests/test_projected_pinch_native.py tests/test_deferred_physical_audit.py
tests/test_owned_clock_projection.py tests/test_dh116_physx_lifetime_debugger_stopped.py
tests/test_r0_health_separation.py tests/test_r0_numerical_health_source.py

私有证据：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/health-native-prep-cpu-apjo32nn。
原始大账本、CPU outputs、GDB夹具日志不提交。所有失败文件保留原样。

## 源码与准备配置指纹（不是新执行授权）


- src/go2_pvcnn/control/dh116/tabletop_health_getter_evidence.py: 7858671e1abc3cffb4618661e1a7478e079fbee00227a321e96be5294ff52c02
- scripts/prepare_dh116_health_projected_pinch.py: e6600b9495b4f2ccc03ae1dce95011045954785d44d9408f1da17d267ab75f46
- scripts/verify_dh116_health_projected_pinch.py: 796b0dc184256dd97e601e4cf40cf603889e251c162584f1558aef7b7631e3c3
- scripts/run_dh116_health_projected_pinch_work_packages.py: 334113348401b6771f9c318c0d54c2ad6a951c13fae32e7a3d0f3e74b4f2d664
- scripts/validate_dh116_health_projected_pinch.py: 81026c48c3315bcf42ee7c3a3ed8445ab4f47dda1e40d7f2a8ac6de996341880
- configs/dh116_execution/deformx_health_projected_pinch_v1.json: f60ddffe9a26920f541da1ed420952dde7b32517d3e70fc8e5dddfdc2b08c7f7
- tests/test_flexible_health_native.py: 0059cd6da160dacca69e5077e1067cd4c34865b1318c5ea415c600582568406e
- tests/test_flexible_health_worker.py: 7d2e3d7df6577d19aa4e314fbeb95819619b787c59e31d3e6566155359618091
- tests/test_flexible_health_owner.py: 2fe963e55608fa7838f1618960417fd6e0ef4ba923d6415f6037cd3c13995af5
- tests/test_flexible_health_postexit.py: e4138484500cdf8585e7d7e68c9c44d4e752dd96e662decd606f71e252fe65ea

日志SHA：

- unit-worker-03.txt: fa234962c037554334fd5554dc9d8c02c46d42735da9fc7c0d137819f61eafa2
- real-replay-02.txt: b3295a42a5d73afdf001c1f8495443e434fe5f2b91fcd203dbe0a8f7f06e8b65
- owner-01.txt: a7f77357ed66fa39b3573518d0bc2c9d4a868aa024f3c17087f95e201a4c6fe5
- old-regression.txt: 0fc202f3856a2544ef52f24c010f4e8547158ee50063cab75a784148d5393af3
- protection.txt: 41edfd5ca62eac8f920c8ff24fe7c51a3d6708ca673572d503f7e3002717c874

## 后继

先完成实际11 mesh reset current/target同步和substep接触pose来源合同，
再由主线程选择独立后继是否接入此薄getter；不得修改或重跑旧234步配方。
本包不自行选择新物理实现、不降低2.1mm、不增加步数或180秒上限，
也不把CPU通过作为原生、Reset、抓取或Teacher准入。

最终轻量日志 unit-worker-final.txt SHA：a33d66cf702f718df6be4e4fcf34533ed3baee83b7f3a596b4231cae86490336
