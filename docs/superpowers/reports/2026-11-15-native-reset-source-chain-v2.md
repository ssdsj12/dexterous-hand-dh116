# W5 Task2：独立长时钟 Reset 源链 v2（CPU）

日期：2026-11-15；基线 `1b6daefceee73e6e5e7246afd125fac4597d4fb9`。
Ubuntu 工作树 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/long-reset-source-v2-20261115`，
分支 `codex/long-reset-source-v2-20261115`。本包待 root SPEC/QUALITY、正式集成和 Windows 发布。

## 范围

仅 Task2：独立双手 Reset、11槽 mesh重建、readback、typed numeric证据与显式
ClockPolicyV2。Task1已正式发布；其文件未改。没有实施 Task3 episode/preflight/
SessionBudget、Task4全部连续消费者、新 worker、原生运行或成功资格。
CPU fixture完整推进不是新原生采集，也不是受控24D policy episode续训证据。

保留原动作、关节上下界、mimic、gain、effort、碰撞/桌面、材料、480/60Hz、
12 solver子步/8迭代及30mm/5s/0.5mm标准；旧v1源码和所有528保护文件及
两个历史Checkpoint不改。无App、GPU、Isaac、安装、下载、原生物理或PPO更新。

## v1 调用图与实际替换

旧链为 Measured.run → capture_mesh_state → order._clock(绝对≤240)；
Rebuilt.run 临时重绑 old dual._clock 后调用旧dual.run；
numeric scope又临时重绑 order._state；numeric._capture也调用旧clock；
readback._validate_clock 独立含≤240，再调用原 shared_clock。
只改顶层上限无法贯通，且会误用旧receipt身份。

新增四个模块全部独立运行：

| 文件 | 实际接口与执行 |
| --- | --- |
| `tabletop_dual_hand_reset_v2.py` | DualHandResetTransactionV2(seed,*,episode_uid,clock_policy)；原物理API与顺序逐条保留，所有实际时钟读取使用read_clock_v2；源字节/策略在构造与执行前复核 |
| `tabletop_mesh_reset_order_v2.py` | RebuiltMeshResetV2(...,mesh_plan,clock_policy,state_reader=None)；独立run与原11槽clear/add；不调旧run、不改旧_clock或_state；默认实例state_reader使用新typed numeric reader |
| `tabletop_mesh_reset_readback_v2.py` | MeasuredMeshResetV2继承新Rebuilt；capture_mesh_state_v2、validate_capture_v2、validate_reset_readback_v2均显式接clock_policy；current/target/solver、XYZW、teleport、pair容量、owner时间逐项验证 |
| `tabletop_mesh_reset_numeric_state_v2.py` | capture_numeric_state_v2与validate_numeric_reset_audit_v2；复用原CAI-aware owned复制、velocity复制、state_differences；不使用全局scope；记录两份实际typed before/after |

每类run均接 `(sim,rod,bindings,*,owner_resets)`，Measured另支持record_readback。
source_lineage与clock_policy_sha256及实际rod reset_id对应的reset_epoch
贯通Reset/readback/numeric。v1 replay不能接纳新clock/schema。
纯_seed/_hands/_targets/_verify_after/_plan数学及原物理写入含义保留；新run绝不调用旧run。
state_reader可显式DI，但自定义reader标为
`explicit_instance_state_reader_unqualified`，不伪称原CAI路径，也不授予native来源资格。

数值检查包含41节点位置/速度/XYZW、40个director+1padding、实际parameter_snapshot、
last_error0、byte equality（含signed zero）。clear前后必须保持同一rope、
实际SDK/solver时间、杆数值和参数；真实mutation clock保留False，完成后才恢复True。
返回数据owned，不外借mesh视图。borrowed_mesh_views_invalidated是锁定clear/add
源码语义，CPU epoch模型同步检查；不是本包测得的CUDA旧指针生命周期资格。

## 冻结源锁

以下每份原v1文件在本包构造和执行时读取实际SHA，纯helper依赖不靠名字保证；
Task1 clock源另锁 `90bc0e7bab5117d5724d35ea5faaee25cb378981b1f3d58815d295c34439a32b`。

| 冻结v1文件 | SHA256 |
| --- | --- |
| tabletop_dual_hand_reset.py | da35b878d6ef4d5236649e2cadd97ae1a66972ce3f0646419d69c0a6d0c6bce2 |
| tabletop_mesh_reset_order.py | 94394cd6a10c528422bda28f7ee8813587ba1ca970dff54d9eabcc41ab19b50a |
| tabletop_mesh_reset_readback.py | 53c362e365df241280caa8d89ebd821a8a541d7c62c845d4956a6c12530ca0cc |
| tabletop_mesh_reset_numeric_state.py | e02853b0e7be2d38f9887d35549f335430fc92ae7380857be931d07b7dda0346 |

运行与磁盘验证沿用实际backend/profile校验；本包未把schema、CPU fixture、
自报scope或typed receipt当成真实backend资格。完整新provider、当前ABI/源闭包
与原生授权仍属于后继门。

## 失败、资源引用与真实进度

- 部分clear/add/slot、几何/velocity/quaternion/参数更改、solver时间漂移和epoch变化立即
  失败；rod保持poisoned，原rod.step拒绝下一步；Reset transaction不可重用，不重试、不rollback。
- KeyboardInterrupt同样保存失败并保持原异常。记录回调失败单独放cleanup_failures，
  不遮盖首次owner/physics错误。两份numeric finalize遇畸形DI样本时也不遮主异常。
- 新run和_state帧中的borrowed sim/rod/bindings/最后binding显式清引用，
  没有新增gc或退出捷径。未修改冻结helper内部生命周期；真实native teardown仍需外层R06/C06验收。
- 失败packet保存独立raw failure_clock_progress（不是qualified clock），包括实际SDK前后
  step/timestamp和solver host time。若SDK2→3，new_sdk_steps=1；不可读取时保留unknown，
  绝不返还或补零。没有从solver_time反推solver调用数。
- 成功路径严格零SDK/solver新步；失败观测与源错误均不能成为成功receipt。

## CPU fixture与实际结果

`tests/test_mesh_reset_v2.py`的long_mesh_fixture复用原native_semantics低层CPU hand/
rod对象，完整补齐41节点typed state、11mesh current/target/teleport与pair容量。
source/runtime替身明确cpu_fixture；native getter的schema字符串仅用于形状/语义测试，
不代表CPU数据由CUDA测得。旧模块的_clock/_state从未被临时重绑。

从原fixture初始SDK2先完整Reset，然后按原rod.step逐步推进：
每次调用sim.step以FP32 DT累加实际fixture时间，再调用原rod.step。
0/256/2400次推进后分别在SDK2/258/2402执行新Reset，比较实际clock原值不变。
没有把step直接设置到边界，没有timestamp=step*DT，也没有硬设rod_tick。
这是Task2 source reader/零步Reset测试，不包含Task3完整control episode、预算owner或下回合验收。

审查前版本84806b1为 **199 passed，19.29s，外层exit0**：38新v2、93既有clock、68旧Reset回归。独立SPEC随后发现下述三组缺陷，因此该版本未获通过，也未发布。
日志SHA256：`092fdbec33de5b00c1da10393c97c3fdf57d305d8d134070692cd1b69c32057b`。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_mesh_reset_v2.py tests/test_native_training_clock.py tests/test_mesh_reset_order.py tests/test_mesh_reset_numeric_state.py tests/test_dual_hand_reset.py
~~~

私有证据 `E=/home/xk/.local/share/dh116-autonomous-3c80166/evidence/long-reset-source-v2-20261115`：

| 日志 | 实际结果 |
| --- | --- |
| task2-import-red | 1 failed / 1 passed（缺新模块；旧nested258拒绝仍有效） |
| task2-chain-red | 29 failed / 1 passed（完整新源链尚缺失） |
| task2-chain-first-green | 30 passed |
| task2-failure-primary-red | 5 failed（entry源检查、畸形DI主异常、interrupt记录缺陷） |
| task2-chain-second-green | 3 failed /32 passed（首次修补遗漏实际缩进，未掩盖） |
| task2-chain-third-green | 35 passed |
| task2-owned-frame-red | 2 failed（最后binding和_state借用参数未清） |
| task2-combined-regression | 198 passed，exit0 |
| task2-schema-red / task2-reader-source-red | 各1 failed（额外schema字段、自定义reader来源标记） |
| task2-final-regression | 199 passed，exit0 |
| task2-failure-sdk-progress-red | 1 failed（实际进度2→3被误记0） |
| task2-final-failure-progress-regression | 199 passed，exit0；最终源代码 |

保护/源图SHA：`dd646fe93e6b12aa92a6d700b29733c0ec2ad12b169dae6ec2500b34d61d551d`。
既有Repository.verify_protection核528源码+2历史Checkpoint通过；
额外逐字节对比四份v1源与基线、核Task1源SHA、AST和新链无旧clock/scope调用。
原始数据与Checkpoint只读，未复制入新工作树或GitHub。git diff --check与精确六文件范围由提交前核验。

## 独立 SPEC 失败与逐项修正

原候选 `84806b1ca3e758aa2d7b4cb3c05c746a1b8f2658` 未发布。
独立审查结果为SPEC_FAIL，私有 `reset-v2-spec-hzlidy9x/spec.json` SHA256：
`fbe16d53123e36450ff3d280bb094483a1c5347b52dbba5cea5ffc955ff29a0e`。
38项新测试通过没有替代语义审查；三组原始失败证据保留，不倒写为当时通过。

1. **P1：失败时solver步数仍错误初始化为0。** owner callback实际执行一次
   sim.step与原rod.step后抛错，SDK2→3、CPU实际solver_calls=1，而两个failed
   packet原来仍记new_solver_steps=0。现在未完成audit/readback的计数从unknown开始；
   失败保留new_solver_steps=None，failure_clock_progress明确
   observed_solver_calls=None / unknown_no_actual_call_counter。本接口没有生产级
   RopeBatch call-counter getter，不借CPU测试专用steps属性冒充原生计数，也不从
   solver_time或rod_tick反推调用次数。成功零步只能在完整Reset/readback时钟源链
   与两份typed状态一致性检查后记exact integer0；失败schema不能通过success replay。
2. **P2：Python相等运算把bool当数值。** 现在sample solver_time、parameter
   time/contact scalar要求有限primitive int/float且排除bool；不再用float(rope.time)
   静默把False变成0。所有numeric counts、reset epoch和零步计数要求exact int，
   包括padding_quaternion_count；mesh plan spacing/friction同样拒绝bool。
   原始v1 validator和Task1源文件均未修改，严格检查只在新v2入口。
3. **P2：cached default flag冒称实际reader来源。** 构造时封存实际callable
   binding与实际默认capture函数identity，执行前及每次_state复核。构造后或
   clear/add期间更换reader，失败后保持poison，不调用未绑定reader。
   来源标签仅按实际封存函数identity决定，调用方改写旧cachedflag不影响标签；
   明确传入的custom reader仍为unqualified，不因此得到native来源资格。

先重新复现同probes：`spec-three-findings-red.log` 实际4 failed。
随后分包验证：solver unknown 1 passed；typed负测先13 failed/10 passed，
修后25 passed；reader binding先3 failed，修后4 passed。
新增29条用例，最终新v2共67条。

42d38c3修正版整组 **228 passed，20.66s，外层exit0**（67新v2 +93既有clock +68旧Reset）；其独立QUALITY随后发现以下唯一早期错误掩盖问题，因此该版仍未发布。
`spec-fixes-full-regression.log` SHA256：
`8ff39b06af39f0016ace756faced0b057e220cfa42c6a2b6fc8dba3c6b0d957d`。
同样独立重放四个审查probe，`spec-recheck-probes.json` SHA256：
`cb3a07379de63c4404640b7914db55615e1150e5b69b7fa78c5d1ca9618c6641`：
实际solver_calls仍1，但两个失败计数均None；两个bool probe明确拒绝；
reader replacement在任何hand event/rod reset前拒绝。主异常、实际SDKdelta1、
rod poison及borrowed引用清理均保留；没有新原生、物理或PPO执行。
本次修正仍待原独立SPEC与后续QUALITY对精确新提交复审。

## QUALITY 早期plan错误掩盖修复

独立QUALITY对 `42d38c3fb5122c59f5494117c13779e80a0a84f8` 的67项新测试
通过，但新增probe发现唯一P2：Measured.run先置started=True，构造receipt时
访问缺失的mesh_plan.sha256抛KeyError，随后except访问不存在的readback_receipt，
二次AttributeError遮盖主异常，status还停在new。该版未发布。
原QUALITY JSON SHA256为
`5af80d17e8e5377ddcb1f7b8c777f4e1c571b48db7cafcdb4b42408b7ddc11ff`；
私有missing-plan probe及失败证据原样保留。

本次只移动初始化顺序：先建owned failure skeleton，mesh_plan_sha256=None，
随后started=True，再执行可能失败的plan读取。没有改后续owner/interrupt/
recorder cleanup。缺sha、None畸形plan、坏recorder，以及预造异常对象的plan
均先真实复现：`quality-early-plan-red.log` **5 failed**。
修复后`quality-early-plan-green.log` **5 passed，2.67s**：
保留原异常（含同一个异常对象）、statusfailed、零hand events/rod reset/
solver调用；clock无可读来源时new_sdk_steps/new_solver_steps均保留None，
不从初始化或名义时间补0。

当前最终相关组 **233 passed，20.55s，外层exit0**：
72新v2 +93既有clock +68旧Reset。日志
`quality-early-plan-full-regression.log` SHA256：
`501fae848feee11ec9c01e154efda15e0618849aad0d743f79a55c4a303db23b`。
执行命令与上方相同五个测试文件，没有扩大到全库。
本次仍待同一QUALITY对精确新提交复核；没有新增Task3/4、原生运行或物理资格。

## 剩余工作

Task2只证明独立软件源链及CPU模型。Task3 guarded exchange完整episode、
真实预算续接、Task4连续消费者兼容与实际SDK精度、当前真实source/固定S0、
provider/自然退出和原生长Reset资格仍未完成。Teacher/MPC/Student成功率、性能、
三抓法与原生成功标准均不能由本包升级。共享readiness/README/CHANGELOG由root集成，
本分支只提交四模块、测试和报告，交独立SPEC/QUALITY，不push。
