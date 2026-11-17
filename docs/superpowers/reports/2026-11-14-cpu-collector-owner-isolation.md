# CPU collector owner 来源隔离后继

日期：2026-11-14。
父提交：0634837368de443b31e9bc6e2621e16804e5aaab。
分支：codex/teacher-collector-owners-20261114。
本后继修正父提交的 CPU fixture 隔离边界；父提交不改写、不 amend。

原构造器只拒绝 ActualBackendMeasuredTeacher 和单个 2fea backend SHA。
“不是这个 backend”不能证明 CPU double。旧 c479、未知后端或隐藏 native owner
仍可能进入 API，并产生 native_steps=0 的误导性元数据。父包在发布前被阻断。

## 本次修复

新增 tabletop_cpu_fixture_owners.py 的正向、受限 CPU owner 协议：

- artifact 必须是既有 fixture 的精确 bytes b"CPU double"、长度和 SHA；
  任何其他后端 SHA 均拒绝，不依赖有限 native 后端 denylist。
- sim 和 PhysX view 必须是精确 SimpleNamespace；rod/base/exchange 必须是原纯 Python
  DualHandRodRuntime、FreeRodRuntime、FlexiblePhysicalExchange。
- solver 必须是已加载且来源核验的精确 test_dual_hand_rod_runtime.FakeRope；
  Hand、Composer、MountView、HandBinding 同样按真实类对象与固定来源检查，
  复制类名或 __module__ 不够。
- 固定 15 个 fixture/adapter 源文件 SHA，并检查 loaded fixture module 的实际路径。
  callback 必须来自这些冻结文件或本 validator；检查 defaults 和 closure 捕获对象。
- 对物理 owner 图中的字段、容器、callback closure 做有界遍历；
  任意未知 owner 类型、_native 包装、非 CPU tensor、tensor subclass、
  非 numeric NumPy/object 数组均拒绝。未知对象在读取其字段或调用 getter 前拒绝。
- 构造前、每次 collect 前、每个物理子步前重查。后者是原 control loop 外的一层
  _step 包装，finally 恢复原函数；不复制、不改变原物理循环。
- CPU journal origin 记录协议版本、fixture artifact SHA、全部 source SHA 和 validator SHA。
  数据仍明确是 CPU doubles；没有新 native、GPU 或 optimizer 调用。

该协议是可信仓库 CPU fixtures 的受限执行入口，不是恶意 Python 的安全沙箱。
不支持第三方 owner、同名任意 fake class 或来自其他文件的任意回调。
修改冻结 fixture/adapter 源时必须独立审查并更新 pins，不能用 caller bool 放行。

构造后换入 owner 在下一 collect 之前被拒绝；子步间挂入 _native.owner 在第二个 SDK
调用之前被拒绝，第一行原 CPU 样本保留。原真实 collector factory 仍拒绝
mesh Reset/R0 来源缺口，不能把此协议用于真实训练资格。

## issued_actions 来源确认

原 PhysicalR0Diagnostic._physical 在 PPO record_step 前：

1. 检查每个 action_readback.request.action24 与同 pre-action 采样动作相等；
2. 按原 6 active + 6 wrist named indices 校验 submission.expected_backend_targets；
3. 校验 post SDK backend_targets 与提交目标数值相等，并校验 q/qd、实体和时钟。

因此最终 executed_actions=sample 发生在交叉核验之后，不是只回填 sample 即视为已发出。
原 request 与 issued-target 两个损坏负例保持通过，均在首行拒绝。
backend target 回读只证明命令提交/目标来源，不证明实际关节已到目标或产生接触力。

## CPU 验证

先红：14 failed / 1 passed（1 个原已拒绝的 2fea 场景）。
15 个新增测试覆盖两个不同真实 backend hash（c479、2fea）及未知 hash、
5 处 _native.owner、Isaac/PhysX/ropesim owner 类伪装、同名 FakeRope、
无 CUDA 分配的 CUDA metadata sentinel、构造后替换、子步间替换。

focused：44 passed in 31.57s。
最终：**137 passed in 52.29s**（15 新 + 父包 122），无 deselection。
原生步数、正式训练更新、CPU synthetic optimizer 更新均为 **0**。
native/Isaac 类负例使用惰性 Python sentinel，不加载或构造真实 SDK/native 实例。

工作目录：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-collector-owners-20261114

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
 tests/test_dh116_no_camera_fixture_owners.py \
 tests/test_dh116_no_camera_collector.py \
 tests/test_dh116_dual_hand_episode.py \
 tests/test_dh116_flexible_physical_exchange.py \
 tests/test_deformable_r0_replay.py \
 tests/test_dh116_deformable_control_step.py \
 tests/test_dh116_actual_backend_teacher.py
~~~

证据根：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collector-owners-20261114

| 文件 | SHA256 |
|---|---|
| red-v2.log | 133c4c9a0a8667f6a2c7d87aaf2e7b4ac5ffda54a5ccaf6ec38e5a1c6d41a9b4 |
| focused-v2.log | 8244d1337228bdf3d23bb1eb1321ae34c896232b0366f87526a3ed6f53eeba3c |
| regression-final.log | 51883957c151157845f48eee3cf91f92f86a30b642cdf711327192da48500895 |
| verified-cpu-two-windows.jsonl | 0e63d2e72c31d2b89cb99bcea776e168251cecfd70ed96e373e950c1ac865f58 |

归档的 16 行 CPU ledger 来自最终 pytest-1669，没有为生成报告重复推进。
本后继仅更改新 collector、增加 owner validator/测试/报告。
原物理、R0、模型、checkpoint、任务/config 源均未改；没有 push。
