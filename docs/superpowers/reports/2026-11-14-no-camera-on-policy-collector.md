# 无相机 on-policy collector 的 CPU 连接验证

日期：2026-11-14。基线：68323f1754041538e52f0813db950b88c3008af3。
独立分支：codex/teacher-onpolicy-collector-20261114。
本包没有修改原任务、PPO、Teacher、checkpoint、物理 owner 或集成工作树。

本包完成的是可执行 CPU 连接验证，以及实际入口构造前的拒绝门。
**新原生步数 0；正式训练更新 0；CPU synthetic optimizer 更新也为 0。**
CPU doubles 的 SDK/rod 计数是测试中的模拟调用，不是新原生物理证据。

## 复用与新增范围

- 新增 tabletop_no_camera_collector.py。
  actual_admission 复用实际 backend adapter 的配置/源验证，并读取原任务的 R0 coefficient_source。
  create_actual_collector 在 policy、optimizer、episode factory 调用前拒绝当前缺源输入，
  不接受 caller 的 native_ready/approved 等布尔开关。
- CPUFixtureCollector 直接调用原 DualHandPhysicalEpisode.control、
  PhysicalR0Diagnostic 的 begin/record/end、DeformablePPOTrainer 的 collect_step/record_step；
  GAE 仍是原 OnPolicyRollout.returns_and_advantages。没有复制 PPO 更新算法。
- _ObserveTerminal 仅包装 exchange.control 的 observe：
  原 episode 先保存实际 physical row 并运行一次 R0，然后把已有 R0 terminal reason 传回原控制循环。
  这解决了原 episode recorder 忽略回调返回值的连接问题；没有改旧函数。
- 新增只读 CLI verify_dh116_no_camera_collector.py、29 个新测试和本报告。

原 R0 receipt 的 training_reward 仍是 None。测试 scalar 来自显式 fixture_reward 回调，
与原诊断结算分开存储；缺失值不转成 0。原 R0 不能结算的 partial guard 窗口
只保存失败，不能给 PPO 编造一条 transition。

CPUFixtureCollector 明确是 doubles harness。其 pack_observation 回调提供带原 66/371
validity/NaN 的独立 synthetic 模型输入，不宣称这些 synthetic 张量来自 live SDK。
真实 ActualBackendMeasuredTeacher 和已知 2fea backend 不能通过此 fixture 入口。
生产路径当前没有成功构造分支；不能将测试 API 用作 native 准入。

## 时序、终止及来源

每个 control：

1. 读取同一个 pre-action，packing 前后完整内容不变，采样 action24。
2. 原 R0 begin 绑定该 pre-action 和 sampled action；原 replay 校验每个 issued target
   与实际 post SDK backend target，并校验 q/qd、名称、完整 clock 和 epoch。
3. 每个 physical row 在验证前写入 exclusive JSONL、flush、fsync。
   验证异常也保留已推进的样本，然后关闭交换器。
4. 原 reducer 每行只计算一次、每个完整 8 行窗口只结算一次；
   原 guard 或 R0 terminal 在第一行出现时立即停，不执行后续 no-op。
5. “settlement” 记录明确为 prepared_before_trainer_record；
   原 PPO record_step 成功后另写 transition_recorded，避免把 prepared 误读为 PPO 已入账。
6. 非 terminal 的 bootstrap 来自同一 post-action/pre-reset endpoint，
   包括 solver_time、policy_step、substep、run/env/episode/reset/rod 和 SDK/rod clocks。
   truncated 可以 bootstrap，但原 GAE 的 done 阻断跨 reset carry；
   terminal 的 next_value 为 0 且 bootstrap_mask=False。
7. 达到 collector 截断预算或遇错后同时关闭底层 exchange，
   直接调用 episode 的零动作也不能额外走一步。

fresh CPU fixture origin 记录 checkpoint_path/sha=None、独立模型 namespace、模型 identity、
初始权重 SHA、trainer session/policy epoch/counters；每次 collection 检查未漂移。
本包不支持把真实已恢复 checkpoint 接入 fixture、不会静默 resume。
真实新旧 checkpoint 格式隔离由既有 actual-backend checkpoint 测试继续覆盖。

本包不执行 trainer.update。未来真实 collector 需要接入合格的实际 pack_actual、
正式训练 R0 source 及经过验证的 mesh Reset/session；不能把 fixture scalar 改名即启用。
当前 source 每次 pack 重核约 72 个源文件，先前实际约 53–55 ms，是吞吐成本；
非实时仿真可以慢于 60 Hz，不应因此修改环境 dt。将来可独立验证 session begin/end 源绑定，
每步继续验证 raw identity；本包没有做该优化。

## 三抓法实际入口审计

| 抓法 | 既有实际输入可读 | 当前拒绝原因 |
|---|---|---|
| Pinch | 是，独立 Pinch adapter binding | mesh Reset 来源不合格；正式 R0 training source 缺失 |
| Tripod | 是，仅自身失败前缀的来源，不是成功 | 同上；不从 Pinch 借成功资格 |
| Power | 否 | 同上，另缺自身 actual model input evidence |

原任务 SHA：
8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63。
原正式 R0 coefficients 全为 null，coefficient_source=null。
ordinal 诊断系数不改变该结论。

Reset 阻断采用基线已记录的真实源发现：teleport=True 仅排 target，
首 step 的 False 覆盖 teleport，首 solver 可能从 add_mesh 构造 pose 插值。
历史 outer PASS 或 Pinch 234 步结果不能证明 mesh Reset 同步。

接触、举升和保持成功是正向任务验收；不应要求每次探索动作先已成功。
但真实 Reset、有效源、明确 R0 数值/未知域合同是普通 rollout 也必须具备的条件。
本包未安装新姿态、未授予抓取成功或 Teacher 资格。

## 验证

先保存失败：
模块缺失 1 failed；底层截断与 PPO commit 记录 2 failed；
持续 bootstrap solver_time/policy_step/substep 漂移 3 failed。
修复后最终 **122 passed in 45.12s**：29 新测试 + 93 相关旧测试，无 deselection。

Ubuntu 工作目录：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-onpolicy-collector-20261114

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
 tests/test_dh116_no_camera_collector.py \
 tests/test_dh116_dual_hand_episode.py \
 tests/test_dh116_flexible_physical_exchange.py \
 tests/test_deformable_r0_replay.py \
 tests/test_dh116_deformable_control_step.py \
 tests/test_dh116_actual_backend_teacher.py
~~~

实际只读入口审计：

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python scripts/verify_dh116_no_camera_collector.py \
 --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-onpolicy-collector-20261114/actual-admission-final.json
~~~

CLI exit=0 表示审计成功写出；admitted_styles=[]，不是训练准入。
policy_instances_created=optimizer_instances_created=new_native_steps=optimizer_updates=0。
已有输出使用 x 模式，不覆盖历史证据。

证据根：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-onpolicy-collector-20261114

| 证据 | SHA256 |
|---|---|
| regression-final.log | 73860b3b8820f8a9838173cc30331e89dd1afef0ad23e023ba961d9539a4dc4c |
| actual-admission-final.json | 9b57ad375aa6bfd214714ebd567513bc6d6dd9d9f458a88c68812af7529881eb |
| actual-admission-final.log | 15dbac8e9216932639afe61c54fb23231ac02dd9fa920b9ab7b2213f58262ab3 |
| cpu-two-windows.jsonl | 211c13ed01032307325ee7e4bbb2e88b2bfaf3200f44f6108c77bc454e907431 |
| cpu-first-terminal.jsonl | ea8d57cc124f330b0dde14382120904f0c0075487d30dc017d1c4aef7d0b364d |
| cpu-r0-first-terminal.jsonl | 1834471b375f185c6ffa4fcc2eddd66e5d080cb6ba5a583952250d7d93e452f3 |

三份 CPU JSONL 从最终 pytest-1660 输出只读复制归档，不重复推进测试物理。
分别保留 16 行、原 joint guard 首行 terminal、R0 instability 首行 terminal。
它们带 scope=cpu_fixture_only、独立 synthetic reward 和原 training_reward=None。

同行 agent 当时在处理优先 Reset/backend 任务，未完成本包审查；
本报告不计未完成同行审查。源码自审和上述 CPU 回归已完成，原生未运行。
